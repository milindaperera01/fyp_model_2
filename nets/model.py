import torch
import torch.nn as nn
import torch.nn.functional as F
from geoopt.optim import RiemannianAdam
from geoopt.tensor import ManifoldParameter

from lib.lorentz.layers import (
    LorentzMLR,
    LorentzConv2d,
    LorentzAvgPool2d,
    LorentzeLU
)
from lib.lorentz.manifold import CustomLorentz
import nets.batchnorm as bn


class HEEGNetStress(nn.Module):

    def __init__(
        self,
        chunk_size=124,
        num_electrodes=32,
        F1=16,
        F2=32,
        D=2,
        kernel_1=32,
        kernel_2=16,
        dropout=0.25,
        domains=None,
        domain_adaptation=True,
        device="cuda",
        dtype=torch.float64,
    ):
        super().__init__()

        self.manifold = CustomLorentz()
        self.domain_adaptation = domain_adaptation
        self.domains = domains or []
        self.device_ = device
        self.dtype_ = dtype

        # ─────────────────────────────────────
        # Phase 1: Euclidean EEG Feature Extractor
        # ─────────────────────────────────────
        self.block1 = nn.Sequential(
            nn.Conv2d(1, F1, (1, kernel_1), padding=(0, kernel_1 // 2), bias=False),
            nn.BatchNorm2d(F1, eps=1e-3, momentum=0.01),
            nn.Conv2d(F1, F1 * D, (num_electrodes, 1),
                      groups=F1, bias=False),
            nn.BatchNorm2d(F1 * D, eps=1e-3, momentum=0.01),
            nn.ELU(),
            nn.AvgPool2d((1, 4)),
            nn.Dropout(dropout),
        )

        self.ec1 = nn.Conv2d(
            F1 * D, F1 * D,
            (1, kernel_2),
            padding=(0, kernel_2 // 2),
            groups=F1 * D,
            bias=False
        )

        self.block2 = nn.Sequential(
            nn.Conv2d(F1 * D, F1 * D,
                      (1, kernel_2),
                      padding=(0, kernel_2 // 2),
                      groups=F1 * D,
                      bias=False),
            nn.Conv2d(F1 * D, F2, 1, bias=False),
            nn.BatchNorm2d(F2, eps=1e-3, momentum=0.01),
            nn.ELU(),
            nn.AvgPool2d((1, 4)),
            nn.Dropout(dropout),
        )

        # ─────────────────────────────────────
        # Hyperbolic Projection & Alignment
        # ─────────────────────────────────────
        self.lc1 = LorentzConv2d(
            self.manifold,
            F1 * D + 1,
            F2 + 1,
            kernel_size=1,
            bias=False
        )

        self.bn = (
            bn.AdaMomDomainSPDBatchNorm(
                self._bn_dim(),
                domains=self.domains,
                dispersion=bn.BatchNormDispersion.SCALAR,
                device=device,
                dtype=dtype
            )
            if domain_adaptation
            else bn.AdaMomSPDBatchNorm(
                self._bn_dim(),
                dispersion=bn.BatchNormDispersion.SCALAR,
                device=device,
                dtype=dtype
            )
        )

        self.elu = LorentzeLU(self.manifold)
        self.avpool = LorentzAvgPool2d(self.manifold, (1, 4))

        # ─────────────────────────────────────
        # Binary Hyperbolic Classifier
        # ─────────────────────────────────────
        self.lmlp = LorentzMLR(
            self.manifold,
            self._feature_dim() + 1,
            num_classes=1   # ← binary stress logit
        )

    # ─────────────────────────────────────
    # Shape helpers
    # ─────────────────────────────────────
    def _bn_dim(self):
        with torch.no_grad():
            x = torch.zeros(1, 1, 32, 124)
            x = self.block1(x)
            x = x.permute(0, 2, 3, 1)
            x = F.normalize(x, dim=-1)
            x = self.manifold.projx(F.pad(x, (1, 0)))
            x = self.lc1(x)
        return x.shape

    def _feature_dim(self):
        with torch.no_grad():
            x = torch.zeros(1, 32, 124)
            x = self.forward_features(x, torch.zeros(1, dtype=torch.long))
        return x.shape[-1]

    # ─────────────────────────────────────
    # Forward
    # ─────────────────────────────────────
    def forward_features(self, x, domains):
        x = x.unsqueeze(1)                 # (B,1,32,124)
        x = self.block1(x)
        x = self.ec1(x)
        x = self.block2(x)

        x = x.permute(0, 2, 3, 1)
        x = F.normalize(x, dim=-1)
        x = self.manifold.projx(F.pad(x, (1, 0)))

        x = self.lc1(x)
        x = self.bn(x, domains)
        x = self.elu(x)
        x = self.avpool(x)

        return self.manifold.lorentz_flatten(x)

    def forward(self, x, domains):
        features = self.forward_features(x, domains)
        logits = self.lmlp(features)
        return logits.squeeze(-1), features

