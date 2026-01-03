import torch
import torch.nn as nn
import torch.nn.functional as F
from geoopt.optim import RiemannianAdam
from geoopt.tensor import ManifoldParameter
from lib.lorentz.layers import LorentzMLR, LorentzConv2d, LorentzAvgPool2d, LorentzeLU
from lib.lorentz.manifold import CustomLorentz
import nets.batchnorm as bn


class HEEGNetStress(nn.Module):
    def __init__(
        self,
        chunk_size: int = 124,
        num_electrodes: int = 32,
        F1: int = 16,
        F2: int = 32,
        D: int = 2,
        bnorm_dispersion=bn.BatchNormDispersion.SCALAR,
        num_classes: int = 2,
        kernel_1: int = 32,
        kernel_2: int = 16,
        dropout: float = 0.25,
        domains=None,
        domain_adaptation=True,
        device='cuda',
        dtype=torch.float64,
        lr=0.01,
        weight_decay=1e-3
    ):
        super().__init__()
        print("[MODEL] Initializing HEEGNetStress")
        self.chunk_size = chunk_size
        self.num_electrodes = num_electrodes
        self.F1 = F1
        self.F2 = F2
        self.D = D
        self.kernel_1 = kernel_1
        self.kernel_2 = kernel_2
        self.dropout = dropout
        self.domain_adaptation = domain_adaptation
        self.domains = domains or list(range(num_electrodes))
        self.device_ = device
        self.dtype_ = dtype
        self.lr = lr
        self.weight_decay = weight_decay

        self.manifold = CustomLorentz()

        # -------- Block 1 --------
        self.block1 = nn.Sequential(
            nn.Conv2d(1, F1, (1, kernel_1), padding=(0, kernel_1//2), bias=False),
            nn.BatchNorm2d(F1, eps=1e-3, momentum=0.01),
            nn.Conv2d(F1, F1*D, (num_electrodes, 1), groups=F1, bias=False),
            nn.BatchNorm2d(F1*D, eps=1e-3, momentum=0.01),
            nn.ELU(),
            nn.AvgPool2d((1,4)),
            nn.Dropout(dropout)
        )

        # -------- Block 2 --------
        self.block2 = nn.Sequential(
            nn.Conv2d(F1*D, F1*D, (1, kernel_2), padding=(0, kernel_2//2), groups=F1*D, bias=False),
            nn.Conv2d(F1*D, F2, 1, bias=False),
            nn.BatchNorm2d(F2, eps=1e-3, momentum=0.01),
            nn.ELU(),
            nn.AvgPool2d((1,4)),
            nn.Dropout(dropout)
        )

        self.ec1 = nn.Conv2d(F1*D, F1*D, (1, kernel_2), padding=(0, kernel_2//2), groups=F1*D, bias=False)
        self.lc1 = LorentzConv2d(self.manifold, F1*D+1, F2+1, kernel_size=1, bias=False)

        # -------- SPD BatchNorm --------
        bn_shape = self._bn_dim()
        print("[MODEL] BN shape:", bn_shape)

        if domain_adaptation:
            self.bn = bn.AdaMomDomainSPDBatchNorm(
                shape=bn_shape,
                batchdim=0,
                domains=torch.tensor(self.domains, dtype=torch.long),
                dispersion=bn.BatchNormDispersion.SCALAR,
                eta=1.0,
                eta_test=0.1
            )
        else:
            self.bn = bn.AdaMomSPDBatchNorm(
                shape=bn_shape,
                batchdim=0,
                dispersion=bn.BatchNormDispersion.SCALAR,
                eta=1.0,
                eta_test=0.1
            )

        self.elu = LorentzeLU(self.manifold)
        self.avpool = LorentzAvgPool2d(self.manifold, (1,4))

        # -------- Classifier --------
        self.lmlp = LorentzMLR(self.manifold, self._feature_dim(), num_classes)
        print("[MODEL] Lorentz flattened dim:", self._feature_dim())

    def _bn_dim(self):
        with torch.no_grad():
            x = torch.zeros(1, 1, self.num_electrodes, self.chunk_size)
            x = self.block1(x)
            x = x.permute(0,2,3,1)
            x = F.normalize(x, dim=-1)
            x = self.manifold.projx(F.pad(x, (1,0)))
            x = self.lc1(x)
            return x.shape

    def _feature_dim(self):
        with torch.no_grad():
            x = torch.zeros(1, self.num_electrodes, self.chunk_size)
            x = self.forward_features(x, torch.zeros(1, dtype=torch.long))
        return x.shape[-1]

    def forward_features(self, x, domains):
        x = x.unsqueeze(1)
        x = self.block1(x)
        x = self.ec1(x)
        x = self.block2(x)
        x = x.permute(0,2,3,1)
        x = F.normalize(x, dim=-1)
        x = self.manifold.projx(F.pad(x, (1,0)))
        x = self.lc1(x)
        x = self.bn(x, domains)
        x = self.elu(x)
        x = self.avpool(x)
        return self.manifold.lorentz_flatten(x)

    def forward(self, inputs, domains):
        features = self.forward_features(inputs, domains)
        logits = self.lmlp(features)
        return logits.squeeze(-1), features

    def configure_optimizers(self, lr=1e-3, weight_decay=0.0):
        return RiemannianAdam(self.parameters(), lr=lr, weight_decay=weight_decay)
