from builtins import NotImplementedError
from enum import Enum
from typing import Tuple
import torch
import torch.nn as nn
from torch.functional import Tensor

from geoopt.tensor import ManifoldParameter, ManifoldTensor
from . import functionals
from Geometry.hyperbolic import Hyperboloid


# ============================================================
# Enums
# ============================================================

class BatchNormTestStatsMode(Enum):
    BUFFER = 'buffer'
    REFIT = 'refit'
    ADAPT = 'adapt'


class BatchNormDispersion(Enum):
    NONE = 'mean'
    SCALAR = 'scalar'
    VECTOR = 'vector'


class BatchNormTestStatsInterface:
    def set_test_stats_mode(self, mode: BatchNormTestStatsMode):
        pass


# ============================================================
# Base classes
# ============================================================

class BaseBatchNorm(nn.Module, BatchNormTestStatsInterface):
    def __init__(self, eta=1.0, eta_test=0.1,
                 test_stats_mode: BatchNormTestStatsMode = BatchNormTestStatsMode.BUFFER):
        super().__init__()
        self.eta = eta
        self.eta_test = eta_test
        self.test_stats_mode = test_stats_mode

    def set_test_stats_mode(self, mode: BatchNormTestStatsMode):
        self.test_stats_mode = mode


class SchedulableBatchNorm(BaseBatchNorm):
    def set_eta(self, eta=None, eta_test=None):
        if eta is not None:
            self.eta = eta
        if eta_test is not None:
            self.eta_test = eta_test


class BaseDomainBatchNorm(nn.Module, BatchNormTestStatsInterface):
    def __init__(self):
        super().__init__()
        self.batchnorm = nn.ModuleDict()

    def set_test_stats_mode(self, mode: BatchNormTestStatsMode):
        for bn in self.batchnorm.values():
            if isinstance(bn, BatchNormTestStatsInterface):
                bn.set_test_stats_mode(mode)

    def add_domain_(self, layer: BaseBatchNorm, domain: Tensor):
        domain_key = str(int(domain.item()))
        self.batchnorm[domain_key] = layer

    def get_domain_obj(self, domain: Tensor):
        domain_key = str(int(domain.item()))
        return self.batchnorm[domain_key]

    @torch.no_grad()
    def initrunningstats(self, X, domain):
        domain_key = str(int(domain.item()))
        self.batchnorm[domain_key].initrunningstats(X)

    def forward_domain_(self, X, domain):
        domain_key = str(int(domain.item()))
        return self.batchnorm[domain_key](X)

    def forward(self, X, d):
        """
        Domain-safe forward pass: splits batch by domain and applies corresponding batchnorm.
        """
        du = d.unique()
        X_normalized = torch.empty_like(X, dtype=X.dtype, device=X.device)

        outs = []
        idxs = []

        for domain in du:
            mask = (d == domain)
            Xd = X[mask]
            if Xd.numel() == 0:
                continue

            Yd = self.forward_domain_(Xd, domain)

            outs.append(Yd.to(dtype=X.dtype, device=X.device))
            idxs.append(torch.nonzero(mask, as_tuple=False).flatten().to(X.device))

        if outs:
            X_out = torch.cat(outs, dim=0)
            ixs = torch.cat(idxs, dim=0)
            X_normalized[ixs] = X_out

        return X_normalized


# ============================================================
# SPD BatchNorm
# ============================================================

class SPDBatchNormImpl(BaseBatchNorm):
    def __init__(
        self,
        shape: Tuple[int, ...] | torch.Size,
        batchdim: int,
        eta=1.0,
        eta_test=0.1,
        karcher_steps: int = 1,
        learn_mean=True,
        learn_std=True,
        dispersion: BatchNormDispersion = BatchNormDispersion.SCALAR,
        eps=1e-5,
        mean=None,
        std=None,
        **kwargs
    ):
        super().__init__(eta, eta_test)

        if dispersion == BatchNormDispersion.VECTOR:
            raise NotImplementedError()

        self.dispersion = dispersion
        self.learn_mean = learn_mean
        self.learn_std = learn_std
        self.batchdim = batchdim
        self.karcher_steps = karcher_steps
        self.eps = eps
        self.manifold = Hyperboloid()

        init_mean = self.manifold.zero(shape[-1])
        init_var = torch.ones((1,), dtype=init_mean.dtype, device=init_mean.device)

        self.register_buffer('running_mean', init_mean)
        self.register_buffer('running_var', init_var)
        self.register_buffer('running_mean_test', init_mean.clone())
        self.register_buffer('running_var_test', init_var.clone())

        self.mean = mean

        if self.dispersion is not BatchNormDispersion.NONE:
            if std is not None:
                self.std = std
            else:
                self.std = nn.Parameter(init_var.clone()) if learn_std else init_var.clone()

    @torch.no_grad()
    def initrunningstats(self, X):
        self.running_mean = self.manifold.frechet_mean(X, max_iter=100)
        self.running_mean_test = self.running_mean.clone()

        if self.dispersion is BatchNormDispersion.SCALAR:
            self.running_var = self.manifold.frechet_variance(X, self.running_mean_test)
            self.running_var_test = self.running_var.clone()

    def forward(self, X):
        bs, h, w, c = X.shape
        X = X.view(-1, c)

        if self.training:
            batch_mean = self.manifold.frechet_mean(X, max_iter=100)
            rm = self.manifold.geodesic(self.running_mean, batch_mean, self.eta)

            if self.dispersion is BatchNormDispersion.SCALAR:
                batch_var = self.manifold.frechet_variance(X, batch_mean)
                rv = (1.0 - self.eta) * self.running_var + self.eta * batch_var
        else:
            rm = self.running_mean_test
            if self.dispersion is BatchNormDispersion.SCALAR:
                rv = self.running_var_test

        inv_mean = self.manifold.gyroinv(rm)
        Xn = self.manifold.gyrotrans(inv_mean, X)

        if self.dispersion is BatchNormDispersion.SCALAR:
            factor = 1.0 / (rv + self.eps).sqrt()
            Xn = self.manifold.gyroscalarprod(Xn, factor)

        if self.training:
            with torch.no_grad():
                self.running_mean = rm.clone()
                self.running_mean_test = self.manifold.geodesic(
                    self.running_mean_test, batch_mean, self.eta_test
                )
                if self.dispersion is not BatchNormDispersion.NONE:
                    self.running_var = rv.clone()
                    self.running_var_test = (
                        (1.0 - self.eta_test) * self.running_var_test
                        + self.eta_test * batch_var
                    )

        return Xn.view(bs, h, w, c)


# ============================================================
# Variants
# ============================================================

class SPDBatchNorm(SPDBatchNormImpl):
    def __init__(self, shape, batchdim, eta=0.1, **kwargs):
        kwargs.setdefault('dispersion', BatchNormDispersion.SCALAR)
        super().__init__(shape, batchdim, eta=1.0, eta_test=eta, **kwargs)


class SPDBatchReNorm(SPDBatchNormImpl):
    def __init__(self, shape, batchdim, eta=0.1, **kwargs):
        kwargs.setdefault('dispersion', BatchNormDispersion.SCALAR)
        super().__init__(shape, batchdim, eta=eta, eta_test=eta, **kwargs)


class AdaMomSPDBatchNorm(SPDBatchNormImpl, SchedulableBatchNorm):
    pass


# ============================================================
# Domain-specific
# ============================================================

class DomainSPDBatchNormImpl(BaseDomainBatchNorm):
    domain_bn_cls = None

    def __init__(
        self,
        shape,
        batchdim,
        learn_mean=True,
        learn_std=True,
        dispersion=BatchNormDispersion.NONE,
        test_stats_mode=BatchNormTestStatsMode.BUFFER,
        eta=1.0,
        eta_test=0.1,
        domains: Tensor = Tensor([]),
        **kwargs
    ):
        super().__init__()

        if dispersion == BatchNormDispersion.VECTOR:
            raise NotImplementedError()

        self.dispersion = dispersion
        self.learn_mean = learn_mean
        self.learn_std = learn_std

        manifold = Hyperboloid()
        self.mean = manifold.zero(shape[-1])

        if self.dispersion is BatchNormDispersion.SCALAR:
            init_var = torch.ones((1,), dtype=self.mean.dtype, device=self.mean.device)
            self.std = nn.Parameter(init_var.clone()) if learn_std else init_var
        else:
            self.std = None

        cls = type(self).domain_bn_cls
        for domain in domains:
            self.add_domain_(
                cls(
                    shape=shape,
                    batchdim=batchdim,
                    learn_mean=learn_mean,
                    learn_std=learn_std,
                    dispersion=dispersion,
                    mean=self.mean,
                    std=self.std,
                    eta=eta,
                    eta_test=eta_test,
                    **kwargs,
                ),
                domain,
            )

        self.set_test_stats_mode(test_stats_mode)


class AdaMomDomainSPDBatchNorm(DomainSPDBatchNormImpl):
    domain_bn_cls = AdaMomSPDBatchNorm
