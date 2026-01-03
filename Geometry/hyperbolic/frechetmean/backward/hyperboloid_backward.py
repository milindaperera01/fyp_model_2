import torch
import torch.nn as nn

from ...utils.utils_common import d2arcosh, darcosh, darcoshsq_diff

def hessian(X, y, w, K):
    """
    Compute the Hessian for the Frechet mean on the hyperboloid.

    Args
    ----
        X (tensor): points, shape [..., points, dim]
        y (tensor): mean point, shape [..., dim]
        w (tensor): weights, shape [..., points]
        K (float): curvature (must be negative)

    Returns
    -------
        hess (tensor): Hessian [..., dim, dim]
    """
    X = X.clone()
    X[..., 0] *= -1
    xlT_M_y = (X * y.unsqueeze(-2)).sum(dim=-1)

    term1 = K**2 * d2arcosh(K * xlT_M_y).unsqueeze(-1).unsqueeze(-1) * (X.unsqueeze(-1) @ X.unsqueeze(-2))

    M = torch.diag(torch.tensor([-1.] + [1]*(term1.shape[-1]-1), device=X.device, dtype=X.dtype))
    M = M.reshape((1,)*(len(term1.shape)-2) + (term1.shape[-1], term1.shape[-1]))
    term2 = (K * darcosh(K * xlT_M_y) * xlT_M_y).unsqueeze(-1).unsqueeze(-1) * M

    return (w.unsqueeze(-1).unsqueeze(-1) * (term1 - K * term2)).sum(dim=-3) / -K


def hess_term(X, y, w, K, eps=1e-3):
    """
    Compute stabilized Hessian term for backward pass.

    Uses pseudo-inverse if Hessian is singular.
    """
    H = hessian(X, y, w, K)

    I = torch.eye(H.size(-1), device=H.device, dtype=H.dtype).expand_as(H)

    try:
        Hi = torch.inverse(H + eps * I)
    except RuntimeError:
        # fallback for singular matrix
        Hi = torch.linalg.pinv(H + eps * I)

    mu = y.clone()
    mu[..., 0] *= -1
    mu = mu.unsqueeze(-1)

    num = Hi @ mu @ mu.transpose(-1, -2) @ Hi
    denom = mu.transpose(-1, -2) @ Hi @ mu
    return (num / denom) - Hi


def gradu(X, y, w, K):
    """
    Gradient of the variance on hyperboloid.
    """
    scalar = torch.zeros_like(X)
    scalar[..., 0] = 2 * torch.ones_like(X[..., 0])
    X_M = X - scalar * X
    xlT_M_y = (X_M * y.unsqueeze(-2)).sum(dim=-1, keepdim=True)

    main_term = -darcoshsq_diff(K * xlT_M_y) * X_M

    return (w.unsqueeze(-1) * main_term).sum(dim=-2)


def frechet_hyperboloid_backward(X, y, grad, w, K):
    """
    Full backward computation for Frechet mean on hyperboloid.

    Args
    ----
        X (tensor): [..., points, dim]
        y (tensor): mean [..., dim]
        grad (tensor): gradient [..., dim]
        w (tensor): weights [..., points]
        K (float): curvature

    Returns
    -------
        dx, dw, dK: gradients
    """
    if not torch.is_tensor(K):
        K = torch.tensor(K, device=X.device, dtype=X.dtype)

    with torch.no_grad():
        # Use slightly larger eps to avoid singular Hessian
        hess_t = hess_term(X, y, w=w, K=K, eps=1e-3)

    with torch.enable_grad():
        # clone variables for autograd
        X = nn.Parameter(X.detach())
        y = y.detach()
        w = nn.Parameter(w.detach())
        K = nn.Parameter(K)

        grad = (hess_t @ grad.unsqueeze(-1)).squeeze(-1)
        gradf = gradu(X, y, w, K)

        dx, dw, dK = torch.autograd.grad(
            gradf, (X, w, K), grad_outputs=grad, allow_unused=True
        )

    return dx, dw, dK
