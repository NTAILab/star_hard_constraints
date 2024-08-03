import torch
from torch import Tensor
from typing import Optional
from functools import partial
from ..ray.marching import RaymarchingFn


def extended_space_sdf(sdf, x: Tensor, nu: Optional[float] = 1.0):
    s = sdf(x[:, 1:])
    if nu is not None:
        return torch.maximum(
            torch.maximum(
                s,
                x[:, 0] - nu,
            ),
            -nu - x[:, 0],
        )
    else:
        a = torch.abs(s)
        return torch.maximum(
            torch.maximum(
                s,
                x[:, 0] - a,
            ),
            -a - x[:, 0],
        )


def make_extended_space_sdf(sdf, nu: Optional[float] = 1.0):
    return partial(extended_space_sdf, sdf, nu=nu)


class RayMarchingExtendedSpaceGenerateInsideLayer(torch.nn.Module):
    def __init__(self, pivot: Tensor, omega, n_iter: int = 20,
                 nu: Optional[float] = None):
        super().__init__()
        # prepend zero to pivot
        self.pivot = torch.concat((torch.zeros_like(pivot[:, :1]), pivot), dim=1)
        self.omega = omega
        self.extended_omega = make_extended_space_sdf(omega, nu=nu)
        self.n_iter = n_iter

    def normalize(self, r):
        return r / torch.norm(r, dim=1, keepdim=True)

    def forward(self, r):
        """Intersect the ray with the boundary in the extended space, (n + 1)-dimensional.

        Args:
            r: Ray directions, (n + 1)-dimensional.

        Returns:
            n-dimensional points inside the original space.
        """
        r_norm = self.normalize(r)
        res_alpha = RaymarchingFn.apply(
            self.pivot,
            r_norm,
            self.extended_omega,
            self.n_iter
        )
        res_x = self.pivot + res_alpha.unsqueeze(1) * r_norm
        res_x = res_x[:, 1:]  # project into n-dimensional space
        return res_x


class RayMarchingESGIWithRayLenLayer(RayMarchingExtendedSpaceGenerateInsideLayer):
    """Calculate ray direction norm and prepend in to the feature vector.
    """
    def forward(self, r):
        norm = torch.norm(r, dim=1, keepdim=True)
        prep = torch.sum(r.square(), dim=1).unsqueeze(1)
        return super().forward(torch.cat((prep, r / norm), dim=1))
