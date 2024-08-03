import torch
from ..ray.marching import RaymarchingFn


class RayMarchingProjectOntoBoundaryLayer(torch.nn.Module):
    def __init__(self, pivot: torch.tensor, omega, n_iter: int = 20):
        super().__init__()
        self.pivot = pivot
        self.omega = omega
        self.n_iter = n_iter

    def forward(self, r):
        r_norm = (r / torch.norm(r, dim=1, keepdim=True))
        res_alpha = RaymarchingFn.apply(self.pivot, r_norm, self.omega, self.n_iter)
        res_x = self.pivot + res_alpha.unsqueeze(1) * r_norm
        return res_x
