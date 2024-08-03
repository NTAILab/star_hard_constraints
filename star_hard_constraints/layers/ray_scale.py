import torch
from ..ray.marching import RaymarchingFn


class RayMarchingGenerateInsideLayer(torch.nn.Module):
    def __init__(self, pivot, omega, n_iter: int = 20):
        super().__init__()
        self.pivot = pivot
        self.omega = omega
        self.n_iter = n_iter

    def forward(self, r, s):
        norm_of_x = torch.norm(r, dim=1)
        r_norm = (r / norm_of_x.unsqueeze(1).detach())
        res_alpha = RaymarchingFn.apply(self.pivot, r_norm, self.omega, self.n_iter)
        scale = torch.sigmoid(s) * res_alpha.unsqueeze(1)
        res_x = self.pivot + scale * r_norm
        return res_x
