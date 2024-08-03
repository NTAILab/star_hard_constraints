from .base import *


class SDFSphere(SDFLeaf):
    def __init__(self, *args, radius: float = 1.0, **kwargs):
        super().__init__(*args, **kwargs)
        self.radius = radius

    def leaf(self, xs: Tensor) -> Tensor:
        return (torch.norm(xs, dim=1) - self.radius)


class SDFCube(SDFLeaf):
    def leaf(self, xs: Tensor) -> Tensor:
        return -(1 - xs.abs().max(dim=1)[0])


class SDFHalfSpace(SDFLeaf):
    """Half-space, determined by a plane normal and a bias.

    In 2d it is also called half-plane.

    """
    def __init__(self, *args, normal: Tensor = None, bias: float = 0.0, **kwargs):
        super().__init__(*args, **kwargs)
        self.normal = normal
        self.bias = bias
        self.normal_norm = torch.norm(normal)
        self.normal_normed = normal / self.normal_norm
        self.bias_normed = bias / self.normal_norm

    def leaf(self, xs: Tensor) -> Tensor:
        sdf = xs @ self.normal_normed - self.bias_normed
        return sdf
        