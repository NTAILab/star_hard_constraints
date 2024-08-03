from .base import *


class TranslationArgOp(ArgOp):
    def __init__(self, shift: torch.Tensor):
        self.shift = shift

    def before(self, xs: torch.Tensor) -> torch.Tensor:
        return (xs - self.shift.unsqueeze(0))

    def after(self, dist: torch.Tensor) -> torch.Tensor:
        return dist


class ScalingArgOp(ArgOp):
    """Scaling with respect to the origin (anchor) placed at 0.
    """
    def __init__(self, scale: float):
        self.scale = scale

    def before(self, xs: torch.Tensor) -> torch.Tensor:
        return xs / self.scale

    def after(self, dist: torch.Tensor) -> torch.Tensor:
        return dist * self.scale


class NonUniformScalingArgOp(ArgOp):
    """Non-uniform scaling with respect to the origin (anchor) placed at 0.

    It affects not only positions, but also rays.
    """
    def __init__(self, scales: torch.Tensor):
        self.scales = scales

    def before(self, xs: torch.Tensor) -> torch.Tensor:
        raise NotImplementedError()
        return xs / self.scale

    def after(self, dist: torch.Tensor) -> torch.Tensor:
        raise NotImplementedError()
        return dist * self.scale


class RotationArgOp(ArgOp):
    """Rotation with respect to the origin (anchor) placed at 0.
    """
    def __init__(self, mat: torch.Tensor):
        self.mat = mat

    def before(self, xs: torch.Tensor) -> torch.Tensor:
        # we are using right multiplication by rotation matrix,
        # which is equivalent to left multiplication by its inverse
        return xs @ self.mat

    def after(self, dist: torch.Tensor) -> torch.Tensor:
        return dist


class SequentialArgOp(ArgOp):
    """Apply Arg Ops sequentially (from left to right).
    """
    def __init__(self, *ops: ArgOp):
        self.ops = ops

    def before(self, xs: torch.Tensor) -> torch.Tensor:
        final_xs = xs
        for op in self.ops:
            final_xs = op.before(final_xs)
        return final_xs

    def after(self, dist: torch.Tensor) -> torch.Tensor:
        final_dist = dist
        for op in reversed(self.ops):
            final_dist = op.after(final_dist)
        return final_dist