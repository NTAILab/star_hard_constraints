from .base import *
from typing import Optional


class SDFInverse(SDFUnaryOp):
    def unary_op(self, values: torch.Tensor) -> torch.Tensor:
        return -values


class SDFIntersection(SDFBinaryOp):
    def binary_op(self, left_values: torch.Tensor, right_values: torch.Tensor) -> torch.Tensor:
        return torch.maximum(left_values, right_values)


class SDFUnion(SDFBinaryOp):
    def binary_op(self, left_values: torch.Tensor, right_values: torch.Tensor) -> torch.Tensor:
        return torch.minimum(left_values, right_values)


class SDFMultiUnion(SDFNaryOp):
    def nary_op(self, *values: torch.Tensor) -> torch.Tensor:
        args_iter = iter(values)
        current = next(args_iter)
        for val in args_iter:
            current = torch.minimum(current, val)
        return current


class SDFMultiIntersection(SDFNaryOp):
    def nary_op(self, *values: torch.Tensor) -> torch.Tensor:
        args_iter = iter(values)
        current = next(args_iter)
        for val in args_iter:
            current = torch.maximum(current, val)
        return current


class SDFTransform(SDFUnaryOp):
    def __init__(self, *args, arg_op: Optional[ArgOp] = None, **kwargs):
        super().__init__(*args, **kwargs)
        if arg_op is None:
            arg_op = IdentityArgOp()
        self.arg_op = arg_op

    def unary_op(self, values: torch.Tensor) -> torch.Tensor:
        return values

    def __call__(self, xs: torch.Tensor, rays: torch.Tensor) -> torch.Tensor:
        return self.unary_op(self.arg_op.after(self.arg(*self.arg_op.before(xs, rays))))


