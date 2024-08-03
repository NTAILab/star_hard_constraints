import torch
from torch import Tensor
from abc import ABC, abstractmethod


class SDF(ABC):
    @abstractmethod
    def __call__(self, xs: Tensor) -> Tensor:
        """
        Args:
            xs: Points at which SDF is estimated, of shape (n_samples, n_features).

        Returns:
            Signed Distance Function values at each pair of inputs,
            of shape (n_samples).

        """
        ...


class ArgOp(ABC):
    @abstractmethod
    def before(self, xs: Tensor) -> Tensor:
        ...

    @abstractmethod
    def after(self, dist: Tensor) -> Tensor:
        ...


class IdentityArgOp(ArgOp):
    def before(self, xs: Tensor) -> Tensor:
        return xs

    def after(self, dist: Tensor) -> Tensor:
        return dist
    

class SDFLeaf(SDF):
    def __init__(self, arg_op: ArgOp = IdentityArgOp()):
        self.arg_op = arg_op

    @abstractmethod
    def leaf(self, xs: Tensor) -> Tensor:
        """
        Calculate leaf value.
        """
        ...

    def __call__(self, xs: Tensor) -> Tensor:
        xs = self.arg_op.before(xs)
        return self.arg_op.after(self.leaf(xs))


class SDFOp(SDF):
    ...


class SDFUnaryOp(SDFOp):
    def __init__(self, arg):
        self.arg = arg

    @abstractmethod
    def unary_op(self, value: Tensor) -> Tensor:
        """
        Apply an unary operator to the values calculated by left operand.
        """
        ...

    def __call__(self, xs: Tensor) -> Tensor:
        return self.unary_op(self.arg(xs))


class SDFBinaryOp(SDFOp):
    def __init__(self, left, right):
        self.left = left
        self.right = right

    @abstractmethod
    def binary_op(self, left_value: Tensor, right_value: Tensor) -> Tensor:
        """
        Apply a binary operator to the values calculated by left and right operands.
        """
        ...

    def __call__(self, xs: Tensor) -> Tensor:
        return self.binary_op(self.left(xs), self.right(xs))


class SDFNaryOp(SDFOp):
    def __init__(self, *args):
        self.args = args

    @abstractmethod
    def nary_op(self, *values: Tensor) -> Tensor:
        """
        Apply a n-ary operator to the values calculated by operands.
        """
        ...

    def __call__(self, xs: Tensor) -> Tensor:
        return self.nary_op(*[
            arg(xs)
            for arg in self.args
        ])


