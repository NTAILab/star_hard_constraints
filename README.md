# Imposing Star-Shaped Hard Constraints on the Neural Network Output

A problem of imposing hard constraints on the neural network output can be met in many applications. We propose a new method for solving this problem for non-convex constraints that are star-shaped. A region produced by constraints is called star-shaped when there exists an origin in the region from which every point is visible. Two tasks are considered: to generate points inside the region and on the region boundary. The key idea behind the method is to generate a shift of the origin towards a ray parametrized by the additional layer of the neural network. The largest admissible shift is determined by the differentiable Ray marching algorithm. This allows us to generate points which are guaranteed to satisfy the constraints. A more accurate modification of the algorithm is also studied. The proposed method can be regarded as a generalization of the methods for convex constraints. Numerical experiments illustrate the method by solving machine learning problems. The code implementing the method is publicly available.

## Installation

It is recommended to use an isolated Python environment, e.g. created with [conda](https://docs.anaconda.com/miniconda/).

`star_hard_constraints` is a regular old-style Python package and can be installed
in development mode with pip after cloning:

```bash
git clone https://github.com/NTAILab/star_hard_constraints.git
cd star_hard_constraints
pip install -e .
```

## Package Contents

The package consists of several submodules, including:

- `sdf` – primitives and operations for differentiable Signed Distance Field construction (it is also allowed to use arbitrary SDFs that can handle PyTorch tensors);
- `ray` – algoirthms for differentiable ray-surface intersection point estimation;
- `layers` – neural network layers that map input feature vectors to the domain (or the domain surface), determined by `sdf` using `ray` algorithms;
- `models` – simple neural networks with constrained `layers` at the end;
- `utils` – utility functions.


## Usage

Usage examples are provided in [notebooks](notebooks/).

Basically, to create a constrained neural network a user has to:

1. Define the domain (admissible set) using `sdf` subpackage.
2. Make a neural network using one of `layers` or existing `models`.

For example, basic usage of layer and predefined neural network:

```python
from star_hard_constraints.sdf import (
    SDFHalfSpace,
    SDFIntersection,
    SDFMultiIntersection,
    SDFMultiUnion,
)
from star_hard_constraints.layers import RayMarchingProjectOntoBoundaryLayer
from star_hard_constraints.models import SimpleConstrainedNN


domain = SDFMultiIntersection(
    SDFHalfSpace(normal=torch.tensor([1.0, 1.0]), bias=1.0),
    SDFHalfSpace(normal=torch.tensor([-1.0, 0.0]), bias=0.0),
    SDFHalfSpace(normal=torch.tensor([0.0, -1.0]), bias=0.0),
)

proj_onto_boundary = RayMarchingProjectOntoBoundaryLayer(origin, omega, n_iter=n_iter)
projected = proj_onto_boundary(torch.randn((100, 2)))
# projected – are on the domain boundary

n_input_features = 3
model = SimpleConstrainedNN(
    domain=domain,
    # point inside the domain (see `utils.pivot` to determine automatically):
    pivot=torch.tensor([[0.1, 0.1]]),
    n_iter=10,  # <- important to set appropriate number of iterations
    encoder=torch.nn.Sequential(
        torch.nn.Linear(n_input_features, 8),
        torch.nn.SiLU(),
        torch.nn.Linear(8, 8),
        torch.nn.SiLU(),
    ),
    encoder_outs=8,
    decoder=lambda x: x,
    decoder_ins=2,
)

interior_points = model(torch.randn((100, n_input_features)))
# interior_opints – are guaranteed to be inside the domain

```

## Citation

```
@article{konstantinov2024imposing,
  title={Imposing Star-Shaped Hard Constraints on the Neural Network Output},
  author={Konstantinov, Andrei and Utkin, Lev and Muliukha, Vladimir},
  journal={Mathematics},
  volume={12},
  number={23},
  pages={3788},
  year={2024},
  publisher={MDPI}
}
```
