import torch
from torch.autograd import Function


class RaymarchingFn(Function):
    """
    Note: input `ray` should be normalized!

    """
    @staticmethod
    def forward(ctx, pivot, ray, sdf_function, n_steps: int = 10):
        """Calculate max allowed shift.

        Args:
            ctx: Context.
            pivot: Ray origin.
            ray: Ray direction, has to be NORMALIZED!
            sdf_function: Differentiable Signed Distance Field function.
                          Accepts 2d array as an input, of shape (n_samples, n_features).
            n_steps: Maximum number of steps.

        Returns:
            Max allowed shift for each input ray direction.

        """
        x = pivot + ray * 0.0  # copy pivot with broadcasting
        initial_signs = torch.sign(sdf_function(x))
        total_alpha = 0
        for i in range(n_steps):
            max_step = torch.clamp_min(initial_signs * sdf_function(x), 0.0)
            x += ray * max_step.unsqueeze(1)
            total_alpha = total_alpha + max_step
        t = total_alpha
        # x = pivot + total_alpha.unsqueeze(1) * ray
        ctx.save_for_backward(pivot, ray, t, x)
        ctx.sdf_function = sdf_function
        return total_alpha

    @staticmethod
    def backward(ctx, grads):
        """Calculate gradients w.r.t. rays.

        Args:
            ctx: Context.
            grads: Input gradients of shape (n_samples, n_features).

        """
        pivot, ray, t, x = ctx.saved_tensors
        sdf_function = ctx.sdf_function
        # pivot_grads = torch.zeros_like(pivot)

        # compute gradient of sdf function at x:
        with torch.set_grad_enabled(True):
            x = x.clone().requires_grad_()
            sdf = sdf_function(x)
            sdf_grad = torch.autograd.grad(sdf, x, grad_outputs=torch.ones(len(x)), only_inputs=True)[0]
        # t (total_alpha) shape: (n_samples)
        # sdf_grad shape: (n_samples, n_features)
        # ray shape: (n_samples, n_features)
        ray_dot_sdf_grad = torch.einsum('sf,sf->s', ray, sdf_grad)  # shape: (n_samples)
        # print(t.shape, sdf_grad.shape, ray.shape, ray_dot_sdf_grad.shape)
        ray_grads = torch.where(
            (torch.abs(ray_dot_sdf_grad) < 1.e-18).unsqueeze(1),
            -t.unsqueeze(1) * sdf_grad,
            -(t / ray_dot_sdf_grad).unsqueeze(1) * sdf_grad
        )
        ray_grads = grads.unsqueeze(1) * ray_grads

        return None, ray_grads, None, None
