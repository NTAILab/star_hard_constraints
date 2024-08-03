import torch


def estimate_pivot(omega,
                   init_point=None,
                   lr=1.e-3,
                   max_n_iter: int = 1000,
                   eps: float = 1.e-7) -> torch.Tensor:
    assert init_point.ndim == 1
    pivot = init_point.unsqueeze(0).clone().requires_grad_(True)
    optim = torch.optim.AdamW([pivot], lr=lr)
    prev_loss = 1.e308
    for i in range(max_n_iter):
        loss = omega(pivot).sum()
        optim.zero_grad()
        loss.backward()
        optim.step()
        if abs(prev_loss - loss.item()) < eps:
            break
        prev_loss = loss.item()
    return pivot.detach()
