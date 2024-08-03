import torch
from ..layers.ray_scale import RayMarchingGenerateInsideLayer


class SimpleConstrainedNN(torch.nn.Module):
    def __init__(self, domain, pivot, n_iter: int,
                 encoder, encoder_outs: int,
                 decoder, decoder_ins: int):
        super().__init__()
        self.domain = domain
        self.generator = RayMarchingGenerateInsideLayer(pivot, domain, n_iter=n_iter)
        self.encoder = encoder
        self.decoder = decoder
        self.latent_to_ray = torch.nn.Linear(encoder_outs, decoder_ins)
        self.latent_to_scale = torch.nn.Linear(encoder_outs, 1)

    def forward(self, raw_features):
        latent = self.encoder(raw_features)
        ray = self.latent_to_ray(latent)
        scale = self.latent_to_scale(latent)
        xs_inside_constraints = self.generator(ray, scale)
        return self.decoder(xs_inside_constraints)
