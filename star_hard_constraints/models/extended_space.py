import torch
from ..layers.extended_space import RayMarchingExtendedSpaceGenerateInsideLayer


class ESConstrainedNN(torch.nn.Module):
    """Ray Marching Extended-Space Constrained NN.
    """
    def __init__(self, domain, pivot, n_iter: int,
                 encoder, encoder_outs: int,
                 decoder, decoder_ins: int):
        super().__init__()
        self.domain = domain
        self.generator = RayMarchingExtendedSpaceGenerateInsideLayer(
            pivot,
            domain,
            n_iter=n_iter,
            nu=None
        )
        self.latent_to_ray = torch.nn.Linear(encoder_outs, decoder_ins + 1)
        self.encoder = encoder
        self.decoder = decoder

    def forward(self, raw_features):
        latent = self.encoder(raw_features)
        ray = self.latent_to_ray(latent)
        xs_inside_constraints = self.generator(ray)
        return self.decoder(xs_inside_constraints)
