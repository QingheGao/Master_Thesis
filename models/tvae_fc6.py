import torch
from torch import nn
from torch.nn import functional as F
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec
from layers.losses import KL_gaussian, KL_exp, KL_laplace,KL_gaussian_exact

class TVAE_fc6(nn.Module):
    def __init__(self, in_channels, s_dim, kernel_size,init_weights=False):
        super(TVAE_fc6, self).__init__()

        self.kernel_size = kernel_size
        self.s_dim = s_dim
        torch.manual_seed(4)
        self.z_encoder = nn.Sequential(
            nn.Linear(in_channels, self.s_dim * 2, bias=True)
        )
        torch.manual_seed(4)
        self.u_encoder = nn.Sequential(
            nn.Linear(in_channels, self.s_dim * 2, bias=True)
        )
        torch.manual_seed(4)
        self.decoder = nn.Sequential(
            nn.Linear(self.s_dim, in_channels, bias=True),
            #nn.Tanh(),
            nn.BatchNorm1d(9216)
        )

        if init_weights:
            self._initialize_weights()

        self.neighborhood_sum = nn.Conv2d(in_channels=1, out_channels=1,
                                          kernel_size=kernel_size,
                                          padding=((kernel_size[0] // 2),
                                                   (kernel_size[1] // 2),
                                                   ),
                                          stride=(1, 1),
                                          padding_mode='circular',
                                          bias=False)


        nn.init.ones_(self.neighborhood_sum.weight)
        # with torch.no_grad():
        #     self.neighborhood_sum.weight.div_(np.prod(kernel_size))
        self.neighborhood_sum.weight.requires_grad = False


    def _initialize_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.uniform_(m.weight, -0.08, 0.08)
                nn.init.constant_(m.bias, 0)


    def encode(self, input):
        z_mu_logvar = torch.flatten(self.z_encoder(input), start_dim=1)
        z_mu = z_mu_logvar[:, :self.s_dim]
        z_logvar = z_mu_logvar[:, self.s_dim:]

        u_mu_logvar = torch.flatten(self.u_encoder(input), start_dim=1)
        u_mu = u_mu_logvar[:, :self.s_dim]
        u_logvar = u_mu_logvar[:, self.s_dim:]

        return z_mu, z_logvar, u_mu, u_logvar

    def decode(self, s):
        x_recon = self.decoder(s)
        return x_recon

    def reparameterize_gauss(self, mu, logvar=None):
        if logvar is not None:
            std = torch.exp(0.5 * logvar)
        else:
            std = 1.0
        eps = torch.normal(0, 1, size=mu.shape).to('cuda')
        return eps * std + mu

    def reparameterize_laplace(self, mu, logvar=None, eps=1e-3):
        self.u_prior = torch.distributions.laplace.Laplace(torch.zeros_like(mu),
                                                           torch.ones_like(mu) * 1)
        y = self.u_prior.sample()
        if logvar is not None:
            std = torch.exp(logvar)
        else:
            std = 1.0
        x = std * y + mu
        return x

    def reparameterize_exp(self, logb, eps=1e-6):
        b = torch.exp(logb)
        prior = torch.distributions.exponential.Exponential(torch.ones_like(b) * 1.0)
        return prior.sample() / b

    def sample(self, inputs):
        ##enconde and sample
        z_mu, z_logvar, u_mu, u_logvar = self.encode(inputs)
        u = self.reparameterize_gauss(u_mu, u_logvar)

        spatial = int(u.shape[1] ** 0.5)

        u_spatial = u.view(u.shape[0], 1, spatial, spatial)

        u_spatial = u_spatial ** 2.0

        v = self.neighborhood_sum(u_spatial).squeeze(1)

        std = 1.0 / torch.sqrt(v + 1e-6)
        z = self.reparameterize_gauss(z_mu.view(std.shape),
                                      z_logvar.view(std.shape))
        s = (z * std+ 40*1/v ).view(z.shape[0], -1)
        return z_mu, z_logvar, u_mu, u_logvar ,u, s

    def forward(self, input):

        z_mu, z_logvar, u_mu, u_logvar,u,s = self.sample(input)
        x_recon = self.decode(s)

        recon_loss = F.mse_loss(x_recon.flatten(start_dim=1),
                                            input.flatten(start_dim=1),
                                            reduction='sum') / input.shape[0]

        KLD = (KL_gaussian_exact(z_mu, z_logvar).sum() + KL_gaussian_exact(u_mu, u_logvar,std_true=10).sum()) / input.shape[0]
        return x_recon, recon_loss, KLD, s

    def normalize_weights(self):
        with torch.no_grad():
            for l in [self.z_encoder[0], self.u_encoder[0], self.decoder[0]]:
                norm_shape = (l.weight.shape[0], l.weight.shape[1])
                norms = torch.sqrt(l.weight.view(norm_shape).pow(2).sum([-1], keepdim=True))
                l.weight.view(norm_shape).div_(norms)




