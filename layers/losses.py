import torch

def KL_gaussian(mu, logvar):
    # assert len(mu.shape) == len(logvar.shape) == 3
    # mu = mu.flatten(start_dim=1)
    # logvar = logvar.flatten(start_dim=1)
    return -0.5 * (1 + logvar - mu ** 2 - logvar.exp()).sum(1)


def KL_gaussian_exact(mu, logvar, mu_true=0.0, std_true=1.0):
    #assert len(mu.shape) == len(logvar.shape) == 3
    #mu = mu.flatten(start_dim=1)
    #logvar = logvar.flatten(start_dim=1)
    return (-0.5 + torch.log(torch.tensor([std_true]).float().to('cuda')) - logvar*0.5 + (logvar.exp() + (mu - mu_true) ** 2)/(2*std_true**2)).sum(1)





