import torch


def gumbel_sample(shape, device, epsilon=1e-20):
    """
    Sample Gumbel(0,1)
    """
    u = torch.rand(shape, device=device)
    return -torch.log(-torch.log(u + epsilon) + epsilon)


def gumbel_softmax_sample(pi, tau, device, epsilon=1e-12):
    """
    Sample Gumbel-softmax
    """
    y = torch.log(pi + epsilon) + gumbel_sample(pi.size(), device)
    return torch.nn.functional.softmax(y / tau, dim=-1)


def gumbel_softmax(pi, tau, device):
    """
    Gumbel-Softmax distribution.
    Implementation from https://github.com/ericjang/gumbel-softmax.
    pi: [B, ..., n_classes] class probs of categorical z
    tau: temperature
    Returns [B, ..., n_classes] as a one-hot vector
    """
    y = gumbel_softmax_sample(pi, tau, device)
    shape = y.size()
    _, ind = y.max(dim=-1)  # [B, ...]
    y_hard = torch.zeros_like(y).view(-1, shape[-1])
    y_hard.scatter_(1, ind.view(-1, 1), 1)
    y_hard = y_hard.view(*shape)
    return (y_hard - y).detach() + y


def gumbel_softmax_argmax(pi, tau, device):
    """
    Gumbel-Softmax distribution.
    pi: [B, ..., n_classes] class probs of categorical z
    tau: temperature
    Returns [B, ..., n_classes] as a one-hot vector
            [B, ..., 1] (argmax over classes)
    """
    y = gumbel_softmax_sample(pi, tau, device)
    shape = y.size()
    _, ind = y.max(dim=-1)  # [B, ...]
    y_hard = torch.zeros_like(y).view(-1, shape[-1])
    y_hard.scatter_(1, ind.view(-1, 1), 1)
    y_hard = y_hard.view(*shape)
    y_ = (y_hard - y).detach() + y
    list_indices = torch.arange(shape[-1]).view(1, 1, shape[-1]).float()
    indices = torch.sum(y_ * list_indices, dim=-1)
    return y_, indices


def weibull_reparametrization(z, k, lambda_):
    """
    here we follow https://www.cse.iitk.ac.in/users/piyush/papers/lgvg_aaai2020.pdf

    the parametrization is the same as in wikipedia, or folowing
    https://arxiv.org/pdf/1803.01328.pdf

    notice that if one were to sample from torch.distribution.Weibull one will sample

    Weibull(lambda_,k)
    """
    L = -torch.log(1 - z)
    L_alpha = L ** (1 / k)
    weibull = lambda_ * (L_alpha)
    return weibull

def reparameterize_kumaraswamy(a, b):
    u = (1e-4 - 0.9999) * torch.rand_like(a) + 0.9999

    return torch.pow(1.0 - torch.pow(u, 1. / (b + EPSILON)), 1. / (a + EPSILON))