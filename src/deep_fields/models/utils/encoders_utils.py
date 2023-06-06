
import torch

def to_one_hot(labels, num_classes):
    """
    Convert tensor of labels to one hot encoding of the labels.
    :param labels: to be encoded
    :param num_classes:
    :return:
    """
    shape = labels.size()
    shape = shape + (num_classes,)
    one_hot = torch.zeros(shape, dtype=torch.float, device=labels.device)
    dim = 1 if len(shape) == 2 else 2
    one_hot.scatter_(dim, labels.unsqueeze(-1), 1)
    return one_hot

def kullback_leibler(mean, logvar, reduction='mean'):
    """
    Kullback-Leibler divergence between Gaussian posterior distr.
    with parameters (mean, sigma) and a fixed Gaussian prior
    with mean = 0 and sigma = 1
    """

    kl = -0.5 * (1 + logvar - mean.pow(2) - logvar.exp())  # [B, D]
    skl = torch.sum(kl, dim=-1)
    if reduction == 'mean':
        return torch.mean(skl)
    elif reduction == 'sum':
        return torch.sum(skl)
    else:
        return skl


def kullback_leibler_two_gaussians(mean1, logvar1, mean2, logvar2, reduction='mean'):
    """
    Kullback-Leibler divergence between two Gaussians
    """
    var1 = logvar1.exp()
    var2 = logvar2.exp()
    kl = -0.5 * (1 - logvar2 + logvar1 - ((mean1 - mean2).pow(2) + var1) / var2)  # [B, D]
    skl = torch.sum(kl, dim=-1)
    if reduction == 'mean':
        return torch.mean(skl)
    elif reduction == 'sum':
        return torch.sum(skl)
    else:
        return skl


def kl_bernoulli(p, q):
    A = p * (torch.log(p + 1e-8) - torch.log(q + 1e-8))
    B = (1. - p) * (torch.log(1. - p) - torch.log(1. - q))
    KL = A + B
    return KL.sum()


def one_step_ahead(batchdata):
    """
    returns
    input_,target_ (batch_size * seq_length, observables_dimension), (batch_size * seq_length, observables_dimension)
    """
    input_, target = batchdata[:, :-1, :].contiguous(), batchdata[:, 1:, :].contiguous()
    batch_size, seq_length, observables_dimension = input_.shape
    input_ = input_.view(batch_size * seq_length, -1)
    target_ = target.view(batch_size * seq_length, -1)
    return input_, target_


def k_forward_l_back(batchdata, k=1, l=1):
    """
    segments time series data
    """
    return batchdata


def reparameterize(mu, logvar):
    """Returns a sample from a Gaussian distribution via reparameterization.
    """

    std = torch.exp(0.5 * logvar)
    eps = torch.randn_like(std)
    return eps.mul_(std).add_(mu)