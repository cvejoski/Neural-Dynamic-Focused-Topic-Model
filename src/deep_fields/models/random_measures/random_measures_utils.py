import numpy as np
import torch
from torch.distributions import MultivariateNormal

def uniform(t,*args):
    if type(t) == type(np.asarray([])):
        return np.repeat(args[0],t.shape[0])
    else:
        return args[0]

def log_beta_function(alpha, beta):
    return torch.lgamma(alpha) + torch.lgamma(beta) - torch.lgamma(alpha + beta)

def kumaraswamy_beta_kullback_leibler(a, b, alpha, beta, reduction='sum'):
    A = ((a - alpha) / a) * (np.euler_gamma - torch.digamma(b) - 1 / b)
    B = torch.log(a * b) + log_beta_function(alpha, beta) - ((b - 1) / b)
    M = 5
    S = (1. / (1 + a * b)) * torch.exp(log_beta_function(1 / a, b))
    for m in range(2, M):
        S = S + (1. / (m + a * b)) * torch.exp(log_beta_function(m / a, b))
    C = (beta - 1.) * b * S
    if reduction == "sum":
        return (torch.sum(A + B + C))
    elif reduction == "mean":
        return (torch.mean(A + B + C))
    else:
        print("Reduction not implemented in Kumaraswamy KL")
        raise Exception


def kumaraswamy_sample(epsilon, a, b):
    v = (1. - epsilon ** (1. / b)) ** (1. / a)
    return v


def stick_breaking(v, device):
    short_sticks = (1. - v)[:, :-1]
    ones_ = torch.ones((v.size()[0], 1)).to(device)
    short_sticks = torch.cat((ones_, short_sticks), dim=1)
    v = torch.cat((v[:, :-1], ones_), dim=1)
    sticks = short_sticks.cumprod(1) * v
    return sticks


def generate_topic_dynamic(W, A, Q, hidden_state_size, number_of_steps):
    """
    h = Ah+ u
    z ~ Categorical(W*h)
    """
    transition_noise_distribution = MultivariateNormal(torch.zeros(hidden_state_size), Q)
    h_0 = MultivariateNormal(torch.zeros(hidden_state_size), Q).sample()
    topic_dynamics = []
    for t in range(number_of_steps):
        # transition
        h_t = torch.matmul(A, h_0)
        noise = transition_noise_distribution.sample()
        h_t = h_t + noise
        # emission
        sticks = torch.sigmoid(W.matmul(h_t)).unsqueeze(0)
        topic_probabilities = stick_breaking(sticks)
        h_0 = h_t
        topic_dynamics.append(topic_probabilities)
    return topic_dynamics
