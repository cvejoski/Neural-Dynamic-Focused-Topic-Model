import numpy as np
import torch


def order_of_magnitude(number):
    if torch.is_tensor(number):
        order_of_magnitude_ = torch.log10(number).long()
    elif isinstance(number, (float, int)):
        order_of_magnitude_ = torch.log10(torch.Tensor([number])).long()
    elif isinstance(number, (np.ndarray, list)):
        order_of_magnitude_ = torch.log10(torch.Tensor(number)).long()
    else:
        print("Order of magnitude of not a number")
        raise Exception
    return order_of_magnitude_


class SigmoidScheduler(object):
    """
    https://arxiv.org/abs/1511.06349
    """

    def __init__(self, lambda_0=1., max_steps=1000, decay_rate=50., percentage_change=0.2):
        self.max_steps = max_steps
        self.decay_rate = decay_rate
        self.percentage_change = percentage_change
        self.lambda_0 = float(lambda_0)

    def __call__(self, step):
        x = (step - (self.max_steps * self.percentage_change)) / self.max_steps
        return self.lambda_0 / (1. + np.exp(-self.decay_rate * x))


class ExponentialScheduler(object):

    def __init__(self, **kwargs):
        self.max_steps = kwargs.get('max_steps', 1000)
        self.decay_rate = kwargs.get('decay_rate', 0.1)
        self.max_value = kwargs.get('max_value', 1.0)

    def __call__(self, step):
        return float(self.max_value / (1. + np.exp(-self.decay_rate * (step - self.max_steps))))


class ExponentialSchedulerGumbel(object):

    def __init__(self, **kwargs):
        self.min_tau = kwargs.get('min_temp', .5)
        self.decay_rate = kwargs.get('decay_rate', 1.)

    def __call__(self, tau_init, step):
        t = np.maximum(tau_init * np.exp(-self.decay_rate * step), self.min_tau)
        return t


class ExponentialSchedulerGumbel2(object):

    def __init__(self, **kwargs):
        self.tau_init = kwargs.get('tau_init', 1.)
        self.min_tau = kwargs.get('min_temp', .5)
        self.percentage_of_training = kwargs.get('percentage_of_training', 0.8)
        self.expected_num_steps = kwargs.get('expected_num_steps', 1000)
        self.fixed = kwargs.get('fixed', False)

        self.steps_to_minimum = self.percentage_of_training * self.expected_num_steps
        self.decay_rate = -(1 / self.steps_to_minimum) * np.log(self.min_tau / self.tau_init)

    def __call__(self, step):
        if not self.fixed:
            t = np.maximum(self.tau_init * np.exp(-self.decay_rate * step), self.min_tau)
            return t
        else:
            return self.tau_init


class LinearScheduler(object):
    def __init__(self, **kwargs):
        self.max_steps = kwargs.get('max_steps', 1000)
        self.start_value = kwargs.get('start_value', 0)
        print("start_value linear scheduler {}".format(self.start_value))

    def __call__(self, step):
        if self.start_value == 0:
            return min(1., float(step) / self.max_steps)
        else:
            return min(1., self.start_value + float(step) / self.max_steps * (1 - self.start_value))


class ConstantScheduler(object):
    def __init__(self, **kwargs):
        self.beta = kwargs.get('beta', 1000)

    def __call__(self, step):
        return self.beta
