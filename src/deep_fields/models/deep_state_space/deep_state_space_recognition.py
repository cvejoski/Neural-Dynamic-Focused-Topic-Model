from unicodedata import bidirectional
import torch
from torch import nn

from deep_fields.models.deep_architectures.deep_nets import MLP
from deep_fields.utils.reparametrizations import gumbel_softmax

KUMARASWAMY_BETA = "kumaraswamy-beta"

BERNOULLI = "bernoulli"

NORMAL = "normal"

EPSILON = 1e-10
EULER_GAMMA = 0.5772156649015329


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


def beta_fn(a, b):
    beta_ab = torch.exp(torch.lgamma(a) + torch.lgamma(b) - torch.lgamma(a + b))
    return beta_ab


def kl_kumaraswamy_beta(a, b, prior_alpha, prior_beta, reduction='mean'):
    # compute taylor expansion for E[log (1-v)] term
    # hard-code so we don't have to use Scan()
    kl = 1. / (1 + a * b) * beta_fn(1. / a, b)
    kl += 1. / (2 + a * b) * beta_fn(2. / a, b)
    kl += 1. / (3 + a * b) * beta_fn(3. / a, b)
    kl += 1. / (4 + a * b) * beta_fn(4. / a, b)
    kl += 1. / (5 + a * b) * beta_fn(5. / a, b)
    kl += 1. / (6 + a * b) * beta_fn(6. / a, b)
    kl += 1. / (7 + a * b) * beta_fn(7. / a, b)
    kl += 1. / (8 + a * b) * beta_fn(8. / a, b)
    kl += 1. / (9 + a * b) * beta_fn(9. / a, b)
    kl += 1. / (10 + a * b) * beta_fn(10. / a, b)
    kl *= (prior_beta - 1.) * b

    # use another taylor approx for Digamma function
    # psi_b_taylor_approx = torch.log(b) - 1. / (2 * b) - 1. / (12 * b ** 2)
    psi_b = torch.digamma(b)

    kl += (a - prior_alpha) / a * (-EULER_GAMMA - psi_b - 1. / b)  # T.psi(self.posterior_b)

    # add normalization constants
    kl += torch.log(a + EPSILON) + torch.log(b + EPSILON) + torch.log(beta_fn(prior_alpha, prior_beta) + EPSILON)

    # final term
    kl += -(b - 1) / b

    skl = kl.sum(dim=-1)
    if reduction == 'mean':
        return torch.mean(skl)
    elif reduction == 'sum':
        return torch.sum(skl)
    return skl


def kl_bernoulli(p, q):
    A = p * (torch.log(p + EPSILON) - torch.log(q + EPSILON))
    B = (1. - p) * (torch.log(1. - p + EPSILON) - torch.log(1. - q + EPSILON))
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


def reparameterize_normal(mu, logvar):
    """Returns a sample from a Gaussian distribution via reparameterization.
    """

    std = torch.exp(0.5 * logvar)
    eps = torch.randn_like(std)
    return eps.mul_(std).add_(mu)


def reparameterize_kumaraswamy(a, b):
    u = (1e-4 - 0.9999) * torch.rand_like(a) + 0.9999

    return torch.pow(1.0 - torch.pow(u, 1. / (b + EPSILON)), 1. / (a + EPSILON))


# ====================================================================
# DEEP KALMAN FILTER ENCODERS
# ====================================================================

class q_INDEP(nn.Module):
    input_dim: int
    par_1: nn.Module
    par_2: nn.Module
    q_infer: nn.Module
    data_2_hidden: MLP
    distribution_type: str

    def __init__(self, device=None, **kwargs):
        nn.Module.__init__(self)
        if device is None:
            self.device = torch.device("cpu")

        self.control_variable_dim: int = kwargs.get("control_variable_dim", 0)
        self.observable_dim: int = kwargs.get("observable_dim")
        self.layers_dim: int = kwargs.get("layers_dim")
        self.hidden_state_dim: int = kwargs.get("hidden_state_dim")
        self.out_dim: int = kwargs.get("output_dim")
        self.dropout: float = kwargs.get("dropout", 0.0)
        self.out_droupout: float = kwargs.get("out_dropout")
        self.distribution_type: float = kwargs.get("distribution_type", "normal")
        self.alpha0: int = kwargs.get("alpha0", 1)
        self.pi0: float = kwargs.get("pi0", 0.5)
        self.define_deep_models_parameters()
        self.num_topics = kwargs.get("num_topics", None)
        self.learn_alpha: bool = kwargs.get("learn_alpha", False)
        self.learn_beta: bool = kwargs.get("learn_beta", False)

    def define_deep_models_parameters(self):
        self.input_dim = self.control_variable_dim + self.observable_dim
        self.data_2_hidden = MLP(input_dim=self.input_dim,
                                 layers_dim=self.layers_dim,
                                 output_dim=self.out_dim,
                                 output_transformation=None,
                                 dropout=self.dropout)
        if self.out_droupout > 0:
            self.out_dropout_layer = torch.nn.Dropout(self.out_droupout)
        self.par_1 = nn.Linear(self.out_dim, self.hidden_state_dim)
        self.par_2 = nn.Linear(self.out_dim, self.hidden_state_dim)
        if self.distribution_type == NORMAL:
            self.q_infer = self._q_normal
        elif self.distribution_type == BERNOULLI:
            self.q_infer = self._q_bernoulli
        elif self.distribution_type == KUMARASWAMY_BETA:
            self.q_infer = self._q_kumaraswamy
        else:
            raise ValueError(f"Invalid distribution type {self.distribution_type}!")

    def forward(self, data, x=None, prior_transform=None):
        """
        data
        ---------
        batch_size,seq_length,observables_size

        parameters
        ---------
        data (batch_size,seq_length,observables_dimension)

        returns
        -------
        z,(z_mean, z_sigma)

        """
        if x is not None:
            data = torch.cat((data, x), dim=-1).unsqueeze(1)
        else:
            x = torch.zeros(data.size(0), self.hidden_state_dim, device=self.device)
        batch_size, seq_length, _ = data.shape
        h = self.data_2_hidden(data)
        h = self.out_dropout_layer(h) if self.out_droupout > 0 else h

        if prior_transform is not None:
            x = prior_transform(x)
        return self.q_infer(h, x, batch_size, seq_length)

    def _q_bernoulli(self, h, x, batch_size, seq_length):
        posterior_pi_logit = self.par_1(h).view(batch_size * seq_length, -1)
        posterior_pi = torch.sigmoid(posterior_pi_logit)

        prior_pi = self.pi0*x

        kl = kl_bernoulli(posterior_pi, prior_pi)
        pi_ = posterior_pi.view(-1).unsqueeze(1)  # [batch_size*number_of_topics,1]
        pi_ = torch.cat((1.0 - pi_, pi_), dim=1)
        b = gumbel_softmax(pi_, 1.0, self.device)[:, 1]  # [batch_size*number_of_topics]
        b = b.view(batch_size, -1)  # [batch_size,number_of_topics]
        return b, posterior_pi, prior_pi, kl

    def _q_normal(self, h, x, batch_size, seq_length):
        mu = self.par_1(h).view(batch_size * seq_length, -1)
        logvar = self.par_2(h).view(batch_size * seq_length, -1)
        z = reparameterize_normal(mu, logvar)
        kl = kullback_leibler_two_gaussians(mu, logvar, x, torch.zeros_like(x), 'sum')
        return z, (mu, logvar), kl

    def _q_kumaraswamy(self, h, x, batch_size, seq_length):
        c = torch.nn.functional.softplus(self.par_1(h).view(batch_size * seq_length, -1))
        d = torch.nn.functional.softplus(self.par_2(h).view(batch_size * seq_length, -1))
        alpha, beta = torch.ones_like(c), torch.ones_like(c)
        if self.learn_alpha:
            alpha = x[0]
        if self.learn_beta:
            beta = x[1]
        kl = kl_kumaraswamy_beta(c, d, alpha, beta, 'sum')
        z = reparameterize_kumaraswamy(c.repeat((1, self.num_topics)), d.repeat((1, self.num_topics)))
        return z, (c, d), (alpha, beta), kl

    def normal_(self):
        torch.nn.init.normal_(self.data_2_hidden)
        torch.nn.init.normal_(self.par_1)
        torch.nn.init.normal_(self.par_2)

    @property
    def device(self):
        return self.__device

    @device.setter
    def device(self, value):
        self.__device = value
        self.to(value)

    @classmethod
    def sample_model_parameters(cls):
        parameters = {"control_variable_dim": 3,
                      "observable_dim": 3,
                      "layers_dim": [10],
                      "hidden_state_dim": 3}
        return parameters

    def init_parameters(self):
        return None


class q_LR(nn.Module):
    input_dim: int
    mean: nn.Module
    log_var: nn.Module
    data_2_hidden: MLP

    def __init__(self, device=None, **kwargs):
        nn.Module.__init__(self)
        if device is None:
            self.device = torch.device("cpu")
        self.control_variable_dim = kwargs.get("control_variable_dim", 0)
        self.observable_dim = kwargs.get("observable_dim")
        self.layers_dim = kwargs.get("layers_dim")
        self.hidden_state_dim = kwargs.get("hidden_state_dim")
        self.dropout = kwargs.get("dropout", .4)
        self.is_bernoulli = kwargs.get("bernoulli", False)
        self.min_logv: float = kwargs.get("min_logv", -10.0)
        self.define_deep_models_parameters()
        self.init_parameters()

    @classmethod
    def sample_model_parameters(self):
        parameters = {"control_variable_dim": 3,
                      "observable_dim": 3,
                      "layers_dim": [10],
                      "hidden_state_dim": 3}
        return parameters

    def forward(self, data):
        """
        parameters
        ---------
        data (batch_size,seq_length,observables_dimension)

        returns
        -------
        z (batch_size * (seq_length - 2), 3*observables_dim)
        """
        batch_size, seq_length, observables_dim = data.shape
        data = data.unfold(1, 3, 1).contiguous()
        data = data.view(batch_size * (seq_length - 2), 3 * observables_dim)
        h = self.data_2_hidden(data)

        z_mean = self.mean(h)
        z_mean = z_mean.view(batch_size * (seq_length - 2), -1)
        z_mean = nn.functional.pad(z_mean, pad=[0, 0, 1, 1], mode="constant", value=1e-10)
        if not self.is_bernoulli:
            z_sigma = torch.exp(.5 * self.log_var(h).clamp(min=self.min_logv)).view(batch_size * (seq_length - 2), -1)
            z_sigma = nn.functional.pad(z_sigma, pad=[0, 0, 1, 1], mode="constant", value=0)
            epsilon = torch.randn(z_mean.shape, device=self.device)
            z = z_mean + epsilon * z_sigma
            return z, (z_mean, z_sigma)  # batch_size*seq_length,hidden_state_dimension
        return torch.sigmoid(z_mean)

    def define_deep_models_parameters(self):
        in_dim = 3 * self.control_variable_dim + 3 * self.observable_dim
        self.data_2_hidden = MLP(input_dim=in_dim,
                                 layers_dim=self.layers_dim,
                                 output_dim=self.hidden_state_dim,
                                 output_transformation=None)
        self.mean = nn.Linear(self.hidden_state_dim, self.hidden_state_dim)
        if not self.is_bernoulli:
            self.log_var = nn.Linear(self.hidden_state_dim, self.hidden_state_dim)

    def init_parameters(self):
        return None


class q_RNN(nn.Module):
    input_dim: int
    data_2_hidden: nn.LSTM
    meanmu: nn.Module
    log_var: nn.Module
    is_past_state: bool
    data_2_latent: MLP

    def __init__(self, device=None, **kwargs):
        nn.Module.__init__(self)
        if device is None:
            self.device = torch.device("cpu")

        self.control_variable_dim = kwargs.get("control_variable_dim", 0)
        self.observable_dim = kwargs.get("observable_dim")
        self.layers_dim = kwargs.get("layers_dim")
        self.hidden_state_dim = kwargs.get("hidden_state_dim")
        self.hidden_state_transition_dim = kwargs.get("hidden_state_transition_dim", 10)
        self.n_rnn_layers = kwargs.get("num_rnn_layers", 1)
        self.dropout = kwargs.get("dropout", .4)
        self.out_droupout: float = kwargs.get("out_dropout")
        self.is_bidirectional: bool = kwargs.get("is_bidirectional", False)
        self.delta = kwargs.get("delta", 0.005)

        self.define_deep_models_parameters()

    @property
    def device(self):
        return self.__device

    @device.setter
    def device(self, value):
        self.__device = value
        self.to(value)

    def define_deep_models_parameters(self):
        # RECOGNITION MODEL
        self.input_dim = self.control_variable_dim + self.observable_dim
        if self.layers_dim is not None:
            self.data_2_latent = nn.Linear(self.input_dim, self.layers_dim)
            self.input_dim = self.layers_dim
        if self.out_droupout > 0:
            self.out_dropout_layer = torch.nn.Dropout(self.out_droupout)
        self.data_2_hidden = nn.LSTM(self.input_dim, self.hidden_state_transition_dim, dropout=self.dropout, batch_first=True, num_layers=self.n_rnn_layers, bidirectional=self.is_bidirectional)
        hidden_state_transition_dim = self.hidden_state_transition_dim*2 if self.is_bidirectional else self.hidden_state_transition_dim
        self.mean = nn.Linear(hidden_state_transition_dim + self.hidden_state_dim, self.hidden_state_dim)
        self.log_var = nn.Linear(hidden_state_transition_dim + self.hidden_state_dim, self.hidden_state_dim)

    def init_hidden_rnn_state(self, batch_size):
        if self.is_bidirectional:
            hidden_init = (torch.randn(2, batch_size, self.hidden_state_transition_dim, device=self.device),
                           torch.randn(2, batch_size, self.hidden_state_transition_dim, device=self.device))
        else:
            hidden_init = (torch.randn(self.n_rnn_layers, batch_size, self.hidden_state_transition_dim, device=self.device),
                           torch.randn(self.n_rnn_layers, batch_size, self.hidden_state_transition_dim, device=self.device))

        return hidden_init

    def forward(self, data, prior=None, prior_data=None):
        """

        :param data: (B, L, D)
        :return:
        """
        batch_size, seq_length, dim = data.shape
        if self.layers_dim is not None:
            data = self.data_2_latent(data)
        # h = self.init_hidden_rnn_state(batch_size)
        out, _ = self.data_2_hidden(data)
        assert not torch.any(torch.isnan(out))
        assert not torch.any(torch.isinf(out))
        out = out.reshape(batch_size * seq_length, -1)
        out = self.out_dropout_layer(out) if self.out_droupout > 0 else out
        assert not torch.any(torch.isnan(out))
        assert not torch.any(torch.isinf(out))

        z = torch.zeros(seq_length, self.hidden_state_dim, device=self.device)
        m_q = torch.zeros(seq_length - 1, self.hidden_state_dim, device=self.device)
        logvar_q = torch.zeros(seq_length - 1, self.hidden_state_dim, device=self.device)

        in_0 = torch.cat([out[0], torch.zeros(self.hidden_state_dim, device=self.device)])
        mu_0: torch.Tensor = self.mean(in_0)
        logvar_0 = self.log_var(in_0)
        z[0] = reparameterize_normal(mu_0, logvar_0)
        assert not torch.any(torch.isnan(z))
        assert not torch.any(torch.isinf(z))
        logvar_t_p = torch.log(self.delta * torch.ones_like(logvar_0))

        kl = kullback_leibler(mu_0, logvar_0, 'sum')

        for t in range(1, seq_length):
            in_t = torch.cat([out[t], z[t - 1]])
            mu_t: torch.Tensor = self.mean(in_t)
            logvar_t = torch.clamp(self.log_var(in_t), max=100)
            m_q[t - 1] = mu_t
            logvar_q[t - 1] = logvar_t
            z[t] = reparameterize_normal(mu_t, logvar_t)
            if prior is None:
                mu_t_p = z[t - 1]
                kl += kullback_leibler_two_gaussians(mu_t, logvar_t, mu_t_p, logvar_t_p, 'sum')
            assert not torch.any(torch.isnan(z))
            assert not torch.any(torch.isinf(z))
        if prior is not None:
            if prior_data is None:
                p_m = prior(z[:-1])
            else:
                p_m = prior(torch.cat((z[:-1], prior_data[:-1]), dim=-1))
            kl += kullback_leibler_two_gaussians(m_q, logvar_q, p_m, torch.log(self.delta * torch.ones_like(z[:-1])), 'sum')
        return z, (mu_t, logvar_t), kl

    @classmethod
    def sample_model_parameters(cls):
        parameters = {"control_variable_dim": 3,
                      "observable_dim": 3,
                      "layers_dim": [10],
                      "hidden_state_dim": 3,
                      "hidden_state_transition_dim": 10}
        return parameters

    def init_parameters(self):
        return None


class q_BRNN(nn.Module):
    input_dim: int
    data_2_hidden: nn.LSTM
    mean: nn.Module
    log_var: nn.Module
    is_past_state: bool
    data_2_latent: MLP

    def __init__(self, device=None, **kwargs):
        nn.Module.__init__(self)
        if device is None:
            self.device = torch.device("cpu")

        self.control_variable_dim = kwargs.get("control_variable_dim", 0)
        self.observable_dim = kwargs.get("observable_dim")
        self.layers_dim = kwargs.get("layers_dim")
        self.hidden_state_dim = kwargs.get("hidden_state_dim")
        self.hidden_state_transition_dim = kwargs.get("hidden_state_transition_dim", 10)
        self.dropout = kwargs.get("dropout", .4)
        self.min_logv: float = kwargs.get("min_logv", -10.0)
        self.is_bernoulli = kwargs.get("bernoulli", False)
        self.define_deep_models_parameters()

    @property
    def device(self):
        return self.__device

    @device.setter
    def device(self, value):
        self.__device = value
        self.to(value)

    def define_deep_models_parameters(self):
        # RECOGNITION MODEL
        self.input_dim = self.control_variable_dim + self.observable_dim
        if self.layers_dim is not None:
            self.data_2_latent = nn.Linear(self.input_dim, self.layers_dim[0])
            self.input_dim = self.layers_dim[-1]

        self.data_2_hidden = nn.LSTM(self.input_dim, self.hidden_state_transition_dim, dropout=self.dropout,
                                     bidirectional=True, batch_first=True)
        self.mean = nn.Linear(self.hidden_state_transition_dim * 2, self.hidden_state_dim)
        if not self.is_bernoulli:
            self.log_var = nn.Linear(self.hidden_state_transition_dim * 2, self.hidden_state_dim)

    def init_hidden_rnn_state(self, batch_size):
        hidden_bi = (torch.randn(2, batch_size, self.hidden_state_transition_dim, device=self.device),
                     torch.randn(2, batch_size, self.hidden_state_transition_dim, device=self.device))

        return hidden_bi

    def forward(self, data):
        """

        :param data: (B, L, D)
        :return:
        """
        batch_size, seq_length, dim = data.shape
        if self.layers_dim is not None:
            data = self.data_2_latent(data)
        h = self.init_hidden_rnn_state(batch_size)
        out, h = self.data_2_hidden(data, h)

        out = out.reshape(batch_size * seq_length, -1)
        z_mean = self.mean(out)
        if not self.is_bernoulli:
            z_sigma = torch.exp(.5 * self.log_var(out).clamp(min=self.min_logv))
            epsilon = torch.randn(z_mean.shape, device=self.device)
            z = z_mean + epsilon * z_sigma
            return z, (z_mean, z_sigma)  # batch_size*seq_leght,hidden_state_dimension
        return torch.sigmoid(z_mean)

    @classmethod
    def sample_model_parameters(self):
        parameters = {"control_variable_dim": 3,
                      "observable_dim": 3,
                      "layers_dim": [10],
                      "hidden_state_dim": 3,
                      "hidden_state_transition_dim": 10}
        return parameters

    def init_parameters(self):
        return None


class q_DIVA(nn.Module):
    input_dim: int
    mean: nn.Module
    log_var: nn.Module
    data_to_hidden: MLP
    is_bernoulli: bool

    def __init__(self, device=None, **kwargs):
        nn.Module.__init__(self)
        if device is None:
            self.device = torch.device("cpu")

        self.control_variable_dim: int = kwargs.get("control_variable_dim", 0)
        self.number_of_categories: int = kwargs.get("num_of_categories")
        self.observable_dim: int = kwargs.get("observable_dim")

        self.out_dim: int = kwargs.get("output_dim")
        self.layers_dim_zx: int = kwargs.get("layers_dim_zx")
        self.hidden_state_dim_zx: int = kwargs.get("hidden_state_dim_zx")
        self.layers_dim_zy: int = kwargs.get("layers_dim_zy")
        self.hidden_state_dim_zy: int = kwargs.get("hidden_state_dim_zy")

        self.layers_dim_classifier: int = kwargs.get("layers_dim_classifier")
        self.reward_type = kwargs.get("reward_type", "classification")

        self.dropout: float = kwargs.get("dropout", 0.0)
        self.out_droupout: float = kwargs.get("out_dropout")
        self.is_bernoulli: float = kwargs.get("is_bernoulli", False)
        self.is_semisupervised: float = kwargs.get("is_semisupervised", False)
        self.llm = kwargs.get("llm")

        self.cov_layers_dim = kwargs.get("cov_layers_dim")
        self.covariates_dim = kwargs.get("covariates_dim")
        self.cov_emb_dim = kwargs.get("cov_emb_dim")

        self.define_deep_models_parameters()

    def define_deep_models_parameters(self):
        self.input_dim = self.control_variable_dim + self.observable_dim
        # ENCODER VARIABLE (ZX for DIVA)
        self.data_to_zx = MLP(input_dim=self.input_dim,
                              layers_dim=self.layers_dim_zx,
                              output_dim=self.out_dim,
                              output_transformation=None,
                              dropout=self.dropout)

        if self.out_droupout > 0:
            self.out_dropout_layer = torch.nn.Dropout(self.out_droupout)
        self.mean_zx = nn.Linear(self.out_dim, self.hidden_state_dim_zx)
        self.lvar_zx = nn.Linear(self.out_dim, self.hidden_state_dim_zx)

        # LABEL PRIOR
        self.label_to_y = MLP(input_dim=self.number_of_categories,
                              layers_dim=self.layers_dim_zy,
                              output_dim=self.out_dim,
                              output_transformation=None,
                              dropout=self.dropout)
        self.mean_prior_y = nn.Linear(self.out_dim, self.hidden_state_dim_zy)
        self.log_var_prior_y = nn.Linear(self.out_dim, self.hidden_state_dim_zy)

        # DATA TO CATEGORIES ENCODing (ZY for DIVA)
        self.data_to_zy = MLP(input_dim=self.input_dim, layers_dim=self.layers_dim_zy,
                              output_dim=self.out_dim, output_transformation=None, dropout=self.dropout)

        self.mean_zy = nn.Linear(self.out_dim, self.hidden_state_dim_zy)
        self.log_var_zy = nn.Linear(self.out_dim, self.hidden_state_dim_zy)

        if self.covariates_dim != 0:
            self.cov2emb = MLP(input_dim=self.covariates_dim, output_dim=self.cov_emb_dim,
                               layers_dim=self.cov_layers_dim, dropout=self.dropout, ouput_transformation=True)
        else:
            self.cov_emb_dim = 0

        # CLASSIFIER
        class_input_dim = self.hidden_state_dim_zy + self.cov_emb_dim + self.hidden_state_dim_zx
        if self.llm is not None:
            class_input_dim += 768

        # LABEL OR REWARD VARIABLE (Y for Kingma)
        self.classifier = MLP(input_dim=class_input_dim,
                              layers_dim=self.layers_dim_classifier,
                              output_dim=1 if self.reward_type == 'regression' else self.number_of_categories,
                              ouput_transformation=1 if self.reward_type == 'regression' else None,
                              dropout=self.dropout)

    def forward(self, bow, reward, llm_encoding, data=None, supervised=None, c_real=None, x=None, prior_transform=None):
        """
        data
        ---------
        batch_size, seq_length, observables_size

        parameters
        ---------
        data (batch_size, seq_length, observables_dimension)

        returns
        -------
        z, (z_mean, z_sigma)

        """
        # ENCODING THE CONTINOUS VARIABLE
        if x is not None:
            bow = torch.cat((bow, x), dim=-1).unsqueeze(1)
        else:
            x = torch.zeros(bow.size(0), self.hidden_state_dim_zx, device=self.device)

        batch_size, observables_dim = bow.shape

        # ENCODE FOR X
        q_zx = self.data_to_zx(bow)
        q_zx = self.out_dropout_layer(q_zx) if self.out_droupout > 0 else q_zx
        q_zx_mean = self.mean_zx(q_zx).view(batch_size, -1)
        q_zx_lvar = self.lvar_zx(q_zx).view(batch_size, -1)
        if prior_transform is not None:
            x = prior_transform(x)
        kl_zx = kullback_leibler_two_gaussians(q_zx_mean, q_zx_lvar, x, torch.zeros_like(x), 'sum')
        zx = reparameterize_normal(q_zx_mean, q_zx_lvar)

        # ENCODING THE CATEGORICAL VARIABLE CLASSIFICATION
        q_zy = self.data_to_zy(bow)
        q_zy = self.out_dropout_layer(q_zy) if self.out_droupout > 0 else q_zy
        q_zy_mean = self.mean_zy(q_zy).view(batch_size, -1)
        q_zy_lvar = self.log_var_zy(q_zy).view(batch_size, -1)

        # LABEL PRIOR
        one_hot_reward = nn.functional.one_hot(reward.long(), self.number_of_categories)
        p_zy = self.label_to_y(one_hot_reward.float())
        p_zy = self.out_dropout_layer(p_zy) if self.out_droupout > 0 else p_zy
        p_zy_mean = self.mean_prior_y(p_zy).view(batch_size, -1)
        p_zy_logvar = self.log_var_prior_y(p_zy).view(batch_size, -1)

        zy = reparameterize_normal(q_zy_mean, q_zy_lvar)
        kl_zy = kullback_leibler_two_gaussians(q_zy_mean, q_zy_lvar, p_zy_mean, p_zy_logvar, 'sum')
        emb = (zy, zx)
        if self.llm is not None:
            emb = (zy, zx, llm_encoding)

        # CLASSIFICATION
        if self.covariates_dim != 0:
            covariates = data['covariates'].to(self.device, non_blocking=True)
            cov_emb = self.cov2emb(covariates)
            y_logits = self.classifier(torch.cat(emb + (cov_emb, ), dim=1))
        else:
            y_logits = self.classifier(torch.cat(emb, dim=1))

        return zx, kl_zx, zy, kl_zy, y_logits

    def normal_(self):
        torch.nn.init.normal_(self.data_to_hidden)
        torch.nn.init.normal_(self.mean)
        torch.nn.init.normal_(self.log_var)

    @property
    def device(self):
        return self.__device

    @device.setter
    def device(self, value):
        self.__device = value
        self.to(value)

    @classmethod
    def sample_model_parameters(self):
        parameters = {
            "observable_dim": 100,
            "num_of_categories": 10,
            "layers_dim_zx": [256],
            "layers_dim_zy": [256],
            "layers_dim_classifier": [256],
            "output_dim": 256,
            "hidden_state_dim_zx": 10,  # number of topics
            "hidden_state_dim_zy": 10,
            "cov_layers_dim": [64, 64],
            "cov_emb_dim": 32,
            "dropout": .1,
            "out_dropout": .1,
            "is_semisupervised": False,
            "llm": "bert"
        }
        return parameters

    def init_parameters(self):
        return None


class q_DynamicDIVA(nn.Module):
    input_dim: int
    mean: nn.Module
    log_var: nn.Module
    data_to_hidden: MLP
    is_bernoulli: bool

    def __init__(self, device=None, **kwargs):
        nn.Module.__init__(self)
        if device is None:
            self.device = torch.device("cpu")

        self.control_variable_dim: int = kwargs.get("control_variable_dim", 0)

        self.number_of_categories: int = kwargs.get("num_of_categories")
        self.observable_dim: int = kwargs.get("observable_dim")

        self.out_dim: int = kwargs.get("output_dim")
        self.layers_dim_zx: int = kwargs.get("layers_dim_zx")
        self.hidden_state_dim_zx: int = kwargs.get("hidden_state_dim_zx")
        self.layers_dim_zy: int = kwargs.get("layers_dim_zy")
        self.hidden_state_dim_zy: int = kwargs.get("hidden_state_dim_zy")

        self.layers_dim_classifier: int = kwargs.get("layers_dim_classifier")
        self.reward_type = kwargs.get("reward_type", "classification")

        self.dropout: float = kwargs.get("dropout", 0.0)
        self.out_droupout: float = kwargs.get("out_dropout")
        self.is_bernoulli: float = kwargs.get("is_bernoulli", False)
        self.is_semisupervised: float = kwargs.get("is_semisupervised", False)
        self.llm = kwargs.get("llm")

        self.cov_layers_dim = kwargs.get("cov_layers_dim")
        self.covariates_dim = kwargs.get("covariates_dim")
        self.cov_emb_dim = kwargs.get("cov_emb_dim")

        self.define_deep_models_parameters()

    def define_deep_models_parameters(self):
        self.input_dim = self.control_variable_dim + self.observable_dim
        # ENCODER VARIABLE (ZX for DIVA)
        self.data_to_zx = MLP(input_dim=self.input_dim,
                              layers_dim=self.layers_dim_zx,
                              output_dim=self.out_dim,
                              output_transformation=None,
                              dropout=self.dropout)

        if self.out_droupout > 0:
            self.out_dropout_layer = torch.nn.Dropout(self.out_droupout)
        self.mean_zx = nn.Linear(self.out_dim, self.hidden_state_dim_zx)
        self.log_var_zx = nn.Linear(self.out_dim, self.hidden_state_dim_zx)

        # LABEL PRIOR
        self.label_to_y = MLP(input_dim=self.number_of_categories+self.control_variable_dim,
                              layers_dim=self.layers_dim_zy,
                              output_dim=self.out_dim,
                              output_transformation=None,
                              dropout=self.dropout)
        self.mean_prior_y = nn.Linear(self.out_dim, self.hidden_state_dim_zy)
        self.log_var_prior_y = nn.Linear(self.out_dim, self.hidden_state_dim_zy)

        # DATA TO CATEGORIES ENCODing (ZY for DIVA)
        self.data_to_zy = MLP(input_dim=self.input_dim, layers_dim=self.layers_dim_zy,
                              output_dim=self.out_dim, output_transformation=None, dropout=self.dropout)
        if self.llm is not None:
            self.llm_to_zy = nn.Linear(768, self.out_dim)
            self.mean_zy = nn.Linear(self.out_dim*2, self.hidden_state_dim_zy)
            self.log_var_zy = nn.Linear(self.out_dim*2, self.hidden_state_dim_zy)
        else:
            self.mean_zy = nn.Linear(self.out_dim, self.hidden_state_dim_zy)
            self.log_var_zy = nn.Linear(self.out_dim, self.hidden_state_dim_zy)

        if self.covariates_dim != 0:
            self.cov2emb = MLP(input_dim=self.covariates_dim, output_dim=self.cov_emb_dim,
                               layers_dim=self.cov_layers_dim, dropout=self.dropout, ouput_transformation=True)
        else:
            self.cov_emb_dim = 0

        # CLASSIFIER
        if self.llm is None:
            # LABEL OR REWARD VARIABLE (Y for Kingma)
            self.classifier = MLP(input_dim=self.hidden_state_dim_zy+self.cov_emb_dim+self.hidden_state_dim_zx,
                                  layers_dim=self.layers_dim_classifier,
                                  output_dim=1 if self.reward_type == 'regression' else self.number_of_categories,
                                  ouput_transformation=1 if self.reward_type == 'regression' else None,
                                  dropout=self.dropout)
        else:
            self.classifier = nn.Linear(self.hidden_state_dim_zy+self.cov_emb_dim+self.hidden_state_dim_zx, 1 if self.reward_type ==
                                        'regression' else self.number_of_categories)  # classification layer

    def forward(self, bow, text_z, reward, latent, data, prior_transform):
        """
        data
        ---------
        batch_size, seq_length, observables_size

        parameters
        ---------
        data (batch_size, seq_length, observables_dimension)

        returns
        -------
        z, (z_mean, z_sigma)

        """
        # ENCODING THE CONTINOUS VARIABLE
        eta_per_d, eta_r_per_d = latent
        eta_transform, eta_r_transform = prior_transform
        bow_x = torch.cat((bow, eta_per_d), dim=-1).unsqueeze(1)
        bow_y = torch.cat((bow, eta_r_per_d), dim=-1).unsqueeze(1)
        batch_size = bow_x.size(0)

        klx, zx = self.encode_x(eta_per_d, eta_transform, bow_x, batch_size)
        assert not torch.any(torch.isnan(zx))
        assert not torch.any(torch.isinf(zx))

        hy = self.data_to_zy(bow_y)
        hy = self.out_dropout_layer(hy) if self.out_droupout > 0 else hy
        # ENCODING THE CATEGORICAL VARIABLE CLASSIFICATION
        if self.llm is not None:
            hy_llm = self.llm_to_zy(text_z)
            hy = torch.cat([hy.squeeze(1), hy_llm], dim=1)
        assert not torch.any(torch.isnan(hy))
        assert not torch.any(torch.isinf(hy))
        zy_mean = self.mean_zy(hy).view(batch_size, -1)
        zy_logvar = self.log_var_zy(hy).view(batch_size, -1)

        # LABEL PRIOR
        one_hot_reward = nn.functional.one_hot(reward.long(), self.number_of_categories)
        hyl = self.label_to_y(torch.cat((one_hot_reward.float(), eta_r_per_d), dim=-1))
        hyl = self.out_dropout_layer(hyl) if self.out_droupout > 0 else hyl
        zyl_prior_mean = self.mean_prior_y(hyl).view(batch_size, -1)
        zyl_prior_logvar = self.log_var_prior_y(hyl).view(batch_size, -1)

        zy = reparameterize_normal(zy_mean, zy_logvar)
        kly = kullback_leibler_two_gaussians(zy_mean, zy_logvar, zyl_prior_mean, zyl_prior_logvar, 'sum')

        # CLASSIFICATION
        if self.covariates_dim != 0:
            covariates = data['covariates'].to(self.device, non_blocking=True)
            cov_emb = self.cov2emb(covariates)
            y_logits = self.classifier(torch.cat((zy, zx, cov_emb), dim=1))
        else:
            y_logits = self.classifier(torch.cat((zy, zx), dim=1))

        return zx, klx, zy, kly, y_logits

    def encode_x(self, eta_per_d,  eta_transform, bow_x, batch_size):
        p_eta_m = eta_transform(eta_per_d)

        # ENCODE FOR X
        hx = self.data_to_zx(bow_x)
        hx = self.out_dropout_layer(hx) if self.out_droupout > 0 else hx
        zx_mean = self.mean_zx(hx).view(batch_size, -1)
        zx_logvar = self.log_var_zx(hx).view(batch_size, -1)

        klx = kullback_leibler_two_gaussians(zx_mean, zx_logvar, p_eta_m, torch.zeros_like(p_eta_m), 'sum')
        zx = reparameterize_normal(zx_mean, zx_logvar)
        return klx, zx

    def normal_(self):
        torch.nn.init.normal_(self.data_to_hidden)
        torch.nn.init.normal_(self.mean)
        torch.nn.init.normal_(self.log_var)

    @property
    def device(self):
        return self.__device

    @device.setter
    def device(self, value):
        self.__device = value
        self.to(value)

    @classmethod
    def sample_model_parameters(self):
        parameters = {
            "observable_dim": 100,
            "num_of_categories": 10,
            "control_variable_dim": 0,

            "layers_dim_zx": [256],
            "layers_dim_zy": [256],
            "layers_dim_classifier": [256],
            "output_dim": 256,
            "hidden_state_dim_zx": 10,  # number of topics
            "hidden_state_dim_zy": 10,
            "cov_layers_dim": [64, 64],
            "cov_emb_dim": 32,
            "dropout": .1,
            "out_dropout": .1,
            "is_semisupervised": False,
            "llm": "bert"
        }
        return parameters

    def init_parameters(self):
        return None


class RecognitionModelFactory(object):
    models: dict

    def __init__(self):
        self._models = {'q-INDEP': q_INDEP,
                        'q-LR': q_LR,
                        'q-RNN': q_RNN,
                        'q-BRNN': q_BRNN,
                        'q-DIVA': q_DIVA,
                        'q-D-DIVA': q_DynamicDIVA}

    def create(self, model_type: str, **kwargs):
        builder = self._models.get(model_type)
        if not builder:
            raise ValueError(f'Unknown recognition model {model_type}')
        return builder(**kwargs)


if __name__ == "__main__":
    from deep_fields.models.topic_models.dynamic import BinaryDynamicTopicModels
    from deep_fields.data.topic_models.dataloaders import DynamicTopicDataloader

    ndt_inference_parameters = BinaryDynamicTopicModels.get_inference_parameters()

    data_dir = "C:/Users/cesar/Desktop/Projects/GeneralData/nips-papers/"
    dataloader_params = {"data_dir": data_dir,
                         "field_name": "nips_text_field.p",
                         "examples_name": "nips_dataset.p",
                         "prediction_split": .2,
                         "validation_split": .2}
    dataloader = DynamicTopicDataloader(inference_parameters=ndt_inference_parameters, **dataloader_params)
    batchdata = next(dataloader.train().__iter__())
    r_hidden_transition_state_size = 35
    r_dynamic_recognition_parameters = {"observable_dim": dataloader.vocabulary_dim,
                                        "layers_dim": [50],
                                        "hidden_state_dim": r_hidden_transition_state_size}

    text, bow, bow_time_indexes, corpora = batchdata
    text = text.squeeze()

    bow_time_indexes = bow_time_indexes.squeeze().long()
    batch_size, _ = bow.shape
    corpora = corpora[0].unsqueeze(0)

    q = q_INDEP(**r_dynamic_recognition_parameters)
    z, _ = q(corpora)
    print(z.shape)
    q = q_BRNN(**r_dynamic_recognition_parameters)
    z, _ = q(corpora)
    print(z.shape)
