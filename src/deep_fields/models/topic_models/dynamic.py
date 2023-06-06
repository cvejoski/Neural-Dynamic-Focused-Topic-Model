from turtle import color
from deep_fields.utils.schedulers import SigmoidScheduler
from deep_fields.models.deep_state_space.deep_state_space_recognition import RecognitionModelFactory
import io
import os
from collections import defaultdict, OrderedDict

import PIL.Image
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
import torch
from torch import nn
from scipy.stats import beta
from torch.distributions import Normal, Multinomial, Poisson, Beta, Bernoulli
from torchvision.transforms import ToTensor
from tqdm import tqdm
from deep_fields.models.basic_utils import create_instance
from deep_fields import project_path
from deep_fields.data.topic_models.dataloaders import ADataLoader
from deep_fields.models.abstract_models import DeepBayesianModel
from deep_fields.models.deep_architectures.deep_nets import MLP
from deep_fields.models.basic_utils import nearest_neighbors
from deep_fields.models.random_measures.random_measures_utils import stick_breaking
from deep_fields.models.topic_models.topics_utils import reparameterize
from deep_fields.utils.loss_utils import get_doc_freq, kullback_leibler_two_gaussians, kullback_leibler

cos = nn.CosineSimilarity(dim=1, eps=1e-6)


def kumaraswamy_pdf(x, a, b):
    if isinstance(a, list) or isinstance(a, np.ndarray):
        results = []
        for i in range(len(a)):
            results.append(a[i]*b[i]*np.power(x, a[i]-1)*np.power((1-np.power(x, a[i])), b[i]-1))
        return np.asarray(results)
    else:
        return a*b*np.power(x, a-1)*np.power((1-np.power(x, a)), b-1)


def beta_pdf(x, a, b):
    if isinstance(a, list) or isinstance(a, np.ndarray):
        results = []
        for i in range(len(a)):
            results.append(beta.pdf(x, a[i], b[i]))
        return np.asfarray(results)
    else:
        return beta.pdf(x, a, b)


class DynamicLDA(DeepBayesianModel):
    topic_recognition_model: nn.Module
    eta_q: nn.Module
    theta_q: nn.Module

    mu_q_alpha: nn.Parameter
    logvar_q_alpha: nn.Parameter

    vocabulary_dim: int
    number_of_documents: int
    num_training_steps: int
    num_prediction_steps: int
    lambda_diversity: float = 0.1
    number_of_topics: int
    delta: float = 0.005

    topic_transition_dim: int

    def __init__(self, model_dir=None, data_loader=None, model_name=None, **kwargs):
        if model_name is None:
            model_name = "dynamic_lda"
        DeepBayesianModel.__init__(self, model_name, model_dir=model_dir, data_loader=data_loader, **kwargs)

    @classmethod
    def get_parameters(cls):
        """
        here we provide an example of the minimum set of parameters requiered to instantiate the model
        """
        number_of_topics = 50

        r_hidden_transition_state_size = 32
        vocabulary_dim = 100
        parameters_sample = {"number_of_topics": number_of_topics,  # z
                             "number_of_documents": 100,
                             "number_of_words_per_document": 30,
                             "vocabulary_dim": vocabulary_dim,

                             "num_training_steps": 48,
                             "lambda_diversity": 0.1,
                             "num_prediction_steps": 2,
                             "topic_proportion_transformation": "gaussian_softmax",
                             "topic_lifetime_tranformation": "gaussian_softmax",
                             "delta": 0.005,
                             "theta_q_type": "q-INDEP",
                             "theta_q_parameters": {
                                 "observable_dim": vocabulary_dim,
                                 "layers_dim": [250, 250],
                                 "output_dim": 250,
                                 "hidden_state_dim": number_of_topics,
                                 "dropout": .1,
                                 "out_dropout": 0.1
                             },
                             "eta_q_type": "q-RNN",
                             "eta_q_parameters": {
                                 "observable_dim": vocabulary_dim,
                                 "layers_dim": 400,
                                 "num_rnn_layers": 4,
                                 "hidden_state_dim": r_hidden_transition_state_size,
                                 "hidden_state_transition_dim": 400,
                                 "dropout": .1,
                                 "out_dropout": 0.1
                             },
                             "model_path": os.path.join(project_path, 'results')}

        return parameters_sample

    def set_parameters(self, **kwargs):
        self.vocabulary_dim = kwargs.get("vocabulary_dim")
        self.number_of_documents = kwargs.get("number_of_documents", None)
        self.num_training_steps = kwargs.get("num_training_steps", None)
        self.num_prediction_steps = kwargs.get("num_prediction_steps", None)

        self.number_of_topics = kwargs.get("number_of_topics", 10)

        self.topic_transition_dim = self.number_of_topics * self.vocabulary_dim  # output text pad ad eos

        self.eta_q_type = kwargs.get("eta_q_type")
        self.eta_q_parameters = kwargs.get("eta_q_parameters")
        self.eta_dim = kwargs.get("eta_q_parameters").get("hidden_state_dim")

        # TEXT
        self.theta_q_type = kwargs.get("theta_q_type")
        self.theta_q_parameters = kwargs.get("theta_q_parameters")

        self.topic_proportion_transformation = kwargs.get("topic_proportion_transformation")

    def update_parameters(self, data_loader, **kwargs):
        kwargs.update({"vocabulary_dim": data_loader.vocabulary_dim})
        kwargs.update({"number_of_documents": data_loader.number_of_documents})
        kwargs.update({"num_training_steps": data_loader.num_training_steps})
        kwargs.update({"num_prediction_steps": data_loader.num_prediction_steps})

        kwargs.get("eta_q_parameters").update({"observable_dim": data_loader.vocabulary_dim, "hidden_state_dim": self.number_of_topics})
        kwargs.get("theta_q_parameters").update({"observable_dim": data_loader.vocabulary_dim, "hidden_state_dim": self.number_of_topics,
                                                 "control_variable_dim": self.number_of_topics})

        return kwargs

    @classmethod
    def get_inference_parameters(cls):
        inference_parameters = super().get_inference_parameters()
        inference_parameters.update({"regularizers": {"nll": None}})
        inference_parameters.update({"model_eval": "PPL-Blei"})
        inference_parameters.update({"gumbel": .0005})

        return inference_parameters

    def initialize_inference(self, data_loader, **inference_parameters):
        super().initialize_inference(data_loader=data_loader, **inference_parameters)
        regularizers = inference_parameters.get("regularizers")

        self.schedulers = {}
        for k, v in regularizers.items():
            if v is not None:
                self.schedulers[k] = create_instance(v)

        self.eta_q.device = self.device
        self.theta_q.device = self.device

    def move_to_device(self):
        self.eta_q.device = self.device

    def define_deep_models(self):
        recognition_factory = RecognitionModelFactory()

        self.eta_q = recognition_factory.create(self.eta_q_type, **self.eta_q_parameters)
        self.theta_q = recognition_factory.create(self.theta_q_type, **self.theta_q_parameters)

        self.mu_q_alpha = nn.Parameter(torch.randn(self.number_of_topics, self.num_training_steps, self.vocabulary_dim))
        self.logvar_q_alpha = nn.Parameter(torch.randn(self.number_of_topics, self.num_training_steps, self.vocabulary_dim))

    def forward(self, x):
        """
        parameters
        ----------
        batchdata ()
        returns
        -------
        z (batch_size*sequence_lenght,self.hidden_state_dim)
        recognition_parameters = (z_mean,z_var)
        (batch_size*sequence_lenght,self.hidden_state_dim)
        likelihood_parameters = (likelihood_mean,likelihood_variance)
        """

        time_idx = x['time'].squeeze().long().to(self.device, non_blocking=True)
        corpora = x['corpus'][:1].float().to(self.device, non_blocking=True)
        if 'bow' in x:
            bow = x['bow'].to(self.device, non_blocking=True)
            normalized_bow = bow.float() / bow.sum(1, True)
        else:
            bow = x['bow_h1'].to(self.device, non_blocking=True)
            normalized_bow = bow.float() / bow.sum(1, True)
            bow = x['bow_h2'].to(self.device, non_blocking=True)

        # Dynamic Stuff

        alpha, kl_alpha = self.q_alpha()  # [total_time, topic_transition_dim]
        eta, eta_params, kl_eta = self.eta_q(corpora)  # [total_time, topic_transition_dim]
        beta = self.get_beta(alpha)
        beta_per_document = beta[time_idx]  # [batch_size, topic_transition_dim]
        eta_per_document = eta[time_idx]  # [batch_size, number_of_topics]

        # Text Recognition
        theta_logits, _, kl_theta = self.theta_q(normalized_bow, eta_per_document)

        theta = self.proportion_transformation(theta_logits)
        nll = self.nll(theta, bow, beta_per_document)

        return nll, kl_theta, kl_alpha, kl_eta, eta_params, theta

    def loss(self, x, forward_results, data_set, epoch):
        """
        nll [batch_size, max_lenght]

        """
        if 'bow' in x:
            text_size = x['bow'].sum(1).double().to(self.device, non_blocking=True)
        else:
            text_size = x['bow_h2'].sum(1).double().to(self.device, non_blocking=True)

        nll, kl_theta, kl_alpha, kl_eta, _, _ = forward_results

        coeff = 1.0
        if self.training:
            coeff = len(data_set)
        loss = (nll.sum() + kl_theta) * coeff + (kl_eta + kl_alpha)  # [number of batches]

        log_perplexity = nll / text_size.float()
        log_perplexity = torch.mean(log_perplexity)
        perplexity = torch.exp(log_perplexity)

        return {"loss": loss,
                "NLL-Loss": nll.sum() * coeff,
                "KL-Loss-Eta": kl_eta,
                "KL-Loss-Alpha": kl_alpha,
                "KL-Loss-Theta": kl_theta * coeff,
                "PPL-Blei": perplexity,
                "PPL": torch.exp(nll.sum() / text_size.sum().float()),
                "Log-Likelihood": log_perplexity}

    def q_alpha(self):
        rho_dim = self.mu_q_alpha.size(-1)
        alphas = torch.zeros(self.num_training_steps, self.number_of_topics, rho_dim).to(self.device)
        alphas[0] = reparameterize(self.mu_q_alpha[:, 0, :], self.logvar_q_alpha[:, 0, :])

        mu_0_p = torch.zeros(self.number_of_topics, rho_dim).to(self.device, non_blocking=True)
        logvar_0_p = torch.zeros(self.number_of_topics, rho_dim).to(self.device, non_blocking=True)
        logvar_t_p = torch.log(self.delta * torch.ones(self.number_of_topics, rho_dim).to(self.device, non_blocking=True))
        kl = kullback_leibler_two_gaussians(self.mu_q_alpha[:, 0, :], self.logvar_q_alpha[:, 0, :], mu_0_p, logvar_0_p, 'sum')

        for t in range(1, self.num_training_steps):
            alphas[t] = reparameterize(self.mu_q_alpha[:, t, :], self.logvar_q_alpha[:, t, :])
            mu_t_p = alphas[t - 1]
            kl += kullback_leibler_two_gaussians(self.mu_q_alpha[:, t, :], self.logvar_q_alpha[:, t, :], mu_t_p, logvar_t_p, 'sum')

        return alphas, kl

    def get_beta(self, alphas):
        topic_embeddings = alphas.view(self.num_training_steps, self.number_of_topics, self.vocabulary_dim)
        beta = torch.softmax(topic_embeddings, dim=2)
        return beta

    def get_beta_eval(self):
        return self.get_beta(self.mu_q_alpha)

    def nll(self, theta, bow, beta):
        loglik = torch.bmm(theta.unsqueeze(1), beta).squeeze(1)
        if self.training:
            assert not torch.isnan(loglik).any()
            assert not torch.isinf(loglik).any()
            loglik = torch.log(loglik + 1e-6)
        else:
            loglik = torch.log(loglik)
        nll = -loglik * bow

        return nll.sum(-1)

    def nll_test(self, theta, bow, beta):
        loglik = torch.matmul(theta, beta)

        loglik = torch.log(loglik)
        nll = -loglik * bow

        return nll.sum(-1)

    def metrics(self, data, forward_results, epoch, mode="evaluation", data_loader=None):
        if mode == "validation_global":
            if epoch % self.metrics_logs == 0:
                top_word_per_topic = self.top_words(data_loader, num_of_top_words=20)
                TEXT = ""
                for topic, value in top_word_per_topic.items():
                    for time, words in value.items():
                        TEXT += f'{topic} --- {time}: {" ".join(words)}\n\n'
                    TEXT += "*" * 1000 + "\n"
                # TEXT = "\n".join(["TOPIC {0}: ".format(j) + " ".join(top_word_per_topic[j]) + "\n" for j in range(len(top_word_per_topic))])
                self.writer.add_text("/TEXT/", TEXT, epoch)
        return {}

    def proportion_transformation(self, proportions_logits):
        if self.topic_proportion_transformation == "gaussian_softmax":
            proportions = torch.softmax(proportions_logits, dim=1)
        elif self.topic_proportion_transformation == "gaussian_stick_break":
            sticks = torch.sigmoid(proportions_logits)
            proportions = stick_breaking(sticks, self.device)
        else:
            raise ValueError(f"{self.topic_proportion_transformation} transformation not implemented!")
        return proportions

    def sample(self):
        """
        returns
        ------
        (documents,documents_z,thetas,phis)
        """
        word_count_distribution = Poisson(self.number_of_words_per_document)
        prior_0 = Normal(torch.zeros(self.number_of_topics), torch.ones(self.number_of_topics))

        word_embeddings = nn.Embedding(self.vocabulary_size, self.word_embeddings_dim)
        topic_embeddings = nn.Embedding(self.number_of_topics, self.word_embeddings_dim)
        beta = torch.matmul(word_embeddings.weight, topic_embeddings.weight.T)
        beta = torch.softmax(beta, dim=0)

        if self.topic_proportion_transformation == "gaussian_softmax":
            proportions_logits = prior_0.sample(torch.Size([self.number_of_documents]))
            topic_proportions = torch.softmax(proportions_logits, dim=1)
        elif self.topic_proportion_transformation == "gaussian_stick_break":
            proportions_logits = prior_0.sample(torch.Size([self.number_of_documents]))
            sticks = torch.sigmoid(proportions_logits)
            proportions = stick_breaking(sticks)

        # LDA Sampling
        documents_z = []
        documents = []
        word_count = word_count_distribution.sample(torch.Size([self.number_of_documents]))
        words_distributions = []
        for k in range(self.number_of_topics):
            words_distributions.append(Multinomial(total_count=1, probs=beta.T[k]))
        # LDA Sampling
        for d_index in range(self.number_of_documents):
            mixing_distribution_document = Multinomial(total_count=1, probs=proportions[d_index])
            # sample word allocation
            number_of_words = int(word_count[d_index].item())
            z_document = mixing_distribution_document.sample(sample_shape=torch.Size([number_of_words]))
            selected_index = torch.argmax(z_document, dim=1)
            documents_z.append(selected_index.numpy())
            document = []
            for w_index in range(number_of_words):
                z_word = words_distributions[selected_index[w_index]].sample()
                document.append(torch.argmax(z_word).item())
            documents.append(document)

        return documents, documents_z

    def get_words_by_importance(self) -> torch.Tensor:
        with torch.no_grad():
            beta = self.get_beta(self.mu_q_alpha)
            important_words = torch.argsort(beta, dim=-1, descending=True)
            if self.model_name == 'dynamic_lda':
                return important_words  # torch.transpose(important_words, 1, 0)
            else:
                return important_words

    # ======================================================
    # POST - PROCESSING
    # ======================================================
    def top_words(self, data_loader, num_of_top_words=20, num_of_time_steps=3):
        important_words = self.get_words_by_importance()
        # from index to words
        vocabulary = data_loader.vocab
        top_word_per_topic = dict()

        tt = important_words.size(0) // 2
        for k in range(self.number_of_topics):
            top_words_per_time = dict()
            for t in [0, tt, important_words.size(0) - 1]:
                topic_words = [vocabulary.itos[w] for w in important_words[t, k, :num_of_top_words].tolist()]
                top_words_per_time[f'Time {t}'] = topic_words
            top_word_per_topic[f'Topic {k}'] = top_words_per_time
        return top_word_per_topic

    def top_topics(self, data_loader):
        with torch.no_grad():
            # most important topics
            number_of_docs = 0.
            proportions = torch.zeros(self.number_of_topics, device=self.device)
            for databatch in data_loader.validation():
                likelihood, theta, theta_parameters, \
                    v_per_document, \
                    v_parameters, r_parameters, \
                    v_parameters_0, r_parameters_0, \
                    v_priors_parameters, r_priors_parameters = self(databatch)
                proportions += theta.sum(dim=0)
                batch_size, _ = theta.shape
                number_of_docs += batch_size

            proportions = proportions / number_of_docs
            proportions_index = torch.argsort(proportions, descending=True)
            proportions_values = proportions[proportions_index].cpu().detach().numpy()
            top_proportions = list(proportions_index.cpu().detach().numpy())
            return top_proportions, proportions_values

        # ===================================================================================
        # PREDICTION POST PROCESSING
        # ===================================================================================

    def words_time_series(self, dataloader):
        return None

    def topics_timeseries(self, dataloader):
        return None

    def get_prior_transitions_distributions(self, v_sample, r_sample):
        """
        returns torch.distributions Normal([montecarlo_samples,number_of_topics])
        """
        v_mean = v_sample  # [total_time-1, number_of_topics]
        v_std = (self.topic_transition_logvar ** 2.) * torch.ones_like(v_mean)  # [total_time-1, number_of_topics]
        v_priors_parameters = v_mean, v_std

        r_mean = r_sample
        r_std = (self.r_transition_logvar ** 2.) * torch.ones_like(r_sample)  # [total_time-1, number_of_topics]
        r_priors_parameters = r_mean, r_std

        transition_prior_v = Normal(*v_priors_parameters)
        transition_prior_r = Normal(*r_priors_parameters)

        return transition_prior_v, transition_prior_r

    def prediction_montecarlo_step(self, current_document_count, alpha_dist, eta_dist):
        alpha_sample = alpha_dist.sample()  # [mc, n_topics, topic_transition_dim]
        eta_sample = eta_dist.sample()  # [mc,n_topics, topic_transition_dim]
        n_mc, _, _ = alpha_sample.shape

        theta_dist = Normal(eta_sample, torch.ones_like(eta_sample))
        theta_sample = theta_dist.sample(sample_shape=torch.Size([current_document_count]))  # [mc,current_doc_count,number_of_topics]
        # [mc,count,number_of_topics]
        # alpha_per_doc = torch.repeat_interleave(alpha_sample, current_document_count, dim=0)
        # alpha_per_doc = alpha_per_doc.view(n_mc, current_document_count, self.number_of_topics, -1).contiguous()  # [mc,current_document_count,
        # number_of_topics]

        theta_sample = theta_sample.view(n_mc * current_document_count, -1).contiguous()
        theta = self.proportion_transformation(theta_sample)  # [mc,count,number_of_topics]
        theta = theta.view(n_mc, current_document_count, -1).contiguous()
        return theta, alpha_sample, eta_sample, alpha_sample

    def get_transitions_dist(self, dist_1_mu, dist_2_mu):
        """
        returns torch.distributions Normal([montecarlo_samples,number_of_topics])
        """
        dist_1_std = torch.ones_like(dist_1_mu) * self.delta
        dist_2_std = torch.ones_like(dist_2_mu) * self.delta

        dist_1_mu = Normal(dist_1_mu, dist_1_std)
        dist_2_mu = Normal(dist_2_mu, dist_2_std)
        return dist_1_mu, dist_2_mu

    def prediction(self, data_loader: ADataLoader, montecarlo_samples=10):
        self.eval()
        x = next(iter(data_loader.train))
        with torch.no_grad():
            forward_results = self.forward(x)

            _, _, _, _, eta_params, _ = forward_results

            last_state_alpha = self.mu_q_alpha[:, -1, :], torch.exp(0.5 * self.logvar_q_alpha[:, -1, :]) * self.delta
            alpha_q_dist = Normal(*last_state_alpha)
            eta_std = torch.exp(0.5 * eta_params[1]) * self.delta
            last_state_eta = eta_params[0], eta_std
            eta_q_dist = Normal(*last_state_eta)

            alpha_sample = alpha_q_dist.sample(sample_shape=torch.Size([montecarlo_samples]))
            eta_sample = eta_q_dist.sample(sample_shape=torch.Size([montecarlo_samples]))
            pred_pp_all = []
            for tt in self.data_loader.prediction_times:
                current_document_count = int(data_loader.prediction_count_per_year)

                alpha_dist, eta_dist = self.get_transitions_dist(alpha_sample, eta_sample)
                theta, alpha_sample, eta_sample, alpha_doc = self.prediction_montecarlo_step(current_document_count, alpha_dist, eta_dist)
                # [mc,count,number_of_topics]
                pred_pp_step = torch.zeros((montecarlo_samples, current_document_count))
                bow = torch.from_numpy(data_loader.predict.dataset.corpus_per_year('prediction')[0], device=self.device)
                text_size = bow.sum(1).double().to(self.device, non_blocking=True)
                for mc_index in range(montecarlo_samples):
                    alpha_per_doc = alpha_sample[mc_index]
                    theta_mc = theta[mc_index]  # [count, number_of_topics]
                    beta = torch.softmax(alpha_per_doc, dim=-1)
                    pred_like = self.nll_test(theta_mc, bow, beta)
                    log_pp = (1. / text_size.float()) * pred_like
                    log_pp = torch.mean(log_pp)

                    pred_pp_step[mc_index] = log_pp
                pred_pp_all.append(pred_pp_step.mean())
            pred_pp = torch.mean(torch.stack(pred_pp_all)).item()
            return pred_pp

    def topic_diversity(self, topk: int = 25) -> float:
        important_words_per_topic = self.get_words_by_importance()
        time_steps = important_words_per_topic.size(1)
        td_all = torch.zeros((time_steps,))
        for tt in range(time_steps):
            list_w = important_words_per_topic[:, tt, :topk]
            n_unique = len(torch.unique(list_w))
            td = n_unique / (topk * self.number_of_topics)
            td_all[tt] = td
        return td_all.mean().item()

    def topic_coherence(self, data: np.ndarray) -> float:
        top_10 = self.get_words_by_importance()[:, :, :10]
        tc_all = []
        p_bar = tqdm(range(top_10.size(0)), desc="Calculating Topics Coherence:")
        for tt in p_bar:
            tc = self._topic_coherence(data, top_10[tt])
            tc_all.append(tc)
            p_bar.set_postfix_str(f'current topic coherence: {np.mean(tc_all)}')
        return np.mean(tc_all)

    def _topic_coherence(self, data: np.ndarray, top_10: torch.Tensor) -> np.ndarray:
        D = data.shape[0]
        TC = []
        # p_bar = tqdm(desc='calculating topics coherence', total=self.number_of_topics)
        for k in range(self.number_of_topics):
            top_10_k = top_10[k].tolist()
            TC_k = 0
            counter = 0
            word_count_ = self.data_loader.vocab.word_count
            for i, word in enumerate(top_10_k):
                # get D(w_i)
                D_wi = word_count_[word]
                j = i + 1
                tmp = 0
                while len(top_10_k) > j > i:
                    # get D(w_j) and D(w_i, w_j)
                    D_wj = word_count_[top_10_k[j]]
                    D_wi_wj = get_doc_freq(data, word, top_10_k[j])
                    # get f(w_i, w_j)
                    if D_wi_wj == 0:
                        f_wi_wj = -1
                    else:
                        f_wi_wj = -1 + (np.log(D_wi) + np.log(D_wj) - 2.0 * np.log(D)) / (np.log(D_wi_wj) - np.log(D))
                    # update tmp:
                    tmp += f_wi_wj
                    j += 1
                    counter += 1
                # update TC_k
                TC_k += tmp
            TC.append(TC_k / counter)
            # p_bar.set_postfix_str(f"TC: {np.mean(TC)}")
            # p_bar.update()
        TC = np.mean(TC)
        # print('Topic coherence is: {}'.format(TC))
        return TC

    def get_topic_entropy_per_document(self, dataset):

        theta_stats = defaultdict(list)
        for x in tqdm(dataset, desc="Building time series"):
            _, _, _, _, _, theta = self.forward(x)

            time_idx = x['time'].squeeze()
            for id, t in zip(time_idx, theta):

                theta_stats[id.item()].append(-(t*torch.log(t)).sum())

        for key, value in theta_stats.items():
            theta_stats[key] = torch.stack(value)

        theta_stats = OrderedDict(sorted(theta_stats.items()))
        # theta_stats = torch.stack(list(theta_stats.values()))

        return theta_stats


class DynamicTopicEmbeddings(DynamicLDA):
    """
    here we follow (overleaf link)

    https://www.overleaf.com/7689487332skzdrpbgmcdm

    """
    word_embeddings_dim: int
    rho: nn.Parameter

    def __init__(self, model_dir=None, data_loader=None, model_name=None, **kwargs):
        if model_name is None:
            model_name = "neural_dynamical_topic_embeddings"
        DeepBayesianModel.__init__(self, model_name, model_dir=model_dir, data_loader=data_loader, **kwargs)

    @classmethod
    def get_parameters(cls):
        """
        here we provide an example of the minimum set of parameters requiered to instantiate the model
        """
        number_of_topics = 50

        r_hidden_transition_state_size = 32
        vocabulary_dim = 67
        parameters_sample = {"number_of_topics": number_of_topics,  # z
                             "number_of_documents": 100,
                             "number_of_words_per_document": 30,
                             "vocabulary_dim": vocabulary_dim,

                             "num_training_steps": 48,
                             "num_prediction_steps": 2,
                             "train_word_embeddings": False,
                             "topic_proportion_transformation": "gaussian_softmax",
                             "topic_lifetime_tranformation": "gaussian_softmax",
                             "delta": 0.005,
                             "theta_q_type": "q-INDEP",
                             "theta_q_parameters": {
                                 "observable_dim": vocabulary_dim,
                                 "layers_dim": [250, 250],
                                 "output_dim": 250,
                                 "hidden_state_dim": number_of_topics,
                                 "dropout": .1,
                                 "out_dropout": 0.1
                             },
                             "eta_q_type": "q-RNN",
                             "eta_q_parameters": {
                                 "observable_dim": vocabulary_dim,
                                 "layers_dim": 400,
                                 "num_rnn_layers": 4,
                                 "hidden_state_dim": r_hidden_transition_state_size,
                                 "hidden_state_transition_dim": 400,
                                 "dropout": .1,
                                 "out_dropout": 0.0
                             },
                             "model_path": os.path.join(project_path, 'results')}

        return parameters_sample

    # def update_parameters(self, data_loader, **kwargs):
    #     kwargs = super().update_parameters(data_loader, **kwargs)
    #     kwargs.get("theta_q_parameters").update({"observable_dim": self.vocabulary_dim,
    #                                              "hidden_state_dim": self.number_of_topics,
    #                                              "control_variable_dim": self.number_of_topics})
    #     return kwargs

    def set_parameters(self, **kwargs):
        self.vocabulary_dim = kwargs.get("vocabulary_dim")
        self.number_of_documents = kwargs.get("number_of_documents", None)
        self.num_training_steps = kwargs.get("num_training_steps", None)
        self.num_prediction_steps = kwargs.get("num_prediction_steps", None)

        self.word_embeddings_dim = kwargs.get("word_embeddings_dim", 100)
        self.number_of_topics = kwargs.get("number_of_topics", 10)

        self.train_word_embeddings = kwargs.get("train_word_embeddings")
        self.topic_transition_dim = self.number_of_topics * self.word_embeddings_dim

        self.eta_q_type = kwargs.get("eta_q_type")
        self.eta_q_parameters = kwargs.get("eta_q_parameters")
        self.eta_dim = kwargs.get("eta_q_parameters").get("hidden_state_dim")

        # TEXT
        self.theta_q_type = kwargs.get("theta_q_type")
        self.theta_q_parameters = kwargs.get("theta_q_parameters")

        self.topic_proportion_transformation = kwargs.get("topic_proportion_transformation")

    def define_deep_models(self):
        recognition_factory = RecognitionModelFactory()
        rho = nn.Embedding(self.vocabulary_dim, self.word_embeddings_dim)
        rho.weight.data = self.data_loader.vocab.vectors[:self.vocabulary_dim]
        self.rho = nn.Parameter(rho.weight.data.clone().float(), requires_grad=self.train_word_embeddings)

        self.eta_q = recognition_factory.create(self.eta_q_type, **self.eta_q_parameters)
        self.theta_q = recognition_factory.create(self.theta_q_type, **self.theta_q_parameters)

        self.mu_q_alpha = nn.Parameter(torch.randn(self.number_of_topics, self.num_training_steps, self.word_embeddings_dim))
        self.logvar_q_alpha = nn.Parameter(torch.randn(self.number_of_topics, self.num_training_steps, self.word_embeddings_dim))

    def get_beta(self, alphas):
        beta = torch.matmul(alphas, self.rho.T)
        beta = torch.softmax(beta, dim=-1)
        return beta

    def get_beta_eval(self):
        return self.get_beta(self.mu_q_alpha)

    def metrics(self, data, forward_results, epoch, mode="evaluation", data_loader=None):
        if mode == "validation_global":
            if epoch % self.metrics_logs == 0:
                super().metrics(data, forward_results, epoch, mode, data_loader)
                queries = []
                # queries = ['economic', 'assembly', 'security', 'management', 'debt', 'rights', 'africa']
                TEXT = ""
                for word in queries:
                    TEXT += 'word: {} .. neighbors: {}\n\n'.format(
                            word, nearest_neighbors(word, self.rho, data_loader.vocab.vocab, 20))

                self.writer.add_text("Nearest-Neighbors/", TEXT, epoch)
        return {}

    # ======================================================
    # POST - PROCESSING
    # ======================================================
    def top_words(self, data_loader, num_of_top_words=20, num_of_time_steps=3):
        important_words = self.get_words_by_importance()
        # from index to words
        vocabulary = data_loader.vocab
        top_word_per_topic = dict()
        steps = np.linspace(0, important_words.size(1)-1, num_of_time_steps, dtype=int)
        for k in range(self.number_of_topics):
            top_words_per_time = dict()
            for t in steps:
                topic_words = [vocabulary.itos[w] for w in important_words[k, t, :num_of_top_words].tolist()]
                top_words_per_time[f'Time {t}'] = topic_words
            top_word_per_topic[f'Topic {k}'] = top_words_per_time
        return top_word_per_topic

    def topic_coherence(self, data: np.ndarray) -> float:
        top_10 = self.get_words_by_importance()[:, :, :10]
        tc_all = []
        p_bar = tqdm(range(top_10.size(1)), desc="Calculating Topics Coherence:")
        for tt in p_bar:
            tc = self._topic_coherence(data, top_10[:, tt])
            tc_all.append(tc)
            p_bar.set_postfix_str(f'current topic coherence: {np.mean(tc_all)}')
        return np.mean(tc_all)

    def _topic_coherence(self, data: np.ndarray, top_10: torch.Tensor) -> np.ndarray:
        D = data.shape[0]
        TC = []
        # p_bar = tqdm(desc='calculating topics coherence', total=self.number_of_topics)
        for k in range(self.number_of_topics):
            top_10_k = top_10[k].tolist()
            TC_k = 0
            counter = 0
            word_count_ = self.data_loader.vocab.word_count
            for i, word in enumerate(top_10_k):
                # get D(w_i)
                D_wi = word_count_[word]
                j = i + 1
                tmp = 0
                while len(top_10_k) > j > i:
                    # get D(w_j) and D(w_i, w_j)
                    D_wj = word_count_[top_10_k[j]]
                    D_wi_wj = get_doc_freq(data, word, top_10_k[j])
                    # get f(w_i, w_j)
                    if D_wi_wj == 0:
                        f_wi_wj = -1
                    else:
                        f_wi_wj = -1 + (np.log(D_wi) + np.log(D_wj) - 2.0 * np.log(D)) / (np.log(D_wi_wj) - np.log(D))
                    # update tmp:
                    tmp += f_wi_wj
                    j += 1
                    counter += 1
                # update TC_k
                TC_k += tmp
            TC.append(TC_k / counter)
            # p_bar.set_postfix_str(f"TC: {np.mean(TC)}")
            # p_bar.update()
        TC = np.mean(TC)
        # print('Topic coherence is: {}'.format(TC))
        return TC

    def prediction(self, data_loader: ADataLoader, montecarlo_samples=10):
        self.eval()
        x = next(iter(data_loader.train))
        with torch.no_grad():
            forward_results = self.forward(x)

            _, _, _, _, eta_params = forward_results

            last_state_alpha = self.mu_q_alpha[:, -1, :], torch.exp(0.5 * self.logvar_q_alpha[:, -1, :]) * self.delta
            alpha_q_dist = Normal(*last_state_alpha)
            eta_std = torch.exp(0.5 * eta_params[1]) * self.delta
            last_state_eta = eta_params[0], eta_std
            eta_q_dist = Normal(*last_state_eta)

            alpha_sample = alpha_q_dist.sample(sample_shape=torch.Size([montecarlo_samples]))
            eta_sample = eta_q_dist.sample(sample_shape=torch.Size([montecarlo_samples]))
            pred_pp_all = []
            for tt in self.data_loader.prediction_times:
                current_document_count = int(data_loader.prediction_count_per_year)

                alpha_dist, eta_dist = self.get_transitions_dist(alpha_sample, eta_sample)
                theta, alpha_sample, eta_sample, alpha_doc = self.prediction_montecarlo_step(current_document_count, alpha_dist, eta_dist)
                # [mc,count,number_of_topics]
                pred_pp_step = torch.zeros((montecarlo_samples, current_document_count))
                bow = torch.from_numpy(data_loader.predict.dataset.corpus_per_year('prediction')[0]).to(self.device)
                text_size = bow.sum(1).double().to(self.device, non_blocking=True)
                for mc_index in range(montecarlo_samples):
                    alpha_per_doc = alpha_sample[mc_index]
                    theta_mc = theta[mc_index]  # [count, number_of_topics]
                    beta = torch.matmul(alpha_per_doc, self.rho.T)
                    beta = torch.softmax(beta, dim=-1)
                    pred_like = self.nll_test(theta_mc, bow, beta)
                    log_pp = (1. / text_size.float()) * pred_like
                    log_pp = torch.mean(log_pp)

                    pred_pp_step[mc_index] = log_pp
                pred_pp_all.append(pred_pp_step.mean())
            pred_pp = torch.mean(torch.stack(pred_pp_all)).item()
            return pred_pp

    def loss_perrone(self, x, forward_results):
        """
        nll [batch_size, max_lenght]

        """
        if 'bow' in x:
            text_size = x['bow'].sum().double()
        else:
            text_size = x['bow_h2'].sum().double()

        nll = forward_results[0]

        return nll.sum().item(), text_size.item()


class DynamicBetaFocusedTopic(DynamicTopicEmbeddings):
    ksi_q: nn.Module
    b_q: nn.Module
    ksi_transform_alpha: nn.Module
    ksi_transform_beta: nn.Module
    eta_transform: nn.Module
    delta: torch.Tensor
    name_: str = 'dynamic_beta_focused_topic_model'

    def __init__(self, model_dir=None, data_loader=None, **kwargs):
        DeepBayesianModel.__init__(self, self.name_, model_dir=model_dir, data_loader=data_loader, **kwargs)

    @classmethod
    def get_parameters(cls):
        """
        here we provide an example of the minimum set of parameters requiered to instantiate the model
        """
        number_of_topics = 50
        vocabulary_dim = 100
        parameters_sample = {"number_of_topics": number_of_topics,  # z
                             "number_of_documents": 100,
                             "number_of_words_per_document": 30,
                             "vocabulary_dim": vocabulary_dim,
                             "num_training_steps": 48,
                             "num_prediction_steps": 2,
                             "lambda_diversity": 0.1,
                             "train_word_embeddings": False,
                             "delta": 0.005,
                             "topic_embeddings": "dynamic",  # static, nonlinear-dynamic
                             "nonlinear_transition_prior": False,
                             "alpha0": 10,
                             "beta_nu": 10.0,
                             "b_q_type": "q-INDEP",
                             "b_q_parameters": {
                                 "distribution_type": "bernoulli",
                                 "observable_dim": vocabulary_dim,
                                 "layers_dim": [250, 250],
                                 "output_dim": 250,
                                 "hidden_state_dim": number_of_topics,
                                 "dropout": .1,
                                 "out_dropout": 0.1
                             },
                             "nu_q_type": "q-INDEP",
                             "nu_q_parameters": {
                                 "distribution_type": "kumaraswamy-beta",
                                 "observable_dim": vocabulary_dim,
                                 "layers_dim": [250, 250],
                                 "output_dim": 250,
                                 "hidden_state_dim": 1,
                                 "dropout": .1,
                                 "out_dropout": 0.1
                             },
                             "theta_q_type": "q-INDEP",
                             "theta_q_parameters": {
                                 "observable_dim": vocabulary_dim,
                                 "layers_dim": [250, 250],
                                 "output_dim": 250,
                                 "hidden_state_dim": number_of_topics,
                                 "dropout": .1,
                                 "out_dropout": 0.1
                             },
                             "ksi_prior_transition": {
                                 "layers_dim": [32, 32],
                                 "output_transformation": None,
                                 "dropout": 0.1
                             },
                             "eta_prior_transition": {
                                 "layers_dim": [32, 32],
                                 "output_transformation": None,
                                 "dropout": 0.1
                             },
                             "ksi_q_type": "q-RNN",
                             "ksi_q_parameters": {
                                 "observable_dim": vocabulary_dim,
                                 "layers_dim": 400,
                                 "num_rnn_layers": 4,
                                 "hidden_state_dim": 64,
                                 "hidden_state_transition_dim": 400,
                                 "dropout": .1,
                                 "out_dropout": 0.1
                             },
                             "eta_q_type": "q-RNN",
                             "eta_q_parameters": {
                                 "observable_dim": vocabulary_dim,
                                 "layers_dim": 400,
                                 "num_rnn_layers": 4,
                                 "hidden_state_dim": 64,
                                 "hidden_state_transition_dim": 400,
                                 "dropout": .1,
                                 "out_dropout": 0.1
                             },
                             "alpha_q_parameters": {
                                 "mean": {
                                     "layers_dim": [256, 256],
                                     "output_transformation": None,
                                     "dropout": 0.0
                                 },
                                 "lvar": {
                                     "layers_dim": [256, 256],
                                     "output_transformation": None,
                                     "dropout": 0.0
                                 }
                             },
                             "alpha_p_parameters": {
                                 "layers_dim": [256, 256],
                                 "output_transformation": None,
                                 "dropout": 0.0
                             },
                             "model_path": os.path.join(project_path, 'results')}

        return parameters_sample

    def update_parameters(self, data_loader, **kwargs):
        kwargs = super().update_parameters(data_loader, **kwargs)
        if self.topic_embeddings == "nonlinear-dynamic":
            kwargs.get("alpha_q_parameters").get("mean").update({"input_dim": self.word_embeddings_dim, "output_dim": self.word_embeddings_dim})
            kwargs.get("alpha_q_parameters").get("lvar").update({"input_dim": self.word_embeddings_dim, "output_dim": self.word_embeddings_dim})
            kwargs.get("alpha_p_parameters").update({"input_dim": self.word_embeddings_dim, "output_dim": self.word_embeddings_dim})

        kwargs.get("b_q_parameters").update({"observable_dim": data_loader.vocabulary_dim, "hidden_state_dim": self.number_of_topics, "control_variable_dim": self.number_of_topics})
        kwargs.get("nu_q_parameters").update({"alpha0": self.alpha0, "observable_dim": data_loader.vocabulary_dim,  "control_variable_dim": self.ksi_dim, "num_topics": self.number_of_topics})
        kwargs.get("theta_q_parameters").update({"hidden_state_dim": self.eta_dim, "control_variable_dim": self.eta_dim})
        kwargs.get("ksi_q_parameters").update({"observable_dim": data_loader.vocabulary_dim})
        kwargs.get("eta_q_parameters").update({"observable_dim": data_loader.vocabulary_dim, "hidden_state_dim": self.eta_dim})
        kwargs.get("ksi_prior_transition").update({"input_dim": self.ksi_dim, "output_dim": self.ksi_dim})
        kwargs.get("eta_prior_transition").update({"input_dim": self.eta_dim, "output_dim": self.eta_dim})
        return kwargs

    def set_parameters(self, **kwargs):
        super().set_parameters(**kwargs)

        self.ksi_q_type = kwargs.get("ksi_q_type")
        self.ksi_q_parameters = kwargs.get("ksi_q_parameters")

        self.ksi_dim = kwargs.get("ksi_q_parameters").get("hidden_state_dim")
        self.zeta_dim = kwargs.get("theta_q_parameters").get("hidden_state_dim")
        self.alpha_q_parameters = kwargs.get("alpha_q_parameters")
        self.alpha_p_parameters = kwargs.get("alpha_p_parameters")

        self.b_q_type = kwargs.get("b_q_type")
        self.b_q_parameters = kwargs.get("b_q_parameters")
        self.nu_q_type = kwargs.get("nu_q_type")
        self.nu_q_parameters = kwargs.get("nu_q_parameters")
        self.lambda_diversity = kwargs.get("lambda_diversity")
        self.ksi_prior_transition_params = kwargs.get("ksi_prior_transition")
        self.eta_prior_transition_params = kwargs.get("eta_prior_transition")

        self.nonlinear_transition_prior = kwargs.get("nonlinear_transition_prior")
        self.topic_embeddings = kwargs.get("topic_embeddings")

        self.alpha0 = kwargs.get("alpha0")
        self.beta_nu = kwargs.get("beta_nu")

    def define_deep_models(self):
        super().define_deep_models()
        self.theta_stats = defaultdict(list)
        self.b_stats = defaultdict(list)
        self.beta_stats = defaultdict(list)
        recognition_factory = RecognitionModelFactory()

        self.ksi_q = recognition_factory.create(self.ksi_q_type, **self.ksi_q_parameters)
        self.nu_q = recognition_factory.create(self.nu_q_type, **self.nu_q_parameters)
        self.b_q = recognition_factory.create(self.b_q_type, **self.b_q_parameters)

        self.ksi_transform_alpha = nn.Sequential(nn.Linear(self.ksi_dim, 1), nn.Softplus())
        self.ksi_transform_beta = nn.Sequential(nn.Linear(self.ksi_dim, 1), nn.Softplus())
        self.eta_transform = nn.Linear(self.eta_dim, self.zeta_dim)
        self.hidden_state_to_topics = nn.Linear(self.zeta_dim, self.number_of_topics)
        if self.topic_embeddings == "dynamic":
            self.div_reg = self.diversity_regularizer_dynamic
            self.alpha_q = super().q_alpha
            self.mu_q_alpha = nn.Parameter(torch.randn(self.number_of_topics, self.num_training_steps, self.word_embeddings_dim))
            self.logvar_q_alpha = nn.Parameter(torch.randn(self.number_of_topics, self.num_training_steps, self.word_embeddings_dim))
        elif self.topic_embeddings == "nonlinear-dynamic":
            self.mu_q_alpha = nn.Parameter(torch.randn(self.number_of_topics, self.word_embeddings_dim))
            self.logvar_q_alpha = nn.Parameter(torch.randn(self.number_of_topics, self.word_embeddings_dim))
            self.alpha_q_m = MLP(**self.alpha_q_parameters.get('mean'))
            self.alpha_q_lvar = MLP(**self.alpha_q_parameters.get('lvar'))
            self.p_alpha_m = MLP(**self.alpha_p_parameters)
            self.alpha_q = self.q_alpha
            self.div_reg = self.diversity_regularizer
        elif self.topic_embeddings == "static":
            self.alpha_q = self.alpha_static_q
            self.mu_q_alpha = nn.Parameter(torch.randn(self.number_of_topics, self.word_embeddings_dim))
            self.div_reg = self.diversity_regularizer
        if self.nonlinear_transition_prior:
            self.p_ksi_m = MLP(**self.ksi_prior_transition_params)
            self.p_eta_m = MLP(**self.eta_prior_transition_params)
        else:
            self.p_ksi_m = None
            self.p_eta_m = None

    def initialize_inference(self, data_loader, **inference_parameters):
        super().initialize_inference(data_loader=data_loader, **inference_parameters)
        self.eta_q.device = self.device
        self.ksi_q.device = self.device
        self.nu_q.device = self.device
        self.b_q.device = self.device

    def move_to_device(self):
        self.eta_q.device = self.device
        self.ksi_q.device = self.device
        self.nu_q.device = self.device
        self.b_q.device = self.device

    def forward(self, x):
        """
        parameters
        ----------
        batchdata ()
        returns
        -------
        z (batch_size*sequence_lenght,self.hidden_state_dim)
        recognition_parameters = (z_mean,z_var)
        (batch_size*sequence_lenght,self.hidden_state_dim)
        likelihood_parameters = (likelihood_mean,likelihood_variance)
        """

        bow_time_indexes = x['time'].squeeze().long().to(self.device, non_blocking=True)
        corpora = x['corpus'][:1].float().to(self.device, non_blocking=True)

        if 'bow' in x:
            bow = x['bow'].to(self.device, non_blocking=True)
            normalized_bow = bow.float() / bow.sum(1, True)
        else:
            bow = x['bow_h1'].to(self.device, non_blocking=True)
            normalized_bow = bow.float() / bow.sum(1, True)
            bow = x['bow_h2'].to(self.device, non_blocking=True)

        ksi, ksi_params, ksi_kl = self.ksi_q(corpora, self.p_ksi_m)
        eta, eta_params, eta_kl = self.eta_q(corpora, self.p_eta_m)
        alpha, alpha_kl = self.alpha_q()

        # inference of the nu variable
        def ksi_transform(x): return (self.ksi_transform_alpha(x), self.ksi_transform_beta(x))
        nu, q_beta_params, p_beta_params, nu_kl = self.nu_q(corpora.squeeze(0), ksi, ksi_transform)

        eta_per_document = eta[bow_time_indexes]
        zeta, zeta_params, zeta_kl = self.theta_q(normalized_bow, eta_per_document, self.eta_transform)

        nu_per_document = nu[bow_time_indexes]
        b, pi, b_kl = self.b_q(normalized_bow, nu_per_document)
        zeta = self.hidden_state_to_topics(zeta)
        theta = b * torch.exp(zeta) / torch.sum(b * torch.exp(zeta), dim=-1, keepdim=True)

        beta = self.get_beta(alpha)
        if self.topic_embeddings != "static":
            beta = beta[bow_time_indexes]
        nll = self.nll(theta, bow, beta)

        return nll, ksi_kl, eta_kl, nu_kl, zeta_kl, b_kl, alpha_kl, torch.softmax(zeta, dim=-1), pi, ksi, eta, ksi_params, eta_params, zeta_params, q_beta_params, p_beta_params

    def nll(self, theta, bow, beta):
        if self.topic_embeddings != "static":
            loglik = torch.bmm(theta.unsqueeze(1), beta).squeeze(1)
        else:
            loglik = torch.matmul(theta, beta)
        if self.training:
            loglik = torch.log(loglik + 1e-6)
        else:
            loglik = torch.log(loglik)
        nll = -loglik * bow

        return nll.sum(-1)

    def nll_test(self, theta, bow, beta):
        loglik = torch.matmul(theta, beta)

        loglik = torch.log(loglik)
        nll = -loglik * bow

        return nll.sum(-1)

    def alpha_static_q(self):
        return self.mu_q_alpha, torch.tensor(0.0, device=self.device)

    def q_alpha(self):
        rho_dim = self.mu_q_alpha.size(-1)
        alphas = torch.zeros(self.num_training_steps, self.number_of_topics, rho_dim, device=self.device)
        alpha_t = reparameterize(self.mu_q_alpha, self.logvar_q_alpha)
        alphas[0] = alpha_t
        q_m = torch.zeros(self.num_training_steps - 1, self.number_of_topics, rho_dim, device=self.device)
        q_lvar = torch.zeros(self.num_training_steps - 1, self.number_of_topics, rho_dim, device=self.device)

        p_lvar = torch.log(self.delta * torch.ones(self.num_training_steps - 1, self.number_of_topics, rho_dim, device=self.device))
        kl = kullback_leibler(self.mu_q_alpha, self.logvar_q_alpha, 'sum')

        for t in range(1, self.num_training_steps):
            mu_t = self.alpha_q_m(alpha_t)
            lvar_t = self.alpha_q_lvar(alpha_t)
            q_m[t - 1] = mu_t
            q_lvar[t - 1] = lvar_t
            alpha_t = reparameterize(mu_t, lvar_t)
            alphas[t] = alpha_t

        kl += kullback_leibler_two_gaussians(q_m, q_lvar, self.p_alpha_m(alphas[:-1]), p_lvar, 'sum')

        return alphas, kl

    def loss(self, x, forward_results, data_set, epoch):
        """
        nll [batch_size, max_lenght]

        """
        if 'bow' in x:
            text_size = x['bow'].sum(1).double().to(self.device, non_blocking=True)
        else:
            text_size = x['bow_h2'].sum(1).double().to(self.device, non_blocking=True)

        nll, ksi_kl, eta_kl, nu_kl, zeta_kl, b_kl, alpha_kl, _, _, _, _, _, _, _, _, _ = forward_results

        coeff = 1.0
        if self.training:
            coeff = len(data_set)
        loss = (nll.sum() + zeta_kl + b_kl + nu_kl * self.beta_nu) * coeff + (eta_kl + ksi_kl + alpha_kl)  # [number of batches]
        topic_diversity_reg = self.div_reg(self.mu_q_alpha)
        loss = loss - self.lambda_diversity * topic_diversity_reg

        log_perplexity = nll / text_size.float()
        log_perplexity = torch.mean(log_perplexity)
        perplexity = torch.exp(log_perplexity)

        return {"loss": loss,
                "NLL-Loss": nll.sum() * coeff,
                "KL-Loss-Alpha": alpha_kl,
                "KL-Loss-Eta": eta_kl,
                "KL-Loss-Ksi": ksi_kl,
                "KL-Loss-Theta": zeta_kl * coeff,
                "KL-Loss-Nu": nu_kl,
                "KL-Loss-B": b_kl * coeff,
                "Diversity": topic_diversity_reg,
                "PPL-Blei": perplexity,
                "PPL": torch.exp(nll.sum() / text_size.sum().float()),
                "Log-Likelihood": log_perplexity}

    def get_words_by_importance(self) -> torch.Tensor:
        with torch.no_grad():
            alpha = self.mu_q_alpha
            if self.topic_embeddings == "nonlinear-dynamic":
                alpha, _ = self.alpha_q()
                alpha = torch.transpose(alpha, 1, 0)
            beta = self.get_beta(alpha)
            important_words = torch.argsort(beta, dim=-1, descending=True)
            return important_words

    def top_words(self, data_loader, num_of_top_words=20):
        important_words = self.get_words_by_importance()
        # from index to words
        vocabulary = data_loader.vocab
        top_word_per_topic = dict()

        for k in range(self.number_of_topics):
            topic_words = [vocabulary.itos[w] for w in important_words[k, :num_of_top_words].tolist()]
            top_word_per_topic[f'TOPIC {k}'] = topic_words
        return top_word_per_topic

    def top_words_dynamic(self, data_loader, num_of_top_words=20):
        important_words = self.get_words_by_importance()
        # from index to words
        vocabulary = data_loader.vocab
        top_word_per_topic = dict()

        tt = important_words.size(1) // 2
        for k in range(self.number_of_topics):
            top_words_per_time = dict()
            for t in [0, tt, important_words.size(1) - 1]:
                topic_words = [vocabulary.itos[w] for w in important_words[k, t, :num_of_top_words].tolist()]
                top_words_per_time[f'TIME {t}'] = topic_words
            top_word_per_topic[f'TOPIC {k}'] = top_words_per_time
        return top_word_per_topic

    def get_time_series(self, dataset):
        b_stats = defaultdict(list)
        theta_stats = defaultdict(list)
        for x in tqdm(dataset, desc="Building time series"):
            _, _, _, _, _, _, _, theta, pi, ksi, eta, _, _, _ = self.forward(x)
            time_idx = x['time'].squeeze()
            for id, t, _nu in zip(time_idx, theta, pi):
                theta_stats[id.item()].append(t)
                b_stats[id.item()].append(_nu)

        for key, value in theta_stats.items():
            theta_stats[key] = torch.stack(value).mean(0)
        for key, value in b_stats.items():
            b_stats[key] = torch.stack(value).mean(0)

        theta_stats = OrderedDict(sorted(theta_stats.items()))
        b_stats = OrderedDict(sorted(b_stats.items()))

        theta_stats = torch.stack(list(theta_stats.values()))
        b_stats = torch.stack(list(b_stats.values()))
        return theta_stats, b_stats

    def metrics(self, data, forward_results, epoch, mode="evaluation", data_loader=None):
        if mode == "validation" and epoch % self.metrics_logs == 0:
            _, _, _, _, _, _, _, theta, pi, _, _, _, _, _, q_beta_params, p_beta_params = forward_results
            self.beta_stats["a"].append(q_beta_params[0].squeeze())
            self.beta_stats["b"].append(q_beta_params[1].squeeze())
            self.beta_stats["alpha"].append(p_beta_params[0].squeeze())
            self.beta_stats["beta"].append(p_beta_params[1].squeeze())
            time_idx = data["time"].squeeze()

            for id, t, _nu in zip(time_idx, theta, pi):
                self.theta_stats[id.item()].append(t)
                self.b_stats[id.item()].append(_nu)

        if mode == "validation_global" and epoch % self.metrics_logs == 0:
            for key, value in self.theta_stats.items():
                self.theta_stats[key] = torch.stack(value).mean(0)
            for key, value in self.b_stats.items():
                self.b_stats[key] = torch.stack(value).mean(0)

            for key, value in self.beta_stats.items():
                self.beta_stats[key] = dict()
                self.beta_stats[key]["mean"] = torch.stack(value).mean(0).cpu().numpy()
                self.beta_stats[key]["std"] = torch.stack(value).std(0).cpu().numpy()

            self.theta_stats = OrderedDict(sorted(self.theta_stats.items()))
            self.b_stats = OrderedDict(sorted(self.b_stats.items()))
            self.theta_stats = torch.stack(list(self.theta_stats.values()))
            self.b_stats = torch.stack(list(self.b_stats.values()))

            self.plot_topics_ts(self.theta_stats.cpu().numpy(), self.b_stats.cpu().numpy(), '/Topic-Time-Series', y_lim=(-0.1, 1.1))
            self.plot_beta_params(self.beta_stats, '/Beta vs Kuma Params', y_lim=(-0.1, 1.1))
            self.plot_beta_kumar_dist(self.beta_stats, '/Beta vs Kuma Dist')
            self.b_stats = defaultdict(list)
            self.theta_stats = defaultdict(list)
            self.beta_stats = defaultdict(list)
            if self.topic_embeddings != "static":
                self._log_top_words_dynamic(data_loader)
            else:
                self._log_top_words(data_loader)
        return {}

    def plot_beta_kumar_dist(self, param, label):
        x = np.sort(np.random.random(100))
        ntime = len(param["a"]["mean"])
        ncols = 5
        nrows = ntime // ncols + ntime % ncols
        fig, axis = plt.subplots(nrows, ncols, sharex=True, figsize=(15, 20))
        axis = axis.flatten()
        color_p = 'tab:red'
        color_q = 'tab:blue'
        for i in range(ntime):
            a_mean = param["a"]["mean"][i]
            a_std = param["a"]["std"][i]
            a_lower = a_mean-1.96*a_std
            a_upper = a_mean+1.96*a_std
            b_mean = param["b"]["mean"][i]
            b_std = param["b"]["std"][i]
            b_lower = b_mean-1.96*b_std
            b_upper = b_mean+1.96*b_std
            prob_post = kumaraswamy_pdf(x, a_mean, b_mean)
            prob_post_lower = kumaraswamy_pdf(x, a_lower, b_lower)
            prob_post_upper = kumaraswamy_pdf(x, a_upper, b_upper)
            axis[i].plot(x, prob_post, label="posterior")
            axis[i].fill_between(x, prob_post_lower, prob_post_upper,  color=color_q, alpha=0.2)

            alpha_mean = param["alpha"]["mean"][i]
            alpha_std = param["alpha"]["std"][i]
            beta_mean = param["beta"]["mean"][i]
            beta_std = param["beta"]["std"][i]
            alpha_lower = alpha_mean - 1.96*alpha_std
            alpha_upper = alpha_mean + 1.96*alpha_std
            beta_lower = beta_mean - 1.96*beta_std
            beta_upper = beta_mean + 1.96*beta_std
            prob_prior = beta_pdf(x, alpha_mean, beta_mean)
            prob_prior_lower = beta_pdf(x, alpha_lower, beta_lower)
            prob_prior_upper = beta_pdf(x, alpha_upper, beta_upper)
            axis[i].fill_between(x, prob_prior_lower, prob_prior_upper,  color=color_p, alpha=0.2)
            axis[i].plot(x, prob_prior, label="prior")
            axis[i].legend()
        plt.tight_layout()
        buf = io.BytesIO()
        plt.savefig(buf, format='jpeg')
        plt.close()
        buf.seek(0)
        image = PIL.Image.open(buf)
        image = ToTensor()(image).unsqueeze(0)
        self.writer.add_images(label, image, self.number_of_iterations)

    def plot_beta_params(self, param, label, y_lim=None):
        ntime = len(param["a"]["mean"])
        x = range(ntime)
        fig, axis = plt.subplots(1, 2, sharex=True, figsize=(15, 8))
        axis = axis.flatten()

        ax0 = axis[0]
        ax1 = axis[1]
        color_p = 'tab:red'
        color_q = 'tab:blue'
        ax0.set_ylabel('value')
        ax0.plot(x, param["alpha"]["mean"], color=color_p, label="prior")
        ax0.plot(x, param["a"]["mean"],  color=color_q, label="posterior")
        lower = param["alpha"]["mean"] - 1.96*param["alpha"]["std"]
        upper = param["alpha"]["mean"] + 1.96*param["alpha"]["std"]
        ax0.fill_between(x, lower, upper,  color=color_p, alpha=0.2)
        lower = param["a"]["mean"] - 1.96*param["a"]["std"]
        upper = param["a"]["mean"] + 1.96*param["a"]["std"]
        ax0.fill_between(x, lower, upper,  color=color_q, alpha=0.2)
        ax0.legend()

        ax1.set_ylabel('value')
        ax1.plot(x, param["beta"]["mean"],  color=color_p, label="prior")
        ax1.plot(x, param["b"]["mean"], color=color_q, label="posterior")
        lower = param["beta"]["mean"] - 1.96*param["beta"]["std"]
        upper = param["beta"]["mean"] + 1.96*param["beta"]["std"]
        ax1.fill_between(x, lower, upper,  color=color_p, alpha=0.2)
        lower = param["b"]["mean"] - 1.96*param["b"]["std"]
        upper = param["b"]["mean"] + 1.96*param["b"]["std"]
        ax1.fill_between(x, lower, upper,  color=color_q, alpha=0.2)
        ax1.legend()
        plt.tight_layout()
        buf = io.BytesIO()
        plt.savefig(buf, format='jpeg')
        plt.close()
        buf.seek(0)
        image = PIL.Image.open(buf)
        image = ToTensor()(image).unsqueeze(0)
        self.writer.add_images(label, image, self.number_of_iterations)

    def plot_topics_ts(self, theta, b, label, y_lim=None):
        ntime, ntopics = theta.shape
        x = range(ntime)
        ncols = 5
        nrows = ntopics // ncols + ntopics % ncols
        fig, axis = plt.subplots(nrows, ncols, sharex=True, figsize=(15, 2*nrows))
        axis = axis.flatten()
        for i in range(ntopics):
            ax = axis[i]
            color = 'tab:red'
            ax.set_ylabel('theta', color=color)
            ax.tick_params(axis='y', labelcolor=color)
            sns.lineplot(x=x, y=theta[:, i], ax=ax, color=color)
            ax2 = ax.twinx()
            color = 'tab:blue'
            ax2.set_ylabel('b', color=color)
            if y_lim is not None:
                ax2.set_ylim(y_lim)
            # print(i, b[:, i])
            sns.lineplot(x=x, y=b[:, i], ax=ax2, color=color)
            ax2.tick_params(axis='y', labelcolor=color)
        plt.tight_layout()
        buf = io.BytesIO()
        plt.savefig(buf, format='jpeg')
        plt.close()
        buf.seek(0)
        image = PIL.Image.open(buf)
        image = ToTensor()(image).unsqueeze(0)
        self.writer.add_images(label, image, self.number_of_iterations)

    def _log_top_words(self, data_loader):
        top_word_per_topic = self.top_words(data_loader, num_of_top_words=20)
        TEXT = "\n".join(["<strong>{0}</strong> ".format(topic) + " ".join(words) + "\n" for topic, words in top_word_per_topic.items()])
        self.writer.add_text("/Topics-Top-Words", TEXT, self.number_of_iterations)

    def _log_top_words_dynamic(self, data_loader):
        top_word_per_topic = self.top_words_dynamic(data_loader, num_of_top_words=20)
        TEXT = ""
        for topic, value in top_word_per_topic.items():
            for time, words in value.items():
                TEXT += f'<strong>{topic} --- {time}</strong>: {" ".join(words)}\n\n'
            TEXT += "*" * 1000 + "\n"
        self.writer.add_text("/Topics-Top-Words", TEXT, self.number_of_iterations)

    def diversity_regularizer(self, x):
        x = x / (torch.norm(x, dim=-1, keepdim=True) + 1e-12)
        cosine_simi = torch.tensordot(x, x, dims=[[1], [1]]).abs()
        angles = torch.acos(torch.clamp(cosine_simi, -1. + 1e-7, 1. - 1e-7))
        angles_mean = angles.mean()
        var = ((angles - angles_mean) ** 2).mean()
        return angles_mean - var

    def diversity_regularizer_dynamic(self, x):  # [K, T, D]
        x = torch.transpose(x, 1, 0)
        x = x / (torch.norm(x, dim=-1, keepdim=True) + 1e-12)
        topics_cosine_similarity = torch.einsum('bij,bgj->big', x, x).abs()
        topics_similarity = torch.acos(torch.clamp(topics_cosine_similarity, -1. + 1e-7, 1. - 1e-7))
        topic_similarity_mean = topics_similarity.mean(-1).mean(-1)
        topic_similarity_variance = ((topics_similarity - topic_similarity_mean.reshape(-1, 1, 1)) ** 2).mean(-1).mean(-1)
        topic_diversity = (topic_similarity_mean - topic_similarity_variance)
        return topic_diversity.mean()

    def topic_coherence(self, data: np.ndarray) -> float:
        if self.topic_embeddings == "static":
            top_10 = self.get_words_by_importance()[:, :10].tolist()
            D = data.shape[0]
            TC = []
            for k in tqdm(range(self.number_of_topics), total=self.number_of_topics, desc="Calculate coherence for topic"):
                top_10_k = top_10[k]
                TC_k = 0
                counter = 0
                word_count_ = self.data_loader.vocab.word_count
                for i, word in enumerate(top_10_k):
                    # get D(w_i)
                    D_wi = word_count_[word]
                    j = i + 1
                    tmp = 0
                    while len(top_10_k) > j > i:
                        # get D(w_j) and D(w_i, w_j)
                        D_wj = word_count_[top_10_k[j]]
                        D_wi_wj = get_doc_freq(data, word, top_10_k[j])
                        # get f(w_i, w_j)
                        if D_wi_wj == 0:
                            f_wi_wj = -1
                        else:
                            f_wi_wj = -1 + (np.log(D_wi) + np.log(D_wj) - 2.0 * np.log(D)) / (np.log(D_wi_wj) - np.log(D))
                        # update tmp:
                        tmp += f_wi_wj
                        j += 1
                        counter += 1
                    # update TC_k
                    TC_k += tmp
                TC.append(TC_k / float(counter))
            # print('Topic coherence is: {}'.format(TC))
            TC = np.mean(TC)
        else:
            TC = super(DynamicBetaFocusedTopic, self).topic_coherence(data)
        return TC

    def topic_diversity(self, topk: int = 25) -> float:
        if self.topic_embeddings == "static":
            important_words_per_topic = self.get_words_by_importance()
            list_w = important_words_per_topic[:, :topk]
            n_unique = len(torch.unique(list_w))
            td = n_unique / (topk * self.number_of_topics)
        else:
            td = super().topic_diversity(topk)
        return td

    def get_transitions_dist(self, alpha_sample, eta_sample, ksi_sample):
        """
        returns torch.distributions Normal([montecarlo_samples,number_of_topics])
        """
        alpha_std = torch.ones_like(alpha_sample) * self.delta
        eta_std = torch.ones_like(eta_sample) * self.delta
        ksi_std = torch.ones_like(ksi_sample) * self.delta

        v_dist = Normal(alpha_sample, alpha_std)
        r_dist = Normal(eta_sample, eta_std)
        ksi_dist = Normal(ksi_sample, ksi_std)
        return v_dist, r_dist, ksi_dist

    def prediction(self, data_loader: ADataLoader, montecarlo_samples=10):
        self.eval()
        x = next(iter(data_loader.train))
        with torch.no_grad():
            forward_results = self.forward(x)

            _, _, _, _, _, _, _, _, pi, _, _, ksi_params, eta_params, zeta_params = forward_results
            if self.topic_embeddings != "static":
                last_state_alpha = self.mu_q_alpha[:, -1, :], torch.exp(0.5 * self.logvar_q_alpha[:, -1, :]) * self.delta
                alpha_q_dist = Normal(*last_state_alpha)
                alpha_sample = alpha_q_dist.sample(sample_shape=torch.Size([montecarlo_samples]))
            eta_std = torch.exp(0.5 * eta_params[1]) * self.delta
            last_state_eta = eta_params[0], eta_std
            eta_q_dist = Normal(*last_state_eta)

            ksi_std = torch.exp(0.5 * ksi_params[1]) * self.delta
            last_state_ksi = ksi_params[0], ksi_std
            ksi_q_dist = Normal(*last_state_ksi)

            eta_sample = eta_q_dist.sample(sample_shape=torch.Size([montecarlo_samples]))
            ksi_sample = ksi_q_dist.sample(sample_shape=torch.Size([montecarlo_samples]))
            pred_pp_all = []
            for tt in self.data_loader.prediction_times:
                current_document_count = int(data_loader.prediction_count_per_year)
                if self.topic_embeddings == 'static':
                    eta_dist, ksi_dist = super().get_transitions_dist(eta_sample, ksi_sample)
                    theta, alpha_sample, eta_sample = self.prediction_montecarlo_step_static(current_document_count, eta_dist, ksi_dist)
                else:
                    alpha_dist, eta_dist, ksi_dist = self.get_transitions_dist(alpha_sample, eta_sample, ksi_sample)
                    theta, alpha_sample, eta_sample = self.prediction_montecarlo_step(current_document_count, alpha_dist, eta_dist, ksi_dist)
                # [mc,count,number_of_topics]
                pred_pp_step = torch.zeros((montecarlo_samples, current_document_count))
                bow = torch.from_numpy(data_loader.predict.dataset.corpus_per_year('prediction')[0], device=self.device)
                text_size = bow.sum(1).double().to(self.device, non_blocking=True)
                for mc_index in range(montecarlo_samples):
                    alpha_per_doc = alpha_sample[mc_index]
                    theta_mc = theta[mc_index]  # [count, number_of_topics]
                    beta = torch.matmul(alpha_per_doc, self.rho.T)
                    beta = torch.softmax(beta, dim=-1)
                    pred_like = self.nll_test(theta_mc, bow, beta)
                    log_pp = (1. / text_size.float()) * pred_like
                    log_pp = self.nanmean(log_pp)

                    pred_pp_step[mc_index] = log_pp
                pred_pp_all.append(pred_pp_step.mean())
            pred_pp = torch.mean(torch.stack(pred_pp_all)).item()
            return pred_pp

    def nanmean(self, v, *args, inplace=False, **kwargs):
        if not inplace:
            v = v.clone()
        is_nan = torch.isnan(v)
        v[is_nan] = 0
        return v.sum(*args, **kwargs) / (~is_nan).float().sum(*args, **kwargs)

    def prediction_montecarlo_step_static(self, current_document_count, eta_dist, ksi_dist):
        eta_sample = eta_dist.sample()  # [mc,n_topics, topic_transition_dim]
        ksi_sample = ksi_dist.sample()  # [mc,n_topics, topic_transition_dim]
        n_mc, _ = eta_sample.shape
        alpha_sample = self.mu_q_alpha.unsqueeze(0).repeat(n_mc, 1, 1)  # [mc, n_topics, topic_transition_dim]
        a = self.nu_q.alpha0 * self.ksi_transform_alpha(ksi_sample)
        nu_dist = Beta(a, torch.ones_like(a))
        nu_sample = nu_dist.sample()
        pi = torch.cumprod(nu_sample, dim=-1)
        b_dist = Bernoulli(pi)
        b_sample = b_dist.sample(sample_shape=torch.Size([current_document_count]))

        zeta_param = self.eta_transform(eta_sample)
        zeta_dist = Normal(zeta_param, torch.ones_like(zeta_param))
        zeta_sample = zeta_dist.sample(sample_shape=torch.Size([current_document_count]))

        theta_sample = b_sample * torch.exp(zeta_sample) / torch.sum(b_sample * torch.exp(zeta_sample), dim=-1, keepdim=True)  # [mc,current_doc_count,
        # number_of_topics]

        theta_sample = theta_sample.view(n_mc, current_document_count, -1).contiguous()

        return theta_sample, alpha_sample, eta_sample

    def prediction_montecarlo_step(self, current_document_count, alpha_dist, eta_dist, ksi_dist):
        eta_sample = eta_dist.sample()  # [mc,n_topics, topic_transition_dim]
        ksi_sample = ksi_dist.sample()  # [mc,n_topics, topic_transition_dim]
        n_mc, _ = eta_sample.shape
        alpha_sample = alpha_dist.sample()  # [mc, n_topics, topic_transition_dim]
        a = self.nu_q.alpha0 * self.ksi_transform_alpha(ksi_sample)
        nu_dist = Beta(a, torch.ones_like(a))
        nu_sample = nu_dist.sample()
        pi = torch.cumprod(nu_sample, dim=-1)
        b_dist = Bernoulli(pi)
        b_sample = b_dist.sample(sample_shape=torch.Size([current_document_count]))

        zeta_param = self.eta_transform(eta_sample)
        zeta_dist = Normal(zeta_param, torch.ones_like(zeta_param))
        zeta_sample = zeta_dist.sample(sample_shape=torch.Size([current_document_count]))

        theta_sample = b_sample * torch.exp(zeta_sample) / torch.sum(b_sample * torch.exp(zeta_sample), dim=-1, keepdim=True)  # [mc,current_doc_count,
        # number_of_topics]

        theta_sample = theta_sample.view(n_mc, current_document_count, -1).contiguous()

        return theta_sample, alpha_sample, eta_sample


class DynamicBinaryFocusedTopic(DynamicTopicEmbeddings):
    ksi_q: nn.Module
    b_q: nn.Module
    ksi_transform_alpha: nn.Module
    ksi_transform_beta: nn.Module
    eta_2_zeta: nn.Module
    delta: torch.Tensor
    name_: str = 'dynamic_binary_focused_topic_model'

    def __init__(self, model_dir=None, data_loader=None, **kwargs):
        DeepBayesianModel.__init__(self, self.name_, model_dir=model_dir, data_loader=data_loader, **kwargs)

    @classmethod
    def get_parameters(cls):
        """
        here we provide an example of the minimum set of parameters requiered to instantiate the model
        """
        number_of_topics = 50
        vocabulary_dim = 100
        parameters_sample = {"number_of_topics": number_of_topics,  # z
                             "number_of_documents": 100,
                             "number_of_words_per_document": 30,
                             "vocabulary_dim": vocabulary_dim,
                             "num_training_steps": 48,
                             "num_prediction_steps": 2,
                             "lambda_diversity": 0.1,
                             "train_word_embeddings": False,
                             "delta": 0.005,
                             "topic_embeddings": "dynamic",  # static, nonlinear-dynamic
                             "nonlinear_transition_prior": False,
                             "pi0": 0.5,
                             "b_q_type": "q-INDEP",
                             "b_q_parameters": {
                                 "distribution_type": "bernoulli",
                                 "observable_dim": vocabulary_dim,
                                 "layers_dim": [250, 250],
                                 "output_dim": 250,
                                 "hidden_state_dim": number_of_topics,
                                 "dropout": .1,
                                 "out_dropout": 0.1
                             },
                             "theta_q_type": "q-INDEP",
                             "theta_q_parameters": {
                                 "observable_dim": vocabulary_dim,
                                 "layers_dim": [250, 250],
                                 "output_dim": 250,
                                 "hidden_state_dim": number_of_topics,
                                 "dropout": .1,
                                 "out_dropout": 0.1
                             },
                             "ksi_2_pi_transform": {
                                 "layers_dim": [32, 32],
                                 "output_transformation": "sigmoid",
                                 "dropout": 0.1
                             },
                             "eta_2_zeta_transform": {
                                 "layers_dim": [32, 32],
                                 "output_transformation": None,
                                 "dropout": 0.1
                             },
                             "ksi_prior_transition": {
                                 "layers_dim": [32, 32],
                                 "output_transformation": None,
                                 "dropout": 0.1
                             },
                             "eta_prior_transition": {
                                 "layers_dim": [32, 32],
                                 "output_transformation": None,
                                 "dropout": 0.1
                             },
                             "ksi_q_type": "q-RNN",
                             "ksi_q_parameters": {
                                 "observable_dim": vocabulary_dim,
                                 "layers_dim": 400,
                                 "num_rnn_layers": 4,
                                 "hidden_state_dim": 64,
                                 "hidden_state_transition_dim": 400,
                                 "dropout": .1,
                                 "out_dropout": 0.1
                             },
                             "eta_q_type": "q-RNN",
                             "eta_q_parameters": {
                                 "observable_dim": vocabulary_dim,
                                 "layers_dim": 400,
                                 "num_rnn_layers": 4,
                                 "hidden_state_dim": 64,
                                 "hidden_state_transition_dim": 400,
                                 "dropout": .1,
                                 "out_dropout": 0.1
                             },
                             "alpha_q_parameters": {
                                 "mean": {
                                     "layers_dim": [256, 256],
                                     "output_transformation": None,
                                     "dropout": 0.0
                                 },
                                 "lvar": {
                                     "layers_dim": [256, 256],
                                     "output_transformation": None,
                                     "dropout": 0.0
                                 }
                             },
                             "alpha_p_parameters": {
                                 "layers_dim": [256, 256],
                                 "output_transformation": None,
                                 "dropout": 0.0
                             },
                             "model_path": os.path.join(project_path, 'results')}

        return parameters_sample

    def update_parameters(self, data_loader, **kwargs):
        kwargs = super().update_parameters(data_loader, **kwargs)
        if self.topic_embeddings == "nonlinear-dynamic":
            kwargs.get("alpha_q_parameters").get("mean").update({"input_dim": self.word_embeddings_dim, "output_dim": self.word_embeddings_dim})
            kwargs.get("alpha_q_parameters").get("lvar").update({"input_dim": self.word_embeddings_dim, "output_dim": self.word_embeddings_dim})
            kwargs.get("alpha_p_parameters").update({"input_dim": self.word_embeddings_dim, "output_dim": self.word_embeddings_dim})

        kwargs.get("b_q_parameters").update({"observable_dim": data_loader.vocabulary_dim, "pi0": self.pi0, "hidden_state_dim": self.number_of_topics, "control_variable_dim": self.ksi_dim})
        kwargs.get("theta_q_parameters").update({"hidden_state_dim": self.number_of_topics, "control_variable_dim": self.eta_dim})
        kwargs.get("ksi_q_parameters").update({"observable_dim": data_loader.vocabulary_dim})
        kwargs.get("eta_q_parameters").update({"observable_dim": data_loader.vocabulary_dim, "hidden_state_dim": self.eta_dim})
        kwargs.get("ksi_prior_transition").update({"input_dim": self.ksi_dim, "output_dim": self.ksi_dim})
        kwargs.get("eta_prior_transition").update({"input_dim": self.eta_dim, "output_dim": self.eta_dim})
        kwargs.get("ksi_2_pi_transform").update({"input_dim": self.ksi_dim, "output_dim": self.number_of_topics})
        kwargs.get("eta_2_zeta_transform").update({"input_dim": self.eta_dim, "output_dim": self.number_of_topics})

        return kwargs

    def set_parameters(self, **kwargs):
        super().set_parameters(**kwargs)

        self.ksi_q_type = kwargs.get("ksi_q_type")
        self.ksi_q_parameters = kwargs.get("ksi_q_parameters")
        self.ksi_dim = kwargs.get("ksi_q_parameters").get("hidden_state_dim")
        self.zeta_dim = kwargs.get("theta_q_parameters").get("hidden_state_dim")
        self.alpha_q_parameters = kwargs.get("alpha_q_parameters")
        self.alpha_p_parameters = kwargs.get("alpha_p_parameters")

        self.b_q_type = kwargs.get("b_q_type")
        self.b_q_parameters = kwargs.get("b_q_parameters")
        self.lambda_diversity = kwargs.get("lambda_diversity")
        self.ksi_prior_transition_params = kwargs.get("ksi_prior_transition")
        self.eta_prior_transition_params = kwargs.get("eta_prior_transition")
        self.ksi_2_pi_transform_params = kwargs.get("ksi_2_pi_transform")
        self.eta_2_zeta_transform_params = kwargs.get("eta_2_zeta_transform")

        self.nonlinear_transition_prior = kwargs.get("nonlinear_transition_prior")
        self.topic_embeddings = kwargs.get("topic_embeddings")
        self.pi0 = kwargs.get("pi0")

    def define_deep_models(self):
        super().define_deep_models()
        self.theta_stats = defaultdict(list)
        self.b_stats = defaultdict(list)
        self.beta_stats = defaultdict(list)
        recognition_factory = RecognitionModelFactory()

        self.ksi_q = recognition_factory.create(self.ksi_q_type, **self.ksi_q_parameters)
        self.b_q = recognition_factory.create(self.b_q_type, **self.b_q_parameters)
        self.ksi_2_pi = MLP(**self.ksi_2_pi_transform_params)
        self.eta_2_zeta = MLP(**self.eta_2_zeta_transform_params)

        if self.topic_embeddings == "dynamic":
            self.div_reg = self.diversity_regularizer_dynamic
            self.alpha_q = super().q_alpha
            self.mu_q_alpha = nn.Parameter(torch.randn(self.number_of_topics, self.num_training_steps, self.word_embeddings_dim))
            self.logvar_q_alpha = nn.Parameter(torch.randn(self.number_of_topics, self.num_training_steps, self.word_embeddings_dim))
        elif self.topic_embeddings == "nonlinear-dynamic":
            self.mu_q_alpha = nn.Parameter(torch.randn(self.number_of_topics, self.word_embeddings_dim))
            self.logvar_q_alpha = nn.Parameter(torch.randn(self.number_of_topics, self.word_embeddings_dim))
            self.alpha_q_m = MLP(**self.alpha_q_parameters.get('mean'))
            self.alpha_q_lvar = MLP(**self.alpha_q_parameters.get('lvar'))
            self.p_alpha_m = MLP(**self.alpha_p_parameters)
            self.alpha_q = self.q_alpha
            self.div_reg = self.diversity_regularizer
        elif self.topic_embeddings == "static":
            self.alpha_q = self.alpha_static_q
            self.mu_q_alpha = nn.Parameter(torch.randn(self.number_of_topics, self.word_embeddings_dim))
            self.div_reg = self.diversity_regularizer
        if self.nonlinear_transition_prior:
            self.p_ksi_m = MLP(**self.ksi_prior_transition_params)
            self.p_eta_m = MLP(**self.eta_prior_transition_params)
        else:
            self.p_ksi_m = None
            self.p_eta_m = None

    def initialize_inference(self, data_loader, **inference_parameters):
        super().initialize_inference(data_loader=data_loader, **inference_parameters)
        self.eta_q.device = self.device
        self.ksi_q.device = self.device
        self.b_q.device = self.device

    def move_to_device(self):
        self.eta_q.device = self.device
        self.ksi_q.device = self.device
        self.b_q.device = self.device

    def forward(self, x):
        """
        parameters
        ----------
        batchdata ()
        returns
        -------
        z (batch_size*sequence_lenght,self.hidden_state_dim)
        recognition_parameters = (z_mean,z_var)
        (batch_size*sequence_lenght,self.hidden_state_dim)
        likelihood_parameters = (likelihood_mean,likelihood_variance)
        """

        bow_time_indexes = x['time'].squeeze().long().to(self.device, non_blocking=True)
        corpora = x['corpus'][:1].float().to(self.device, non_blocking=True)

        if 'bow' in x:
            bow = x['bow'].to(self.device, non_blocking=True)
            normalized_bow = bow.float() / bow.sum(1, True)
        else:
            bow = x['bow_h1'].to(self.device, non_blocking=True)
            normalized_bow = bow.float() / bow.sum(1, True)
            bow = x['bow_h2'].to(self.device, non_blocking=True)

        ksi, ksi_params, ksi_kl = self.ksi_q(corpora, self.p_ksi_m)
        eta, eta_params, eta_kl = self.eta_q(corpora, self.p_eta_m)
        eta_per_document = eta[bow_time_indexes]
        ksi_per_document = ksi[bow_time_indexes]
        alpha, alpha_kl = self.alpha_q()

        zeta, zeta_params, zeta_kl = self.theta_q(normalized_bow, eta_per_document, self.eta_2_zeta)

        b, q_b_param, p_b_param, b_kl = self.b_q(normalized_bow, ksi_per_document, self.ksi_2_pi)

        theta = b * torch.exp(zeta) / torch.sum(b * torch.exp(zeta), dim=-1, keepdim=True)

        beta = self.get_beta(alpha)
        if self.topic_embeddings != "static":
            beta = beta[bow_time_indexes]
        nll = self.nll(theta, bow, beta)

        return nll, ksi_kl, eta_kl, zeta_kl, b_kl, alpha_kl, torch.softmax(zeta, dim=-1), ksi, eta, ksi_params, eta_params, zeta_params, q_b_param, p_b_param

    def nll(self, theta, bow, beta):
        if self.topic_embeddings != "static":
            loglik = torch.bmm(theta.unsqueeze(1), beta).squeeze(1)
        else:
            loglik = torch.matmul(theta, beta)
        if self.training:
            loglik = torch.log(loglik + 1e-6)
        else:
            loglik = torch.log(loglik)
        nll = -loglik * bow

        return nll.sum(-1)

    def nll_test(self, theta, bow, beta):
        loglik = torch.matmul(theta, beta)

        loglik = torch.log(loglik)
        nll = -loglik * bow

        return nll.sum(-1)

    def alpha_static_q(self):
        return self.mu_q_alpha, torch.tensor(0.0, device=self.device)

    def q_alpha(self):
        rho_dim = self.mu_q_alpha.size(-1)
        alphas = torch.zeros(self.num_training_steps, self.number_of_topics, rho_dim, device=self.device)
        alpha_t = reparameterize(self.mu_q_alpha, self.logvar_q_alpha)
        alphas[0] = alpha_t
        q_m = torch.zeros(self.num_training_steps - 1, self.number_of_topics, rho_dim, device=self.device)
        q_lvar = torch.zeros(self.num_training_steps - 1, self.number_of_topics, rho_dim, device=self.device)

        p_lvar = torch.log(self.delta * torch.ones(self.num_training_steps - 1, self.number_of_topics, rho_dim, device=self.device))
        kl = kullback_leibler(self.mu_q_alpha, self.logvar_q_alpha, 'sum')

        for t in range(1, self.num_training_steps):
            mu_t = self.alpha_q_m(alpha_t)
            lvar_t = self.alpha_q_lvar(alpha_t)
            q_m[t - 1] = mu_t
            q_lvar[t - 1] = lvar_t
            alpha_t = reparameterize(mu_t, lvar_t)
            alphas[t] = alpha_t

        kl += kullback_leibler_two_gaussians(q_m, q_lvar, self.p_alpha_m(alphas[:-1]), p_lvar, 'sum')

        return alphas, kl

    def loss(self, x, forward_results, data_set, epoch):
        """
        nll [batch_size, max_lenght]

        """
        if 'bow' in x:
            text_size = x['bow'].sum(1).double().to(self.device, non_blocking=True)
        else:
            text_size = x['bow_h2'].sum(1).double().to(self.device, non_blocking=True)

        nll, ksi_kl, eta_kl, zeta_kl, b_kl, alpha_kl, _, _, _, _, _, _, _, _ = forward_results

        coeff = 1.0
        if self.training:
            coeff = len(data_set)
        loss = (nll.sum() + zeta_kl + b_kl) * coeff + (eta_kl + ksi_kl + alpha_kl)  # [number of batches]
        topic_diversity_reg = self.div_reg(self.mu_q_alpha)
        loss = loss - self.lambda_diversity * topic_diversity_reg

        log_perplexity = nll / text_size.float()
        log_perplexity = torch.mean(log_perplexity)
        perplexity = torch.exp(log_perplexity)

        return {"loss": loss,
                "NLL-Loss": nll.sum() * coeff,
                "KL-Loss-Alpha": alpha_kl,
                "KL-Loss-Eta": eta_kl,
                "KL-Loss-Ksi": ksi_kl,
                "KL-Loss-Theta": zeta_kl * coeff,
                "KL-Loss-B": b_kl * coeff,
                "Diversity": topic_diversity_reg,
                "PPL-Blei": perplexity,
                "PPL": torch.exp(nll.sum() / text_size.sum().float()),
                "Log-Likelihood": log_perplexity}

    def loss_perrone(self, x, forward_results):
        """
        nll [batch_size, max_lenght]

        """
        if 'bow' in x:
            text_size = x['bow'].sum().double()
        else:
            text_size = x['bow_h2'].sum().double()

        nll = forward_results[0]

        return nll.sum().item(), text_size.item()

    def get_words_by_importance(self) -> torch.Tensor:
        with torch.no_grad():
            alpha = self.mu_q_alpha
            if self.topic_embeddings == "nonlinear-dynamic":
                alpha, _ = self.alpha_q()
                alpha = torch.transpose(alpha, 1, 0)
            beta = self.get_beta(alpha)
            important_words = torch.argsort(beta, dim=-1, descending=True)
            return important_words

    def top_words(self, data_loader, num_of_top_words=20):
        important_words = self.get_words_by_importance()
        # from index to words
        vocabulary = data_loader.vocab
        top_word_per_topic = dict()

        for k in range(self.number_of_topics):
            topic_words = [vocabulary.itos[w] for w in important_words[k, :num_of_top_words].tolist()]
            top_word_per_topic[f'TOPIC {k}'] = topic_words
        return top_word_per_topic

    def top_words_dynamic(self, data_loader, num_of_top_words=20):
        important_words = self.get_words_by_importance()
        # from index to words
        vocabulary = data_loader.vocab
        top_word_per_topic = dict()

        tt = important_words.size(1) // 2
        for k in range(self.number_of_topics):
            top_words_per_time = dict()
            for t in [0, tt, important_words.size(1) - 1]:
                topic_words = [vocabulary.itos[w] for w in important_words[k, t, :num_of_top_words].tolist()]
                top_words_per_time[f'TIME {t}'] = topic_words
            top_word_per_topic[f'TOPIC {k}'] = top_words_per_time
        return top_word_per_topic

    def get_time_series(self, dataset):
        b_stats = defaultdict(list)
        theta_stats = defaultdict(list)
        for x in tqdm(dataset, desc="Building time series"):
            _, _, _, _, _, _, theta, _, _, _, _, _, pi, _ = self.forward(x)

            time_idx = x['time'].squeeze()
            for id, t, _nu in zip(time_idx, theta, pi):
                theta_stats[id.item()].append(t)
                b_stats[id.item()].append(_nu)

        for key, value in theta_stats.items():
            theta_stats[key] = torch.stack(value).mean(0)
        for key, value in b_stats.items():
            b_stats[key] = torch.stack(value).mean(0)

        theta_stats = OrderedDict(sorted(theta_stats.items()))
        b_stats = OrderedDict(sorted(b_stats.items()))

        theta_stats = torch.stack(list(theta_stats.values()))
        b_stats = torch.stack(list(b_stats.values()))
        return theta_stats, b_stats

    def metrics(self, data, forward_results, epoch, mode="evaluation", data_loader=None):
        if mode == "validation" and epoch % self.metrics_logs == 0:
            _, _, _, _, _, _, theta, _, _, _, _, _, q_pi, p_pi = forward_results

            time_idx = data["time"].squeeze()

            for id, t, _nu in zip(time_idx, theta, q_pi):
                self.theta_stats[id.item()].append(t)
                self.b_stats[id.item()].append(_nu)

        if mode == "validation_global" and epoch % self.metrics_logs == 0:
            for key, value in self.theta_stats.items():
                self.theta_stats[key] = dict()
                self.theta_stats[key]["mean"] = torch.stack(value).mean(0).cpu().numpy()
                self.theta_stats[key]["std"] = torch.stack(value).std(0).cpu().numpy()
            for key, value in self.b_stats.items():
                self.b_stats[key] = dict()
                self.b_stats[key]["mean"] = torch.stack(value).mean(0).cpu().numpy()
                self.b_stats[key]["std"] = torch.stack(value).std(0).cpu().numpy()

            self.theta_stats = OrderedDict(sorted(self.theta_stats.items()))
            self.b_stats = OrderedDict(sorted(self.b_stats.items()))
            theta_stats = defaultdict(list)
            b_stats = defaultdict(list)

            for key, value in self.theta_stats.items():
                theta_stats["mean"].append(value["mean"])
                theta_stats["std"].append(value["std"])
            for key, value in self.b_stats.items():
                b_stats["mean"].append(value["mean"])
                b_stats["std"].append(value["std"])

            theta_stats["mean"] = np.stack(theta_stats["mean"])
            theta_stats["std"] = np.stack(theta_stats["std"])
            b_stats["mean"] = np.stack(b_stats["mean"])
            b_stats["std"] = np.stack(b_stats["std"])

            self.plot_topics_ts(theta_stats, b_stats, '/Topic-Time-Series', y_lim=(-0.1, 1.1))

            self.b_stats = defaultdict(list)
            self.theta_stats = defaultdict(list)

            if self.topic_embeddings != "static":
                self._log_top_words_dynamic(data_loader)
            else:
                self._log_top_words(data_loader)
        return {}

    def plot_topics_ts(self, theta, b, label, y_lim=None):
        ntime, ntopics = theta["mean"].shape
        x = range(ntime)
        ncols = 5
        nrows = ntopics // ncols + ntopics % ncols
        fig, axis = plt.subplots(nrows, ncols, sharex=True, figsize=(15, 2*nrows))
        axis = axis.flatten()
        for i in range(ntopics):
            ax = axis[i]
            color = 'tab:red'
            ax.set_ylabel('theta', color=color)
            ax.tick_params(axis='y', labelcolor=color)
            sns.lineplot(x=x, y=theta["mean"][:, i], ax=ax, color=color)
            # lower = theta["mean"][:, i] - 1.96*theta["std"][:, i]
            # upper = theta["mean"][:, i] + 1.96*theta["std"][:, i]
            # ax.fill_between(x, lower, upper,  color=color, alpha=0.2)
            ax2 = ax.twinx()
            color = 'tab:blue'
            ax2.set_ylabel('b', color=color)
            if y_lim is not None:
                ax2.set_ylim(y_lim)
            # print(i, b[:, i])
            sns.lineplot(x=x, y=b["mean"][:, i], ax=ax2, color=color)
            # lower = b["mean"][:, i] - 1.96*b["std"][:, i]
            # upper = b["mean"][:, i] + 1.96*b["std"][:, i]
            # ax2.fill_between(x, lower, upper,  color=color, alpha=0.2)
            ax2.tick_params(axis='y', labelcolor=color)
        plt.tight_layout()
        buf = io.BytesIO()
        plt.savefig(buf, format='jpeg')
        plt.close()
        buf.seek(0)
        image = PIL.Image.open(buf)
        image = ToTensor()(image).unsqueeze(0)
        self.writer.add_images(label, image, self.number_of_iterations)

    def _log_top_words(self, data_loader):
        top_word_per_topic = self.top_words(data_loader, num_of_top_words=20)
        TEXT = "\n".join(["<strong>{0}</strong> ".format(topic) + " ".join(words) + "\n" for topic, words in top_word_per_topic.items()])
        self.writer.add_text("/Topics-Top-Words", TEXT, self.number_of_iterations)

    def _log_top_words_dynamic(self, data_loader):
        top_word_per_topic = self.top_words_dynamic(data_loader, num_of_top_words=20)
        TEXT = ""
        for topic, value in top_word_per_topic.items():
            for time, words in value.items():
                TEXT += f'<strong>{topic} --- {time}</strong>: {" ".join(words)}\n\n'
            TEXT += "*" * 1000 + "\n"
        self.writer.add_text("/Topics-Top-Words", TEXT, self.number_of_iterations)

    def diversity_regularizer(self, x):
        x = x / (torch.norm(x, dim=-1, keepdim=True) + 1e-12)
        cosine_simi = torch.tensordot(x, x, dims=[[1], [1]]).abs()
        angles = torch.acos(torch.clamp(cosine_simi, -1. + 1e-7, 1. - 1e-7))
        angles_mean = angles.mean()
        var = ((angles - angles_mean) ** 2).mean()
        return angles_mean - var

    def diversity_regularizer_dynamic(self, x):  # [K, T, D]
        x = torch.transpose(x, 1, 0)
        x = x / (torch.norm(x, dim=-1, keepdim=True) + 1e-12)
        topics_cosine_similarity = torch.einsum('bij,bgj->big', x, x).abs()
        topics_similarity = torch.acos(torch.clamp(topics_cosine_similarity, -1. + 1e-7, 1. - 1e-7))
        topic_similarity_mean = topics_similarity.mean(-1).mean(-1)
        topic_similarity_variance = ((topics_similarity - topic_similarity_mean.reshape(-1, 1, 1)) ** 2).mean(-1).mean(-1)
        topic_diversity = (topic_similarity_mean - topic_similarity_variance)
        return topic_diversity.mean()

    def topic_coherence(self, data: np.ndarray) -> float:
        if self.topic_embeddings == "static":
            top_10 = self.get_words_by_importance()[:, :10].tolist()
            D = data.shape[0]
            TC = []
            for k in tqdm(range(self.number_of_topics), total=self.number_of_topics, desc="Calculate coherence for topic"):
                top_10_k = top_10[k]
                TC_k = 0
                counter = 0
                word_count_ = self.data_loader.vocab.word_count
                for i, word in enumerate(top_10_k):
                    # get D(w_i)
                    D_wi = word_count_[word]
                    j = i + 1
                    tmp = 0
                    while len(top_10_k) > j > i:
                        # get D(w_j) and D(w_i, w_j)
                        D_wj = word_count_[top_10_k[j]]
                        D_wi_wj = get_doc_freq(data, word, top_10_k[j])
                        # get f(w_i, w_j)
                        if D_wi_wj == 0:
                            f_wi_wj = -1
                        else:
                            f_wi_wj = -1 + (np.log(D_wi) + np.log(D_wj) - 2.0 * np.log(D)) / (np.log(D_wi_wj) - np.log(D))
                        # update tmp:
                        tmp += f_wi_wj
                        j += 1
                        counter += 1
                    # update TC_k
                    TC_k += tmp
                TC.append(TC_k / float(counter))
            # print('Topic coherence is: {}'.format(TC))
            TC = np.mean(TC)
        else:
            TC = super(DynamicBetaFocusedTopic, self).topic_coherence(data)
        return TC

    def topic_diversity(self, topk: int = 25) -> float:
        if self.topic_embeddings == "static":
            important_words_per_topic = self.get_words_by_importance()
            list_w = important_words_per_topic[:, :topk]
            n_unique = len(torch.unique(list_w))
            td = n_unique / (topk * self.number_of_topics)
        else:
            td = super().topic_diversity(topk)
        return td

    def get_transitions_dist(self, alpha_sample, eta_sample, ksi_sample):
        """
        returns torch.distributions Normal([montecarlo_samples,number_of_topics])
        """
        alpha_std = torch.ones_like(alpha_sample) * self.delta
        eta_std = torch.ones_like(eta_sample) * self.delta
        ksi_std = torch.ones_like(ksi_sample) * self.delta

        v_dist = Normal(alpha_sample, alpha_std)
        r_dist = Normal(eta_sample, eta_std)
        ksi_dist = Normal(ksi_sample, ksi_std)
        return v_dist, r_dist, ksi_dist

    def prediction(self, data_loader: ADataLoader, montecarlo_samples=10):
        self.eval()
        x = next(iter(data_loader.train))
        with torch.no_grad():
            forward_results = self.forward(x)

            _, _, _, _, _, _, _, ksi, eta, _, _, _, _, _ = forward_results

            current_document_count = int(data_loader.prediction_count_per_year)
            if self.topic_embeddings != "static":
                last_state_alpha = self.mu_q_alpha[:, -1, :], torch.exp(0.5 * self.logvar_q_alpha[:, -1, :]) * self.delta
                alpha_q_dist = Normal(*last_state_alpha)
                alpha_sample = alpha_q_dist.sample(sample_shape=torch.Size([montecarlo_samples]))
            else:
                alpha_sample = self.mu_q_alpha.unsqueeze(0).repeat(montecarlo_samples, 1, 1)
            eta_mu = eta[-1]
            ksi_mu = ksi[-1]
            if self.nonlinear_transition_prior:
                eta_mu = self.p_eta_m(eta_mu)
                ksi_mu = self.p_ksi_m(ksi_mu)

            eta_q, ksi_q = super().get_transitions_dist(eta_mu, ksi_mu)

            eta_sample = eta_q.sample(sample_shape=torch.Size([montecarlo_samples]))
            ksi_sample = ksi_q.sample(sample_shape=torch.Size([montecarlo_samples]))
            pred_pp_all = []
            for i in self.data_loader.prediction_times:
                pred_pp_step = torch.zeros((montecarlo_samples, current_document_count))
                bow = torch.from_numpy(data_loader.predict.dataset.corpus_per_year('prediction')[0]).to(self.device)
                text_size = bow.sum(1).double().to(self.device, non_blocking=True)

                for mc_index in range(montecarlo_samples):
                    if self.topic_embeddings == 'static':
                        theta = self.prediction_montecarlo_step_static(current_document_count, eta_sample[mc_index], ksi_sample[mc_index])
                    else:
                        raise RuntimeError("Prediction with dynamic topic embeddings not implemented!")
                    alpha_per_doc = alpha_sample[mc_index]
                    beta = torch.matmul(alpha_per_doc, self.rho.T)
                    beta = torch.softmax(beta, dim=-1)
                    pred_like = self.nll_test(theta, bow, beta)
                    log_pp = (1. / text_size.float()) * pred_like
                    log_pp = self.nanmean(log_pp)

                    pred_pp_step[mc_index] = log_pp
                pred_pp_all.append(pred_pp_step.mean())
                if self.nonlinear_transition_prior:
                    eta_mu = self.p_eta_m(eta_sample)
                    ksi_mu = self.p_ksi_m(ksi_sample)

                eta_q, ksi_q = super().get_transitions_dist(eta_mu, ksi_mu)

                eta_sample = eta_q.sample()
                ksi_sample = ksi_q.sample()

                # [mc,count,number_of_topics]
            pred_pp = torch.mean(torch.stack(pred_pp_all)).item()
            return pred_pp

    def nanmean(self, v, *args, inplace=False, **kwargs):
        if not inplace:
            v = v.clone()
        is_nan = torch.isnan(v)
        v[is_nan] = 0
        return v.sum(*args, **kwargs) / (~is_nan).float().sum(*args, **kwargs)

    def prediction_montecarlo_step_static(self, current_document_count, eta_sample, ksi_sample):
        pi = self.b_q.pi0 * self.ksi_2_pi(ksi_sample)

        b_dist = Bernoulli(pi)
        b_sample = b_dist.sample(sample_shape=torch.Size([current_document_count]))

        zeta_mu = self.eta_2_zeta(eta_sample)
        zeta_dist = Normal(zeta_mu, torch.ones_like(zeta_mu))
        zeta_sample = zeta_dist.sample(sample_shape=torch.Size([current_document_count]))
        theta_sample = b_sample * torch.exp(zeta_sample) / torch.sum(b_sample * torch.exp(zeta_sample), dim=-1, keepdim=True)  # [mc,current_doc_count,
        # number_of_topics]

        theta_sample = theta_sample.view(current_document_count, -1).contiguous()

        return theta_sample

    def prediction_montecarlo_step(self, current_document_count, alpha_dist, eta_dist, ksi_dist):
        eta_sample = eta_dist.sample()  # [mc,n_topics, topic_transition_dim]
        ksi_sample = ksi_dist.sample()  # [mc,n_topics, topic_transition_dim]
        n_mc, _ = eta_sample.shape
        alpha_sample = alpha_dist.sample()  # [mc, n_topics, topic_transition_dim]
        a = self.nu_q.alpha0 * self.ksi_transform_alpha(ksi_sample)
        nu_dist = Beta(a, torch.ones_like(a))
        nu_sample = nu_dist.sample()
        pi = torch.cumprod(nu_sample, dim=-1)
        b_dist = Bernoulli(pi)
        b_sample = b_dist.sample(sample_shape=torch.Size([current_document_count]))

        zeta_param = self.eta_2_zeta(eta_sample)
        zeta_dist = Normal(zeta_param, torch.ones_like(zeta_param))
        zeta_sample = zeta_dist.sample(sample_shape=torch.Size([current_document_count]))

        theta_sample = b_sample * torch.exp(zeta_sample) / torch.sum(b_sample * torch.exp(zeta_sample), dim=-1, keepdim=True)  # [mc,current_doc_count,
        # number_of_topics]

        theta_sample = theta_sample.view(n_mc, current_document_count, -1).contiguous()

        return theta_sample, alpha_sample, eta_sample

    def get_topic_entropy_per_document(self, dataset):

        theta_stats = defaultdict(list)
        for x in tqdm(dataset, desc="Building time series"):
            theta = self.forward(x)[6]

            time_idx = x['time'].squeeze()
            for id, t in zip(time_idx, theta):

                theta_stats[id.item()].append(-(t*torch.log(t)).sum())

        for key, value in theta_stats.items():
            theta_stats[key] = torch.stack(value)

        theta_stats = OrderedDict(sorted(theta_stats.items()))
        # theta_stats = torch.stack(list(theta_stats.values()))

        return theta_stats
