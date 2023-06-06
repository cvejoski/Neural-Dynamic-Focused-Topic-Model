import os

import numpy as np
import torch
from deep_fields import project_path
from deep_fields.data.topic_models.dataloaders import ADataLoader, TopicDataloader
from deep_fields.models.abstract_models import DeepBayesianModel
from deep_fields.models.deep_state_space.deep_state_space_recognition import RecognitionModelFactory
from deep_fields.models.random_measures.random_measures_utils import stick_breaking
from deep_fields.utils.loss_utils import get_doc_freq
from deep_fields.utils.schedulers import SigmoidScheduler
from torch import nn
from tqdm import tqdm


class DiscreteLatentTopicNVI(DeepBayesianModel):
    """
    This is an implementation of https://arxiv.org/pdf/1706.00359.pdf.
    """
    theta_q: nn.Module
    lambda_diversity: float = 0.1

    def __init__(self, model_dir=None, data_loader: ADataLoader = None, model_name=None, **kwargs):
        if model_name is None:
            model_name = "discrete_latent_topic"
        DeepBayesianModel.__init__(self, model_name, model_dir=model_dir, data_loader=data_loader, **kwargs)

    @classmethod
    def get_parameters(cls):
        """
        here we provide an example of the minimum set of parameters requiered to instantiate the model
        """
        number_of_topics = 50
        vocabulary_dim = 100
        parameters_sample = {"number_of_topics": number_of_topics,  # z
                             "no_topics": False,
                             "no_embeddings": False,
                             "number_of_documents": 100,
                             "number_of_words_per_document": 30,
                             "vocabulary_dim": vocabulary_dim,
                             "word_embeddings_dim": 20,
                             "lambda_diversity": 0.1,
                             "delta": 0.005,
                             "topic_proportion_transformation": "gaussian_softmax",
                             "topic_lifetime_tranformation": "gaussian_softmax",
                             "theta_q_type": "q-INDEP",
                             "theta_q_parameters": {
                                 "observable_dim": vocabulary_dim,
                                 "layers_dim": [256, 256],
                                 "output_dim": 256,
                                 "hidden_state_dim": number_of_topics,
                                 "dropout": .1,
                                 "out_dropout": .1
                             },
                             "model_path": os.path.join(project_path, 'results')}

        return parameters_sample

    def set_parameters(self, **kwargs):
        self.vocabulary_dim = kwargs.get("vocabulary_dim")
        self.number_of_documents = kwargs.get("number_of_documents", None)

        self.word_embeddings_dim = kwargs.get("word_embeddings_dim", 100)
        self.number_of_topics = kwargs.get("number_of_topics", 10)
        self.no_topics = kwargs.get("no_topics", True)
        self.no_embeddings = kwargs.get("no_embeddings", False)
        self.lambda_diversity = kwargs.get("lambda_diversity")
        # TEXT
        self.theta_q_type = kwargs.get("theta_q_type")
        self.theta_q_param = kwargs.get("theta_q_parameters")

        self.theta_q_param.update({"observable_dim": self.vocabulary_dim,
                                   "hidden_state_dim": self.number_of_topics
                                   })

        self.topic_proportion_transformation = kwargs.get("topic_proportion_transformation")

    def update_parameters(self, data_loader: ADataLoader, **kwargs):
        kwargs.update({"vocabulary_dim": data_loader.vocabulary_dim})
        kwargs.update({"number_of_documents": data_loader.number_of_documents})
        kwargs.update({"word_embeddings_dim": data_loader.word_embeddings_dim})

        return kwargs

    @classmethod
    def get_inference_parameters(cls):
        inference_parameters = super().get_inference_parameters()
        inference_parameters.update({"regularizers": {"nll": None}})
        inference_parameters.update({"model_eval": "PPL-Blei"})

        return inference_parameters

    def define_deep_models(self):
        # THE DECODING DATA IS IN LANGUAGE MODEL TYPE (THE PAD ARE HOWEVER REMOVED IN LOSS)
        recognition_factory = RecognitionModelFactory()

        if not self.no_topics:
            self.topic_prop_transform = nn.Sequential(nn.Linear(self.number_of_topics, self.number_of_topics), nn.Softmax(dim=1))
            if not self.no_embeddings:
                self.topic_word_embeddings = torch.nn.Parameter(self.data_loader.vocab.vectors[:self.vocabulary_dim].clone(), requires_grad=True)
                self.topic_embeddings = nn.Parameter(torch.randn((self.number_of_topics, self.word_embeddings_dim)), requires_grad=True)
            else:
                self.topic_word_logits = nn.Parameter(torch.randn((self.number_of_topics, self.vocabulary_dim)), requires_grad=True)
        else:
            # for neural variational, just bag of words
            self.H = nn.Embedding(num_embeddings=self.vocabulary_dim, embedding_dim=self.number_of_topics)
            self.bias = nn.Parameter(torch.Tensor(size=torch.Size([self.vocabulary_dim])))

        self.theta_q = recognition_factory.create(self.theta_q_type, **self.theta_q_param)

    def move_to_device(self):
        self.theta_q.device = self.device

    def forward(self, x):
        """
        parameters
        ----------
        data ()
        returns
        -------
        z (batch_size*sequence_lenght,self.hidden_state_dim)
        recognition_parameters = (z_mean,z_var)
        (batch_size*sequence_lenght,self.hidden_state_dim)
        likelihood_parameters = (likelihood_mean,likelihood_variance)
        """
        if 'bow' in x:
            bow = x['bow'].to(self.device, non_blocking=True)
            normalized_bow = bow.float() / bow.sum(1, True)
        else:
            bow = x['bow_h1'].to(self.device)
            normalized_bow = bow.float() / bow.sum(1, True)
            bow = x['bow_h2'].to(self.device, non_blocking=True)

        theta_logits, _, kl_theta = self.theta_q(normalized_bow.unsqueeze(1))

        if not self.no_topics:
            theta = self.proportion_transformation(theta_logits)  # [batch_size, number_of_topics]
        else:
            theta = theta_logits
        beta = self.get_beta()
        nll = self.nll(theta, bow, beta)

        return nll, kl_theta

    def get_beta(self):
        if not self.no_topics:
            if not self.no_embeddings:
                # Decoding and language
                beta = torch.matmul(self.topic_embeddings, self.topic_word_embeddings.T)
                beta = torch.softmax(beta, dim=1)
                return beta
            else:
                return torch.softmax(self.topic_word_logits, dim=1)
        else:
            # Energy
            H = self.H.weight
            return H

    def nll(self, theta, bow, beta):
        if not self.no_topics:
            loglik = torch.matmul(theta, beta)
            if self.training:
                loglik = loglik + 1e-6

            loglik = torch.log(loglik)
        else:
            # Energy
            H = self.H.weight
            E = torch.matmul(theta, H.T)
            logits = E + self.bias
            loglik = torch.log_softmax(logits, dim=1)

        loglik = loglik * bow
        return -loglik.sum(-1)

    def topic_diversity_regularizer(self):
        if not self.no_embeddings:
            topic_ = self.topic_embeddings
            # topic_ = torch.nn.functional.normalize(TOPIC_EMBEDDINGS, dim=1, eps=1e-8)
        else:
            topic_ = self.get_beta()
        topic_ = topic_ / (torch.norm(topic_, dim=-1, keepdim=True) + 1e-12)
        cosine_simi = torch.tensordot(topic_, topic_, dims=[[1], [1]]).abs()
        angles = torch.acos(torch.clamp(cosine_simi, -1. + 1e-7, 1. - 1e-7))
        angles_mean = angles.mean()
        var = ((angles - angles_mean) ** 2).mean()
        return angles_mean - var

    def proportion_transformation(self, proportions_logits):
        if self.topic_proportion_transformation == "gaussian_softmax":
            proportions = self.topic_prop_transform(proportions_logits)
        elif self.topic_proportion_transformation == "gaussian_stick_break":
            sticks = torch.sigmoid(proportions_logits)
            proportions = stick_breaking(sticks, self.device)
        else:
            raise ValueError(f"{self.topic_proportion_transformation} transformation not implemented")
        return proportions

    def metrics(self, data, forward_results, epoch, mode="evaluation", data_loader=None):
        if mode == "validation_global" and epoch % self.metrics_logs == 0:
            top_word_per_topic = self.top_words(data_loader, num_of_top_words=20)
            TEXT = "\n".join(["TOPIC {0}: ".format(j) + " ".join(top_word_per_topic[j]) + "\n" for j in range(len(top_word_per_topic))])
            self.writer.add_text("/TEXT/", TEXT, epoch)
        return {}

    def loss(self, x, forward_results, data_loader, epoch):
        """
        nll [batch_size, max_lenght]

        """
        topic_diversity_reg = 0.0
        if 'bow' in x:
            text_size = x['bow'].sum(1).double().to(self.device, non_blocking=True)
        else:
            text_size = x['bow_h2'].sum(1).double().to(self.device, non_blocking=True)

        nll, kl_theta = forward_results
        batch_size = nll.size(0)

        loss = (nll.sum() + kl_theta) / batch_size

        if not self.no_topics:
            topic_diversity_reg = self.topic_diversity_regularizer()
            loss = loss - self.lambda_diversity * topic_diversity_reg  # / batch_size

        log_perplexity = nll / text_size.float()
        log_perplexity = torch.mean(log_perplexity)
        perplexity = torch.exp(log_perplexity)
        return {"loss": loss,
                "Topic-Div-Reg": topic_diversity_reg,
                "NLL-Loss": nll.sum().item() / batch_size,
                "KL-Loss-Theta": kl_theta / batch_size,
                "PPL-Blei": perplexity,
                "PPL": torch.exp(nll.sum() / text_size.sum().float()),
                "Log-Likelihood": log_perplexity}

    def initialize_inference(self, data_loader: ADataLoader, parameters=None, **inference_parameters) -> None:
        super().initialize_inference(data_loader=data_loader, parameters=parameters, **inference_parameters)
        regularizers = inference_parameters.get("regularizers")

        self.schedulers = {}
        for k, v in regularizers.items():
            if v is not None:
                lambda_0 = v["lambda_0"]
                percentage = v["percentage"]
                self.schedulers[k] = SigmoidScheduler(lambda_0=lambda_0,
                                                      max_steps=self.expected_num_steps,
                                                      decay_rate=50.,
                                                      percentage_change=percentage)
        self.theta_q.device = self.device

    def topic_coherence(self, data: np.ndarray) -> float:
        top_10 = self.get_words_by_importance()[:, :10].tolist()
        D = data.shape[0]
        TC = []
        data = data > 0
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
        return TC

    def topic_diversity(self, topk: int = 25) -> float:
        important_words_per_topic = self.get_words_by_importance()
        list_w = important_words_per_topic[:, :topk]
        n_unique = len(torch.unique(list_w))
        td = n_unique / (topk * self.number_of_topics)
        return td

    def get_words_by_importance(self) -> torch.Tensor:
        with torch.no_grad():
            if not self.no_topics:
                beta = self.get_beta()
                important_words = torch.argsort(beta, dim=1, descending=True)
            else:
                W = self.H.weight.T + self.bias  # [topics, vocabulary_dim+2]
                important_words = torch.argsort(W, dim=1, descending=True)
            return important_words

    def top_words(self, data_loader: ADataLoader, num_of_top_words=20):
        important_words = self.get_words_by_importance()
        # from index to words
        vocabulary = data_loader.vocab
        top_word_per_topic = []
        for k in range(self.number_of_topics):
            topic_words = []
            for i in range(num_of_top_words):
                topic_words.append(vocabulary.itos[important_words[k, i].item()])
            top_word_per_topic.append(topic_words)
        return top_word_per_topic

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


if __name__ == "__main__":
    from deep_fields import test_data_path

    data_dir = os.path.join(test_data_path, "preprocessed", "yelp", "language")
    dataloader_params = {"path_to_data": data_dir, "batch_size": 128}

    data_loader = TopicDataloader('cpu', **dataloader_params)
    databatch = next(data_loader.train.__iter__())

    model_parameters = DiscreteLatentTopicNVI.get_parameters()
    model = DiscreteLatentTopicNVI(data_loader=data_loader, **model_parameters)
    forward_data = model(databatch)
