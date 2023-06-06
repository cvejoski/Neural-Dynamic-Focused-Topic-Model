import os
import torch
import numpy as np

import pprint

from torch.distributions import MultivariateNormal, Multinomial


def words_tfidf(bag_of_words, number_of_documents, corpora, vocabulary_size):
    score = (corpora / vocabulary_size) * torch.log(number_of_documents / bag_of_words)
    return score


def count_words_in_topics(LDA, documents, current_z):
    """
    documents = [[w_1,w_2,...]]
    current_z = [[]]
    """
    N_wk_dict = {w: {} for w in range(LDA.vocabulary_size)}
    for j, (document_j, z_ij) in enumerate(zip(documents, current_z)):
        for word in torch.unique(document_j):
            w = word.item()
            topics_with_w = z_ij[document_j == w]
            for topic in topics_with_w:
                try:
                    N_wk_dict[w][topic.item()] += 1
                except:
                    N_wk_dict[w][topic.item()] = 1
    return N_wk_dict


def obtain_sparse_word_topic_counts(LDA, documents, current_z):
    N_wk_dict = count_words_in_topics(LDA, documents, current_z)
    """
    N_wk:{w_1:{k_1:n_k1,...,K:n_K},w_2:{...},...}
    returns N_wk,N_k
    """
    row = []
    columns = []
    values = []
    for w in range(LDA.vocabulary_size):
        topics_ = list(N_wk_dict[w].keys())
        topics_.sort()
        row.extend(list(np.repeat(w, len(topics_))))
        columns.extend(topics_)
        values.extend([N_wk_dict[w][topic] for topic in topics_])
    indices = torch.Tensor([row, columns]).type(torch.long)
    N_wk = torch.sparse.FloatTensor(indices=indices,
                                    values=torch.tensor(values),
                                    size=torch.Size([LDA.vocabulary_size, LDA.number_of_topics]))
    N_wk = N_wk.coalesce()

    sparse_Nk = torch.sparse.sum(N_wk, dim=0)
    non_zero_topics = sparse_Nk.indices()
    topic_counts = sparse_Nk.values()
    N_k = torch.zeros(size=(LDA.number_of_topics,), dtype=torch.long)[non_zero_topics] = topic_counts
    return N_wk, N_k


def calculate_nkj(LDA, z_ij_row):
    """
    word is a tensor
    """
    non_zero_topics, non_zero_topics_counts = torch.unique(z_ij_row, return_counts=True)
    N_kj = torch.zeros(size=torch.Size([LDA.number_of_topics]), dtype=torch.long)
    N_kj[non_zero_topics] = non_zero_topics
    return N_kj


def calculate_nkw_row(LDA, word, N_wk):
    """
    word is a tensor
    """
    topics_with_w = N_wk.indices()[1][N_wk.indices()[0] == word.item()]
    count_topics_with_w = N_wk.values()[N_wk.indices()[0] == word.item()]
    N_wk_row = torch.zeros(size=torch.Size([LDA.number_of_topics]), dtype=torch.long)
    N_wk_row[topics_with_w] = count_topics_with_w
    return N_wk_row


def calculate_categorical_gibbs_probabilities(LDA, z_ij, N_k, N_kj, N_wk_row):
    N_k_minus = N_k.clone()

    N_k_minus[z_ij.item()] = N_k_minus[z_ij.item()] - 1.
    N_kj[z_ij.item()] = N_kj[z_ij.item()] - 1.
    N_wk_row[z_ij.item()] = N_wk_row[z_ij.item()] - 1.

    a_kj = (N_wk_row + LDA.alpha).type(torch.float)
    b_wk = (N_wk_row + LDA.beta)
    b_wk = b_wk.type(torch.float)
    denominator = (N_k_minus + LDA.vocabulary_size * LDA.beta).type(torch.float)
    b_wk = b_wk / denominator
    logits = a_kj * b_wk
    categorical_probabilities = torch.softmax(logits, dim=0)
    return categorical_probabilities


def update_counts(word, old_zij, new_zij, N_wk, N_k, current_z):
    # substract
    N_k[old_zij] -= 1
    N_wk.values()[N_wk.indices()[0] == word.item()][old_zij] -= 1
    # add
    N_k[new_zij] += 1
    N_wk.values()[N_wk.indices()[0] == word.item()][new_zij] += 1
    return N_wk, N_k


def generate_topic_filter(W, A, Q, hidden_state_size, number_of_steps):
    """
    h = Ah+ u
    z ~ Categorical(W*h)
    """
    transition_noise_distribution = MultivariateNormal(torch.zeros(hidden_state_size), Q)
    h_0 = MultivariateNormal(torch.zeros(hidden_state_size), Q).sample()
    z = []
    for t in range(number_of_steps):
        # transition
        h_t = torch.matmul(A, h_0)
        noise = transition_noise_distribution.sample()
        h_t = h_t + noise
        # emission
        categorical_logits = torch.softmax(W.matmul(h_t), dim=0)
        emission_probability = Multinomial(probs=categorical_logits)
        z_t = torch.argmax(emission_probability.sample()).item()
        h_0 = h_t
        z.append(z_t)
    return z


def reparameterize(mu, logvar):
    """Returns a sample from a Gaussian distribution via reparameterization.
    """

    std = torch.exp(0.5 * logvar)
    eps = torch.randn_like(std)
    return eps.mul_(std).add_(mu)
