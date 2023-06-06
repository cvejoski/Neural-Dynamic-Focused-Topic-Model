import csv
import glob
import os
import pickle
import math

import warnings
from multiprocessing import Pool
from operator import itemgetter
from typing import List
import click
import operator
import numpy as np
import tqdm
from deep_fields.models.generative_models.text.utils import SPECIAL_TOKENS, tokenize_doc_transformers, count_lines_in_file, preprocess_text
from deep_fields.models.generative_models.text import utils
from scipy import sparse
from sklearn.feature_extraction.text import CountVectorizer, TfidfTransformer
from sklearn.model_selection import train_test_split
from tqdm.contrib.concurrent import process_map
from functools import partial

parent_path = os.path.dirname(__file__)

warnings.filterwarnings("ignore")
# Read stopwords
with open(os.path.join(parent_path, 'stops.txt'), 'r') as f:
    STOPS = f.read().split('\n')


def fix_nulls(s):
    for line in s:
        yield line.replace('\0', ' ')


@click.command()
@click.option('-i', '--input', type=click.Path(exists=True), required=True, help="Input directory of the raw data.")
@click.option('-e', '--embeddings', 'embeddings_path', type=click.Path(exists=True), required=True, help="Input directory of the embeddings.")
@click.option('-o', '--output', type=click.Path(exists=False), required=True, help="Output directory of the pre-processed data.")
@click.option('-min-df-tp', '--min-df-tp', default=100, type=int, required=True, help="Minimum document frequency for the topic model.")
@click.option('-max-df', '--max-df', default=0.95, type=float, required=True, help="Maximum document frequency.")
@click.option('-psteps', '--prediction-steps', 'p_steps', default=1, type=int, required=True, help="Split the data along the time axis. Number of timesteps for prediction")
@click.option('-tt-ratio', '--train-test-ratio', 'tt_ratio', nargs=2, type=float, required=True, help="Split the train data into train/validate.")
@click.option('-max-doc-len', '--max-doc-len', default=None, type=int, help="Max len of a document")
@click.option('--date-field-name', default='date', type=str, help='Name of the date field')
@click.option('--reward-column', type=str, required=False, help='Name of the reward column.')
@click.option('--random-seed', default=87, type=int, required=True, help='Random seed for spliting the data')
@click.option('--num-workers', default=1, type=int, required=True, help='Number of workers for paralle preprocessing and tokenizatioin')
@click.option('--binarize-reward', '-br', multiple=True, default=[], type=int, help="Binarize the reward into bins (ex. [0, 1, 2, 11, 101, 1001, 5001, 10001, 50000])")
@click.option('--covariates', '-c', multiple=True, required=False, type=(str, str), help="Name of the dataset attributes that are used as covariates")
@click.option('--total-num-time-steps', default=None, type=int, help='Total number of timesteps to be used')
@click.option('--samples-per-timestep', default=None, type=int, help="Subsample the documents per each time step")
@click.option('--split-by-paragraph', is_flag=True, help='Take the abstracts as document text')
def preprocess(input: str, output: str, min_df_tp: int, max_df: float, max_doc_len: int, p_steps: int, tt_ratio: tuple,  embeddings_path: str, reward_column: str, covariates: List[tuple], binarize_reward: list, random_seed: int, num_workers: int, date_field_name: str, total_num_time_steps: int, samples_per_timestep: int, split_by_paragraph: bool):

    csv.field_size_limit(1310720)

    all_docs, all_timestamps, all_rewards, all_rewards_bin, all_covariates = _read_docs(
        input, False, reward_column, binarize_reward, covariates, date_field_name, samples_per_timestep, total_num_time_steps)

    print("Sorting Documents")

    all_timestamps_, all_docs_, all_rewards_, all_rewards_bin_, all_covariates_ = zip(*sorted(zip(all_timestamps, all_docs, all_rewards, all_rewards_bin, all_covariates), key=itemgetter(0)))
    if split_by_paragraph:
        print('splitting by paragraphs ...')
        all_docs = []
        all_timestamps = []
        all_rewards = []
        all_rewards_bin = []
        all_covariates = []
        for dd, doc in tqdm.tqdm(enumerate(all_docs_), total=len(all_docs_)):
            splitted_doc = doc.split('.\n')
            for ii in splitted_doc:
                all_docs.append(ii)
                all_timestamps.append(all_timestamps_[dd])
                all_rewards.append(all_rewards_[dd])
                all_rewards_bin.append(all_rewards_bin_[dd])
                all_covariates.append(all_covariates_[dd])
    else:
        all_docs = all_docs_
        all_timestamps = all_timestamps_
        all_rewards = all_rewards_
        all_rewards_bin = all_rewards_bin_
        all_covariates = all_covariates_

    print('SENTENCE: ')
    print('         tokenization and preprocessing ...')

    pool = Pool(num_workers)
    all_docs_sent = []
    for r in tqdm.tqdm(pool.imap(partial(tokenize_doc_transformers, max_doc_len=max_doc_len), all_docs, 100), total=len(all_docs)):
        all_docs_sent.append(r)
    pool.close()
    pool.join()
    # all_docs_sent = tokenize_doc_transformers(all_docs, max_doc_len)

    # Remove punctuation
    print('BOW: \n'
          '         removing punctuation ...')

    # all_docs_bow = []
    # for r in process_map(preprocess_text, all_docs, max_workers=num_workers):
    #     all_docs_bow.append(r)
    pool = Pool(num_workers)
    all_docs_bow = []
    for r in tqdm.tqdm(pool.imap(preprocess_text, all_docs, 100), total=len(all_docs)):
        all_docs_bow.append(r)
    pool.close()
    pool.join()
    del all_docs

    print('         counting document frequency of words ...')
    cvectorizer = CountVectorizer(min_df=min_df_tp, max_df=max_df, stop_words=None)
    cvz = cvectorizer.fit_transform(all_docs_bow).sign()
    vocab_bow = cvectorizer.vocabulary_
    del cvectorizer

    print(f'         vocabulary size: {len(vocab_bow)}')
    # Filter out stopwords (if any)
    vocab_bow = [w for w in vocab_bow.keys() if w not in STOPS]
    vocab_bow_size = len(vocab_bow)
    word2id = dict([(w, j) for j, w in enumerate(vocab_bow)])

    print(f'         vocabulary size after removing stopwords from list: {vocab_bow_size}')

    # Create mapping of timestamps
    all_times = sorted(set(all_timestamps))
    time2id = dict([(t, i) for i, t in enumerate(all_times)])
    id2time = dict([(i, t) for i, t in enumerate(all_times)])
    time_list = [id2time[i] for i in range(len(all_times))]

    # Split in train/test/valid
    print('tokenizing documents and splitting into train/test/valid/prediction...')

    num_time_points = len(time_list)
    tr_size = num_time_points - p_steps
    pr_size = num_time_points - tr_size
    print(f'total number of time steps: {num_time_points}')
    print(f'total number of train time steps: {tr_size}')
    del cvz

    tr_docs = list(filter(lambda x: x[2] in all_times[:tr_size], zip(all_docs_bow,  all_docs_sent, all_timestamps, all_rewards, all_rewards_bin, all_covariates)))
    pr_docs = list(filter(lambda x: x[2] in all_times[tr_size:tr_size + pr_size], zip(all_docs_bow, all_docs_sent, all_timestamps, all_rewards, all_rewards_bin, all_covariates)))

    ts_tr = [t for t in all_timestamps if t in all_times[:tr_size]]
    ts_pr = [t for t in all_timestamps if t in all_times[tr_size:tr_size + pr_size]]

    tr_size = int(np.floor(len(tr_docs) * tt_ratio[0]))
    ts_size = int(np.floor(len(tr_docs) * tt_ratio[1]))
    va_size = int(len(tr_docs) - tr_size - ts_size)

    tr_docs, va_docs, ts_tr, ts_va = train_test_split(tr_docs, ts_tr, train_size=tt_ratio[0], random_state=random_seed)
    va_docs, te_docs, ts_va, ts_te = train_test_split(va_docs, ts_va, train_size=va_size, random_state=random_seed)

    print('  removing words from vocabulary not in training set ...')
    vocab_bow = list(set([w for doc in tr_docs for w in doc[0].split() if w in word2id]))
    vocab_bow_size = len(vocab_bow)
    print('         bow vocabulary after removing words not in train: {}'.format(vocab_bow_size))

    vocab = vocab_bow
    vocab.extend(SPECIAL_TOKENS)
    print(f' total vocabulary size: {len(vocab)}')

    # Create dictionary and inverse dictionary
    print('  create dictionary and inverse dictionary ')
    word2id = dict([(w, j) for j, w in enumerate(vocab)])
    id2word = dict([(j, w) for j, w in enumerate(vocab)])

    print('tokenizing bow ...')
    docs_b_tr = tokenize_bow(tr_docs, vocab_bow_size, word2id)
    docs_b_va = tokenize_bow(va_docs, vocab_bow_size, word2id)
    docs_b_te = tokenize_bow(te_docs, vocab_bow_size, word2id)
    docs_b_pr = tokenize_bow(pr_docs, vocab_bow_size, word2id)
    del all_docs_bow

    docs_r_tr = [float(doc[-3]) for doc in tr_docs]
    docs_r_va = [float(doc[-3]) for doc in va_docs]
    docs_r_te = [float(doc[-3]) for doc in te_docs]
    docs_r_pr = [float(doc[-3]) for doc in pr_docs]

    docs_r_bin_tr = [float(doc[-2]) for doc in tr_docs]
    docs_r_bin_va = [float(doc[-2]) for doc in va_docs]
    docs_r_bin_te = [float(doc[-2]) for doc in te_docs]
    docs_r_bin_pr = [float(doc[-2]) for doc in pr_docs]

    docs_covar_tr = [doc[-1] for doc in tr_docs]
    docs_covar_va = [doc[-1] for doc in va_docs]
    docs_covar_te = [doc[-1] for doc in te_docs]
    docs_covar_pr = [doc[-1] for doc in pr_docs]

    docs_s_tr = [doc[1] for doc in tr_docs]
    docs_s_va = [doc[1] for doc in va_docs]
    docs_s_te = [doc[1] for doc in te_docs]
    docs_s_pr = [doc[1] for doc in pr_docs]

    # Remove empty documents
    print('removing empty documents ...')

    def remove_empty(in_docs_b, in_docs_trans, in_timestamps, in_reward, in_reward_bin, in_covar):
        out_docs_b = []

        out_docs_trans = []
        out_timestamps = []
        out_reward = []
        out_reward_bin = []
        out_covar = []
        for ii, doc in enumerate(in_docs_b):
            if doc:
                out_docs_b.append(doc)
                out_docs_trans.append(in_docs_trans[ii])
                out_timestamps.append(in_timestamps[ii])
                if in_reward:
                    out_reward.append(in_reward[ii])
                    out_reward_bin.append(in_reward_bin[ii])
                if in_covar:
                    out_covar.append(in_covar[ii])

        return out_docs_b, out_docs_trans, out_timestamps, out_reward, out_reward_bin, out_covar

    def remove_by_threshold(in_docs_b, in_docs_trans, in_timestamps, in_docs_r, in_docs_r_bin, in_docs_c, thr):
        out_docs_b = []
        out_docs_trans = []
        out_timestamps = []
        out_docs_r = []
        out_docs_r_bin = []
        out_docs_c = []
        for ii, doc in enumerate(in_docs_b):
            if len(doc) > thr:
                out_docs_b.append(doc)
                out_docs_trans.append(in_docs_trans[ii])
                out_timestamps.append(in_timestamps[ii])
                if reward_column is not None:
                    out_docs_r.append(in_docs_r[ii])
                    out_docs_r_bin.append(in_docs_r_bin[ii])
                if covariates:
                    out_docs_c.append(in_docs_c[ii])
        return out_docs_b, out_docs_trans, out_timestamps, out_docs_r, out_docs_r_bin, out_docs_c

    docs_b_tr, docs_s_tr, ts_tr, docs_r_tr, docs_r_bin_tr, docs_covar_tr = remove_empty(docs_b_tr, docs_s_tr, ts_tr, docs_r_tr, docs_r_bin_tr, docs_covar_tr)
    docs_b_va, docs_s_va, ts_va, docs_r_va, docs_r_bin_va, docs_covar_va = remove_empty(docs_b_va, docs_s_va, ts_va, docs_r_va, docs_r_bin_va, docs_covar_va)
    docs_b_te, docs_s_te, ts_te, docs_r_te, docs_r_bin_te, docs_covar_te = remove_empty(docs_b_te, docs_s_te, ts_te, docs_r_te, docs_r_bin_te, docs_covar_te)
    docs_b_pr, docs_s_pr, ts_pr, docs_r_pr, docs_r_bin_pr, docs_covar_pr = remove_empty(docs_b_pr,  docs_s_pr, ts_pr, docs_r_pr, docs_r_bin_pr, docs_covar_pr)

    # Remove prediction and test documents with length=1
    docs_b_pr, docs_s_pr, ts_pr, docs_r_pr, docs_r_bin_pr, docs_covar_pr = remove_by_threshold(docs_b_pr, docs_s_pr, ts_pr, docs_r_pr, docs_r_bin_pr, docs_covar_pr, 1)
    docs_b_te, docs_s_te, ts_te, docs_r_te, docs_r_bin_te, docs_covar_te = remove_by_threshold(docs_b_te, docs_s_te, ts_te, docs_r_te, docs_r_bin_te, docs_covar_te, 1)

    # Split test set in 2 halves
    print('splitting test documents in 2 halves...')
    docs_te_h1 = [[w for i, w in enumerate(doc) if i <= len(doc) / 2.0 - 1] for doc in docs_b_te]
    docs_te_h2 = [[w for i, w in enumerate(doc) if i > len(doc) / 2.0 - 1] for doc in docs_b_te]

    # Getting lists of words and doc_indices
    print('creating lists of words...')

    def create_list_words(in_docs):
        return [x for y in in_docs for x in y]

    words_tr = create_list_words(docs_b_tr)
    words_va = create_list_words(docs_b_va)
    words_te = create_list_words(docs_b_te)
    words_te_h1 = create_list_words(docs_te_h1)
    words_te_h2 = create_list_words(docs_te_h2)
    words_pr = create_list_words(docs_b_pr)

    print('  len(words_tr): ', len(words_tr))
    print('  len(words_va): ', len(words_va))
    print('  len(words_va): ', len(words_te))
    print('  len(words_te_h1): ', len(words_te_h1))
    print('  len(words_te_h2): ', len(words_te_h2))
    print('  len(words_pr): ', len(words_pr))

    # Get doc indices
    print('getting doc indices...')

    def create_doc_indices(in_docs):
        aux = [[j for _ in range(len(doc))] for j, doc in enumerate(in_docs)]
        return [int(x) for y in aux for x in y]

    doc_indices_tr = create_doc_indices(docs_b_tr)
    doc_indices_va = create_doc_indices(docs_b_va)
    doc_indices_te = create_doc_indices(docs_b_te)
    doc_indices_te_h1 = create_doc_indices(docs_te_h1)
    doc_indices_te_h2 = create_doc_indices(docs_te_h2)
    doc_indices_pr = create_doc_indices(docs_b_pr)

    print('  len(np.unique(doc_indices_tr)): {} [this should be {}]'.format(len(np.unique(doc_indices_tr)), len(docs_b_tr)))
    print('  len(np.unique(doc_indices_va)): {} [this should be {}]'.format(len(np.unique(doc_indices_va)), len(docs_b_va)))
    print('  len(np.unique(doc_indices_te)): {} [this should be {}]'.format(len(np.unique(doc_indices_te)), len(docs_b_te)))
    print('  len(np.unique(doc_indices_te_h1)): {} [this should be {}]'.format(len(np.unique(doc_indices_te_h1)), len(docs_te_h1)))
    print('  len(np.unique(doc_indices_te_h2)): {} [this should be {}]'.format(len(np.unique(doc_indices_te_h2)), len(docs_te_h2)))
    print('  len(np.unique(doc_indices_pr)): {} [this should be {}]'.format(len(np.unique(doc_indices_pr)), len(docs_b_pr)))

    # Number of documents in each set
    n_docs_tr = len(docs_b_tr)
    n_docs_va = len(docs_b_va)
    n_docs_te = len(docs_b_te)
    n_docs_te_h1 = len(docs_te_h1)
    n_docs_te_h2 = len(docs_te_h2)
    n_docs_pr = len(docs_b_pr)

    # Create bow representation
    print('creating bow representation...')

    def create_bow(doc_indices, words, n_docs, vocab_size):
        return sparse.coo_matrix(([1] * len(doc_indices), (doc_indices, words)), shape=(n_docs, vocab_size)).tocsr()

    bow_tr = create_bow(doc_indices_tr, words_tr, n_docs_tr, vocab_bow_size)
    bow_va = create_bow(doc_indices_va, words_va, n_docs_va, vocab_bow_size)
    bow_te = create_bow(doc_indices_te, words_te, n_docs_te, vocab_bow_size)
    bow_te_h1 = create_bow(doc_indices_te_h1, words_te_h1, n_docs_te_h1, vocab_bow_size)
    bow_te_h2 = create_bow(doc_indices_te_h2, words_te_h2, n_docs_te_h2, vocab_bow_size)
    bow_pr = create_bow(doc_indices_pr, words_pr, n_docs_pr, vocab_bow_size)

    del words_tr
    del words_pr
    del words_va
    del words_te_h1
    del words_te_h2
    del doc_indices_tr
    del doc_indices_pr
    del doc_indices_va
    del doc_indices_te_h1
    del doc_indices_te_h2
    del doc_indices_te

    print('bow => tf idf')
    tfidf_trfm = TfidfTransformer(norm=None)
    tfidf_tr = tfidf_trfm.fit_transform(bow_tr)
    tfidf_va = tfidf_trfm.transform(bow_va)
    tfidf_te = tfidf_trfm.transform(bow_te)
    tfidf_te_h1 = tfidf_trfm.transform(bow_te_h1)
    tfidf_te_h2 = tfidf_trfm.transform(bow_te_h2)
    tfidf_pr = tfidf_trfm.transform(bow_pr)

    if not os.path.exists(output):
        os.makedirs(output, exist_ok=True)

    print('exporting training dataset...')
    ts_tr_id = [time2id[t] for t in ts_tr]
    train_dataset = {'text': docs_s_tr, 'time': ts_tr_id, 'bow': bow_tr, 'tfidf': tfidf_tr}

    if reward_column is not None:
        train_dataset.update({'reward': docs_r_tr, 'reward_bin': docs_r_bin_tr})
    if covariates:
        train_dataset.update({'covariates': np.asarray(docs_covar_tr, dtype=np.float32)})

    with open(os.path.join(output, f'train.pkl'), 'wb') as out:
        pickle.dump(train_dataset, out, protocol=4)

    del train_dataset
    del docs_s_tr
    del ts_tr_id
    del docs_covar_tr
    del tfidf_tr
    del docs_r_tr
    del docs_r_bin_tr

    print('exporting validation dataset...')
    ts_va_id = [time2id[t] for t in ts_va]

    validation_dataset = {'text': docs_s_va, 'time': ts_va_id, 'bow': bow_va, 'tfidf': tfidf_va}
    if reward_column is not None:
        validation_dataset.update({'reward': docs_r_va, 'reward_bin': docs_r_bin_va})
    if covariates:
        validation_dataset.update({'covariates': np.asarray(docs_covar_va, dtype=np.float32)})

    with open(os.path.join(output, f'validation.pkl'), 'wb') as out:
        pickle.dump(validation_dataset, out, protocol=4)

    del validation_dataset
    del docs_s_va
    del ts_va_id
    del bow_va
    del tfidf_va
    del docs_r_bin_va
    del docs_r_va

    print('exporting test dataset...')
    ts_te_id = [time2id[t] for t in ts_te]

    test_dataset = {'text': docs_s_te, 'time': ts_te_id, 'bow_h1': bow_te_h1, 'bow_h2': bow_te_h2, 'bow': bow_te, 'tfidf_h1': tfidf_te_h1,
                    'tfidf_h2': tfidf_te_h2, 'tfidf': tfidf_te}
    if reward_column is not None:
        test_dataset.update({'reward': docs_r_te, 'reward_bin': docs_r_bin_te})
    if covariates:
        test_dataset.update({'covariates': np.asarray(docs_r_bin_te, dtype=np.float32)})
    with open(os.path.join(output, f'test.pkl'), 'wb') as out:
        pickle.dump(test_dataset, out, protocol=4)

    del test_dataset
    del docs_s_te
    del ts_te_id
    del bow_te_h1
    del bow_te_h2
    del bow_te
    del tfidf_te_h1
    del tfidf_te_h2
    del tfidf_te
    del docs_r_te
    del docs_r_bin_te

    print('export prediction dataset...')
    ts_pr_id = [time2id[t] for t in ts_pr]

    prediction_dataset = {'text': docs_s_pr, 'time': ts_pr_id, 'bow': bow_pr, 'tfidf': tfidf_pr}
    if reward_column is not None:
        prediction_dataset.update({'reward': docs_r_pr, 'reward_bin': docs_r_bin_pr})
    if covariates:
        prediction_dataset.update({'covariates': np.asarray(docs_covar_pr, dtype=np.float32)})

    with open(os.path.join(output, f'prediction.pkl'), 'wb') as out:
        pickle.dump(prediction_dataset, out, protocol=4)

    del prediction_dataset
    del docs_s_pr
    del ts_pr_id
    del bow_pr
    del tfidf_pr
    del docs_r_bin_pr

    time = {'all_time': all_times, 'time2id': time2id, 'id2time': id2time}
    with open(os.path.join(output, 'time.pkl'), 'wb') as out:
        pickle.dump(time, out)

    del time
    del all_times
    del time2id
    del id2time

    print('counting words ...')

    word_counts = np.squeeze(np.asarray((bow_tr > 0).sum(axis=0)))
    word_counts = dict(zip(range(len(word_counts)), word_counts.tolist()))

    embeddings = __load_embeddings(embeddings_path)
    e_size = embeddings['the'].shape[0]
    vectors = list(map(lambda x: embeddings.get(x, np.random.randn(e_size)), vocab))

    vocabulary = {'vocab': vocab, 'stoi': word2id, 'itos': id2word, 'word_count': dict(word_counts), 'vectors': np.asarray(vectors, dtype=np.float32)}

    with open(os.path.join(output, 'vocabulary.pkl'), 'wb') as out:
        pickle.dump(vocabulary, out)


def tokenize_bow(tr_docs, vocab_bow_size, word2id):
    docs_b_tr = [[word2id[w] for w in doc[0].split() if w in word2id and word2id[w] < vocab_bow_size] for doc in tr_docs]
    return docs_b_tr


def __load_embeddings(embeddings_path):
    print("load embeddings...")
    embeddings = dict()
    with open(embeddings_path, 'rb') as f:
        for row in f:
            line = row.decode().split()
            word = line[0]
            vect = np.array(line[1:]).astype(np.float)
            embeddings[word] = vect
    e_size = embeddings['the'].shape[0]
    for tok in SPECIAL_TOKENS:
        embeddings[tok] = np.zeros(e_size)
    return embeddings


def get_type_conversion_function(type: str):
    if type == 'str':
        return lambda x: str(x)
    elif type == 'bool':
        return lambda x: 1.0 if x == 'True' else 0.0
    elif type == 'int':
        return lambda x: -1.0 if x == '' else int(x)
    elif type == 'float':
        return lambda x: -1.0 if x == '' else float(x)
    else:
        raise TypeError(f"Unknown type {type}")


def _read_docs(input, is_abstract: bool, reward: str, binarize_reward: list, covariates: List[tuple], date_field_name: str,  n_docs_per_time: int, total_num_time_steps: int):
    all_timestamps, all_docs, all_rewards, all_rewards_bin, all_covar = [], [], [], [], []
    if total_num_time_steps is None:
        total_num_time_steps = math.inf
    if os.path.isfile(input):
        files = [input]
    else:
        files = filter(os.path.isfile, glob.glob(os.path.join(input, "*")))
    for file_path in files:

        with open(file_path, 'r', encoding='utf-8', newline=None) as out:
            csv_reader = csv.reader(fix_nulls(out), delimiter=',', quotechar='"')
            header: list = next(csv_reader)
            date_ix = header.index(date_field_name)
            text_ix = header.index('text')
            if covariates:
                cov_ix = [(header.index(n[0]), get_type_conversion_function(n[1])) for n in covariates]
            if reward is not None:
                reward_id = header.index(reward)
            if is_abstract:
                text_ix = header.index('abstract')

            print(f"Reading data with header: {header}")
            count = 0
            p_bar = tqdm.tqdm(desc=f"Reading documents: {file_path}", unit="document")
            while True:
                try:
                    row = next(csv_reader)
                    date = row[date_ix]

                    if '-' in date:
                        year, month = date.split('-')
                        date = int(year) * 100 + int(month)
                    else:
                        date = int(date)
                    if date > total_num_time_steps:
                        continue
                    all_timestamps.append(date)
                    all_docs.append(row[text_ix])
                    if reward is not None:
                        all_rewards_bin.append(np.digitize(int(row[reward_id]), binarize_reward)-1)
                        all_rewards.append(int(row[reward_id]))
                    else:
                        all_rewards.append(0)
                        all_rewards_bin.append(0)
                    if covariates:
                        try:
                            all_covar.append([convert_fn(row[ix]) for ix, convert_fn in cov_ix])
                        except Exception as e:
                            print(e)
                    else:
                        all_covar.append([1])

                except StopIteration:
                    break
                except Exception as e:
                    print(f"Exception for row {count}: {e}")

                finally:
                    count += 1
                    p_bar.update()
    if n_docs_per_time is not None:
        all_docs, all_timestamps, all_rewards, all_rewards_bin, all_covar = subsample_per_time_period(n_docs_per_time, all_timestamps, all_docs, all_rewards, all_rewards_bin, all_covar)

    return all_docs, all_timestamps, all_rewards, all_rewards_bin, all_covar


def subsample_per_time_period(n_documents: int, time_ids: list, docs: list,  all_rewards: list, all_rewards_bin: list, all_covar: list):
    _ids,  _docs,  _covar, _reward, _reward_bin = [], [], [], [], []
    unique_time_ix = set(time_ids)
    time_ids = np.asarray(time_ids)

    all_rewards = np.asarray(all_rewards)
    all_rewards_bin = np.asarray(all_rewards_bin)
    all_covar = np.asarray(all_covar)
    for time_ix in unique_time_ix:
        ids = np.where(time_ids == time_ix)[0]
        sample_ids = np.random.choice(ids, n_documents)
        _docs.extend(operator.itemgetter(*sample_ids)(docs))
        _ids.extend([time_ix]*n_documents)

        _covar.extend(all_covar[sample_ids].tolist())

        _reward.extend(all_rewards[sample_ids].tolist())
        _reward_bin.extend(all_rewards_bin[sample_ids].tolist())

    return _docs, _ids, _reward, _reward_bin, _covar


if __name__ == '__main__':
    preprocess()
