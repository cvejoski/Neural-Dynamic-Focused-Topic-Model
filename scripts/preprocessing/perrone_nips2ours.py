# %%
import pandas as pd
from deep_fields import data_path
import os
from tqdm import tqdm
import numpy as np
import pickle
from scipy.sparse import csr_matrix
from sklearn.model_selection import train_test_split

# %%


def __load_embeddings(embeddings_path):
    print("load embeddings:")
    embeddings = dict()
    with open(embeddings_path, 'rb') as f:
        for row in f:
            line = row.decode().split()
            word = line[0]
            vect = np.array(line[1:]).astype(np.float)
            embeddings[word] = vect
    return embeddings


NIPS_PATH = os.path.join(data_path, 'raw', 'nips', 'NIPS_1987-2015.csv')
p_steps = 1
tt_ratio = (0.8, 0.05, 0.15)

data = pd.read_csv(NIPS_PATH)
output = 'data/preprocessed/nips-perrone/language'
embeddings_path = 'embeddings/glove.6B.300d.txt'
heldout = [0.5, 0.6, 0.7, 0.8]

# %%
data.iloc[6863, 0] = 'null'
data.iloc[6600, 0] = 'nan'
data.head()
vocab_bow = list(data['Unnamed: 0'].values)
data = data.set_index('Unnamed: 0')
# %%
all_timestamps = []
all_bows = []
for column in tqdm(data.columns):
    bow = data[column].values
    if sum(bow) == 0:
        continue

    ts = int(column.split('_')[0])
    if ts > 1999:
        break
    all_timestamps.append(ts)
    all_bows.append(data[column].values)


# %%
word2id = dict([(w, j) for j, w in enumerate(vocab_bow)])
all_times = sorted(set(all_timestamps))
time2id = dict([(t, i) for i, t in enumerate(all_times)])
id2time = dict([(i, t) for i, t in enumerate(all_times)])
time_list = [id2time[i] for i in range(len(all_times))]
# %%
# Split in train/test/valid
print('tokenizing documents and splitting into train/test/valid/prediction...')

num_time_points = len(time_list)
# tr_size = num_time_points - p_steps
# pr_size = num_time_points - tr_size
# print(f'total number of time steps: {num_time_points}')
# print(f'total number of train time steps: {tr_size}')

# %%
tr_docs_all = list(filter(lambda x: x[1] in all_times, zip(all_bows, all_timestamps)))
# pr_docs_all = list(filter(lambda x: x[1] in all_times[tr_size:tr_size + pr_size], zip(all_bows, all_timestamps)))


timestamps_tr_all = [t for t in all_timestamps if t in all_times]
# timestamps_pr = [t for t in all_timestamps if t in all_times[tr_size:tr_size + pr_size]]

embeddings = __load_embeddings(embeddings_path)
e_size = embeddings['the'].shape[0]
for per in heldout:
    bow_size = tr_docs_all[0][0].shape
    output = 'data/preprocessed/nips-perrone/language'
    data = list(filter(lambda x: x[1] != 1999, tr_docs_all))
    data_1999 = list(filter(lambda x: x[1] == 1999, tr_docs_all))
    train_1999 = []
    va_docs = []
    for d, _ in data_1999:
        mask = np.random.binomial(size=bow_size, n=1, p=per)
        train_1999.append((1-mask*d, 1999))
        va_docs.append((mask*d, 1999))
    tr_docs = data + train_1999
    # tr_docs, va_docs, timestamps_tr, timestamps_va = train_test_split(tr_docs_all, timestamps_tr_all, train_size=tt_ratio[0], random_state=random_seed)
    # va_docs, te_docs, timestamps_va, timestamps_te = train_test_split(va_docs, timestamps_va, train_size=va_size, random_state=random_seed)

    # va_size, te_size = len(va_docs), len(te_docs)
    vocab = vocab_bow
    print(f' total vocabulary size: {len(vocab)}')

    bow_tr = csr_matrix(np.stack([doc[0] for doc in tr_docs]))
    bow_va = csr_matrix(np.stack([doc[0] for doc in va_docs]))
    # bow_te = csr_matrix(np.stack([doc[0] for doc in te_docs]))
    # # bow_pr = csr_matrix(np.stack([doc[0] for doc in pr_docs_all]))

    # Create dictionary and inverse dictionary
    print('  create dictionary and inverse dictionary ')
    word2id = dict([(w, j) for j, w in enumerate(vocab)])
    id2word = dict([(j, w) for j, w in enumerate(vocab)])

    # %%
    timestamps_tr_id = np.asarray([time2id[t] for t in timestamps_tr_all])
    train_dataset = {'time': timestamps_tr_id, 'bow': bow_tr}
    output = os.path.join(output, str(per))
    if not os.path.exists(output):
        os.makedirs(output, exist_ok=True)

    with open(os.path.join(output, f'train.pkl'), 'wb') as out:
        pickle.dump(train_dataset, out, protocol=4)

    del train_dataset
    del timestamps_tr_id

    print('padding validation dataset')
    timestamps_va_id = np.ones(bow_va.shape[0], dtype=np.int)*time2id[1999]  # np.asarray([time2id[t] for t in timestamps_tr_all])
    validation_dataset = {'time': timestamps_va_id, 'bow': bow_va}

    with open(os.path.join(output, f'validation.pkl'), 'wb') as out:
        pickle.dump(validation_dataset, out, protocol=4)

    del validation_dataset
    # del timestamps_va_id
    # del bow_va

    print('padding test dataset')
    timestamps_te_id = np.asarray([time2id[t] for t in timestamps_tr_all])
    test_dataset = {'time': timestamps_va_id, 'bow_h1': bow_va, 'bow_h2': bow_va, 'bow': bow_va}

    with open(os.path.join(output, f'test.pkl'), 'wb') as out:
        pickle.dump(test_dataset, out, protocol=4)

    del test_dataset
    del timestamps_te_id
    # del bow_te

    print('padding prediction dataset')
    timestamps_pr_id = np.asarray([time2id[t] for t in timestamps_tr_all])
    prediction_dataset = {'time': timestamps_va_id, 'bow': bow_va}

    with open(os.path.join(output, f'prediction.pkl'), 'wb') as out:
        pickle.dump(prediction_dataset, out, protocol=4)

    del prediction_dataset
    del timestamps_pr_id

    time = {'all_time': all_times, 'time2id': time2id, 'id2time': id2time}
    with open(os.path.join(output, 'time.pkl'), 'wb') as out:
        pickle.dump(time, out)

    print('counting words ...')

    word_counts = np.squeeze(np.asarray((bow_tr > 0).sum(axis=0)))
    word_counts = dict(zip(range(len(word_counts)), word_counts.tolist()))

    vectors = list(map(lambda x: embeddings.get(x, np.random.randn(e_size)), vocab))

    vocabulary = {'vocab': vocab, 'stoi': word2id, 'itos': id2word, 'word_count': dict(word_counts), 'vectors': np.asarray(vectors, dtype=np.float32)}

    with open(os.path.join(output, 'vocabulary.pkl'), 'wb') as out:
        pickle.dump(vocabulary, out)
