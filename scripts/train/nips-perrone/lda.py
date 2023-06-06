# %%
import gensim
import gensim.corpora as corpora
from gensim.models import LdaModel
from gensim.matutils import Sparse2Corpus
import pickle
import os
import numpy as np
from deep_fields import data_path
from pprint import pprint
from tqdm import tqdm


import logging
# logging.basicConfig(format='%(asctime)s : %(levelname)s : %(message)s', level=logging.INFO)

# %%
data = pickle.load(open(os.path.join(data_path, 'preprocessed/nips-perrone/language/0.5/train.pkl'), 'rb'))
vocabulary = pickle.load(open(os.path.join(data_path, 'preprocessed/nips-perrone/language/0.5/vocabulary.pkl'), 'rb'))
print(vocabulary.keys())
# %%

text = []
for doc in tqdm(data['bow'].toarray().tolist(), desc="Building Corpus"):
    text.append([vocabulary['itos'][w] for w in np.nonzero(doc)[0]])

# Create Dictionary
id2word = corpora.Dictionary(text)

# Create Corpus


# Term Document Frequency
corpus = [id2word.doc2bow(text) for text in text]

lda_model = LdaModel(corpus=corpus,
                     id2word=id2word,
                     num_topics=50,
                     random_state=100,
                     update_every=1,
                     chunksize=100,
                     passes=10,
                     alpha='auto',
                     per_word_topics=True)

pprint(lda_model.print_topics())

top_topics = lda_model.top_topics(corpus)  # , num_words=20)

# Average topic coherence is the sum of topic coherences of all topics, divided by the number of topics.
avg_topic_coherence = sum([t[1] for t in top_topics]) / 10
print('Average topic coherence: %.4f.' % avg_topic_coherence)
perplexity = lda_model.log_perplexity(corpus)
print('perplexity: %.4f.' % perplexity)

# %%
