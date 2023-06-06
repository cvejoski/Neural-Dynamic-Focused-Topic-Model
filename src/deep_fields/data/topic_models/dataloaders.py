from abc import ABC
from functools import partial

import numpy as np
import spacy
import torch
from nltk import TweetTokenizer

from torch.utils.data.dataloader import DataLoader

from deep_fields.data.topic_models.dataset import TopicLanguageDataset, DynamicTopicLanguageDataset, LanguageDataset, TopicDataset, \
    TopicTransformerLanguageDataset
from deep_fields.data.topic_models.language_modeling import PennTreebank
from deep_fields.data.utils import Vocab

sampler = torch.utils.data.RandomSampler
DistributedSampler = torch.utils.data.distributed.DistributedSampler

spacy_en = spacy.load('en_core_web_sm')
tokenizer = TweetTokenizer(preserve_case=False).tokenize


class ADataLoader(ABC):
    _train_iter: DataLoader
    _valid_iter: DataLoader
    _test_iter: DataLoader
    _predict_iter: DataLoader

    def __init__(self, device, rank: int = 0, world_size: int = -1, **kwargs):
        self.dataset_kwargs = kwargs
        self.device = device
        self.batch_size = kwargs.get('batch_size')
        self.path_to_vectors = kwargs.pop('path_to_vectors', None)
        self.emb_dim = kwargs.pop('emb_dim', None)
        self.world_size = world_size
        self.rank = rank

    @property
    def train(self):
        return self._train_iter

    @property
    def validate(self):
        return self._valid_iter

    @property
    def test(self):
        return self._test_iter

    @property
    def predict(self):
        return self._predict_iter

    @property
    def n_train_batches(self):
        return len(self.train.dataset) // self.batch_size // abs(self.world_size)

    @property
    def n_test_batches(self):
        return len(self.test.dataset) // self.batch_size // abs(self.world_size)

    @property
    def n_validate_batches(self):
        return len(self.validate.dataset) // self.batch_size // abs(self.world_size)

    @property
    def train_set_size(self):
        return len(self.train.dataset)

    @property
    def validation_set_size(self):
        return len(self.validate.dataset)

    @property
    def test_set_size(self):
        return len(self.test.dataset)

    @property
    def prediction_set_size(self):
        return len(self.predict.dataset)

    @property
    def number_of_documents(self):
        try:
            return self.train_set_size + self.validation_set_size + self.prediction_set_size + self.test_set_size
        except:
            return self.train_set_size + self.validation_set_size + self.test_set_size

    @property
    def vocab(self) -> Vocab:
        return self.train.dataset.vocab

    @property
    def vocabulary_dim(self):
        voc_l = len(self.vocab.vocab)
        for special in ['<pad>', '<sos>', '<eos>', '<unk>']:
            if special in self.vocab.stoi:
                voc_l -= 1
        return voc_l

    @property
    def max_doc_len(self):
        raise AttributeError('max_doc_len is only defined for bag of sentences dataloaders!')

    @property
    def max_sent_len(self):
        return len(self.train.dataset.data['text'][0][0])

    @property
    def num_training_steps(self):
        return len(set(self.train.dataset.data['time']))

    @property
    def num_prediction_steps(self):
        return len(set(self.predict.dataset.data['time']))

    @property
    def training_times(self):
        return list(set(self.train.dataset.data['time']))

    @property
    def prediction_times(self):
        return list(set(self.predict.dataset.data['time']))

    @property
    def prediction_count_per_year(self):
        return len(self.predict.dataset)


class TopicDataloader(ADataLoader):
    """
    first we create and store the examples and text fields
    here, then, the dataset is divided in prediction and training
    prediction is only in the future (respects causality)

    the training is then divided in train and validation, which is a random split
    """

    def __init__(self, device, rank: int = 0, world_size=-1, **kwargs):
        super().__init__(device, rank, world_size, **kwargs)

        data_dir = kwargs.get("path_to_data")
        is_dynamic = kwargs.get("is_dynamic", False)
        use_covariates = kwargs.get("use_covariates", False)
        use_tmp_covariates = kwargs.get("use_tmp_covariates", False)
        word_emb_type = kwargs.get("word_emb_type", "bow")
        normalize = kwargs.get("normalize", False)
        n_workers = kwargs.get("n_workers", 8)
        transformer_name = kwargs.get("transformer_name", None)
        tokenizer = self.get_transformer_tokenizer(transformer_name)
        train_dataset = TopicDataset(data_dir, "train", is_dynamic, use_covariates, use_tmp_covariates, normalize, word_emb_type, tokenizer)
        valid_dataset = TopicDataset(data_dir, "validation", is_dynamic, use_covariates, use_tmp_covariates, train_dataset.cov_stand, word_emb_type, tokenizer)
        test_dataset = TopicDataset(data_dir, "test", is_dynamic, use_covariates, use_tmp_covariates, train_dataset.cov_stand, word_emb_type, tokenizer)
        predict_dataset = TopicDataset(data_dir, "prediction", is_dynamic, use_covariates, use_tmp_covariates, train_dataset.cov_stand, word_emb_type, tokenizer)
        if is_dynamic:
            valid_dataset.corpus_per_time_period_avg = train_dataset.corpus_per_time_period_avg
            test_dataset.corpus_per_time_period_avg = train_dataset.corpus_per_time_period_avg
            if train_dataset.number_of_rewards_categories() is not None:
                valid_dataset.reward_per_time_period_avg = train_dataset.reward_per_time_period_avg
                test_dataset.reward_per_time_period_avg = train_dataset.reward_per_time_period_avg

        train_sampler = None
        valid_sampler = None
        test_sampler = None
        predict_sampler = None
        # if is_dynamic:

        if self.world_size != -1:
            train_sampler = DistributedSampler(train_dataset, self.world_size, self.rank)
            valid_sampler = DistributedSampler(valid_dataset, self.world_size, self.rank)
            test_sampler = DistributedSampler(test_dataset, self.world_size, self.rank)
            predict_sampler = DistributedSampler(test_dataset, self.world_size, self.rank)

        self._train_iter = DataLoader(train_dataset, drop_last=False, sampler=train_sampler, shuffle=train_sampler is None, batch_size=self.batch_size, num_workers=n_workers, pin_memory=True)
        self._valid_iter = DataLoader(valid_dataset, drop_last=False, sampler=valid_sampler, shuffle=valid_sampler is None, batch_size=self.batch_size, num_workers=n_workers, pin_memory=True)
        self._test_iter = DataLoader(test_dataset, drop_last=False, sampler=test_sampler, shuffle=test_sampler is None, batch_size=self.batch_size, num_workers=n_workers, pin_memory=True)
        self._predict_iter = DataLoader(predict_dataset, drop_last=False, sampler=predict_sampler, shuffle=predict_sampler is None, batch_size=self.batch_size)

        self._rewards_values = train_dataset.rewards_categories()
        self._number_of_reward_categories = train_dataset.number_of_rewards_categories()
        self._type_of_rewards = train_dataset.type_of_rewards()

    @property
    def vocabulary_dim(self):
        return self.train.dataset.data['bow'].shape[1]

    @property
    def word_embeddings_dim(self):
        return self.train.dataset.vocab.vectors.shape[1]

    @property
    def number_of_documents(self):
        return self.train_set_size + self.validation_set_size + self.test_set_size

    @property
    def rewards_values(self):
        if self._type_of_rewards == "discrete":
            return self._rewards_values
        else:
            return None

    @property
    def number_of_reward_categories(self):
        if self._type_of_rewards == "discrete":
            return self._number_of_reward_categories

        return np.inf

    @property
    def type_of_rewards(self):
        return self._type_of_rewards

    @property
    def word_emb_type(self):
        return self.train.dataset.word_emb_type

    def get_transformer_tokenizer(self, tokenizer_name):
        if tokenizer_name == None:
            return None
        elif tokenizer_name == 'bert':
            from transformers import BertTokenizer
            tokenizer = BertTokenizer.from_pretrained('bert-base-uncased', do_lower_case=True)
            tokenizer.name = 'bert-base-uncased'
        elif tokenizer_name == 'roberta':
            from transformers import RobertaTokenizer
            tokenizer = RobertaTokenizer.from_pretrained('roberta-base', do_lower_case=True)
            tokenizer.name = 'roberta-base'
        elif tokenizer_name == 'albert':
            from transformers import AlbertTokenizer
            tokenizer = AlbertTokenizer.from_pretrained('albert-base-v2', do_lower_case=True)
            tokenizer.name = 'albert-base-v2'
        else:
            raise ValueError('No matching backbone network')
        tokenizer = partial(tokenizer, add_special_tokens=True, truncation=True, padding='max_length', return_attention_mask=True, return_tensors='pt')
        return tokenizer


class LanguageDataloader(ADataLoader):
    """
    first we create and store the examples and text fields
    here, then, the dataset is divided in prediction and training
    prediction is only in the future (respects causality)

    the training is then divided in train and validation, which is a random split
    """

    def __init__(self, device, rank: int = 0, world_size=-1, **kwargs):
        super().__init__(device, rank, world_size, **kwargs)
        data_dir = kwargs.get("path_to_data")

        train_dataset = LanguageDataset(data_dir, "train")
        valid_dataset = LanguageDataset(data_dir, "validation")
        test_dataset = LanguageDataset(data_dir, "test")

        train_sampler = None
        valid_sampler = None
        test_sampler = None

        if self.world_size != -1:
            train_sampler = DistributedSampler(train_dataset, self.world_size, self.rank)
            valid_sampler = DistributedSampler(valid_dataset, self.world_size, self.rank)
            test_sampler = DistributedSampler(test_dataset, self.world_size, self.rank)

        self._train_iter = DataLoader(train_dataset, drop_last=True, sampler=train_sampler, shuffle=train_sampler is None, batch_size=self.batch_size)
        self._valid_iter = DataLoader(valid_dataset, drop_last=True, sampler=valid_sampler, shuffle=valid_sampler is None, batch_size=self.batch_size)
        self._test_iter = DataLoader(test_dataset, drop_last=True, sampler=test_sampler, shuffle=test_sampler is None, batch_size=self.batch_size)


class TopicLanguageDataloader(ADataLoader):
    """
    first we create and store the examples and text fields
    here, then, the dataset is divided in prediction and training
    prediction is only in the future (respects causality)

    the training is then divided in train and validation, which is a random split
    """

    def __init__(self, device, rank: int = 0, world_size=-1, **kwargs):
        super().__init__(device, rank, world_size, **kwargs)
        data_dir = kwargs.get("path_to_data")

        train_dataset = TopicLanguageDataset(data_dir, "train")
        valid_dataset = TopicLanguageDataset(data_dir, "validation")
        test_dataset = TopicLanguageDataset(data_dir, "test")

        train_sampler = None
        valid_sampler = None
        test_sampler = None

        if self.world_size != -1:
            train_sampler = DistributedSampler(train_dataset, self.world_size, self.rank)
            valid_sampler = DistributedSampler(valid_dataset, self.world_size, self.rank)
            test_sampler = DistributedSampler(test_dataset, self.world_size, self.rank)

        self._train_iter = DataLoader(train_dataset, drop_last=True, sampler=train_sampler, shuffle=train_sampler is None, batch_size=self.batch_size)
        self._valid_iter = DataLoader(valid_dataset, drop_last=True, sampler=valid_sampler, shuffle=valid_sampler is None, batch_size=self.batch_size)
        self._test_iter = DataLoader(test_dataset, drop_last=True, sampler=test_sampler, shuffle=test_sampler is None, batch_size=self.batch_size)

    @property
    def bow_size(self):
        return self.train.dataset.data['bow'].shape[1]


class TopicTransformerLanguageDataloader(ADataLoader):
    def __init__(self, device, rank: int = 0, world_size=-1, name="bert", **kwargs):
        super().__init__(device, rank, world_size, **kwargs)
        data_dir = kwargs.get("path_to_data")
        is_dynamic = kwargs.get("is_dynamic", False)
        use_covariates = kwargs.get("use_covariates", False)
        use_tmp_covariates = kwargs.get("use_tmp_covariates", False)
        word_emb_type = kwargs.get("word_emb_type", "bow")
        normalize = kwargs.get("normalize", False)
        if name == 'bert':
            from transformers import BertTokenizer
            tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
            tokenizer.name = 'bert-base-uncased'
        elif name == 'roberta':
            from transformers import RobertaTokenizer
            tokenizer = RobertaTokenizer.from_pretrained('roberta-base')
            tokenizer.name = 'roberta-base'
        elif name == 'albert':
            from transformers import AlbertTokenizer
            tokenizer = AlbertTokenizer.from_pretrained('albert-base-v2')
            tokenizer.name = 'albert-base-v2'
        else:
            raise ValueError('No matching backbone network')

        tokenizer = partial(tokenizer, add_special_tokens=True, truncation=True, padding='max_length',
                            return_attention_mask=True, return_tensors='pt')

        train_dataset = TopicTransformerLanguageDataset(data_dir, "train", tokenizer, is_dynamic, use_covariates, use_tmp_covariates, normalize, word_emb_type)
        valid_dataset = TopicTransformerLanguageDataset(data_dir, "validation", tokenizer, is_dynamic, use_covariates, use_tmp_covariates, train_dataset.cov_stand, word_emb_type)
        test_dataset = TopicTransformerLanguageDataset(data_dir, "test", tokenizer, is_dynamic, use_covariates, use_tmp_covariates, train_dataset.cov_stand, word_emb_type)

        train_sampler = None
        valid_sampler = None
        test_sampler = None

        if self.world_size != -1:
            train_sampler = DistributedSampler(train_dataset, self.world_size, self.rank)
            valid_sampler = DistributedSampler(valid_dataset, self.world_size, self.rank)
            test_sampler = DistributedSampler(test_dataset, self.world_size, self.rank)

        self._train_iter = DataLoader(train_dataset, drop_last=True, sampler=train_sampler, shuffle=train_sampler is None, batch_size=self.batch_size)
        self._valid_iter = DataLoader(valid_dataset, drop_last=True, sampler=valid_sampler, shuffle=valid_sampler is None, batch_size=self.batch_size)
        self._test_iter = DataLoader(test_dataset, drop_last=True, sampler=test_sampler, shuffle=test_sampler is None, batch_size=self.batch_size)

        self._rewards_values = train_dataset.rewards_categories()
        self._number_of_reward_categories = train_dataset.number_of_rewards_categories()
        self._type_of_rewards = train_dataset.type_of_rewards()

    @property
    def bow_size(self):
        return self.train.dataset.data['bow'].shape[1]

    @property
    def rewards_values(self):
        if self._type_of_rewards == "discrete":
            return self._rewards_values
        else:
            return None

    @property
    def number_of_reward_categories(self):
        if self._type_of_rewards == "discrete":
            return self._number_of_reward_categories
        else:
            return np.inf

    @property
    def type_of_rewards(self):
        return self._type_of_rewards

    @property
    def word_embeddings_dim(self):
        return self.train.dataset.vocab.vectors.shape[1]

    @property
    def word_emb_type(self):
        return self.train.dataset.word_emb_type


class DynamicTopicLanguageDataloader(ADataLoader):
    """
    first we create and store the examples and text fields
    here, then, the dataset is divided in prediction and training
    prediction is only in the future (respects causality)

    the training is then divided in train and validation, which is a random split
    """

    def __init__(self, device, rank: int = 0, world_size=-1, **kwargs):
        super().__init__(device, rank, world_size, **kwargs)
        data_dir = kwargs.get("path_to_data")

        train_dataset = DynamicTopicLanguageDataset(data_dir, "train")
        valid_dataset = DynamicTopicLanguageDataset(data_dir, "validation")
        test_dataset = DynamicTopicLanguageDataset(data_dir, "test")
        predict_dataset = DynamicTopicLanguageDataset(data_dir, "prediction")

        train_sampler = None
        valid_sampler = None
        test_sampler = None
        predict_sampler = None
        if self.world_size != -1:
            train_sampler = DistributedSampler(train_dataset, self.world_size, self.rank)
            valid_sampler = DistributedSampler(valid_dataset, self.world_size, self.rank)
            test_sampler = DistributedSampler(test_dataset, self.world_size, self.rank)
            predict_sampler = DistributedSampler(predict_dataset, self.world_size, self.rank)

        self._train_iter = DataLoader(train_dataset, drop_last=True, sampler=train_sampler, shuffle=train_sampler is None, batch_size=self.batch_size)
        self._valid_iter = DataLoader(valid_dataset, drop_last=True, sampler=valid_sampler, shuffle=valid_sampler is None, batch_size=self.batch_size)
        self._test_iter = DataLoader(test_dataset, drop_last=True, sampler=test_sampler, shuffle=valid_sampler is None, batch_size=self.batch_size)
        self._predict_iter = DataLoader(predict_dataset, drop_last=True, sampler=predict_sampler, shuffle=predict_sampler is None, batch_size=self.batch_size)

    @property
    def corpora(self):
        return self.train.dataset.corpus_per_year_avg

    @property
    def bow_size(self):
        return self.train.dataset.data['bow'].shape[1]


class DataLoaderPTB(ADataLoader):
    def __init__(self, device, rank: int = 0, world_size=-1, **kwargs):
        path_to_data = kwargs.pop('path_to_data')
        super().__init__(device, rank, world_size, **kwargs)
        path_to_vectors = kwargs.pop('path_to_vectors')
        emb_dim = kwargs.pop('emb_dim')
        voc_size = kwargs.pop('voc_size')
        min_freq = kwargs.pop('min_freq')
        min_len = kwargs.pop('min_len')
        fix_len = kwargs.pop('fix_len')
        train_dataset, test_dataset, valid_dataset = PennTreebank(root=path_to_data, tokenizer=tokenizer,
                                                                  path_to_vectors=path_to_vectors,
                                                                  emb_dim=emb_dim,
                                                                  voc_size=voc_size,
                                                                  min_freq=min_freq,
                                                                  fix_len=fix_len,
                                                                  min_len=min_len)

        train_sampler = None
        valid_sampler = None
        test_sampler = None
        if self.world_size != -1:
            train_sampler = DistributedSampler(train_dataset, self.world_size, self.rank)
            valid_sampler = DistributedSampler(valid_dataset, self.world_size, self.rank)
            test_sampler = DistributedSampler(test_dataset, self.world_size, self.rank)

        self._train_iter = DataLoader(train_dataset, sampler=train_sampler, shuffle=train_sampler is None, **kwargs)
        self._valid_iter = DataLoader(valid_dataset, sampler=valid_sampler, shuffle=valid_sampler is None, **kwargs)
        self._test_iter = DataLoader(test_dataset, sampler=test_sampler, shuffle=test_sampler is None, **kwargs)
        vocab = train_dataset.get_vocab()
        self.train_vocab = vocab
        self._fix_length = fix_len

    @property
    def train(self):
        return self._train_iter

    @property
    def test(self):
        return self._test_iter

    @property
    def validate(self):
        return self._valid_iter

    @property
    def vocab(self):
        return self.train_vocab

    @property
    def max_sent_len(self):
        return self._fix_length


if __name__ == "__main__":
    import os
    from deep_fields import test_data_path

    kwargs = {
        "path_to_data": os.path.join(test_data_path, "preprocessed", "yelp", "language-transformer"),
        "batch_size": 32
    }
    dl = TopicTransformerLanguageDataloader('cpu', **kwargs)
    batch = next(iter(dl.train))
    assert batch.text['input_ids'].size() == (32, 1, 512)
    assert batch.bow.size() == (32, 46)
    assert batch.reward.size() == (32,)
    batch = next(iter(dl.validate))
    assert batch.text['input_ids'].size() == (32, 1, 512)
    assert batch.bow.size() == (32, 46)
    assert batch.reward.size() == (32,)
    batch = next(iter(dl.test))
    assert batch.text['input_ids'].size() == (32, 1, 512)
    assert batch.bow_h1.size() == (32, 46)
    assert batch.bow_h2.size() == (32, 46)
    assert batch.reward.size() == (32,)
