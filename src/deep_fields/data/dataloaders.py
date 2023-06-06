from abc import ABC, abstractmethod

import torch
import torchtext
from torchtext.data import BucketIterator
from torchtext.data import Field, Example


class ADataLoader(ABC):
    def __init__(self, device, rank: int = 0, world_size: int = -1, **kwargs):
        self.device = device
        self.batch_size = kwargs.pop('batch_size')
        self.path_to_vectors = kwargs.pop('path_to_vectors', None)
        self.emb_dim = kwargs.pop('emb_dim', None)
        self.voc_size = kwargs.pop('voc_size', None)
        self.min_freq = kwargs.pop('min_freq', 1)
        self._fix_length = kwargs.pop('fix_len', None)
        self.min_len = kwargs.pop('min_len', None)
        self.max_len = kwargs.pop('max_len', None)
        self.lower = kwargs.pop('lower', False)
        self.punctuation = kwargs.pop('punctuation', True)
        self.dataset_kwargs = kwargs
        self.world_size = world_size
        self.rank = rank

    @property
    @abstractmethod
    def train(self): ...

    @property
    @abstractmethod
    def validate(self): ...

    @property
    @abstractmethod
    def test(self): ...

    @property
    def n_train_batches(self):
        return len(self.train) // abs(self.world_size)

    @property
    def n_validate_batches(self):
        return len(self.validate) // abs(self.world_size)

    @property
    def n_test_batches(self):
        return len(self.test) // abs(self.world_size)

    @property
    def train_set_size(self):
        return len(self.train.dataset)

    @property
    def validation_set_size(self):
        return len(self.validate.dataset)

    @property
    def test_set_size(self):
        return len(self.test.dataset)


def iterators_from_allocations(current_z, ALLOCATION_FIELD=None, **kwargs):
    """
    here we create an interator for the allocatios, due to problems regarding the BBPT requiering
    just one example, as well as the necesity for a test set  defined with the future part of the
    time series we create our own BBT from the buckets.
    """
    batch_size = kwargs.get("batch_size")
    current_z = [{"text": z} for z in current_z]
    # we make sure that we reuse the same field (same mapping), since we don't want to confuse
    # the LSTM that we are training everytime
    new_field = False
    if ALLOCATION_FIELD is None:
        ALLOCATION_FIELD = Field(sequential=True)
        new_field = True
    examples = [Example.fromdict(z, {"text": ("text", ALLOCATION_FIELD)}) for z in current_z]
    allocation_data_set = torchtext.data.Dataset(examples, {"text": ALLOCATION_FIELD})
    if new_field:
        ALLOCATION_FIELD.build_vocab(allocation_data_set)
    train_iter, = BucketIterator.splits((allocation_data_set,),  # we pass in the datasets we want the iterator to draw data from
                                        batch_size=batch_size,
                                        device=torch.device("cpu"),  # if you want to use the GPU, specify the GPU number here
                                        sort_key=lambda x: len(x),  # the BucketIterator needs to be told what function it should use to group the data.
                                        sort_within_batch=False,
                                        repeat=False)  # we pass repeat=False because we want to wrap this Iterator layer.)
    return train_iter, ALLOCATION_FIELD

class allocation_dataloader():

    def __init__(self, current_z, LLA, **kwargs):
        """
        created for latent lstm allocation
        """
        # parameters
        self.batch_size = kwargs.get("batch_size")
        self.btt = kwargs.get("bbt")
        self.number_topics = LLA.number_of_topics
        self.train_percentage = kwargs.get("train_percentage")
        # field
        self.ALLOCATION_FIELD = Field(sequential=True)
        # data set
        current_z = [{"text": z} for z in current_z]
        examples = [Example.fromdict(z, {"text": ("text", self.ALLOCATION_FIELD)}) for z in current_z]
        self.allocation_data_set = torchtext.data.Dataset(examples, {"text": self.ALLOCATION_FIELD})

        self.ALLOCATION_FIELD.build_vocab(self.allocation_data_set)

        # iterator
        self.train_iter, = BucketIterator.splits((self.allocation_data_set,),
                                                 batch_size=self.batch_size,
                                                 device=torch.device("cpu"),
                                                 sort_key=lambda x: len(x),
                                                 sort_within_batch=False,
                                                 repeat=False)  # we pass repeat=False because we want to wrap this Iterator layer.

    def update_dataset(self, j_document, i_word):
        return None

    def train_and_validation(self):
        """
        train_percentage: float (percentage of training slice)

        here we split the data set such that the fisrt part corresponds to beggining in time of the time arrow
        and the end validation split is at the end of the arrow, so we validate in the future
        """
        for batch, _ in self.train_iter.__iter__():
            training_size = int(batch.shape[0] * self.train_percentage)
            training_batch = batch[:training_size, :]
            validation_batch = batch[training_size:, :]

            training_input, training_output = training_batch[:-1, :], training_batch[1:, :]
            validation_input, validation_ouput = validation_batch[:-1, :], validation_batch[1:, :]
            yield (training_input, training_output, validation_input, validation_ouput)

# examples = [Example.fromdict(current_z[0], {"text": ("text", ALLOCATION_FIELD)})]
# train_iter, = BPTTIterator.splits((allocation_data_set,),
#                                      batch_size=4,
#                                      bptt_len=3,  # this is where we specify the sequence length
#                                      device=torch.device("cpu"),
#                                      repeat=False)
