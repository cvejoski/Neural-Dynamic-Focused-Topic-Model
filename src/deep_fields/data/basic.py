import torch
from deep_fields.data.utils import divide_data

class Dataset(torch.utils.data.Dataset):
    # Characterizes a dataset for PyTorch'
    def __init__(self, X, labels=None):
        # Initialization
        self.labels = labels
        self.X = X

    def __len__(self):
        # Denotes the total number of samples
        return self.X.shape[0]

    def __getitem__(self, index):
        # Generates one sample of data
        # Select sample
        if self.labels is None:
            return self.X[index]
        else:
            return self.X[index], self.y[index]

def obtain_dataloaders(X,**kwargs):
    train_, test_, validation_ = divide_data(X, train_p=0.8, test_p=0.1)

    train_ = Dataset(train_)
    test_ = Dataset(test_)
    validation_ = Dataset(validation_)

    training_generator = torch.utils.data.DataLoader(train_, **kwargs)
    test_generator = torch.utils.data.DataLoader(test_, **kwargs)
    val_generator = torch.utils.data.DataLoader(validation_, **kwargs)

    return training_generator, test_generator, val_generator


if __name__=="__main__":

    from deep_fields.models.random_fields.poisson_covariance import PoissonCovariance
    from torch.distributions import MultivariateNormal

    model_param = PoissonCovariance.get_parameters()
    inference_param = PoissonCovariance.get_inference_parameters()
    inference_param.update({"nmc": 1000})
    pc_s = PoissonCovariance(None, None, None, **model_param)
    data_loader = pc_s.sample()
    Covariance = data_loader["K"].detach()
    gaussian = MultivariateNormal(torch.zeros(Covariance.shape[0]), Covariance)
    gaussian_sample = gaussian.sample(sample_shape=(100,))
    dimension = gaussian_sample.shape[1]

    params = {'batch_size': 8,
              'shuffle': True,
              'num_workers': 6}

    training_generator, test_generator, val_generator = obtain_dataloaders(gaussian_sample,**params)

    batch = next(training_generator.__iter__())
    print(batch)