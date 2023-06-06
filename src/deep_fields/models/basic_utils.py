import os
from importlib import import_module
from typing import List

import torch
from torch.utils.tensorboard import SummaryWriter
import numpy as np

def all_metrics_to_floats(all_metrics):
    all_metrics_new = {}
    for metric_name, metric_value in all_metrics.items():
        if type(metric_value) == torch.Tensor:
            all_metrics_new[metric_name] = metric_value.item()
        else:
            all_metrics_new[metric_name] = metric_value
    return all_metrics_new


def set_debugging_scheme(model, inference_parameters, inference_variables):
    model.debug = inference_parameters.get("debug").get("debug")
    if model.debug and inference_parameters.get("true_data") is not None:
        inference_variables["debug_stats"] = {}
        inference_variables["true_data"] = inference_parameters.get("true_data")
        print("Debug Mode")
        for parameter, value in inference_parameters.get("debug").items():
            if parameter != "debug":
                if value:
                    inference_variables["debug_stats"][parameter] = []
    return inference_variables


def define_model_paths(model, results_path, model_name):
    model.class_dir = os.path.join(results_path, model_name)
    model.model_dir = os.path.join(results_path, model_name, model.model_identifier)
    if not os.path.isdir(model.class_dir):
        os.mkdir(model.class_dir)
    if not os.path.isdir(model.model_dir):
        os.mkdir(model.model_dir)

    model.log_dir = os.path.join(results_path, model_name, model.model_identifier, "tensorboard_log")
    model.parameter_path = os.path.join(results_path, model_name, model.model_identifier, "parameters.json")
    model.best_model_path = os.path.join(results_path, model_name, model.model_identifier, "best_model.p")
    model.writer = SummaryWriter(model.log_dir)


def generate_training_message():
    print("#------------------------------")
    print("# Start of Training")
    print("#------------------------------")


def set_cuda(object, **inference_parameters):
    object.cuda = inference_parameters.get("cuda")
    # check https://discuss.pytorch.org/t/how-to-load-all-data-into-gpu-for-training/27609/7
    if object.cuda is not None:
        if torch.cuda.is_available():
            object.device = torch.device(object.cuda)
        else:
            print("Cuda Not Available")
            object.device = torch.device("cpu")
    else:
        object.device = torch.device("cpu")


def dict_mean(dict_list: List[dict]) -> dict:
    mean_dict = {}
    for key in dict_list[0].keys():
        mean_dict[key] = float(sum(d[key] for d in dict_list)) / len(dict_list)
    return mean_dict


def create_class_instance(module_name, class_name, kwargs, *args):
    """Create an instance of a given class.

    :param module_name: where the class is located
    :param class_name:
    :param kwargs: arguments needed for the class constructor
    :returns: instance of 'class_name'

    """
    module = import_module(module_name)
    clazz = getattr(module, class_name)
    if kwargs is None:
        instance = clazz(*args)
    else:
        instance = clazz(*args, **kwargs)

    return instance


def create_instance(params, *args):
    """Creates an instance of class given configuration.

    :param name: of the module we want to create
    :param params: dictionary containing information how to instantiate the class
    :returns: instance of a class
    :rtype:

    """
    if type(params) is list:
        instance = [create_class_instance(
            p['module'], p['name'], p['args'], *args) for p in params]
    else:
        instance = create_class_instance(
            params['module'], params['name'], params['args'], *args)
    return instance

def nearest_neighbors(word, embeddings, vocab, num_words):
    vectors = embeddings.cpu().numpy()
    index = vocab.index(word)
    query = embeddings[index].cpu().numpy()
    ranks = vectors.dot(query).squeeze()
    denom = query.T.dot(query).squeeze()
    denom = denom * np.sum(vectors ** 2, 1)
    denom = np.sqrt(denom)
    ranks = ranks / denom
    mostSimilar = []
    [mostSimilar.append(idx) for idx in ranks.argsort()[::-1]]
    nearest_neighbors = mostSimilar[:num_words]
    nearest_neighbors = [vocab[comp] for comp in nearest_neighbors]
    return nearest_neighbors
