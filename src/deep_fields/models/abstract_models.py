from asyncio.constants import LOG_THRESHOLD_FOR_CONNLOST_WRITES
import json
import os
import sys
import time
from datetime import datetime
from typing import Dict, Union

import numpy as np
import torch
from deep_fields.data.topic_models.dataloaders import ADataLoader
from deep_fields.models.basic_utils import set_cuda, all_metrics_to_floats, dict_mean
from torch.optim import SGD, Adam
from torch.optim.adadelta import Adadelta
from torch.optim.adagrad import Adagrad
from torch.optim.asgd import ASGD
from torch.optim.rmsprop import RMSprop
from torch.utils.tensorboard import SummaryWriter
from tqdm import tqdm


class DeepBayesianModel(torch.nn.Module):

    def __init__(self, model_name, model_dir=None, data_loader: ADataLoader = None, metalearning=None, **kwargs):
        super(DeepBayesianModel, self).__init__()
        self.nan_counter: int = 0
        self.model_name = model_name
        self.device = torch.device("cpu")
        self.number_of_iterations = None
        self.number_of_iterations_val = None
        self.metalearning = metalearning

        # first check if results
        if model_dir is not None:
            print("Loading Model No Inference")
            self.model_path_from_results_folder(model_dir)
            kwargs = json.load(open(self.parameter_path, "r"))
            self.set_parameters(**kwargs)
            self.data_loader = data_loader
            self.define_deep_models()
            self.load_model()
            try:
                self.dataloader_parameters = json.load(open(self.dataloaders_parameters_path, "r"))
                self.inference_parameters = json.load(open(self.inference_parameters_path, "r"))
            except:
                pass
        else:
            print("New Model Set For Inference")
            # file handling
            if kwargs == {}:
                kwargs = self.get_parameters()
            self.model_path = kwargs.get("model_path")
            self.experiment_name = kwargs.get("experiment_name", "")
            self.model_identifier = str(int(time.time()))
            self.define_model_paths(self.model_path, self.experiment_name)
            self.set_parameters(**kwargs)
            if data_loader is not None:
                kwargs = self.update_parameters(data_loader, **kwargs)
                self.set_parameters(**kwargs)
                json.dump(kwargs, open(self.parameter_path, "w"))
                self.model_parameters = kwargs
                self.data_loader = data_loader
                self.define_deep_models()
                try:
                    json.dump(data_loader.dataset_kwargs, open(self.dataloaders_parameters_path, "w"))
                except:
                    print("Data Loader has No Parameters")
                    pass
                # json.dump(dataloader.inference_parameters, open(self.inference_parameters_path, "w"))
            else:
                try:
                    self.update_parameters(data_loader, **kwargs)
                except:
                    print(sys.exc_info())
                    pass
                self.set_parameters(**kwargs)
                json.dump(kwargs, open(self.parameter_path, "w"))
                self.define_deep_models()

    # ====================================
    # PARAMETERS HANDLE
    # ====================================
    @classmethod
    def get_parameters(cls):
        return None

    @classmethod
    def set_parameters(self, **kwargs):
        return None

    @classmethod
    def update_parameters(self, dataloader, **kwargs):
        return None

    @classmethod
    def get_inference_parameters(cls):
        inference_parametes = {"number_of_epochs": 250,
                               "metrics_log": 10,
                               "cuda": None,
                               "bbt": 3,
                               "learning_rate": .0001,
                               "model_eval": "loss",
                               "reduced_num_batches": 10,
                               "clip_norm": None,
                               "debug": False,
                               "min_lr_rate": None,
                               "anneal_lr_after_epoch": 50,
                               "aggregate_num_batches": 1,
                               "optimizer_name": "Adam",
                               "lr_scheduler": {
                                   "counter": 1000,
                                   "step_size": 1,
                                   "gamma": 0.25}
                               }

        return inference_parametes

    # ====================================
    # MODELS HANDLE
    # ====================================
    def define_deep_models(self):
        return None

    def init_parameters(self):
        return None

    # ====================================
    # INFERENCE HANDLE
    # ====================================
    def sample(self):
        return None

    def loss(self, databatch, forward_results, dataloader, epoch) -> Dict[str, torch.Tensor]:
        return {"loss": torch.tensor(0.0)}

    def metrics(self, databatch, forward_results, epoch, mode="evaluation", data_loader=None):
        return {}

    def forward(self, databatch):
        return None

    def inference_step(self, optimizer, data_loader: ADataLoader, epoch, inference_variables, **inference_parameters):
        self.train()
        bath_number = 1
        optimizer.zero_grad()
        for data in tqdm(data_loader.train, total=len(data_loader.train), unit='batch', desc='Training minibatch'):
            with torch.autograd.set_detect_anomaly(self.debug):
                self.initialize_steps(data)
                data = self.data_to_device(data)
                forward_results = self(data)
                losses = self.loss(data, forward_results, data_loader.train, epoch)
                metrics = self.metrics(data, forward_results, epoch, mode="train")
                loss = losses["loss"]
                # print(losses)
                if bath_number % self.aggregate_num_batches == 0:
                    loss.backward()
                    if self.clip_norm is not None:
                        torch.nn.utils.clip_grad_norm_(self.parameters(), max_norm=self.clip_norm)
                    optimizer.step()
                    optimizer.zero_grad()
                    self.detach_history()
                    # break
                else:
                    loss.backward()
                parameters_and_grad_stats = {}
                # if self.debug:
                # self.check_gradients_and_params()
                # parameters_and_grad_stats = self.check_gradients_and_params_stats()
                self.update_writer({**losses, **metrics, **parameters_and_grad_stats}, label="train")
                self.number_of_iterations += 1
                bath_number += 1
                if self.debug and (bath_number > self.reduced_num_batches):
                    break
                if torch.isinf(loss) or torch.isnan(loss):
                    if self.nan_counter >= 2000:
                        raise ValueError(f"Useless loss {loss.item()}!")
                    self.nan_counter += 1
            # break
        self.metrics(data, forward_results, epoch, mode="train_global", data_loader=data_loader)

    def _anneal_lr(self) -> None:
        if self.lr_scheduler['counter'] > 0:
            self.lr_scheduler['counter'] -= 1
        else:
            self.lr_scheduler['scheduler'].step()
            self.lr_scheduler['counter'] = self.lr_scheduler['default_counter']

    def __init_lr_scheduler(self, optimizer, **inference_parameters) -> Union[None, dict]:
        lr_scheduler_param = inference_parameters.get("lr_scheduler")
        step_size = lr_scheduler_param.get('step_size')
        gamma = lr_scheduler_param.get('gamma')
        counter = lr_scheduler_param.get('counter')
        if lr_scheduler_param is None:
            return None
        opt_scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size, gamma)
        lr_scheduler = {'counter': counter,
                        'default_counter': counter,
                        'scheduler': opt_scheduler}
        return lr_scheduler

    def _early_stop(self) -> bool:
        return self.min_lr_rate is not None and self.optimizer.param_groups[0]['lr'] < float(self.min_lr_rate)

    def initialize_steps(self, data):
        return None

    def detach_history(self):
        return None

    def data_to_device(self, data):
        return data

    def validation_step(self, optimizer, data_loader: ADataLoader, epoch, inference_variables, **inference_parameters):
        self.eval()
        with torch.no_grad():
            model_evaluation = []
            for data in tqdm(data_loader.validate, total=len(data_loader.validate), unit='batch', desc='Validation minibatch'):
                data = self.data_to_device(data)
                self.initialize_steps(data)
                forward_results = self(data)
                losses = self.loss(data, forward_results, data_loader.validate, epoch)
                metrics = self.metrics(data, forward_results, epoch, mode="validation", data_loader=data_loader)
                all_metrics = {**losses, **metrics}
                model_evaluation.append(all_metrics)

                self.number_of_iterations_val += 1

            all_metrics = dict_mean(model_evaluation)
            self.update_writer(all_metrics, label="validation")
            metric = self.metrics(data, forward_results, epoch, mode="validation_global", data_loader=data_loader)
            self.update_writer(metric, label="validation")
            return all_metrics

    def initialize_inference(self, data_loader, parameters=None, **inference_parameters):
        self.inference_results = {}
        self.learning_rate = inference_parameters.get("learning_rate", None)
        self.min_lr_rate = inference_parameters.get("min_lr_rate")
        self.number_of_epochs = inference_parameters.get("number_of_epochs", 10)
        self.anneal_lr_after_epoch = inference_parameters.get("anneal_lr_after_epoch")
        self.aggregate_num_batches = inference_parameters.get("aggregate_num_batches")
        self.is_loss_minimized = inference_parameters.get("is_loss_minimized", True)

        try:
            self.expected_num_steps = len(data_loader.train) * self.number_of_epochs
        except:
            self.expected_num_steps = self.number_of_epochs

        self.metrics_logs = inference_parameters.get("metrics_log", 10)
        self.time_0 = datetime.now()

        self.best_eval = np.inf
        self.best_eval_key = inference_parameters.get("model_eval", "loss")
        self.debug = inference_parameters.get("debug")
        self.reduced_num_batches = inference_parameters.get("reduced_num_batches")
        self.clip_norm = inference_parameters.get("clip_norm")

        set_cuda(self, **inference_parameters)
        self.to(self.device)
        self.number_of_iterations = 0
        self.number_of_iterations_val = 0
        json.dump(inference_parameters, open(self.inference_parameters_path, "w"))
        try:
            self.optimizer = self.__create_optimizer(inference_parameters.get('optimizer_name'),
                                                     self.learning_rate, inference_parameters.get('weight_decay', 1.2 * 1e-6), parameters=parameters)
            self.lr_scheduler = self.__init_lr_scheduler(self.optimizer, **inference_parameters)
        except:
            print(sys.exc_info())
            print("Optimizer Not Created In Abstract Initialize Inference")

    def __create_optimizer(self, optmimizer_name: str, lr: float, weight_decay: float, parameters=None):
        if parameters is None:
            parameters = self.parameters()
        if optmimizer_name.lower() == "adam":
            return Adam(parameters, lr=lr, weight_decay=weight_decay, eps=1e-8)
        elif optmimizer_name.lower() == "adadelta":
            return Adadelta(parameters, lr=lr, weight_decay=weight_decay)
        elif optmimizer_name.lower() == "adagrad":
            return Adagrad(parameters, lr=lr, weight_decay=weight_decay)
        elif optmimizer_name.lower() == "asgd":
            return ASGD(parameters, lr=lr, weight_decay=weight_decay)
        elif optmimizer_name.lower() == "rmsprop":
            return RMSprop(parameters, lr=lr, weight_decay=weight_decay)
        else:
            return SGD(parameters, lr=lr, weight_decay=weight_decay)

    def inference(self, data_loader: ADataLoader, **inference_parameters):
        if self.INFERENCE:
            self.generate_training_message()
            inference_variables = self.initialize_inference(data_loader, **inference_parameters)

            with open(self.inference_history_path, "a+") as fh:
                with open(self.inference_path, "a+") as f:
                    for epoch in tqdm(range(1, self.number_of_epochs + 1), desc="Train Epoch", unit='epoch'):
                        self.inference_step(self.optimizer, data_loader, epoch, inference_variables, **inference_parameters)
                        model_evaluation = self.validation_step(self.optimizer, data_loader, epoch, inference_variables, **inference_parameters)
                        model_evaluation = all_metrics_to_floats(model_evaluation)
                        current_eval = model_evaluation[self.best_eval_key]
                        if not self.is_loss_minimized:
                            current_eval *= -1
                        if current_eval < self.best_eval:
                            self.lr_scheduler['counter'] = self.lr_scheduler['default_counter']
                            self.best_eval = current_eval
                            self.time_f = datetime.now()
                            self.inference_results["epoch"] = epoch
                            self.inference_results["lr"] = self.optimizer.param_groups[0]['lr']
                            self.inference_results["best_eval_time"] = (self.time_f - self.time_0).total_seconds()
                            self.inference_results["best_eval_criteria"] = self.best_eval
                            self.inference_results.update(model_evaluation)
                            json.dump(self.inference_results, f)
                            f.write("\n")
                            f.flush()
                            self.save_model()
                        else:
                            if epoch > self.anneal_lr_after_epoch:
                                self._anneal_lr()
                        model_evaluation.update({'eval_time': (datetime.now() - self.time_0).total_seconds(),
                                                 'epoch': epoch, 'lr': self.optimizer.param_groups[0]['lr']})
                        json.dump(model_evaluation, fh)
                        fh.write("\n")
                        fh.flush()
                        if self._early_stop():
                            print("Early Stopping!")
                            break
                    # final_time = datetime.now()
                    # self.inference_results["final_time"] = (final_time - self.time_0).total_seconds()
                    # self.inference_results.update(model_evaluation)
                    # json.dump(self.inference_results, f)
                    final_time = datetime.now()
                    self.inference_results["final_time"] = (final_time - self.time_0).total_seconds()
                    return self.inference_results
        else:
            print("MODEL OPEN IN RESULTS FOLDER, INFERENCE WILL OVERRIDE OLD RESULTS")
            print("CREATE NEW MODEL")
            raise Exception

    def update_writer(self, metrics, label="train"):
        for metric_name, metric_value in metrics.items():
            if label == "train":
                self.writer.add_scalar(label + "/" + metric_name, metric_value, self.number_of_iterations)
            elif label == "validation":
                self.writer.add_scalar(label + "/" + metric_name, metric_value, self.number_of_iterations_val)
            else:
                print("Wrong label in writer update")
                raise Exception

    def save_model(self, end=False):
        state = {
            'model_name': self.model_name,
            'state_dict': self.state_dict()
        }
        if end:
            torch.save(state, self.best_model_path_end)
        else:
            torch.save(state, self.best_model_path)

    def load_model(self):
        state = torch.load(self.best_model_path, map_location=self.device)
        self.load_state_dict(state['state_dict'])

    def define_model_paths(self, results_path, experiment_name):
        self.model_dir = os.path.join(results_path, self.model_name, experiment_name, self.model_identifier)

        if not os.path.isdir(self.model_dir):
            os.makedirs(self.model_dir, exist_ok=True)

        self.log_dir = os.path.join(self.model_dir, "tensorboard_log")
        self.parameter_path = os.path.join(self.model_dir, "parameters.json")
        self.best_model_path = os.path.join(self.model_dir, "best_model.p")
        self.best_model_path_end = os.path.join(self.model_dir, "best_model_end.p")

        self.inference_path = os.path.join(self.model_dir, "inference_results.json")
        self.inference_history_path = os.path.join(self.model_dir, "inference_history.json")

        self.inference_parameters_path = os.path.join(self.model_dir, "inference_parameters.json")
        self.dataloaders_parameters_path = os.path.join(self.model_dir, "dataloaders_parameters.json")

        self.writer = SummaryWriter(self.log_dir)
        self.INFERENCE = True

    def model_path_from_results_folder(self, model_dir):
        self.model_dir = model_dir
        if os.path.isdir(self.model_dir):
            self.log_dir = os.path.join(self.model_dir, "tensorboard_log")
            self.parameter_path = os.path.join(self.model_dir, "parameters.json")
            self.inference_parameters_path = os.path.join(self.model_dir, "inference_parameters.json")
            self.dataloaders_parameters_path = os.path.join(self.model_dir, "dataloaders_parameters.json")

            if not os.path.isfile(self.parameter_path):
                print("NO PARAMETERS FILE IN RESULTS FOLDER")
                raise Exception
            self.best_model_path = os.path.join(self.model_dir, "best_model.p")
            self.best_model_path_end = os.path.join(self.model_dir, "best_model_end.p")

            if not os.path.isfile(self.best_model_path):
                print("NO BEST MODEL FILE IN RESULTS FOLDER")
                raise Exception
            self.INFERENCE = False
        else:
            print("RESULTS FOLDER NON EXISTING")
            raise Exception

    def generate_training_message(self):
        training__format = "# Start of Training {0}".format(self.model_name)
        print("#" + "-" * len(training__format))
        print(training__format)
        print("#" + "-" * len(training__format))

    def check_gradients_and_params(self):
        stop = False
        for j, A in enumerate(self.named_parameters()):
            name, param = A
            if param.requires_grad:
                if torch.isnan(param.data).sum() > 1.:
                    print(j)
                    print(name)
                    print("Param Nan")
                    stop = True
                if torch.isnan(param.grad).sum() > 1.:
                    print(j)
                    print(name)
                    print("Grad Param Nan")
                    stop = True
                if torch.isinf(param.data).sum() > 1.:
                    print(j)
                    print(name)
                    print("Param Inf")
                    stop = True
                if torch.isinf(param.grad).sum() > 1.:
                    print(j)
                    print(name)
                    print("Grad Param Inf")
                    stop = True
        if stop:
            raise Exception

    def check_gradients_and_params_stats(self):
        param_grad_stats = {}
        for j, A in enumerate(self.named_parameters()):
            name, param = A
            if param.requires_grad:
                param_mean = param.data.mean()
                param_grad = param.grad.mean()
                param_grad_stats[name + "_data"] = param_mean
                param_grad_stats[name + "_grad"] = param_grad
        return param_grad_stats
