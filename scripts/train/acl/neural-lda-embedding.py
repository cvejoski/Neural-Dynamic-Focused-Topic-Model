import os

import numpy as np
import torch
from deep_fields import data_path
from deep_fields.data.topic_models.dataloaders import TopicDataloader
from deep_fields.models.topic_models import DiscreteLatentTopicNVI

debug = False

data_dir = os.path.join(data_path, "preprocessed", "acl", "language")

training = True
if __name__ == "__main__":
    for number_of_topics in [50]:
        for rec_layers in [[512]]:
            for lr in [0.001]:
                for bs in [100]:
                    dataloader_params = {"path_to_data": data_dir,
                                         "batch_size": bs}

                    data_loader = TopicDataloader('cpu', **dataloader_params)
                    ndt_inference_parameters = DiscreteLatentTopicNVI.get_inference_parameters()

                    ndt_inference_parameters.get("lr_scheduler").update({"counter": 3})
                    ndt_inference_parameters.update({
                        "learning_rate": lr,
                        "cuda": 0,
                        "clip_norm": 2.0,
                        'number_of_epochs': 2000,
                        'metrics_log': 5,
                        "min_lr_rate": 1e-12,
                        "anneal_lr_after_epoch": 600,
                        "gumbel": None})

                    ndt_parameters = DiscreteLatentTopicNVI.get_parameters()
                    ndt_parameters.update({"word_embeddings_dim": 300})
                    ndt_parameters.update({"lambda_diversity": 0.1})
                    ndt_parameters.update({"no_topics": False})
                    ndt_parameters.update({"no_embeddings": False})
                    ndt_parameters.update({"number_of_topics": number_of_topics})

                    ndt_parameters.get("theta_q_parameters").update({"layers_dim": rec_layers})
                    ndt_parameters.get("theta_q_parameters").update({"output_dim": rec_layers[-1]})
                    ndt_parameters.get("theta_q_parameters").update({"dropout": 0.0})
                    ndt_parameters.get("theta_q_parameters").update({"out_dropout": 0.1})

                    ndt_parameters.update({'experiment_name': 'acl'})

                try:
                    np.random.seed(2021)
                    torch.backends.cudnn.deterministic = True
                    torch.manual_seed(2021)
                    model = DiscreteLatentTopicNVI(data_loader=data_loader, **ndt_parameters)
                    print('\nDLTNVI architecture: {}'.format(model))
                    model.inference(data_loader, **ndt_inference_parameters)
                except Exception as e:
                    print(e)

                # model = DiscreteLatentTopicNVI(data_loader=data_loader, **ndt_parameters)
                # print('\nDLTNVI architecture: {}'.format(model))
                # model.inference(data_loader, **ndt_inference_parameters)
