import os

import numpy as np
import torch

from deep_fields import data_path
from deep_fields.data.topic_models.dataloaders import TopicDataloader
from deep_fields.models.topic_models.dynamic import DynamicLDA

debug = False

training = True

if __name__ == "__main__":
    for number_of_topics in [100]:
        for rec_layers in [[800]]:
            for topic_layers_dim in [400]:  # [400, 200, 100]:
                for lr in [0.001]:
                    for bs in [200]:
                        for i in range(87, 89):
                            data_dir = os.path.join(data_path, "preprocessed", "acl", "language", str(i))
                            dataloader_params = {"path_to_data": data_dir,
                                                 "batch_size": bs,
                                                 "is_dynamic": True,
                                                 "n_workers": 0}

                            data_loader = TopicDataloader('cpu', **dataloader_params)
                            ndt_parameters = DynamicLDA.get_parameters()
                            ndt_inference_parameters = DynamicLDA.get_inference_parameters()
                            ndt_parameters.update({"word_embeddings_dim": 300})

                            ndt_inference_parameters.update({
                                "learning_rate": lr,
                                "cuda": 0,
                                "clip_norm": 2.0,
                                'number_of_epochs': 400,
                                'metrics_log': 1,
                                "tau": 0.75,
                                "min_lr_rate": 1e-12,
                                "anneal_lr_after_epoch": 50,
                                "gumbel": None})
                            ndt_inference_parameters.get("lr_scheduler").update({"counter": 5})

                            ndt_parameters.update({"number_of_topics": number_of_topics})
                            ndt_parameters.update({'experiment_name': f'acl-split-{i}-number-topics-{number_of_topics}'})

                            ndt_parameters.get("theta_q_parameters").update({"layers_dim": rec_layers})
                            ndt_parameters.get("theta_q_parameters").update({"output_dim": rec_layers[-1]})
                            ndt_parameters.get("theta_q_parameters").update({"num_rnn_layers": 4})
                            ndt_parameters.get("theta_q_parameters").update({"dropout": 0.0})

                            ndt_parameters.get("eta_q_parameters").update({"hidden_state_transition_dim": topic_layers_dim})
                            ndt_parameters.get("eta_q_parameters").update({"layers_dim": 400})
                            ndt_parameters.get("eta_q_parameters").update({"num_rnn_layers": 4})
                            ndt_parameters.get("eta_q_parameters").update({"dropout": 0.0})

                            # try:
                            #     np.random.seed(2021)
                            #     torch.backends.cudnn.deterministic = True
                            #     torch.manual_seed(2021)
                            #     model = DynamicLDA(data_loader=data_loader, **ndt_parameters)
                            #     print('\nD-LDA architecture: {}'.format(model))
                            #     model.inference(data_loader, **ndt_inference_parameters)
                            # except Exception as e:
                            #     print(e)
                            np.random.seed(2019)
                            torch.backends.cudnn.deterministic = True
                            torch.manual_seed(2019)
                            model = DynamicLDA(data_loader=data_loader, **ndt_parameters)
                            print('\nDETM architecture: {}'.format(model))
                            model.inference(data_loader, **ndt_inference_parameters)
