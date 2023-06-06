import os

import numpy as np
import torch

from deep_fields import data_path
from deep_fields.data.topic_models.dataloaders import TopicDataloader
from deep_fields.models.topic_models.dynamic import DynamicBinaryFocusedTopic

debug = False

if __name__ == "__main__":
    for number_of_topics in [50, 100]:
        for rec_layers in [[800]]:
            for topic_layers_dim in [400]:  # [400, 200, 100]:
                for lr in [0.001]:
                    for bs in [128]:
                        for emb_dim in [64]:
                            for dynamic_topics in ["static"]:
                                for diversity in [0.1]:
                                    for alpha0 in [0.1, 1.0]:
                                        for i in range(87, 89):
                                            data_dir = os.path.join(data_path, "preprocessed", "acl", "language", str(i))

                                            dataloader_params = {"path_to_data": data_dir, "batch_size": bs, "is_dynamic": True, "n_workers": 0}

                                            model_parameters = DynamicBinaryFocusedTopic.get_parameters()
                                            model_inference_parameters = DynamicBinaryFocusedTopic.get_inference_parameters()

                                            model_inference_parameters.update({"learning_rate": lr,
                                                                               "cuda": 0,
                                                                               "clip_norm": 2.0,
                                                                               'number_of_epochs': 500,
                                                                               'metrics_log': 10,
                                                                               "min_lr_rate": 1e-12,
                                                                               "anneal_lr_after_epoch": 10,
                                                                               "gumbel": None,
                                                                               "debug": False,
                                                                               "reduced_num_batches": 2,
                                                                               })
                                            model_inference_parameters.get("lr_scheduler").update({"counter": 5})

                                            model_parameters.update({
                                                "word_embeddings_dim": 300,
                                                "lambda_diversity": diversity,
                                                "train_word_embeddings": False,
                                                "number_of_topics": number_of_topics,
                                                "topic_embeddings": dynamic_topics,
                                                "nonlinear_transition_prior": False,
                                                "pi0": alpha0
                                            })

                                            experiment_name = f'acl-{dynamic_topics}-linear-transition-n-topics-{number_of_topics}-pi-{alpha0}-lr-{lr}-split-{i}'

                                            model_parameters.update({
                                                'experiment_name': experiment_name,
                                                # 'model_path': '/raid/Results/Topic/UN/dynamic_binary_focused_topic_model-v1/'
                                            })

                                            model_parameters.get("theta_q_parameters").update({
                                                "layers_dim": rec_layers,
                                                "output_dim": rec_layers[-1],
                                                "dropout": 0.3,
                                                "hidden_state_dim": emb_dim
                                            })

                                            model_parameters.get("b_q_parameters").update({
                                                "layers_dim": rec_layers,
                                                "output_dim": rec_layers[-1],
                                                "dropout": 0.3
                                            })
                                            model_parameters.get("ksi_2_pi_transform").update({
                                                "layers_dim": [512, 512],
                                                "output_dim": rec_layers[-1],
                                                "dropout": 0.3
                                            })
                                            model_parameters.get("eta_2_zeta_transform").update({
                                                "layers_dim": [512, 512],
                                                "output_dim": rec_layers[-1],
                                                "dropout": 0.3
                                            })

                                            model_parameters.get("eta_q_parameters").update({
                                                "hidden_state_transition_dim": topic_layers_dim,
                                                "layers_dim": 400,
                                                "num_rnn_layers": 1,
                                                "dropout": 0.5,
                                                "out_dropout": 0.1,
                                                "is_bidirectional": True,
                                                "hidden_state_dim": emb_dim
                                            })

                                            model_parameters.get("ksi_q_parameters").update({
                                                "hidden_state_transition_dim": topic_layers_dim,
                                                "layers_dim": 400,
                                                "num_rnn_layers": 1,
                                                "dropout": 0.5,
                                                "out_dropout": 0.1,
                                                "is_bidirectional": True,
                                                "hidden_state_dim": emb_dim
                                            })

                                            model_parameters.get("ksi_prior_transition").update({
                                                "layers_dim": [512, 512],
                                                "output_transformation": None,
                                                "dropout": 0.3
                                            })
                                            model_parameters.get("eta_prior_transition").update({
                                                "layers_dim": [512, 512],
                                                "output_transformation": None,
                                                "dropout": 0.3
                                            })

                                            if os.path.exists(
                                                    os.path.join(model_parameters["model_path"], DynamicBinaryFocusedTopic.name_, experiment_name)):
                                                print(f"Skipping {experiment_name}")
                                                continue
                                            data_loader = TopicDataloader('cpu', **dataloader_params)

                                            try:
                                                np.random.seed(2021)
                                                torch.backends.cudnn.deterministic = True
                                                torch.manual_seed(2021)
                                                model = DynamicBinaryFocusedTopic(data_loader=data_loader, **model_parameters)
                                                print('\nDynamicBinaryTopicEmbeddings architecture: {}'.format(model))
                                                model.inference(data_loader, **model_inference_parameters)
                                            except Exception as e:
                                                print(e)
                                            # np.random.seed(2021)
                                            # torch.backends.cudnn.deterministic = True
                                            # torch.manual_seed(2021)
                                            # model = DynamicBinaryFocusedTopic(data_loader=data_loader, **model_parameters)
                                            # print('\nDETM architecture: {}'.format(model))
                                            # model.inference(data_loader, **model_inference_parameters)
