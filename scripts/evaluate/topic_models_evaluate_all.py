from collections import defaultdict
import glob
import logging
import os
import sys
import pickle
import numpy as np
import torch
from tqdm import tqdm
import pandas as pd
from deep_fields import data_path
from deep_fields.data.topic_models.dataloaders import TopicDataloader
from deep_fields.models.basic_utils import set_cuda
from deep_fields.models.topic_models import ModelFactory

logging.basicConfig(stream=sys.stdout, level=logging.DEBUG)


def main():
    results = defaultdict(lambda: defaultdict(list))
    count = 0
    for model_dir in glob.iglob(os.path.join("C:\\Projects\\CommonWork\\deep_random_fields\\results\\neural_dynamical_topic_embeddings", 'un*')):
        try:

            split = os.path.split(model_dir)[1].split('-')[-1]
            experiment_name = "-".join(os.path.split(model_dir)[1].split('-')[:-2])
            model_path = glob.glob(os.path.join(model_dir, '**', '*.p'), recursive=True)[0]

            state = torch.load(model_path, map_location='cuda:0')
            model_name = state['model_name']
            del state
            print("Split " + split)
            data_p = os.path.join(data_path, 'preprocessed', 'un', 'language', split)
            # data_p = os.path.join(data_path, 'preprocessed', 'acl', 'language')
            dataloader_params = {"path_to_data": data_p, 'batch_size': 200, "is_dynamic": True, 'n_workers': 0}
            data_loader = TopicDataloader('cuda:0', **dataloader_params)
            model_dir = os.path.split(model_path)[0]
            model = ModelFactory.get_instance(model_type=model_name, **{'model_dir': model_dir, 'data_loader': data_loader})

            model.inference_parameters['cuda'] = 0
            set_cuda(model, **model.inference_parameters)

            model.eval()
            model.to(model.device)
            model.move_to_device()
        except Exception as e:
            print(e)
            continue

        with torch.no_grad():

            logging.info("saving beta ...")
            beta = model.get_beta_eval().detach().cpu().numpy()
            pickle.dump(beta, open(os.path.join(model_dir, 'beta.pkl'), 'bw'))

            logging.info("saving theta and b ...")

            # theta, b = model.get_time_series(data_loader.train)
            theta_per_doc = model.get_topic_entropy_per_document(data_loader.train)
            # pickle.dump(theta.detach().cpu().numpy(), open(os.path.join(model_dir, 'theta.pkl'), 'bw'))
            # pickle.dump(b.detach().cpu().numpy(), open(os.path.join(model_dir, 'b.pkl'), 'bw'))
            pickle.dump(theta_per_doc, open(os.path.join(model_dir, 'theta_per_doc.pkl'), 'bw'))
            # pickle.dump(b_per_doc.detach().cpu().numpy(), open(os.path.join(model_path, 'b_per_doc.pkl'), 'bw'))

        #     # print("saving alpha ...")
        #     # mu_alpha = model.mu_q_alpha.detach().numpy()
        #     # logvar_alpha = model.logvar_q_alpha.detach().numpy()
        #     # pickle.dump((mu_alpha, logvar_alpha), open(os.path.join(model_path, 'topic-embeddings.pkl'), 'bw'))
        #
        #     logging.info(f"Evaluating the model stored at {model_path}")
        #     logging.info("======================================================")
        #     # print("=================== Train ============================")
        #     # eval_dataset(data_loader.train, model)
        #
        #     logging.info("=================== Valid  ============================")
        #     # eval_dataset(data_loader.validate, model)
        #     logging.info("======================================================")
        #
    #     logging.info("=================== Test  ============================")
    #     ppl = eval_dataset(data_loader.test, model)
    #     # PPL.append(ppl)
    #     results[experiment_name]['ppl'].append(ppl)
    #     logging.info("======================================================")
    #     try:
    #         predictive_ppl = model.prediction(data_loader)
    #         print(f"Negative Log Likelihood: {predictive_ppl}")
    #         results[experiment_name]['p_ppl'].append(predictive_ppl)
    #     except Exception as e:
    #         print(e)

    #     td = model.topic_diversity()
    #     logging.info(f"Topic Diversity: {td}")
    #     # TD.append(td)
    #     results[experiment_name]['TD'].append(td)
    #     tc = model.topic_coherence(data_loader.train.dataset.data['bow'])
    #     results[experiment_name]['TC'].append(tc)
    #     logging.info(f"Topic Coherence: {tc}")
    #     logging.info(f"Topic Quality: {td * tc}")
    #     results[experiment_name]['TC'].append(td * tc)
    #     count += 1

    # results_agg = []
    # for key, value in results.items():
    #     results_agg.append({'model_name': key,
    #                         'ppl_m': np.mean(value['ppl']),
    #                         'ppl_s': np.std(value['ppl']),
    #                         'p_ppl_m': np.mean(value['p_ppl']),
    #                         'p_ppl_s': np.std(value['p_ppl']),
    #                         'td_m': np.mean(value['TD']),
    #                         'td_s': np.std(value['TD']),
    #                         'tc_m': np.mean(value['TC']),
    #                         'tc_s': np.std(value['TC']),
    #                         'tq_m': np.mean(value['TQ']),
    #                         'tq_s': np.std(value['TQ'])})

    # results = pd.DataFrame(results_agg)
    # results.to_latex('acl-lda.tex')


def eval_dataset(data_loader, ndt):
    ndt.eval()
    with torch.no_grad():
        ppl = []
        try:
            for data in tqdm(data_loader):
                forward_results = ndt.forward(data)
                loss_stats = ndt.loss(data, forward_results, data_loader, 0)

                ppl.append(loss_stats['Log-Likelihood'].item())
        except Exception as e:
            print(e)

        # print(f"NLL-Loss: {loss_stats['NLL-Loss']}")
        logging.info(f"PPL: {np.exp(np.mean(ppl))}")
    return np.exp(np.mean(ppl))


if __name__ == '__main__':
    main()
