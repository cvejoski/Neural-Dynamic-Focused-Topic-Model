from collections import defaultdict
from email.policy import default
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
    results = defaultdict(list)
    for model_dir in glob.iglob(os.path.join("C:\\Projects\\CommonWork\\deep_random_fields\\results\\nips-perrone-1999\\neural_dynamical_topic_embeddings\\", 'nips*')):

        split = os.path.split(model_dir)[1].split('-')[-1]
        experiment_name = "-".join(os.path.split(model_dir)[1].split('-')[:-2])
        model_path = glob.glob(os.path.join(model_dir, '**', '*.p'), recursive=True)[0]
        logging.info(f"****** Experiment Name {experiment_name} ***************")
        state = torch.load(model_path, map_location='cuda:0')
        model_name = state['model_name']
        del state
        print("Split " + split)
        data_p = os.path.join(data_path, 'preprocessed', 'nips-perrone', 'language', split)
        results['model'].append(experiment_name)
        results['split'].append(split)
        dataloader_params = {"path_to_data": data_p, 'batch_size': 200, "is_dynamic": True, 'n_workers': 0}
        data_loader = TopicDataloader('cuda:0', **dataloader_params)
        model_dir = os.path.split(model_path)[0]
        model = ModelFactory.get_instance(model_type=model_name, **{'model_dir': model_dir, 'data_loader': data_loader})

        model.inference_parameters['cuda'] = 0
        set_cuda(model, **model.inference_parameters)

        model.eval()
        model.to(model.device)
        model.move_to_device()

        with torch.no_grad():

            logging.info(f"Evaluating the model stored at {model_path}")
            logging.info("saving beta ...")
            # beta = model.get_beta_eval().detach().cpu().numpy()
            # pickle.dump(beta, open(os.path.join(model_dir, 'beta.pkl'), 'bw'))

            logging.info("saving theta and b ...")

            # theta, b = model.get_time_series(data_loader.train)
            # theta_per_doc = model.get_topic_entropy_per_document(data_loader.train)
            # pickle.dump(theta.detach().cpu().numpy(), open(os.path.join(model_dir, 'theta.pkl'), 'bw'))
            # pickle.dump(b.detach().cpu().numpy(), open(os.path.join(model_dir, 'b.pkl'), 'bw'))
            # pickle.dump(theta_per_doc, open(os.path.join(model_dir, 'theta_per_doc.pkl'), 'bw'))
            logging.info("=================== Test  ============================")
            ppl = eval_dataset(data_loader.test, model)
            results['ppl'].append(ppl)
            top_words = model.top_words(data_loader, 30)
            # for ix, value in enumerate(top_words):
            #     results[f'Topic {ix}'].append(", ".join(value))
            for key, value in top_words.items():
                results[key].append(", ".join(value))
            logging.info(f"Perplexity {ppl}")
            logging.info("======================================================")

    results = pd.DataFrame(results)
    results.to_csv('perrone_results_dte.csv', index=False)


def eval_dataset(data_loader, ndt):
    ndt.eval()
    with torch.no_grad():
        nll = 0
        size = 0
        try:
            for data in tqdm(data_loader):
                forward_results = ndt.forward(data)
                n, t = ndt.loss_perrone(data, forward_results)
                nll += n
                size += t

        except Exception as e:
            print(e)

    return np.exp(nll/size)


if __name__ == '__main__':
    main()
