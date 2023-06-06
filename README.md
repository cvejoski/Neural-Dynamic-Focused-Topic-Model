# Neural Dynamic Focused Topic Model

Code accompanying the paper **Neural Dynamic Focused Topic Model**.

## Installation of the library

In order to set up the necessary environment:

1. create an environment `drf` with the help of [conda],

   ```
   conda env create -f environment.yaml
   ```

2. activate the new environment with

   ```
   conda activate drf
   ```

3. install `drf` with:

   ```
   python setup.py install # or `develop`
   python setup.py install # or

## Data Preprocessing and preparation

All the data used should be in the folder `{project_location}/data`. The preprocessing of the **NIPS, ACL and UN* datasets is done using the script located in `{project_location}/scripts/preprocessing/dataset_preprocessor.py`.

## Training of models

The training of the models used in the paper is done by using the training scripts located in the `{project_location}/scripts/train/{acl|nips|un|nips_perrone}` folder.

## Evaluating of the trained models

For evaluation of the trained models we use the script `{project_location}/scripts/evaluate/topic_models_evaluate_all.py`
