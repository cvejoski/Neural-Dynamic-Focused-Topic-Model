
import os
import time
import shutil
from deep_fields import models_path
from torch.utils.tensorboard import SummaryWriter

def create_dir_and_writer(model_name="particles_schrodinger",experiments_class="ou",model_identifier=None,delete=False):
    if model_identifier is None:
        model_identifier = str(int(time.time()))
    results_path = os.path.join(models_path, model_name, "{0}_{1}".format(experiments_class,model_identifier))
    if not os.path.isdir(results_path):
        os.makedirs(results_path)
    else:
        if delete:
            shutil.rmtree(results_path)
            os.makedirs(results_path)

    tensorboard_path = os.path.join(results_path, "tensorboard")

    if os.path.isdir(tensorboard_path):
        shutil.rmtree(tensorboard_path)
    writer = SummaryWriter(tensorboard_path)
    best_model_path = os.path.join(results_path, "best_model")

    return writer,best_model_path
