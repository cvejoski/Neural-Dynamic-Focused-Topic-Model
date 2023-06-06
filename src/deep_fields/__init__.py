# -*- coding: utf-8 -*-
import os

from pkg_resources import get_distribution, DistributionNotFound

try:
    # Change here if project is renamed and does not equal the package name
    dist_name = __name__
    __version__ = get_distribution(dist_name).version
except DistributionNotFound:
    __version__ = 'unknown'
finally:
    del get_distribution, DistributionNotFound

base_path = os.path.dirname(os.path.abspath(__file__))
project_path = os.path.abspath(os.path.join(base_path, '..', '..'))
data_path = os.path.join(project_path, 'data')
models_path = os.path.join(project_path, 'results')
test_data_path = os.path.join(project_path, 'tests', 'data')
