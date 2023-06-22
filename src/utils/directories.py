import os
from sacred import Ingredient
from utils.project import project, get_project_name

directories = Ingredient('directories', ingredients=[project])

@directories.config
def default_config():
    data_dir = os.path.join(os.path.expanduser('~'), 'shared')
    shm_dir = os.path.join(os.path.abspath(os.sep), 'dev', 'shm')

@directories.capture
def get_model_dir(data_dir):
    return make_if_not_exits(os.path.join(data_dir, get_project_name(), 'model_checkpoints'))

@directories.capture
def get_dataset_dir(data_dir):
    return make_if_not_exits(data_dir)

@directories.capture
def get_persistent_cache_dir(data_dir):
    return make_if_not_exits(os.path.join(data_dir, 'tmp'))


@directories.capture
def get_ram_cache_dir(shm_dir):
    if os.getenv("MEM_FOLDER") is not None:
        return make_if_not_exits(os.getenv("MEM_FOLDER"))
    return make_if_not_exits(os.path.join(shm_dir, 'tmp_'))


def make_if_not_exits(path):
    os.makedirs(path, exist_ok=True)
    return path
