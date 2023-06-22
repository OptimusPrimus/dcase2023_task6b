import os
from sacred import Ingredient

project = Ingredient('project')


@project.config
def default_config():
    name = 'dcase_2023_audio_retrieval'


@project.capture
def get_project_name(name):
    return name