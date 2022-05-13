import importlib


# find the dataset definition by name, used in train.py, evaluate_1.py
def find_dataset_def(dataset_name):
    module_name = 'datasets.{}'.format(dataset_name)
    module = importlib.import_module(module_name)
    return getattr(module, "MVSDataset")


# find the SceneData class, used in optimize process
def find_scenedata_def(dataset_name):
    module_name = 'datasets.{}'.format(dataset_name)
    module = importlib.import_module(module_name)
    return getattr(module, "SceneData")

