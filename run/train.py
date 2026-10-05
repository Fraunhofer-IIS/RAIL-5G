import os

import torch
torch.use_deterministic_algorithms(True, warn_only=False)
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False

from pathlib import Path
import random

import hydra
from torch.utils.data import DataLoader

from lightning.pytorch.callbacks import LearningRateMonitor

from neural_pos.lighning_utils import Logger
from neural_pos.models import Models
from neural_pos.datasets import Datasets
from eval import eval

import lightning as L

def load_dataset(dataset_type, dataset_params = {}):
    return Datasets[dataset_type](**dataset_params)

def load_model(model_type, model_params = {}, data_shape=None):
    return Models[model_type](**model_params, data_shape=data_shape)

def load_logger(logger_type, logger_params = {}):
    return Logger[logger_type](**logger_params)

@hydra.main(version_base=None, config_path="../configs/model_training", config_name="default")
def train(cfg):
    seed = cfg.get("seed")
    if seed is None:
        out_dir = Path(cfg["output_dir"])
        out_dir.mkdir(parents=True, exist_ok=True)
        seed = random.randint(0, int(2**16 -1))
        with open(out_dir / "seed.txt", "w") as f:
            f.write(str(seed))

    L.seed_everything(seed, workers=True)

    ds_train = load_dataset(**cfg["dataset_config"]["train"])
    data_shape = ds_train.get_data_shape()
    dl_train = DataLoader(ds_train, **cfg["dataloader_config"]["train"])

    dl_valid = None # TODO implement

    model = load_model(**cfg["model_config"], data_shape=data_shape)
    
    logger = load_logger(**cfg["logger_config"])

    callbacks = [LearningRateMonitor()]

    trainer = trainer = L.Trainer(
        logger=logger,
        callbacks=callbacks,
        **cfg["trainer_config"]
    )

    print(dl_train)
    print(model)

    try:
        trainer.fit(model, dl_train, dl_valid)
    except KeyboardInterrupt:
        # Allow manual stopping with Ctrl + C
        pass

    for test_name, test_params in cfg["eval_config"].items():
        print("Running test", test_name)
        ds_test = load_dataset(**test_params["dataset_config"])
        dl_test = DataLoader(ds_test, **cfg["dataloader_config"]["eval"])
        eval(model, dl_test, ds_test.reference, ds_test.timestamps, output_path=test_params["output_path"], valid_samples=ds_test.ref_valid)


if __name__ == "__main__":
    train()
