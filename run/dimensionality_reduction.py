from pathlib import Path
import random

from omegaconf import DictConfig, OmegaConf
import hydra
from torch.utils.data import DataLoader

from lightning.pytorch.callbacks import LearningRateMonitor

from neural_pos.datasets import Datasets
from eval import eval
import lightning as L

from neural_pos.dimensionality_reduction import DimensionalityReduction

def load_dataset(dataset_type, dataset_params = {}):
    return Datasets[dataset_type](**dataset_params)

@hydra.main(version_base=None, config_path="../configs/model_training", config_name="dimensionality_reduction")
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
    data_shape = ds_train[0]["inputs"].shape
    dl_train = DataLoader(ds_train, **cfg["dataloader_config"]["train"])

    dim_reduction = DimensionalityReduction(**cfg["dimensionality_reduction_config"], seed=seed)

    dim_reduction.fit(dl_train)

    for test_name, test_params in cfg["eval_config"].items():
        print("Running test", test_name)
        ds_test = load_dataset(**test_params["dataset_config"])
        dl_test = DataLoader(ds_test, **cfg["dataloader_config"]["eval"])
        eval(dim_reduction, dl_test, ds_test.reference, ds_test.timestamps, output_path=test_params["output_path"], valid_samples=ds_test.ref_valid, is_dim_reduction=True)


if __name__ == "__main__":
    train()