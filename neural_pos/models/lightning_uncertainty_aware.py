import io

import lightning as L
import torch

from neural_pos.lighning_utils import load_optimizer
from neural_pos.models.backbones import Backbones
from PIL import Image
import torchvision.transforms.functional as TF
import numpy as np
import matplotlib.pyplot as plt
import torch.nn as nn

def plot_colorized(positions, groundtruth_positions, title=None, show=True, alpha=1.0):
    center_point = np.zeros(2, dtype=np.float32)
    center_point[0] = 0.5 * (np.min(groundtruth_positions[:, 0]) + np.max(groundtruth_positions[:, 0]))
    center_point[1] = 0.5 * (np.min(groundtruth_positions[:, 1]) + np.max(groundtruth_positions[:, 1]))
    NormalizeData = lambda d: (d - np.min(d)) / (np.max(d) - np.min(d))
    rgb_values = np.zeros((groundtruth_positions.shape[0], 3))
    rgb_values[:, 0] = 1 - 0.9 * NormalizeData(groundtruth_positions[:, 0])
    rgb_values[:, 1] = 0.8 * NormalizeData(np.square(np.linalg.norm(groundtruth_positions - center_point, axis=1)))
    rgb_values[:, 2] = 0.9 * NormalizeData(groundtruth_positions[:, 1])

    plt.figure(figsize=(6, 6))
    if title is not None:
        plt.title(title, fontsize=16)
    plt.scatter(positions[:, 0], positions[:, 1], c=rgb_values, alpha=alpha, s=10, linewidths=0)
    plt.xlabel("x coordinate")
    plt.ylabel("y coordinate")
    if show:
        plt.show()


def affine_transform_channel_chart(groundtruth_pos, channel_chart_pos):
    pad = lambda x: np.hstack([x, np.ones((x.shape[0], 1))])
    unpad = lambda x: x[:, :-1]
    A, res, rank, s = np.linalg.lstsq(pad(channel_chart_pos), pad(groundtruth_pos), rcond=None)
    transform = lambda x: unpad(np.dot(pad(x), A))
    return transform(channel_chart_pos)

class ChannelChartingLossTorch(nn.Module):
    def __init__(
        self,
        acceleration_variance: float = 1.7,
        acceleration_weight: float = 0.01,
        triangle_penalty_weight: float = 0.001,
        acceleration_reduction: int = 10,
    ):
        super().__init__()
        self.acceleration_variance = acceleration_variance
        self.acceleration_weight = acceleration_weight
        self.acceleration_reduction = acceleration_reduction
        self.triangle_penalty_weight = triangle_penalty_weight
        self.interation = 0

    def _acceleration_loss(self, all_positions: torch.Tensor, timestamps: torch.Tensor) -> torch.Tensor:
        reduction_offset = self.interation % self.acceleration_reduction
        self.interation += 1
        dt = torch.diff(timestamps[reduction_offset::self.acceleration_reduction], dim=0)
        valid = dt > 1e-8
        velocities = torch.diff(all_positions[reduction_offset::self.acceleration_reduction], dim=0) / dt.unsqueeze(-1).clamp(min=1e-8)
        
        # Only compute acceleration where both consecutive velocity intervals are valid
        valid_acc = valid[:-1] & valid[1:]
        dt2 = dt[:-1].clamp(min=1e-8)
        accelerations = torch.diff(velocities, dim=0) / dt2.unsqueeze(-1)
        
        accelerations = accelerations[valid_acc]
        if accelerations.numel() == 0:
            return torch.tensor(0.0, device=all_positions.device)
        
        acc_sq = (accelerations ** 2).sum(dim=-1)
        return (acc_sq / self.acceleration_variance).mean()

    def forward(
        self,
        all_positions: torch.Tensor,
        timestamps: torch.Tensor,
        paths: torch.Tensor,
        path_hops: torch.Tensor,
        path_means: torch.Tensor,
        path_variances: torch.Tensor,
    ) -> dict[str, torch.Tensor]:
        B = paths.shape[0]

        path_positions = all_positions[paths]
        deltas = path_positions[:, 1:, :] - path_positions[:, :-1, :]
        step_dist = torch.sqrt((deltas ** 2).sum(dim=-1) + 1e-6)
        geodesic_distance = step_dist.sum(dim=1)

        geodesic_loss = (
            (geodesic_distance - path_means) ** 2 / (path_variances + 1e-6)
        ).mean()

        idx_a = paths[:, 0]
        endpoint_rows = torch.stack([torch.arange(B, device=paths.device), path_hops], dim=1)
        idx_b = paths[endpoint_rows[:, 0], endpoint_rows[:, 1]]
        endpoint_dist = torch.sqrt(
            ((all_positions[idx_a] - all_positions[idx_b]) ** 2).sum(dim=-1) + 1e-6
        )
        triangle_penalty = torch.clamp(
            step_dist - endpoint_dist.unsqueeze(-1), min=0.0
        ).sum()

        acc_loss = self._acceleration_loss(all_positions, timestamps)

        return {
            "geodesic":  geodesic_loss,
            "triangle":  self.triangle_penalty_weight * triangle_penalty,
            "acceleration": self.acceleration_weight * acc_loss,
        }

class GDMModel(L.LightningModule):

    is_cc = True

    def __init__(
        self,
        backbone_type: str,
        data_shape,
        criterion_params: dict[str, any] = {},
        backbone_params: dict[str, any] = {},
        optim_config: dict[str, any] = {},
        plot_cc: bool = False,
        plot_cc_period: int = 200,
        plot_cc_paths_to_plot_count: int = 50,
    ):
        super().__init__()
        self.save_hyperparameters(ignore=["plot_cc_groundtruth_positions"])
        self.optim_config = optim_config
        self.backbone = Backbones.get(backbone_type)(**backbone_params, data_shape=data_shape)
        self.criterion = ChannelChartingLossTorch(**criterion_params)

        # Channel chart plotting
        self.plot_cc = plot_cc
        self.plot_cc_period = plot_cc_period
        self.plot_cc_paths_to_plot_count = plot_cc_paths_to_plot_count
        self._cc_step_counter = 0

    def forward(self, inputs: torch.Tensor, valid_mask: torch.Tensor | None = None) -> torch.Tensor:
        assert inputs.ndim == 4
        if valid_mask is not None:
            assert valid_mask.shape[1] == inputs.shape[2]
        return self.backbone(inputs, valid_mask=valid_mask)

    def _step(self, batch: dict, batch_idx: int) -> tuple[dict, torch.Tensor]:
        inputs     = batch["inputs"][0]
        timestamps = batch["timestamps"][0]
        paths      = batch["paths"][0]
        path_hops  = batch["path_hops"][0]
        means      = batch["means"][0]
        variances  = batch["variances"][0]

        valid_mask = None

        all_positions = self(inputs, valid_mask=valid_mask)

        losses = self.criterion(
            all_positions=all_positions,
            timestamps=timestamps,
            paths=paths,
            path_hops=path_hops,
            path_means=means,
            path_variances=variances,
        )

        return losses, all_positions

    def training_step(self, batch: dict, batch_idx: int) -> torch.Tensor:
        losses, all_positions = self._step(batch, batch_idx)
        total = sum(losses.values())
        for name, value in losses.items():
            self.log(f"training/{name}_loss", value)
        self.log("training/loss", total, prog_bar=True)

        if self.plot_cc:
            self._cc_step_counter += 1
            if self._cc_step_counter % self.plot_cc_period == 0:
                self._log_channel_chart(
                    all_positions.detach().cpu().numpy(),
                    batch["paths"][0].detach().cpu().int().numpy(),
                )

        return total

    def validation_step(self, batch: dict, batch_idx: int) -> torch.Tensor:
        losses, _ = self._step(batch, batch_idx)
        total = sum(losses.values())
        for name, value in losses.items():
            self.log(f"validation/{name}_loss", value)
        self.log("validation/loss", total, prog_bar=True)
        return total

    def predict_step(self, batch: dict, batch_idx: int) -> torch.Tensor:
        inputs = batch["inputs"]
        valid_mask = batch["valid_mask"] if "valid_mask" in batch else None
        return self(inputs, valid_mask=valid_mask)

    def configure_optimizers(self):
        return load_optimizer(self.parameters(), **self.optim_config)

    # ---- Channel chart plotting ----

    def _get_groundtruth_positions(self) -> np.ndarray | None:
        try:
            dataloader = self.trainer.train_dataloader
            dataset = dataloader.dataset
            # If using a Subset wrapper, unwrap it
            while hasattr(dataset, "dataset"):
                dataset = dataset.dataset
            return dataset.reference
        except Exception:
            return None

    def _log_channel_chart(self, pred_positions: np.ndarray, paths: np.ndarray):
        gt = self._get_groundtruth_positions()
        if gt is None:
            return

        transformed = affine_transform_channel_chart(gt, pred_positions)
        errorvectors = gt - transformed
        errors = np.sqrt(errorvectors[:, 0] ** 2 + errorvectors[:, 1] ** 2)
        mae = float(np.mean(errors))
        cep = float(np.median(errors))

        plot_colorized(
            pred_positions, gt,
            title=f"MAE = {mae:.4f}m, CEP = {cep:.4f}m",
            show=False,
        )

        for path_indices in paths[:self.plot_cc_paths_to_plot_count]:
            plt.plot(pred_positions[path_indices, 0], pred_positions[path_indices, 1])

        fig = plt.gcf()
        img_tensor = self._fig_to_tensor(fig)
        plt.close(fig)

        tb_logger = self._get_tb_logger()
        if tb_logger is not None:
            tb_logger.add_image("channel_chart", img_tensor, global_step=self.global_step)

        self.log("cc/mae", mae, prog_bar=False)
        self.log("cc/cep", cep, prog_bar=False)

    def _get_tb_logger(self):
        loggers = self.loggers if hasattr(self, "loggers") else [self.logger]
        for logger in loggers:
            if logger is None:
                continue
            if hasattr(logger, "experiment") and hasattr(logger.experiment, "add_image"):
                return logger.experiment
        return None

    @staticmethod
    def _fig_to_tensor(fig):
        buf = io.BytesIO()
        fig.savefig(buf, format="png", bbox_inches="tight", dpi=100)
        buf.seek(0)
        image = Image.open(buf)
        tensor = TF.to_tensor(image)[:3]
        buf.close()
        return tensor