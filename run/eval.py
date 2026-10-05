import hydra
import torch
import lightning as L
from pathlib import Path
import numpy as np
import json
from neural_pos.eval_utils import affine_transform_channel_chart, plot_cdf, plot_colorized, plot_trajectory, plot_hexbin, trustworthiness, continuity, kruskal_stress


def eval(model: torch.nn.Module, dataloader: torch.utils.data.DataLoader, ground_truth: torch.Tensor, timestamps: torch.Tensor, output_path: str, valid_samples = None, accelerator="gpu", trajectory_plot_max_points: int = 5000, is_dim_reduction = False):

    out_path = Path(output_path)
    out_path.mkdir(parents=True)

    if is_dim_reduction:
        pred = model.predict(dataloader)
        if not model.isParametric():
            # If the model is not parametric, we will refit all data and thus fit_reduction will be applied
            ground_truth = ground_truth[::model.fit_reduction]
            timestamps = timestamps[::model.fit_reduction]
            if valid_samples is not None:
                valid_samples = valid_samples[::model.fit_reduction]
    else:
        trainer = L.Trainer(accelerator=accelerator, deterministic=True)

        pred = trainer.predict(model, dataloader)
        pred: np.array = torch.cat(pred, dim=0).numpy()

    pred_no_transform = None
    pred_ekf_no_transform = None

    is_cc = model.is_cc

    if is_cc:
        pred_no_transform = pred
        pred = affine_transform_channel_chart(ground_truth, pred)


    if valid_samples is None:
        valid_samples = np.ones_like(timestamps) == 1

    np.save(out_path / "predictions.npy", pred)
    np.save(out_path / "ground_truth.npy", ground_truth)
    np.save(out_path / "timestamps.npy", timestamps)
    np.save(out_path / "valid.npy", valid_samples)

    errors = np.linalg.norm(pred - ground_truth, axis=-1)

    fig, ax = plot_cdf(errors[valid_samples])
    fig.savefig(out_path / f"cdf.pdf", bbox_inches="tight")

    # As we store our plots as pdf, we will need to reduce numbers to make them renderable
    limit_samples_trajectory = max(len(pred) // trajectory_plot_max_points, 1)

    fig, ax = plot_trajectory(prediction = pred[::limit_samples_trajectory], ground_truth=ground_truth[::limit_samples_trajectory]) 
    fig.savefig(out_path / f"trajectory.pdf", bbox_inches="tight")

    fig, ax = plot_hexbin(ground_truth[valid_samples], errors[valid_samples], vmax=3)
    fig.savefig(out_path / f"hexbin.pdf", bbox_inches="tight")

    if is_cc:
        fig = plot_colorized(pred_no_transform[::limit_samples_trajectory], ground_truth[::limit_samples_trajectory])
        fig.savefig(out_path / f"channel_chart.pdf", bbox_inches="tight")

    metrics = {f"CE{int(ceXX * 100)}": float(np.quantile(errors[valid_samples], ceXX)) for ceXX in [0.5, 0.75, 0.9, 0.95, 0.99]}
    metrics["MAE"] = float(errors[valid_samples].mean())

    if is_cc:

        fig = plot_colorized(ground_truth[::limit_samples_trajectory], ground_truth[::limit_samples_trajectory])
        fig.savefig(out_path / f"channel_chart_gt.pdf", bbox_inches="tight")

        limit_samples_metric = max(valid_samples.sum() // trajectory_plot_max_points, 1)

        metrics["trustworthiness"] = trustworthiness(ground_truth[valid_samples][::limit_samples_metric], pred_no_transform[valid_samples][::limit_samples_metric])
        metrics["continuity"] = continuity(ground_truth[valid_samples][::limit_samples_metric], pred_no_transform[valid_samples][::limit_samples_metric])
        metrics["kruskal_stress"] = kruskal_stress(ground_truth[valid_samples][::limit_samples_metric], pred_no_transform[valid_samples][::limit_samples_metric])

    with open(out_path / "metrics.json", "w") as f:
        json.dump(metrics, f)
   

@hydra.main(version_base=None, config_path="../configs/", config_name="")
def main(cfg):
    raise NotImplementedError("TODO implement standalone version")
