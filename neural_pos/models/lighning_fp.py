import lightning as L
import torch
import torch.nn.functional as F

from neural_pos.lighning_utils import load_optimizer
from neural_pos.models.criterions import Criterions
from neural_pos.models.backbones import Backbones


class FingerprintingModel(L.LightningModule):

    is_cc = False

    def __init__(
        self, 
        backbone_type: str, 
        data_shape,
        criterion_type, 
        criterion_params: dict[str, any] = {}, 
        backbone_params: dict[str, any] = {}, 
        optim_config: dict[str, any] = {},
        estimate_uncertainty: bool = False,
        return_var: bool = False,
        output_dim = 2,
    ):
        super().__init__()
        self.save_hyperparameters(ignore=[])
        self.optim_config = optim_config
        self.estimate_uncertainty = estimate_uncertainty
        self.return_var = return_var
        self.output_dim = output_dim * 2 if estimate_uncertainty else output_dim
        self.backbone = Backbones.get(backbone_type)(**backbone_params, out_dim=self.output_dim, data_shape=data_shape)
        self.criterion = Criterions.get(criterion_type)(**criterion_params)

    def forward(self, inputs: torch.Tensor, valid_mask: torch.Tensor | None = None) -> torch.Tensor:
        return self.backbone(inputs, valid_mask = valid_mask)

    def _step(self, batch: dict, batch_idx: int) -> torch.Tensor:
        inputs = batch["inputs"]
        tgt = batch["targets"]

        valid_mask = None
        if "valid_mask" in batch.keys():
            valid_mask = batch["valid_mask"]

        outputs = self(inputs, valid_mask=valid_mask)
        if self.estimate_uncertainty:
            # Split outputs: first half = means, second half = log(σ²)
            n_dims = outputs.shape[-1] // 2
            mean = outputs[..., :n_dims]
            log_var = outputs[..., n_dims:]
            
            # Uncertainty regularization: penalize log(σ²) to prevent trivial solutions
            # and weight the prediction error by uncertainty
            mse = F.mse_loss(mean, tgt).mean(dim=-1, keepdim=True)  # per-sample MSE
            uncertainty_loss = (log_var + mse / torch.exp(log_var)).mean()
            
            # Total loss
            loss = uncertainty_loss
            
            # Log components
            stage = "training" if self.trainer.state.stage == "train" else "validation"

            self.log(f"{stage}/mse_loss", mse.mean())
            return loss
        else:
            return self.criterion(outputs, tgt)

    def training_step(self, batch: dict, batch_idx: int) -> torch.Tensor:
        loss = self._step(batch, batch_idx)

        self.log("training/loss", loss, prog_bar=True)
        return loss
    
    def validation_step(self, batch, batch_idx):
        loss = self._step(batch, batch_idx)

        self.log("validation/loss", loss, prog_bar=True)
        return loss
    
    def predict_step(self, batch, batch_idx) -> torch.Tensor | tuple[torch.Tensor, torch.Tensor]:
        inputs = batch["inputs"]
        valid_mask = None
        if "valid_mask" in batch.keys():
            valid_mask = batch["valid_mask"]

        outputs = self(inputs, valid_mask=valid_mask)
        
        if self.estimate_uncertainty:
            # Split outputs: first half = means, second half = log(σ²)
            n_dims = outputs.shape[-1] // 2
            mean = outputs[..., :n_dims]
            log_var = outputs[..., n_dims:]
            
            if self.return_var:
                var = torch.exp(log_var)
                return mean, var
            else:
                return mean
        else:
            return outputs
    
    def configure_optimizers(self):
        return load_optimizer(self.parameters(), **self.optim_config)