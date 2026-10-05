import lightning as L
import torch

from neural_pos.lighning_utils import load_optimizer
from neural_pos.models.criterions import Criterions
from neural_pos.models.backbones import Backbones


class SiameseChannelChartingModel(L.LightningModule):

    is_cc = True

    def __init__(self, backbone_type: str, data_shape, criterion_type: str, criterion_params: dict[str, any] = {}, backbone_params: dict[str, any] = {}, optim_config: dict[str, any] = {}, reg_criterion_type: str = "MSE", reg_criterion_params: dict[str, any]= {}):
        super().__init__()
        self.save_hyperparameters(ignore=[])
        self.optim_config = optim_config
        self.backbone = Backbones.get(backbone_type)(**backbone_params, data_shape=data_shape)
        self.criterion = Criterions.get(criterion_type)(**criterion_params)
        self.reg_criterion = Criterions.get(reg_criterion_type)(**reg_criterion_params)

    def forward(self, inputs: torch.Tensor, valid_mask: torch.Tensor | None = None) -> torch.Tensor:
        assert inputs.ndim == 4 # Needs shape batch, channels (e.g. real/imag), num antennas, num taps / 

        if valid_mask != None:
            assert valid_mask.shape[1] == inputs.shape[2] # If we have a validity mask, it needs one value per antenna

        return self.backbone(inputs, valid_mask = valid_mask)

    def _step(self, batch: dict, batch_idx: int) -> torch.Tensor:
        inputs = batch["inputs"]
        inputs2 = batch["inputs2"]

        tgt = batch["targets"]
        weights = batch.get("weights")

        valid_mask = None
        valid_mask2 = None
        if "valid_mask" in batch.keys():
            valid_mask = batch["valid_mask"]
            valid_mask2 = batch["valid_mask2"]

        preds: torch.Tensor = self(inputs, valid_mask=valid_mask)
        preds2: torch.Tensor = self(inputs2, valid_mask=valid_mask2)

        dist = torch.linalg.norm(preds - preds2, dim=-1)
        
        if weights is None:
            loss_dist = self.criterion(dist, tgt)
        else:
            loss_dist = self.criterion(dist, tgt, weights = weights)

        if "regularization_mask" in batch.keys():
            reg_mask = batch["regularization_mask"]
            reg_mask2 = batch["regularization_mask2"]

            # Making two additional new passes is inefficient, as some models, e.g. AdaPos can reuse intermediate results.
            # However, due to better framework generalization, we will keep this.
            reg_preds = self(inputs, valid_mask=reg_mask)
            reg_preds_2 = self(inputs2, valid_mask=reg_mask2)

            total_reg_preds = torch.cat([reg_preds, reg_preds_2])
            total_preds = torch.cat([preds.detach(), preds2.detach()]) # We want to push reg to "normal", not both to each other
            loss_reg = self.reg_criterion(total_preds, total_reg_preds)

            return loss_dist, loss_reg
        
        else:
            return loss_dist, None

    def training_step(self, batch: dict, batch_idx: int) -> torch.Tensor:
        loss, reg_loss = self._step(batch, batch_idx)
        self.log("training/siamese_loss", loss)
        if reg_loss is not None:
            self.log("training/reg_loss", reg_loss)
            self.loss += reg_loss
        self.log("training/loss", loss, prog_bar=True)
        return loss
    
    def validation_step(self, batch, batch_idx):
        loss = self._step(batch, batch_idx)

        self.log("validation/loss", loss, prog_bar=True)
        return loss
    
    def predict_step(self, batch, batch_idx) -> torch.Tensor:
        inputs = batch["inputs"]
        valid_mask = None
        if "valid_mask" in batch.keys():
            valid_mask = batch["valid_mask"]

        return self(inputs, valid_mask=valid_mask)
    
    def configure_optimizers(self):
        return load_optimizer(self.parameters(), **self.optim_config)