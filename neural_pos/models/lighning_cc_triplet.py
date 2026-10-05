import lightning as L
import torch

from neural_pos.lighning_utils import load_optimizer
from neural_pos.models.criterions import Criterions
from neural_pos.models.backbones import Backbones


class TripletChannelChartingModel(L.LightningModule):
    """
    Lightning module for triplet-based channel charting.
    
    Trains a siamese network using triplet loss where each batch contains:
    - Anchor: Reference sample
    - Positive: Sample similar to anchor (e.g., temporally close)
    - Negative: Dissimilar sample (e.g., random sample)
    
    Args:
        backbone_type (str): Type of backbone network from Backbones registry
        criterion_type (str): Type of triplet loss from Criterions registry (e.g., "TripletMarginLoss")
        criterion_params (dict): Parameters for the triplet loss criterion
        backbone_params (dict): Parameters for the backbone network
        optim_config (dict): Optimizer configuration
        reg_criterion_type (str): Type of regularization criterion. Default: "MSE"
        reg_criterion_params (dict): Parameters for regularization criterion
    """

    is_cc = True

    def __init__(
        self, 
        backbone_type: str, 
        data_shape,
        criterion_type: str, 
        criterion_params: dict[str, any] = {}, 
        backbone_params: dict[str, any] = {}, 
        optim_config: dict[str, any] = {}, 
        reg_criterion_type: str = "MSE", 
        reg_criterion_params: dict[str, any] = {}
    ):
        super().__init__()
        self.save_hyperparameters(ignore=[])
        self.optim_config = optim_config
        self.backbone = Backbones.get(backbone_type)(**backbone_params, data_shape=data_shape)
        self.criterion = Criterions.get(criterion_type)(**criterion_params)
        self.reg_criterion = Criterions.get(reg_criterion_type)(**reg_criterion_params)

    def forward(self, inputs: torch.Tensor, valid_mask: torch.Tensor | None = None) -> torch.Tensor:
        assert inputs.ndim == 4  # Needs shape batch, channels (e.g. real/imag), num antennas, num taps

        if valid_mask is not None:
            assert valid_mask.shape[1] == inputs.shape[2]  # If we have a validity mask, it needs one value per antenna

        return self.backbone(inputs, valid_mask=valid_mask)

    def _step(self, batch: dict, batch_idx: int) -> tuple[torch.Tensor, torch.Tensor | None]:
        inputs = batch["inputs"]
        inputs_positive = batch["inputs_positive"]
        inputs_negative = batch["inputs_negative"]

        # Get validity masks if available
        valid_mask = batch.get("valid_mask")
        valid_mask_positive = batch.get("valid_mask_positive")
        valid_mask_negative = batch.get("valid_mask_negative")

        # Forward pass for all three samples
        anchor_embeddings = self(inputs, valid_mask=valid_mask)
        positive_embeddings = self(inputs_positive, valid_mask=valid_mask_positive)
        negative_embeddings = self(inputs_negative, valid_mask=valid_mask_negative)

        # Compute triplet loss
        loss_triplet = self.criterion(anchor_embeddings, positive_embeddings, negative_embeddings)

        # Optional regularization
        if "regularization_mask" in batch.keys():
            reg_mask = batch["regularization_mask"]
            reg_mask_positive = batch["regularization_mask_positive"]
            reg_mask_negative = batch["regularization_mask_negative"]

            # Making additional passes with regularization masks
            reg_anchor = self(inputs, valid_mask=reg_mask)
            reg_positive = self(inputs_positive, valid_mask=reg_mask_positive)
            reg_negative = self(inputs_negative, valid_mask=reg_mask_negative)

            # Concatenate all embeddings
            total_reg_preds = torch.cat([reg_anchor, reg_positive, reg_negative])
            total_preds = torch.cat([
                anchor_embeddings.detach(), 
                positive_embeddings.detach(), 
                negative_embeddings.detach()
            ])
            
            loss_reg = self.reg_criterion(total_preds, total_reg_preds)
            return loss_triplet, loss_reg
        
        return loss_triplet, None

    def training_step(self, batch: dict, batch_idx: int) -> torch.Tensor:
        loss_triplet, loss_reg = self._step(batch, batch_idx)
        
        self.log("training/triplet_loss", loss_triplet)
        
        total_loss = loss_triplet
        if loss_reg is not None:
            self.log("training/reg_loss", loss_reg)
            total_loss = total_loss + loss_reg
        
        self.log("training/loss", total_loss, prog_bar=True)
        return total_loss
    
    def validation_step(self, batch: dict, batch_idx: int) -> torch.Tensor:
        loss_triplet, loss_reg = self._step(batch, batch_idx)
        
        total_loss = loss_triplet
        if loss_reg is not None:
            total_loss = total_loss + loss_reg
        
        self.log("validation/loss", total_loss, prog_bar=True)
        return total_loss
    
    def predict_step(self, batch: dict, batch_idx: int) -> torch.Tensor:
        inputs = batch["inputs"]
        valid_mask = batch.get("valid_mask")

        return self(inputs, valid_mask=valid_mask)
    
    def configure_optimizers(self):
        return load_optimizer(self.parameters(), **self.optim_config)