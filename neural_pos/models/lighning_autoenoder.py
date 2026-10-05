import math

import lightning as L
import torch
import torch.nn as nn
import torch.nn.functional as F

from neural_pos.lighning_utils import load_optimizer
from neural_pos.models.criterions import Criterions
from neural_pos.models.backbones import Backbones


class VectorQuantizer(nn.Module):
    """Vector Quantization layer for VQ-VAE"""
    def __init__(self, num_embeddings: int, embedding_dim: int, commitment_cost: float = 0.25):
        super().__init__()
        self.num_embeddings = num_embeddings
        self.embedding_dim = embedding_dim
        self.commitment_cost = commitment_cost
        
        self.embeddings = nn.Embedding(num_embeddings, embedding_dim)
        self.embeddings.weight.data.uniform_(-1/num_embeddings, 1/num_embeddings)
    
    def forward(self, z: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        """
        Args:
            z: encoder output of shape (B, embedding_dim)
        Returns:
            quantized: quantized embeddings
            vq_loss: vector quantization loss
        """
        # Calculate distances to embeddings
        distances = (
            torch.sum(z**2, dim=1, keepdim=True)
            + torch.sum(self.embeddings.weight**2, dim=1)
            - 2 * torch.matmul(z, self.embeddings.weight.t())

        )
        
        # Get nearest embedding indices
        encoding_indices = torch.argmin(distances, dim=1)
        quantized = self.embeddings(encoding_indices)
        
        # Calculate VQ losses
        e_latent_loss = F.mse_loss(quantized.detach(), z)
        q_latent_loss = F.mse_loss(quantized, z.detach())
        vq_loss = q_latent_loss + self.commitment_cost * e_latent_loss
        
        # Straight-through estimator
        quantized = z + (quantized - z).detach()
        
        return quantized, vq_loss


class AutoencoderModel(L.LightningModule):

    is_cc = True

    def __init__(
        self,
        backbone_type: str,
        decoder_type: str,
        data_shape,
        criterion_type,
        criterion_params: dict[str, any] = {},
        backbone_params: dict[str, any] = {},
        decoder_params: dict[str, any] = {},
        optim_config: dict[str, any] = {},
        return_var: bool = False,
        output_dim: int = 2,
        model_type: str = "ae",           # "ae", "vae", or "vqvae"
        vae_beta: float = 1.0,
        vqvae_num_embeddings: int = 512,
        vqvae_commitment_cost: float = 0.25,
        vqvae_return_quantized: bool = True,
        # --- Regularization ---
        regularization_type: str | None = None,   # None, "FRD", "MRD"
        regularization_weight: float = 1.0,
        regularization_max_time_diff: float = 1.0,
    ):
        super().__init__()
        self.save_hyperparameters(ignore=[])
        self.optim_config = optim_config
        self.data_shape = data_shape
        self.return_var = return_var
        self.output_dim = output_dim
        self.model_type = model_type
        self.vae_beta = vae_beta
        self.vqvae_return_quantized = vqvae_return_quantized

        assert regularization_type in (None, "FRD", "MRD"), \
            "regularization_type must be None, 'FRD', or 'MRD'"
        self.regularization_type = regularization_type
        self.regularization_weight = regularization_weight
        self.regularization_max_time_diff = regularization_max_time_diff

        backbone_out_dim = self.output_dim * 2 if model_type == "vae" else self.output_dim
        self.backbone = Backbones.get(backbone_type)(
            **backbone_params,
            out_dim=backbone_out_dim,
            data_shape=data_shape,
        )

        if model_type == "vqvae":
            self.vq_layer = VectorQuantizer(
                num_embeddings=vqvae_num_embeddings,
                embedding_dim=self.output_dim,
                commitment_cost=vqvae_commitment_cost,
            )

        self.decoder = Backbones.get(decoder_type)(
            **decoder_params,
            out_dim=math.prod(data_shape),
            data_shape=[self.output_dim, 1, 1, 1],
        )
        self.criterion = Criterions.get(criterion_type)(**criterion_params)

    # ------------------------------------------------------------------
    # Encoding helpers
    # ------------------------------------------------------------------

    def encode(self, inputs, valid_mask=None):
        z = self.backbone(inputs, valid_mask=valid_mask)
        if self.model_type == "vae":
            mean, logvar = torch.chunk(z, 2, dim=-1)
            return mean, logvar
        return z

    def reparameterize(self, mean, logvar):
        std = torch.exp(0.5 * logvar)
        return mean + torch.randn_like(std) * std

    def forward(self, inputs, valid_mask=None):
        if self.model_type == "vae":
            mean, _ = self.encode(inputs, valid_mask=valid_mask)
            return mean
        elif self.model_type == "vqvae":
            z = self.encode(inputs, valid_mask=valid_mask)
            z_q, _ = self.vq_layer(z)
            return z_q if self.vqvae_return_quantized else z
        else:
            return self.encode(inputs, valid_mask=valid_mask)

    # ------------------------------------------------------------------
    # Regularization
    # ------------------------------------------------------------------

    def _regularization_loss(
        self,
        embeddings: torch.Tensor,   # (B, D)
        timestamps: torch.Tensor,   # (B,) or (B, 1)  – arbitrary time units
    ) -> torch.Tensor:
        """
        Compute FRD or MRD regularization loss for pairs whose
        timestamp difference is <= regularization_max_time_diff.

        FRD: Huber( ||y_i - y_j||,  Δt_ij )          – enforce equality
        MRD: mean( max(||y_i - y_j|| - Δt_ij, 0)² )  – enforce upper bound
        """
        B = embeddings.size(0)
        if B < 2:
            return embeddings.sum() * 0.0   # keeps grad graph alive, value 0

        ts = timestamps.float().reshape(B)

        # --- Pairwise timestamp differences (upper-triangular, flat) ---
        # Shape: (B, B) -> upper-tri indices
        rows, cols = torch.triu_indices(B, B, offset=1, device=embeddings.device)
        t_diff = (ts[rows] - ts[cols]).abs()          # (num_pairs,)

        # --- Filter to pairs within the time window ---
        mask = t_diff <= self.regularization_max_time_diff
        if mask.sum() == 0:
            return embeddings.sum() * 0.0

        t_diff_sel = t_diff[mask]                      # (P,)
        emb_dist = torch.linalg.norm(
            embeddings[rows[mask]] - embeddings[cols[mask]], dim=-1
        )                                              # (P,)

        if self.regularization_type == "FRD":
            # Smooth L1 (Huber) between embedding distance and time difference
            loss = F.huber_loss(emb_dist, t_diff_sel, reduction="mean")

        else:  # MRD
            # One-sided penalty: only penalise when embedding dist > time diff
            violation = F.relu(emb_dist - t_diff_sel)
            loss = (violation ** 2).mean()

        return loss

    # ------------------------------------------------------------------
    # Step
    # ------------------------------------------------------------------

    def _step(self, batch, batch_idx):
        inputs    = batch["inputs"]
        timestamps = batch["timestamps"]   # (B,) expected
        valid_mask = batch.get("valid_mask", None)

        # --- Encode ---
        if self.model_type == "vae":
            mean, logvar = self.encode(inputs, valid_mask=valid_mask)
            z = self.reparameterize(mean, logvar)
            kl_loss = -0.5 * torch.sum(
                1 + logvar - mean.pow(2) - logvar.exp(), dim=-1
            ).mean()
            embeddings_for_reg = mean          # use mean (deterministic) for reg
        elif self.model_type == "vqvae":
            z = self.encode(inputs, valid_mask=valid_mask)
            z_q, vq_loss = self.vq_layer(z)
            z = z_q
            embeddings_for_reg = z_q
        else:
            z = self.encode(inputs, valid_mask=valid_mask)
            embeddings_for_reg = z

        # --- Decode ---
        recon: torch.Tensor = self.decoder(z)
        recon = recon.reshape((recon.shape[0],) + self.data_shape)

        # --- Apply valid mask ---
        if valid_mask is not None:
            B, C, A, T = recon.shape
            recon  = recon.permute(0, 2, 1, 3).reshape(B * A, C * T)
            inputs = inputs.permute(0, 2, 1, 3).reshape(B * A, C * T)
            vm     = valid_mask.reshape(-1)
            recon  = recon[vm]
            inputs = inputs[vm]

        recon_loss = self.criterion(recon, inputs)

        # --- Regularization ---
        reg_loss = torch.tensor(0.0, device=recon_loss.device)
        if self.regularization_type is not None:
            reg_loss = self._regularization_loss(embeddings_for_reg, timestamps)

        # --- Total loss ---
        if self.model_type == "vae":
            total = recon_loss + self.vae_beta * kl_loss + self.regularization_weight * reg_loss
            return total, recon_loss, kl_loss, reg_loss
        elif self.model_type == "vqvae":
            total = recon_loss + vq_loss + self.regularization_weight * reg_loss
            return total, recon_loss, vq_loss, reg_loss
        else:
            total = recon_loss + self.regularization_weight * reg_loss
            return total, recon_loss, reg_loss

    # ------------------------------------------------------------------
    # Lightning steps
    # ------------------------------------------------------------------

    def _log_losses(self, prefix, loss_output):
        if self.model_type in ("vae", "vqvae"):
            total, recon, aux, reg = loss_output
            self.log(f"{prefix}/loss",       total, prog_bar=True)
            self.log(f"{prefix}/recon_loss", recon)
            aux_name = "kl_loss" if self.model_type == "vae" else "vq_loss"
            self.log(f"{prefix}/{aux_name}", aux)
        else:
            total, recon, reg = loss_output
            self.log(f"{prefix}/loss",       total, prog_bar=True)
            self.log(f"{prefix}/recon_loss", recon)

        if self.regularization_type is not None:
            self.log(f"{prefix}/reg_loss", reg)

        return total

    def training_step(self, batch, batch_idx):
        return self._log_losses("training", self._step(batch, batch_idx))

    def validation_step(self, batch, batch_idx):
        return self._log_losses("validation", self._step(batch, batch_idx))

    def predict_step(self, batch, batch_idx):
        inputs     = batch["inputs"]
        valid_mask = batch.get("valid_mask", None)
        embeddings = self(inputs, valid_mask=valid_mask)
        assert embeddings.shape[-1] == self.output_dim
        return embeddings

    def configure_optimizers(self):
        return load_optimizer(self.parameters(), **self.optim_config)