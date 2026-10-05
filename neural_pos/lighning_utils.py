import lightning as L
from torch.optim import AdamW

from lightning.pytorch.loggers import TensorBoardLogger
from torch.optim.lr_scheduler import LinearLR, CosineAnnealingLR, SequentialLR

def cosine_with_warmup(optimizer, start_factor=0.1, warmup_epochs=5, T_max=50):
    warmup_scheduler = LinearLR(optimizer, start_factor=start_factor, total_iters=warmup_epochs)

    main_scheduler = CosineAnnealingLR(optimizer, T_max=T_max - warmup_epochs)

    scheduler = SequentialLR(
        optimizer, 
        schedulers=[warmup_scheduler, main_scheduler], 
        milestones=[warmup_epochs]  # Switch after 5 epochs
    )

    return scheduler

Optimizer = {
    "AdamW": AdamW,
}

LR_Scheduler = {
    "LinearLR": LinearLR,
    "CosineAnnealingLR": CosineAnnealingLR,
    "CosineWithWarmupLR": cosine_with_warmup,
}

Logger = {
    "TensorBoardLogger": TensorBoardLogger
}

def load_optimizer(params, optimizer_type, optimizer_params = {}, lr_scheduler_type = None, lr_scheduler_params = {}):
    optimizer = [Optimizer[optimizer_type](params, **optimizer_params)]
    lr_scheduler = []
    if lr_scheduler_type is not None:
        lr_scheduler.append(LR_Scheduler[lr_scheduler_type](optimizer[0], **lr_scheduler_params))
    return optimizer, lr_scheduler


