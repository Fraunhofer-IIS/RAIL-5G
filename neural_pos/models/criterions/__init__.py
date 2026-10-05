import torch


Criterions: dict[str, object] = {
    "MSE": torch.nn.MSELoss,
    "HuberLoss": torch.nn.HuberLoss,
    "TripletMarginLoss": torch.nn.TripletMarginLoss
}