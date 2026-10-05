import torch
from neural_pos.models.lighning_autoenoder import AutoencoderModel
from neural_pos.models.lighning_cc_siamese import SiameseChannelChartingModel
from neural_pos.models.lighning_fp import FingerprintingModel
from neural_pos.models.lighning_cc_triplet import TripletChannelChartingModel
from neural_pos.models.lightning_uncertainty_aware import GDMModel

Models = {
    "FingerprintingModel": FingerprintingModel,
    "SiameseChannelChartingModel": SiameseChannelChartingModel,
    "TripletChannelChartingModel": TripletChannelChartingModel,
    "AutoencoderModel": AutoencoderModel,
    "GDMModel": GDMModel
}