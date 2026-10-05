import torch
from sklearn.decomposition import PCA
from sklearn.manifold import Isomap, TSNE
#import umap
import numpy as np
from umap import UMAP

reductions = {
    "Isomap": Isomap,
    "t-SNE": TSNE,
    "PCA": PCA,
    "UMAP": UMAP
}

non_parametric = ["t-SNE", "SammonMapping", "Isomap", "UMAP"]

class DimensionalityReduction:

    is_cc = True

    def __init__(self, dimensionality_reduction_type: str, dimensionality_reduction_params: dict[str, any], fit_reduction = None, seed = None):
        if dimensionality_reduction_type == "UMAP":
            self.reducer = reductions[dimensionality_reduction_type](**dimensionality_reduction_params, random_state=seed) # Umap does not respect global seeds
        else:
            self.reducer = reductions[dimensionality_reduction_type](**dimensionality_reduction_params)
        self.fit_reduction = fit_reduction
        self.type = dimensionality_reduction_type

    def fit(self, dl: torch.utils.data.DataLoader):
        if not self.isParametric():
            print("WARN: non parametric, skipping fit...")
            return
        with torch.no_grad():
            all_data = torch.cat([batch["inputs"] for batch in dl], dim=0).cpu().numpy()
            all_data = all_data.reshape(all_data.shape[0], -1)
        if self.fit_reduction is not None:
            all_data = all_data[::self.fit_reduction]
        self.reducer.fit(all_data)
        

    def predict(self, dl: torch.utils.data.DataLoader):
        if not self.isParametric():
            with torch.no_grad():
                all_data = torch.cat([batch["inputs"] for batch in dl], dim=0).cpu().numpy()
                all_data = all_data.reshape(all_data.shape[0], -1)
                if self.fit_reduction is not None:
                    all_data = all_data[::self.fit_reduction]
            return self.reducer.fit_transform(all_data)
        outputs = []
        with torch.no_grad():
            for batch in dl:
                inputs = batch["inputs"].cpu().numpy()
                inputs = inputs.reshape(inputs.shape[0], -1)
                outputs.append(self.reducer.transform(inputs))
        return np.concat(outputs, axis=0)
    
    def isParametric(self):
        return not (self.type in non_parametric)
