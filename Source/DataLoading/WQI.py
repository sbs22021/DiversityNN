import pandas as pd
import numpy as np
from ast import literal_eval
from torch.utils.data.dataset import Dataset

class ObjectView(object):
    def __init__(self, d):
        self.__dict__ = d

def get_dataset_args(as_dict=False):
    arg_dict = {
        "num_samples": 1000,  # modified for meta-learning
        "train_split": 0.8,
        "template_len": 12,
        "scale_coeff": 0.4,
        "max_translation": 48,
        "corr_noise_scale": 0.25,
        "iid_noise_scale": 2e-2,
        "shear_scale": 0.75,
        "final_seq_length": 40,
    }
    return arg_dict if as_dict else ObjectView(arg_dict)

class WQIDataset(Dataset):
    """Pytorch custom dataset for loading WQI dataset"""

    def __init__(
        self,
        root_path,
        train,
        transform=None,
        **kwargs
    ):
        """
        Args:
            root_path (string): path to the dataset pickle file
            train (bool): selects between training and testing dataset
            regenerate (bool): Whether to regenerate the dataset
            download (bool): Downloads original 1D mnist dataset
            transform (Function): Applies transformation to PIL image
            target_transform (Function): Applies transformation to target values
        """
        self.args = get_dataset_args()
        self.path = root_path + "/DSVersion1Scalled.csv"
        self.train = train
        self.transform = transform


        wqi_df = pd.read_csv(self.path)
        wqi_df['scaled_features_array'] = wqi_df['scaled_features'].apply(literal_eval)
        wqi_df['scaled_features_numpy_array'] = wqi_df['scaled_features_array'].apply(np.array)
        X = np.vstack(wqi_df.scaled_features_numpy_array.to_numpy())
        Y = wqi_df.ClassificationStatus_index.to_numpy()
        split_idx = int(0.8 * X.shape[0])
        X_train = X[:split_idx]
        X_test = X[split_idx:]

        Y_train = Y[:split_idx]
        Y_test = Y[split_idx:]

        self.dataset = {
            "x": X_train,
            "y": Y_train,
            "x_test": X_test,
            "y_test": Y_test,
        }        
        
        if train:
            self.imgs = self.dataset["x"]
            self.targets = self.dataset["y"]
        else:
            self.imgs = self.dataset["x_test"]
            self.targets = self.dataset["y_test"]

    def __len__(self):
        return len(self.targets)

    def __getitem__(self, idx):
        img, target = self.imgs[idx].astype("float32"), int(self.targets[idx])

        if self.transform is not None:
            img = self.transform(img)

        return img, target