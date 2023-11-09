from pathlib import Path
from typing import Any, List, Union, Optional

import scipy.io

import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset, TensorDataset

from pytorch_lightning import LightningDataModule

from einops import rearrange


class Burgers(LightningDataModule):

    file_name = 'burgers_data_R10.mat'

    def __init__(self, data_dir, ntrain=1000, ntest=100, subsampling_rate=1,
                 batch_size=32, shuffle=False, pin_memory=False, drop_last=False):
        super().__init__()
        self.data_dir = Path(data_dir).expanduser()
        self.ntrain = ntrain
        self.ntest = ntest
        self.subsampling_rate = subsampling_rate
        self.batch_size = batch_size
        self.shuffle = shuffle
        self.pin_memory = pin_memory
        self.drop_last = drop_last

    def prepare_data(self):
        if not (self.data_dir / self.file_name).is_file():
            raise FileNotFoundError(f"Data file was not found in {self.data_dir / self.file_name}")

    def setup(self, stage: Optional[str] = None) -> None:
        if stage == 'test' and hasattr(self, 'dataset_test'):
            return

        data = scipy.io.loadmat(self.data_dir / self.file_name)
        x_data = torch.tensor(data['a'], dtype=torch.float)[:, ::self.subsampling_rate]
        y_data = torch.tensor(data['u'], dtype=torch.float)[:, ::self.subsampling_rate]
        x_train, y_train = x_data[:self.ntrain], y_data[:self.ntrain]
        x_test, y_test = x_data[-self.ntest:], y_data[-self.ntest:]
        self.dataset_train = TensorDataset(x_train, y_train)
        self.dataset_test = TensorDataset(x_test, y_test)
        self.dataset_val = self.dataset_test

    def train_dataloader(self, *args: Any, **kwargs: Any) -> DataLoader:
        """ The train dataloader """
        return self._data_loader(self.dataset_train, shuffle=self.shuffle)

    def val_dataloader(self, *args: Any, **kwargs: Any) -> Union[DataLoader, List[DataLoader]]:
        """ The val dataloader """
        return self._data_loader(self.dataset_val)

    def test_dataloader(self, *args: Any, **kwargs: Any) -> Union[DataLoader, List[DataLoader]]:
        """ The test dataloader """
        return self._data_loader(self.dataset_test)

    def _data_loader(self, dataset: Dataset, shuffle: bool = False) -> DataLoader:
        return DataLoader(
            dataset,
            batch_size=self.batch_size,
            shuffle=shuffle,
            drop_last=self.drop_last,
            pin_memory=self.pin_memory
        )


# normalization, pointwise gaussian
class UnitGaussianNormalizer(nn.Module):

    def __init__(self, x, eps=0.00001):
        super().__init__()
        # x could be in shape of ntrain*n or ntrain*T*n or ntrain*n*T
        self.register_buffer('mean', torch.mean(x, 0))
        self.register_buffer('std', torch.std(x, 0))
        self.eps = eps

    def encode(self, x):
        return (x - self.mean) / (self.std + self.eps)

    def decode(self, x):
        return x * (self.std + self.eps) + self.mean


class Darcy(LightningDataModule):

    file_name_train = 'piececonst_r421_N1024_smooth1.mat'
    file_name_test = 'piececonst_r421_N1024_smooth2.mat'

    def __init__(self, data_dir, ntrain=1000, ntest=100, subsampling_rate=1,
                 batch_size=32, shuffle=False, pin_memory=False, drop_last=False):
        super().__init__()
        self.data_dir = Path(data_dir).expanduser()
        self.ntrain = ntrain
        self.ntest = ntest
        self.subsampling_rate = subsampling_rate
        self.batch_size = batch_size
        self.shuffle = shuffle
        self.pin_memory = pin_memory
        self.drop_last = drop_last

    def prepare_data(self):
        if not (self.data_dir / self.file_name_train).is_file():
            raise FileNotFoundError(f"Data file was not found in {self.data_dir / self.file_name_train}")
        if not (self.data_dir / self.file_name_test).is_file():
            raise FileNotFoundError(f"Data file was not found in {self.data_dir / self.file_name_test}")

    def setup(self, stage: Optional[str] = None) -> None:
        if stage == 'test' and hasattr(self, 'dataset_test'):
            return

        data_train = scipy.io.loadmat(self.data_dir / self.file_name_train)
        data_test = scipy.io.loadmat(self.data_dir / self.file_name_test)
        rate = self.subsampling_rate
        x_train = torch.tensor(data_train['coeff'], dtype=torch.float)[:self.ntrain, ::rate, ::rate]
        y_train = torch.tensor(data_train['sol'], dtype=torch.float)[:self.ntrain, ::rate, ::rate]

        x_test = torch.tensor(data_test['coeff'], dtype=torch.float)[:self.ntest, ::rate, ::rate]
        y_test = torch.tensor(data_test['sol'], dtype=torch.float)[:self.ntest, ::rate, ::rate]

        x_normalizer = UnitGaussianNormalizer(x_train)
        x_train = x_normalizer.encode(x_train)
        x_test = x_normalizer.encode(x_test)

        self.y_normalizer = UnitGaussianNormalizer(y_train)

        self.dataset_train = TensorDataset(x_train, y_train)
        self.dataset_test = TensorDataset(x_test, y_test)
        self.dataset_val = self.dataset_test

    def train_dataloader(self, *args: Any, **kwargs: Any) -> DataLoader:
        """ The train dataloader """
        return self._data_loader(self.dataset_train, shuffle=self.shuffle)

    def val_dataloader(self, *args: Any, **kwargs: Any) -> Union[DataLoader, List[DataLoader]]:
        """ The val dataloader """
        return self._data_loader(self.dataset_val)

    def test_dataloader(self, *args: Any, **kwargs: Any) -> Union[DataLoader, List[DataLoader]]:
        """ The test dataloader """
        return self._data_loader(self.dataset_test)

    def _data_loader(self, dataset: Dataset, shuffle: bool = False) -> DataLoader:
        return DataLoader(
            dataset,
            batch_size=self.batch_size,
            shuffle=shuffle,
            drop_last=self.drop_last,
            pin_memory=self.pin_memory
        )
