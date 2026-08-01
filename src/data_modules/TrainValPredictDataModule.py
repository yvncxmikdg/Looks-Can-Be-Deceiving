import torch
from pytorch_lightning import LightningDataModule
from torch.utils.data import DataLoader

from modeling.Sample import collate_fn

class TrainValPredictDataModule(LightningDataModule):
    def __init__(self,
                 train_dataset=None,
                 valid_dataset=None,
                 predict_dataset=None,
                 num_workers=1,
                 train_batch_size=1,
                 valid_batch_size=1,
                 predict_batch_size=1,
                 collate_func=None):

        self.__train_dataset = train_dataset
        self.__valid_dataset = valid_dataset
        self.__predict_dataset = predict_dataset
        self.__num_workers = num_workers
        self.__train_batch_size = train_batch_size
        self.__valid_batch_size = valid_batch_size
        self.__predict_batch_size = predict_batch_size
        if collate_func is None:
            self.__collate_func = collate_fn
        else:
            self.__collate_func = collate_func

        super().__init__()

    def train_dataset(self):
        return self.__train_dataset

    def val_dataset(self):
        return self.__valid_dataset

    def predict_dataset(self):
        return self.__predict_dataset

    def train_dataloader(self) -> DataLoader:
        return torch.utils.data.DataLoader(
            self.train_dataset(),
            batch_size=self.__train_batch_size,
            shuffle=True,
            pin_memory=True,
            drop_last=True,
            num_workers=self.__num_workers,
            persistent_workers=self.__num_workers > 0,
            collate_fn=self.__collate_func
        )

    def val_dataloader(self) -> DataLoader:
        return torch.utils.data.DataLoader(
            self.val_dataset(),
            batch_size=self.__valid_batch_size,
            shuffle=False,
            pin_memory=True,
            drop_last=False,
            num_workers=self.__num_workers,
            persistent_workers=self.__num_workers > 0,
            collate_fn=self.__collate_func,
        )

    def predict_dataloader(self) -> DataLoader:
        return torch.utils.data.DataLoader(
            self.predict_dataset(),
            batch_size=self.__predict_batch_size,
            shuffle=False,
            pin_memory=True,
            drop_last=False,
            num_workers=self.__num_workers,
            persistent_workers=self.__num_workers > 0,
            collate_fn=self.__collate_func,
        )
