from per_class_augmentation import model
from per_class_augmentation.data import ImageNetDataModule
from torch.utils.data import DataLoader
import pytorch_lightning as pl


DATA_DIR = "/datasets01/imagenet_full_size/061417"


def test_data():
    data = ImageNetDataModule(data_dir=DATA_DIR)
    assert isinstance(data.train_dataloader(), DataLoader)


def test_model():
    # runs 1 batch for train and val
    trainer = pl.Trainer(fast_dev_run=True)
    data = ImageNetDataModule(data_dir=DATA_DIR)
    trainer.fit(model.ResNet18Classifier(), data)
