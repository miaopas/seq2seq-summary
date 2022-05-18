from pytorch_lightning import Trainer
import torch
from pytorch_lightning.callbacks import ModelCheckpoint, LearningRateMonitor
# print(sys.executable)
from pytorch_lightning.strategies import DDPStrategy
from scipy.special import softmax
from pytorch_lightning.loggers import TensorBoardLogger
import os
import pickle
import torch
from lib.seq2seq_model import TCNModel, RNNModel
from math import floor

from ml_collections import FrozenConfigDict
CONFIG = FrozenConfigDict({'shift': dict(LENGTH = 100,
                                                NUM = 20,
                                                SHIFT = 30),
                            'convo': dict(LENGTH = 100,
                                                NUM = 20,
                                                FILTER = [0.002, 0.022, 0.097, 0.159, 0.097, 0.022, 0.002]),
                                'lorentz': dict(NUM = 10, 
                                                K=1, J=10, 
                                                LENGTH=32 ), 'train_size':9500, 'valid_size': 500})


def train_model(name, model, input, output, train_test_split, epochs=300, batch_size = 128):
    """_summary_

    Args:
        name (str): Name of this run
        model (Model):The model
        input (ndarray): input array
        output (ndarray): output array
        train_test_split (float): ratio of train test split
    """

    input = torch.tensor(input, dtype=torch.float32)
    output = torch.tensor(output, dtype=torch.float32)

    dataset = torch.utils.data.TensorDataset(input, output)
    total = len(dataset)
    train_size = floor(total*train_test_split)
    test_size = total - train_size

    train_dataset, valid_dataset = torch.utils.data.random_split(dataset, [train_size, test_size])
    train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=batch_size,drop_last=True, num_workers=os.cpu_count(), pin_memory=True)
    valid_loader = torch.utils.data.DataLoader(valid_dataset, batch_size=batch_size,drop_last=True, num_workers=os.cpu_count(), pin_memory=True)

    lr_monitor = LearningRateMonitor(logging_interval='epoch')
    checkpoint_callback = ModelCheckpoint(dirpath="checkpoints", save_top_k=4, monitor="valid_loss",filename=name + "-{epoch:02d}-{valid_loss:.2e}")
    trainer = Trainer(accelerator="gpu", devices=1,
                max_epochs=epochs,
                precision=32,
                logger=TensorBoardLogger("runs", name=name),
                callbacks=[checkpoint_callback, lr_monitor])
    trainer.fit(model=model, train_dataloaders=train_loader, val_dataloaders=valid_loader)



def train_TCN_shift():

    model = TCNModel(input_size=1, output_size=1,num_channels=[10]*7, kernel_size=4, dropout=0.1)

    with open('resources/data/shift/shift_32_128.pkl', 'rb') as f:
        input, output = pickle.load(f)

    train_model('Shift-TCN', model, input, output, 0.8, epochs=5000)


def train_TCN_lorentz():

    model = TCNModel(input_size=1, output_size=1,num_channels=[30]*7, kernel_size=4, dropout=0.1)

    with open('resources/data/lorentz/lorentz_1_10_128.pkl', 'rb') as f:
        input, output = pickle.load(f)

    train_model('Lorentz-TCN', model, input, output, 0.8, epochs=5000)


def train_rnn_lorentz():

    model = RNNModel(hid_dim=256, num_layers=2, input_dim=1, output_dim=1)

    with open('resources/data/lorentz/lorentz_1_10_128.pkl', 'rb') as f:
        input, output = pickle.load(f)

    train_model('Lorentz-RNN', model, input, output, 0.8, epochs=5000)



train_rnn_lorentz()