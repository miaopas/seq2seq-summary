from ast import Constant
import pytorch_lightning as pl
import torch.nn.functional as F
from pytorch_lightning import Trainer
import torch
from torch import nn
from pytorch_lightning.callbacks import ModelCheckpoint, LearningRateMonitor
# print(sys.executable)
from pytorch_lightning.strategies import DDPStrategy
from scipy.special import softmax
from pytorch_lightning.loggers import TensorBoardLogger
import os
from lib.lfgenerator import LorenzRandFGenerator
from lib.tcn import TemporalConvNet
from lib.layers import ConstantPositionalEncoding

from ml_collections import FrozenConfigDict
CONFIG = FrozenConfigDict({'shift': dict(LENGTH = 100,
                                                NUM = 20,
                                                SHIFT = 30),
                            'convo': dict(LENGTH = 100,
                                                NUM = 20,
                                                FILTER = [0.002, 0.022, 0.097, 0.159, 0.097, 0.022, 0.002]),
                                'lorentz': dict(NUM = 10, 
                                                K=1, J=10, 
                                                LENGTH=32 ), 'train_size':49500, 'valid_size': 500})


class ShiftDataset(torch.utils.data.Dataset):

    def __init__(self,size, seq_len,shift, dtype=torch.float32):
        input = []
        output = []
        for _ in range(size):
            data = self._generate_gaussian(seq_len)
            input.append(data)
            output.append(np.concatenate((np.zeros(shift), data[:-shift])))

        input = np.array(input)
        output = np.array(output)
        self.X = torch.tensor(input, dtype=dtype).unsqueeze(-1)
        self.Y = torch.tensor(output, dtype=dtype).unsqueeze(-1)

    def __len__(self):
        return len(self.X)                

    def __getitem__(self, idx):
        return (self.X[idx], self.Y[idx])

    def _generate_gaussian(self, seq_length):
        def rbf_kernel(x1, x2, variance = 1):
            from math import exp
            return exp(-1 * ((x1-x2) ** 2) / (2*variance))
        def gram_matrix(xs):
            return [[rbf_kernel(x1,x2) for x2 in xs] for x1 in xs]
        xs = np.arange(seq_length)*0.1
        mean = [0 for _ in xs]
        gram = gram_matrix(xs)
        ys = np.random.multivariate_normal(mean, gram)
        return ys


class Seq2SeqModel(pl.LightningModule):
    def __init__(self):
        super().__init__() 
        
        self.save_hyperparameters()

    def forward(self, x):
        pass

    def configure_optimizers(self):
        optimizer = torch.optim.Adam(self.parameters(), lr=1e-5)
        scheduler = {
                        "scheduler": torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer), 
                        "interval": "epoch",
                        "frequency": 1,
                        "monitor": "train_loss_epoch"
                    }
        return {"optimizer": optimizer, "lr_scheduler":scheduler}

    def training_step(self, batch, batch_idx):
        x, y = batch
        y_hat = self(x)
        trainloss = nn.MSELoss()(y_hat, y)
        self.log("train_loss", trainloss, on_epoch=True, prog_bar=True, logger=True)
        return trainloss

    def validation_step(self, batch, batch_idx):
        x, y = batch
        y_hat = self(x)
        validloss = nn.MSELoss()(y_hat, y)
        self.log("valid_loss", validloss, prog_bar=True, logger=True)
        return validloss

    def predict(self, x):
        x = torch.tensor(x, dtype=torch.float32)
        pred = self(x)
        return pred.detach().cpu().numpy()


class RNNModel(Seq2SeqModel):
    def __init__(self, hid_dim, num_layers, input_dim, output_dim):
        super().__init__()
        self.rnn = nn.LSTM(input_size=input_dim, hidden_size=hid_dim, num_layers=num_layers)
        self.dense = nn.Linear(hid_dim, output_dim)

    def forward(self, x):
        y = self.rnn(x)[0]
        y = nn.Tanh()(self.dense(y))
        output = y
        return output

class TCNModel(Seq2SeqModel):
    def __init__(self, input_size, output_size, num_channels, kernel_size, dropout):
        super(TCNModel, self).__init__()
        self.tcn = TemporalConvNet(input_size, num_channels, kernel_size=kernel_size, dropout=dropout)
        self.linear = nn.Linear(num_channels[-1], output_size)
        self.init_weights()

    def init_weights(self):
        self.linear.weight.data.normal_(0, 0.01)

    def forward(self, x):
        if len(x.shape) ==2:
            x = x.unsqueeze(0)
        x = x.permute(0,2,1)
        y1 = self.tcn(x)
        y1 = y1.permute(0,2,1)
        return self.linear(y1)

class TransformerModel(Seq2SeqModel):
    def __init__(self, input_dim, output_dim, num_layers, hid_dim, nhead, src_length, dropout=0.1):
        super(TransformerModel, self).__init__()

        self.input_ff = nn.Linear(input_dim, hid_dim)
        self.output_ff =  nn.Linear(hid_dim, output_dim)
        transformerlayer = nn.TransformerEncoderLayer(d_model=hid_dim, nhead=nhead, dropout=dropout, batch_first=True)
        self.transformer = nn.TransformerEncoder(transformerlayer, num_layers=num_layers)
        self.pos_encoder = ConstantPositionalEncoding(hid_dim, max_len=src_length)
        mask = self._generate_square_subsequent_mask(src_length)
        self.register_buffer('mask', mask)

    def forward(self, x):
        x = self.input_ff(x)
        x = self.pos_encoder(x)
        y = self.transformer(x, self.mask)
        output = self.output_ff(y)
        
        return output

    def _generate_square_subsequent_mask(self, sz):
        mask = (torch.triu(torch.ones(sz, sz)) == 1).transpose(0, 1)
        mask = mask.float().masked_fill(mask == 0, float('-inf')).masked_fill(mask == 1, float(0.0))
        return mask




