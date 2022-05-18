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


# class ShiftSequenceModel(pl.LightningModule):
#     def __init__(self, hid_dim, num_layers):
#         super().__init__() 
#         input_dim = output_dim = 1
#         self.rnn = nn.LSTM(input_size=input_dim, hidden_size=hid_dim, num_layers=num_layers)
#         self.dense = nn.Linear(hid_dim, output_dim)
#         self.save_hyperparameters()


#     def forward(self, x):
#         y = self.rnn(x)[0]
#         y = self.dense(y)
#         output = y
#         return output

    
#     def configure_optimizers(self):
#         optimizer = torch.optim.Adam(self.parameters(), lr=1e-3)
#         return [optimizer]

#     def training_step(self, batch, batch_idx):
#         x, y = batch
#         y_hat = self(x)
#         trainloss = nn.MSELoss()(y_hat, y)
#         self.log("train_loss", trainloss, on_epoch=True, prog_bar=True, logger=True)
#         return trainloss

#     def validation_step(self, batch, batch_idx):
#         x, y = batch
#         y_hat = self(x)
#         validloss = nn.MSELoss()(y_hat, y)
#         self.log("valid_loss", validloss, prog_bar=True, logger=True)
#         return validloss

#     def predict(self, x):
#         x = torch.tensor(x, dtype=torch.float32).unsqueeze(-1)
#         pred = self(x)
#         return pred.squeeze(-1).detach().cpu().numpy()


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
        self.save_hyperparameters()

    def forward(self, x):
        y = self.rnn(x)[0]
        y = nn.Tanh()(self.dense(y))
        output = y
        return output





if __name__ == "__main__":
    # model = TextGeneration(hid_dim=256, num_layers=1)
    

    # model = TextGeneration.load_from_checkpoint('checkpoints/Text-epoch=113-train__loss_epoch=1.74.ckpt')



    # # trainer = Trainer(accelerator="gpu", devices=4, strategy=DDPStrategy(find_unused_parameters=False),
    # #                 max_epochs=300,
    # #                 precision=32,
    # #                 logger=TensorBoardLogger("runs", name="text_generation"))
    # checkpoint_callback = ModelCheckpoint(dirpath="checkpoints", save_top_k=4, monitor="train__loss_epoch",filename="Text-{epoch:02d}-{train__loss_epoch:.2f}")
    # trainer = Trainer(accelerator="gpu", devices=1,
    #             max_epochs=300,
    #             precision=32,
    #             logger=TensorBoardLogger("runs", name="text_generation"),
    #             callbacks=[checkpoint_callback])
    # # print(trainer.callback_metrics)
    # trainer.fit(model=model)




















    # model = LorentzModel(hid_dim=256, num_layers=2)
    model = RNNModel(hid_dim=256, num_layers=2, input_dim=1, output_dim=1)
    # train_dataset = ShiftDataset(size=CONFIG.train_size, seq_len=100, shift=20)
    # valid_dataset = ShiftDataset(size=CONFIG.valid_size, seq_len=100, shift=20)




    import pickle
    with open('resources/data/shift/shift_8_32.pkl', 'rb') as f:
        input, output = pickle.load( f)
    input = input[:CONFIG.train_size+CONFIG.valid_size]
    output = output[:CONFIG.train_size+CONFIG.valid_size]

    input = torch.tensor(input, dtype=torch.float32)
    output = torch.tensor(output, dtype=torch.float32)

    dataset = torch.utils.data.TensorDataset(input, output)
    train_dataset, valid_dataset = torch.utils.data.random_split(dataset, [CONFIG.train_size, CONFIG.valid_size] )


    train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=128,drop_last=False, num_workers=os.cpu_count(), pin_memory=True)
    valid_loader = torch.utils.data.DataLoader(valid_dataset, batch_size=128,drop_last=False, num_workers=os.cpu_count(), pin_memory=True)
    lr_monitor = LearningRateMonitor(logging_interval='epoch')

    checkpoint_callback = ModelCheckpoint(dirpath="checkpoints", save_top_k=4, monitor="valid_loss",filename="ShiftRNN-{epoch:02d}-{valid_loss:.2e}")
    trainer = Trainer(accelerator="gpu", devices=1,
                max_epochs=1000,
                precision=32,
                logger=TensorBoardLogger("runs", name="shift_seq_rnn"),
                callbacks=[checkpoint_callback, lr_monitor])
    trainer.fit(model=model, train_dataloaders=train_loader, val_dataloaders=valid_loader)

