import pytorch_lightning as pl
import torch.nn.functional as F
from initialize import *
from pytorch_lightning import Trainer
import sys
from torch import nn
from pytorch_lightning.callbacks import ModelCheckpoint
print(sys.executable)
from pytorch_lightning.strategies import DDPStrategy
from scipy.special import softmax
from pytorch_lightning.loggers import TensorBoardLogger


class TextGenerationModel(pl.LightningModule):
    def __init__(self, hid_dim, num_layers):
        super().__init__() 
        input_dim = output_dim = self.data_setup()
        self.rnn = nn.LSTM(input_size=input_dim, hidden_size=hid_dim, num_layers=num_layers)
        self.dense = nn.Linear(hid_dim, output_dim)
        self.rnn.apply(self.init_weights)
        self.dense.apply(self.init_weights)
        self.save_hyperparameters()

    def init_weights(self, m):
        for param in m._parameters.keys():
            if 'weight' in param:
                torch.nn.init.orthogonal_(m._parameters[param])
            elif 'bias' in param:
                m._parameters[param].data.fill_(0.00)


    def data_setup(self):
        data_name = 'wiki'
        with open(f'data/{data_name}/{data_name}.txt', encoding='utf-8') as f:
            self.text = f.read()

        self.chars = sorted(list(set(self.text)))
        self.char_indices = {c: i for i, c in enumerate(self.chars)}
        self.indices_char = {i: c for i, c in enumerate(self.chars)}
        self.maxlen = 100
        step = 5
        sentences = []
        next_chars = []
        for i in range(0, len(self.text) - self.maxlen, step):
            sentences.append(self.text[i: i + self.maxlen])
            next_chars.append(self.text[i + self.maxlen])
        x = np.zeros((len(sentences), self.maxlen, len(self.chars)))
        y = np.zeros(len(sentences))
        for i, sentence in enumerate(sentences):
            for t, char in enumerate(sentence):
                x[i, t, self.char_indices[char]] = 1
            y[i] = self.char_indices[next_chars[i]]

        # import pickle
        # with open(f'data/{data_name}/x.pkl', 'rb') as f:
        #     x = pickle.load(f)
        # with open(f'data/{data_name}/y.pkl', 'rb') as f:
        #     y = pickle.load(f)
            
        self.dataset = TensorDataset(torch.tensor(x, dtype=torch.float32), torch.LongTensor(y))[:1000]
        return x.shape[-1]

    def forward(self, x):
        #src = [batch size, input len, input dim]
        y = self.rnn(x)[0][:,-1,:]
        y = self.dense(y)
        output = y
        return output

    def configure_optimizers(self):
        optimizer = torch.optim.RMSprop(self.parameters(), lr=1e-2)
        return [optimizer]

    def training_step(self, batch, batch_idx):
        x, y = batch
        y_hat = self(x)
        loss = nn.CrossEntropyLoss()(y_hat, y)
        self.log("train_loss", loss, on_step=False, on_epoch=True, prog_bar=True, logger=True)
        return loss

    def predict(self, sentence=None, start_index=None, length=100, diversity = 0.5):
        def sample(preds, temperature=1.0):
            preds = np.asarray(preds).astype('float64')
            preds = np.log(preds) / temperature
            exp_preds = np.exp(preds)
            preds = exp_preds / np.sum(exp_preds)
            probas = np.random.multinomial(1, preds, 1)
            return np.argmax(probas)
        generated = ''
        if start_index is None:
            start_index = np.random.randint(0, len(self.text)-self.maxlen)
        if sentence is None:
            sentence = self.text[start_index: start_index + self.maxlen]
        else:
            assert len(sentence) > self.maxlen, f'Need at least {self.maxlen} characters to start'
            sentence = sentence[:self.maxlen]

        generated += sentence
        for i in range(length):
            x_pred = np.zeros((1, len(sentence), len(self.chars)))
            for t, char in enumerate(sentence):
                x_pred[0, t, self.char_indices[char]] = 1.

            preds = self(torch.tensor(x_pred, dtype=torch.float32))[0].detach().cpu().numpy()
            preds = softmax(preds)
            next_index = sample(preds, diversity)

            
            next_char = self.indices_char[next_index]
            sentence = sentence[1:] + next_char
            generated += next_char
        return generated


    def train_dataloader(self):
        return torch.utils.data.DataLoader(self.dataset, batch_size=128,drop_last=True, num_workers=18, pin_memory=True)


if __name__ == "__main__":
    model = TextGenerationModel(hid_dim=128, num_layers=1)

    # model = TextGenerationModel.load_from_checkpoint('runs/text_generation/version_0/checkpoints/epoch=29-step=21780.ckpt')



    # trainer = Trainer(accelerator="gpu", devices=4, strategy=DDPStrategy(find_unused_parameters=False),
    #                 max_epochs=300,
    #                 precision=32,
    #                 logger=TensorBoardLogger("runs", name="text_generation"))

    trainer = Trainer(accelerator="gpu", devices=1,
                max_epochs=300,
                precision=32,
                logger=TensorBoardLogger("runs", name="text_generation"))

    trainer.fit(model=model)
