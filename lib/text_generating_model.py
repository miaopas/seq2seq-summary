import pytorch_lightning as pl
import torch.nn.functional as F
import torch
from pytorch_lightning import Trainer
from torch import nn
from pytorch_lightning.callbacks import ModelCheckpoint
from pytorch_lightning.strategies import DDPStrategy
from scipy.special import softmax
from torch.utils.data import TensorDataset
from pytorch_lightning.loggers import TensorBoardLogger
import numpy as np

import os

class TextGeneration(pl.LightningModule):
    def __init__(self, hid_dim, num_layers, load_data=False):
        super().__init__() 
        self.load_data = load_data
        input_dim = output_dim = self.data_setup()
        self.rnn = nn.LSTM(input_size=input_dim, hidden_size=hid_dim, num_layers=num_layers)
        self.dense = nn.Linear(hid_dim, output_dim)
        self.save_hyperparameters()

    def data_setup(self):
        data_name = 'wiki'
        with open(f'resources/data/text/{data_name}.txt', encoding='utf-8') as f:
            self.text = f.read().lower()

        self.chars = sorted(list(set(self.text)))
        self.char_indices = {c: i for i, c in enumerate(self.chars)}
        self.indices_char = {i: c for i, c in enumerate(self.chars)}

        if self.data_setup:
            self.maxlen = 100
            step = 5
            sentences = []
            next_chars = []
            for i in range(0, len(self.text) - self.maxlen, step):
                sentences.append(self.text[i: i + self.maxlen])
                next_chars.append(self.text[i + self.maxlen])

            x = np.zeros((len(sentences), self.maxlen, len(self.chars)), dtype=bool)
            y = np.zeros(len(sentences))
            for i, sentence in enumerate(sentences):
                for t, char in enumerate(sentence):
                    x[i, t, self.char_indices[char]] = 1
                y[i] = self.char_indices[next_chars[i]]

            self.dataset = TensorDataset(torch.tensor(x, dtype=torch.float32), torch.LongTensor(y))
        return len(self.chars)

    def forward(self, x):
        y = self.rnn(x)[0][:,-1,:]
        y = self.dense(y)
        output = y
        return output

    def configure_optimizers(self):
        optimizer = torch.optim.SGD(self.parameters(), lr=1e-5)
        return [optimizer]

    def training_step(self, batch, batch_idx):
        x, y = batch
        y_hat = self(x)
        loss = nn.CrossEntropyLoss()(y_hat, y)
        self.log("train__loss", loss, on_epoch=True, prog_bar=True, logger=True)
        return loss

    def predict(self, sentence=None, start_index=None, length=400, diversity = 0.5):
        def sample(preds, temperature=1.0):
            preds = np.asarray(preds).astype('float64')
            preds = np.log(preds) / temperature
            exp_preds = np.exp(preds)
            preds = exp_preds / np.sum(exp_preds)
            probas = np.random.multinomial(1, preds, 1)
            return np.argmax(probas)
        
        generated = ''
        if start_index is None and sentence is None:
            start_index = np.random.randint(0, len(self.text)-self.maxlen)
        if sentence is None:
            sentence = self.text[start_index: start_index + self.maxlen]
        sentence=sentence.lower()
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
        return torch.utils.data.DataLoader(self.dataset, batch_size=128,drop_last=True, num_workers=os.cpu_count(), pin_memory=True)


class WordGeneration(TextGeneration):
    def __init__(self, hid_dim, num_layers, load_data=True):
        super().__init__(hid_dim, num_layers, load_data)

    def data_setup(self):
        from collections import Counter, defaultdict
        import numpy as np
        data_name = 'wiki'
        with open(f'resources/data/text/{data_name}.txt', encoding='utf-8') as f:
            text = f.read().lower()
        text = list(filter(lambda x: x!='', text.split(' ')))
        self.chars = {word for word, count in Counter(text).items() if count >=80}
        # print(f'Have {len(self.chars)} of words')
        self.char_indices = defaultdict(lambda:0, {c: i for i, c in enumerate(sorted(self.chars),1)})
        self.indices_char = defaultdict(lambda:'', {c: i for i, c in self.char_indices.items()})
        maxlen = 100
        step = 5
        sentences = []
        next_word = []

        if self.load_data:
            for i in range(0, len(text) - maxlen, step):
                sentences.append(text[i: i + maxlen])
                next_word.append(text[i + maxlen])
            x = np.zeros((len(sentences)+1, maxlen+1, len(self.chars)+1),dtype=bool)
            y = np.zeros(len(sentences)+1)
            for i, sentence in enumerate(sentences):
                for t, word in enumerate(sentence):
                    x[i, t, self.char_indices[word]] = 1
                y[i] = self.char_indices[next_word[i]]
            
            self.dataset = TensorDataset(torch.tensor(x, dtype=torch.float32), torch.LongTensor(y))
        return len(self.chars)+1

    def predict(self, sentence=None, length=100, diversity = 0.5):
        def sample(preds, temperature=1.0):
            preds = np.asarray(preds).astype('float64')
            preds = np.log(preds) / temperature
            exp_preds = np.exp(preds)
            preds = exp_preds / np.sum(exp_preds)
            probas = np.random.multinomial(1, preds, 1)
            return np.argmax(probas)
        
        generated = ''
        sentence=sentence.lower()
        generated  = generated + sentence +' '
        for i in range(length):
            x_pred = np.zeros((1, len(sentence), len(self.chars)+1))
            for t, char in enumerate(sentence.split(' ')):
                x_pred[0, t, self.char_indices[char]] = 1.

            preds = self(torch.tensor(x_pred, dtype=torch.float32))[0].detach().cpu().numpy()
            preds = softmax(preds)
            next_index = sample(preds, diversity)

            
            next_char = self.indices_char[next_index]
            sentence = sentence[1:] + next_char
            generated  = generated + next_char +' '
        return generated







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



    model = WordLevelTextGeneration(hid_dim=256, num_layers=1)

    checkpoint_callback = ModelCheckpoint(dirpath="checkpoints", save_top_k=4, monitor="train__loss_epoch",filename="TextWord-{epoch:02d}-{train__loss_epoch:.2f}")
    trainer = Trainer(accelerator="gpu", devices=1,
                max_epochs=300,
                precision=32,
                logger=TensorBoardLogger("runs", name="text_generation"),
                callbacks=[checkpoint_callback])
    # print(trainer.callback_metrics)
    trainer.fit(model=model)

