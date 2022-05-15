from pydoc import describe
import torch
import torch.nn as nn
from torch.utils.tensorboard import SummaryWriter
import time
import numpy as np
import pathlib
from tqdm import tqdm
from lib.model import Encoder
from math import exp


class Dataset(torch.utils.data.Dataset):
    '''
        Helper class to put numpy arrays into tensor datasets
    '''
    def __init__(self,X,Y, dtype, device):
        self.X = torch.tensor(X, dtype=dtype, device=device)                           # set data
        self.Y = torch.tensor(Y, dtype=dtype, device=device)                              # set lables

    def __len__(self):
        return len(self.X)                   # return length

    def __getitem__(self, idx):
        return (self.X[idx], self.Y[idx])

def train_model(name, model, train_data, test_data, epochs=1000):
    LEARNING_RATE = 0.001

    optimizer = torch.optim.Adam(model.parameters(), lr = LEARNING_RATE)

    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer,verbose=False, cooldown=100)

    criterion = nn.MSELoss()

    def epoch_time(start_time, end_time):
        elapsed_time = end_time - start_time
        elapsed_mins = int(elapsed_time / 60)
        elapsed_secs = int(elapsed_time - (elapsed_mins * 60))
        return elapsed_mins, elapsed_secs


    writer = SummaryWriter(f'runs/{name}')
    
    CLIP = 1
    best_valid_loss = float('inf')
    best_train_loss = float('inf')
    valid_loss = float('inf')
    # teacher_forcing = 0.5

    save_every = 300
    start_eopch = 0
    pathlib.Path(f"saved_model/{name}").mkdir(parents=True, exist_ok=True)

    epoch_iterator = tqdm(range(start_eopch, epochs),bar_format='{desc}{r_bar}')
    # writer.add_graph(model, next(iter(train_data))[0])
    for epoch in epoch_iterator:

        start_time = time.time()

        train_loss = train(model, train_data, optimizer, criterion)
        valid_loss = inference( model, test_data)

        train_loss = np.mean(train_loss)
        valid_loss = np.mean(valid_loss)

        writer.add_scalar("train/loss", scalar_value=train_loss, global_step=epoch)
        writer.add_scalar("test/loss", scalar_value=valid_loss, global_step=epoch)

        end_time = time.time()

        epoch_mins, epoch_secs = epoch_time(start_time, end_time)

        if valid_loss < best_valid_loss:
            best_valid_loss = valid_loss
            torch.save(model.state_dict(), f'saved_model/{name}/best_valid.pt')

        if train_loss < best_train_loss:
            best_train_loss = train_loss
            torch.save(model.state_dict(), f'saved_model/{name}/best_train.pt')

        if epoch % save_every == 0:
            torch.save(model.state_dict(), f'saved_model/{name}/saved_model.pt')

            
        scheduler.step(train_loss)
        lr = optimizer.param_groups[0]['lr']
        epoch_iterator.set_description(f"Train Loss: {train_loss:.3e} \t Val. Loss: {valid_loss:.3e} \t Best Loss: {best_train_loss:.3e} \t Current lr: {lr:.3e}")
        # print(f'Epoch: {epoch+1:02} | Time: {epoch_mins}m {epoch_secs}s || Best Loss: {best_train_loss:.3e} || Current lr: {lr:.3e}')
        # print(f'\tTrain Loss: {train_loss:.3e}')
        # print(f'\t Val. Loss: {valid_loss:.3e}')


def train(model, dataset, optimizer, criterion):
    train_loss = []
    model.train()


    # iterate through batches
    for data in dataset:
        # Shape of _input : [batch, input_length, feature]
        # Desired input for model: [input_length, batch, feature]
        
        optimizer.zero_grad()
        src = data[0] #[train len, batch, feature]
        target = data[1]

        if isinstance(model, Encoder):
            pred, _ = model(src)
        else:
            pred = model(src)

        loss = criterion(pred, target)



        loss.backward()
        optimizer.step()
        train_loss.append(loss.detach().item())

    
    return train_loss


def inference( model, dataset):
    criterion = torch.nn.MSELoss()


    val_loss = []
    with torch.no_grad():

        for data in dataset:
            model.eval()
            src = data[0]#[train len, batch, feature]
            target = data[1]

            if isinstance(model, Encoder):
                pred, _ = model(src)
            else:
                pred = model(src)

            loss = criterion(pred, target)

            val_loss.append(loss.detach().item())

    return val_loss