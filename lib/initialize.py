import imp
import torch
from torch import nn
import numpy as np
import random
from lib.layers import Module
from lib.seq2seq_model import RNNModel
from lib.text_generating_model import TextGeneration, WordGeneration
from lib.plot import *
from lib.utils import Dataset,train_model, inference
from math import exp
from lib.lfgenerator import Shift, LorenzRandFGenerator
from lib.plot import LorentzEvaluation
from torch.utils.data import TensorDataset

SEED = 1234
DTYPE = torch.float32
random.seed(SEED)
np.random.seed(SEED)
torch.manual_seed(SEED)
torch.cuda.manual_seed(SEED)
torch.backends.cudnn.benchmark=True

# DataPlotter