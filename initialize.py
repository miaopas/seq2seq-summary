import imp
import torch
import torch.nn as nn
import numpy as np
import random
from lib.model import Module
from lib.plot import *
from lib.utils import Dataset,train_model, inference
from math import exp
from lib.generator import Shift, LorenzRandFGenerator
from lib import *

SEED = 1234
DTYPE = torch.float64
random.seed(SEED)
np.random.seed(SEED)
torch.manual_seed(SEED)
torch.cuda.manual_seed(SEED)
torch.backends.cudnn.benchmark=True

# DataPlotter