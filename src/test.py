import configparser, sys
import numpy as np
from collections import OrderedDict
from math import ceil
import torch
from torch import nn
from torch import optim
import torch.nn.functional as F
from ontology import *
from data_preprocessing import *

ont_name1, ont_name2 = sys.argv[1], sys.argv[2]

# Read `config.ini` and initialize parameter values
config = configparser.ConfigParser()
config.read('config.ini')

# Initialize variables from config
model_path = str(config["Paths"]["model_path"])
USE_folder = str(config["USE Embeddings"]["USE_folder"])
spellcheck = config["Preprocessing"]["has_spellcheck"] == "True"

