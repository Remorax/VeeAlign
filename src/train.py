import configparser
import numpy as np
from collections import OrderedDict
from math import ceil
import torch
from torch import nn
from torch import optim
import torch.nn.functional as F
from ontology import *
from data_preprocessing import *

# Load reference alignments 
def load_alignments(folder):
    alignments = []
    for f in os.listdir(folder):
        doc = minidom.parse(folder + f)
        ls = list(zip(doc.getElementsByTagName('entity1'), doc.getElementsByTagName('entity2')))
        alignments.extend([(a.getAttribute('rdf:resource'), b.getAttribute('rdf:resource')) for (a,b) in ls])
    return alignments

# Read `config.ini` and initialize parameter values
config = configparser.ConfigParser()
config.read('config.ini')

# Initialize variables from config
alignment_folder = str(config["Paths"]["alignment_folder"])
train_folder = str(config["Paths"]["train_folder"])
model_path = str(config["Paths"]["model_path"])

USE_folder = str(config["USE Embeddings"]["USE_folder"])
spellcheck = config["Preprocessing"]["has_spellcheck"] == "True"

max_neighbours = int(config["Parameters"]["max_neighbours"])
min_neighbours = int(config["Parameters"]["min_neighbours"])

lr = float(config["Hyperparameters"]["lr"])
num_epochs = int(config["Hyperparameters"]["num_epochs"])
weight_decay = float(config["Hyperparameters"]["weight_decay"])
batch_size = int(config["Hyperparameters"]["batch_size"])


reference_alignments = load_alignments(alignment_folder)
gt_mappings = [tuple([elem.split("/")[-1] for elem in el]) for el in reference_alignments]

ontologies_in_alignment = [[train_folder + el  + ".owl" for el in l.split(".")[0].split("-")]
                           for l in os.listdir(alignment_folder)]

# Preprocessing and parsing input data for training
preprocessing = DataParser(ontologies_in_alignment, USE_folder, gt_mappings)
train_data, emb_indexer, emb_indexer_inv, emb_vals, neighbours_dicts = preprocessing.process(spellcheck)


class SiameseNetwork(nn.Module):
    # Defines the Siamese Network model
    def __init__(self):
        super().__init__() 
        self.embedding_dim = np.array(emb_vals).shape[1]
        self.n = max_neighbours
        self.v = nn.Parameter(torch.DoubleTensor([1/(self.n-1) for i in range(self.n-1)]))
        self.output = nn.Linear(2*self.embedding_dim, 300)
        self.cosine_sim_layer = nn.CosineSimilarity(dim=1)

    def masked_softmax(self, inp):
        # To softmax all non-zero tensor values
        inp = inp.double()
        mask = ((inp != 0).double() - 1) * 9999  # for -inf
        return (inp + mask).softmax(dim=-1)

    def forward(self, inputs):
        results = []
        inputs = inputs.permute(1,0,2,3)
        for i in range(2):
            x = inputs[i]
            node = x.permute(1,0,2)[:1].permute(1,0,2) # batch_size * 1 * 512
            neighbours = x.permute(1,0,2)[1:].permute(1,0,2) # batch_size * max_neighbours * 512
            
            att_weights = torch.bmm(neighbours, node.permute(0, 2, 1)).squeeze()
            att_weights = self.masked_softmax(att_weights).unsqueeze(-1)
            context = torch.matmul(self.v, att_weights * neighbours)

            x = torch.cat((node.reshape(-1, self.embedding_dim), context.reshape(-1, self.embedding_dim)), dim=1)
            x = self.output(x)
            results.append(x)
        x = self.cosine_sim_layer(results[0], results[1])
        return x

def generate_data(elem_tuple):
    return np.array([[emb_indexer[el] for el in neighbours_dicts[elem.split("#")[0]][elem]] for elem in elem_tuple])

def generate_input(elems, target):
    inputs, targets = [], []
    print ("Generating input data to model...")
    for elem in list(elems):
        try:
            inputs.append(generate_data(elem))
            targets.append(target)
        except Exception as e:
            continue
    return np.array(inputs), np.array(targets)

def count_non_unk(elem):
    return len([l for l in elem if l!="<UNK>"])

def embed(inputs):
    return np.array([emb_vals[idx] for idx in inputs])

neighbours_dicts = {ont: {el: neighbours_dicts[ont][el][:max_neighbours] for el in neighbours_dicts[ont]
       if count_non_unk(neighbours_dicts[ont][el]) > min_neighbours} for ont in neighbours_dicts}

data_items = train_data.items()
np.random.shuffle(list(data_items))
train_data = OrderedDict(data_items)

print ("Number of entities:", len(train_data))

torch.set_default_dtype(torch.float64)

train_data_t = [key for key in train_data if train_data[key]]
train_data_f = [key for key in train_data if not train_data[key]]
train_data_t = np.repeat(train_data_t, ceil(len(train_data_f)/len(train_data_t)), axis=0)
train_data_t = train_data_t[:len(train_data_f)].tolist()
np.random.shuffle(train_data_f)

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

model = SiameseNetwork().to(device)

optimizer = optim.Adam(model.parameters(), lr=lr, weight_decay=weight_decay)

print ("Starting training...")

for epoch in range(num_epochs):
    inputs_pos, targets_pos = generate_input(train_data_t, 1)
    inputs_neg, targets_neg = generate_input(train_data_f, 0)
    inputs_all = list(inputs_pos) + list(inputs_neg)
    targets_all = list(targets_pos) + list(targets_neg)
    
    indices_all = np.random.permutation(len(inputs_all))
    inputs_all = np.array(inputs_all)[indices_all]
    targets_all = np.array(targets_all)[indices_all]

    batch_size = min(batch_size, len(inputs_all))
    num_batches = int(ceil(len(inputs_all)/batch_size))

    for batch_idx in range(num_batches):
        batch_start = batch_idx * batch_size
        batch_end = (batch_idx+1) * batch_size
        
        inputs = inputs_all[batch_start: batch_end]
        targets = targets_all[batch_start: batch_end]
        inputs = np.apply_along_axis(embed, 2, inputs)
        
        inp_elems = torch.DoubleTensor(inputs).to(device)
        targ_elems = torch.DoubleTensor(targets).to(device)
        optimizer.zero_grad()
        outputs = model(inp_elems)
        loss = F.mse_loss(outputs, targ_elems)
        loss.backward()
        optimizer.step()

        if batch_idx%1000 == 0:
            print ("Epoch: {} Idx: {} Loss: {}".format(epoch, batch_idx, loss.item()))

print ("Training complete!")
torch.save(model.state_dict(), model_path)
