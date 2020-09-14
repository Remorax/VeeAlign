import configparser, logging
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
ontologies_in_alignment = []
def load_alignments(folder):
    global ontologies_in_alignment
    alignments = []
    for f in os.listdir(folder):
        doc = minidom.parse(folder + f)
        ls = list(zip(doc.getElementsByTagName('entity1'), doc.getElementsByTagName('entity2')))
        src = train_folder + doc.getElementsByTagName('Ontology')[0].getAttribute("rdf:about").split("/")[-1].split(".")[0] + ".owl"
        targ = train_folder + doc.getElementsByTagName('Ontology')[1].getAttribute("rdf:about").split("/")[-1].split(".")[0] + ".owl"
        ontologies_in_alignment.append((src, targ))
        alignments.extend([(a.getAttribute('rdf:resource'), b.getAttribute('rdf:resource')) for (a,b) in ls])
    return alignments

# Read `config.ini` and initialize parameter values
config = configparser.ConfigParser()
config.read('config.ini')

prefix_path = "/".join(os.path.dirname(os.path.abspath(__file__)).split("/")[:-1])

print ("Prefix path: ", prefix_path)

# Initialize variables from config

language = str(config["General"]["Language"])
K = str(config["General"]["K"])


alignment_folder = prefix_path + str(config["Paths"]["alignment_folder"])
train_folder = prefix_path + str(config["Paths"]["train_folder"])
model_path = prefix_path + str(config["Paths"]["model_path"])

spellcheck = config["Preprocessing"]["has_spellcheck"] == "True"

max_neighbours = int(config["Parameters"]["max_paths"])
min_neighbours = int(config["Parameters"]["max_pathlen"])
bag_of_neighbours = int(config["Parameters"]["bag_of_neighbours"])
weighted_average = int(config["Parameters"]["weighted_average"])

lr = float(config["Hyperparameters"]["lr"])
num_epochs = int(config["Hyperparameters"]["num_epochs"])
weight_decay = float(config["Hyperparameters"]["weight_decay"])
batch_size = int(config["Hyperparameters"]["batch_size"])

reference_alignments = load_alignments(alignment_folder)
gt_mappings = [tuple([elem.split("/")[-1] for elem in el]) for el in reference_alignments]
print ("Ontologies being aligned are: ", ontologies_in_alignment)

# Preprocessing and parsing input data for training
preprocessing = DataParser(ontologies_in_alignment, language, gt_mappings)
train_data, emb_indexer, emb_indexer_inv, emb_vals, neighbours_dicts, max_paths, max_pathlen, max_types = preprocessing.process(spellcheck, bag_of_neighbours)


class SiameseNetwork(nn.Module):
    # Defines the Siamese Network model
    def __init__(self, threshold=0.9):
        super().__init__() 
        self.n_neighbours = max_types
        self.max_paths = max_paths
        self.max_pathlen = max_pathlen
        self.embedding_dim = np.array(emb_vals).shape[1]
        
        self.name_embedding = nn.Embedding(len(emb_vals), self.embedding_dim)
        self.name_embedding.load_state_dict({'weight': torch.from_numpy(np.array(emb_vals))})
        self.name_embedding.weight.requires_grad = False
        
        self.threshold = threshold
        
        self.cosine_sim_layer = nn.CosineSimilarity(dim=1)
        self.output = nn.Linear(2*self.embedding_dim, 300)
        
        self.v = nn.Parameter(torch.DoubleTensor([1/(self.max_pathlen) for i in range(self.max_pathlen)]))
        if max_types == 4:
            self.w_rootpath = nn.Parameter(torch.DoubleTensor([0.25]))
            self.w_children = nn.Parameter(torch.DoubleTensor([0.25]))
            self.w_obj_neighbours = nn.Parameter(torch.DoubleTensor([0.25]))
        elif max_types == 3:
            self.w_rootpath = nn.Parameter(torch.DoubleTensor([0.33]))
            self.w_children = nn.Parameter(torch.DoubleTensor([0.33]))
        else:
            self.w_rootpath = nn.Parameter(torch.DoubleTensor([0.5]))

    def masked_softmax(self, inp):
        # To softmax all non-zero tensor values
        inp = inp.double()
        mask = ((inp != 0).double() - 1) * 9999  # for -inf
        return (inp + mask).softmax(dim=-1)

    def forward(self, nodes, features):
        '''
        Arguments:
            - nodes: entities being compared. dim: (batch_size, 2)
            - features: demarcated neighbourhood features of these entities. 
                        dim: (batch_size, 2, max_types, max_paths, max_pathlen)
        '''
        results = []
        nodes = nodes.permute(1,0) # dim: (2, batch_size)
        features = features.permute(1,0,2,3,4) # dim: (2, batch_size, max_types, max_paths, max_pathlen)
        for i in range(2):
            node_emb = self.name_embedding(nodes[i]) # dim: (2, batch_size)
            feature_emb = self.name_embedding(features[i]) #  dim: (2, batch_size, max_types, max_paths, max_pathlen, 512)
            
            feature_emb_reshaped = feature_emb.permute(0,4,1,2,3).reshape(-1, self.embedding_dim, self.n_neighbours * self.max_paths * self.max_pathlen)
            path_weights = torch.bmm(node_emb[:, None, :], feature_emb_reshaped)
            path_weights = path_weights.squeeze(1).reshape(-1, self.n_neighbours, self.max_paths, self.max_pathlen)
            path_weights = torch.sum(path_weights, dim=-1)
            
            if weighted_average:
                # Calculate unified path representation as a weighted sum of all paths.
                path_weights = masked_softmax(path_weights)
                feature_emb_reshaped = feature_emb.reshape(-1, self.max_paths, self.max_pathlen * self.embedding_dim)
                best_path = torch.bmm(path_weights.reshape(-1, 1, self.max_paths), feature_emb_reshaped)
                best_path = best_path.squeeze(1).reshape(-1, self.n_neighbours, self.max_pathlen, self.embedding_dim)
                # best_path has dim: (batch_size, max_types, max_pathlen, 512)
            else:
                # Calculate unified path representation by applying max-pool over the attended weights
                best_path_indices = torch.max(path_weights, dim=-1)[1][(..., ) + (None, ) * 3]
                best_path_indices = best_path_indices.expand(-1, -1, -1, self.max_pathlen,  self.embedding_dim)
                best_path = torch.gather(feature_emb, 2, best_path_indices).squeeze(2)
                # best_path has dim: (batch_size, max_types, max_pathlen, 512)

            best_path_reshaped = best_path.permute(0,3,1,2).reshape(-1, self.embedding_dim, self.n_neighbours * self.max_pathlen)
            node_weights = torch.bmm(node_emb.unsqueeze(1), best_path_reshaped) # dim: (batch_size, 4, max_pathlen)
            node_weights = masked_softmax(node_weights.squeeze(1).reshape(-1, self.n_neighbours, self.max_pathlen)) # dim: (batch_size, 4, max_pathlen)
            attended_path = node_weights.unsqueeze(-1) * best_path # dim: (batch_size, 4, max_pathlen, 512)

            distance_weighted_path = torch.sum((self.v[None,None,:,None] * attended_path), dim=2) # batch_size * 4 * 512
            
            if self.n_neighbours == 4:
                self.w_data_neighbours = (1-self.w_rootpath-self.w_children-self.w_obj_neighbours)
                context_emb = self.w_rootpath * distance_weighted_path[:,0,:] \
                        + self.w_children * distance_weighted_path[:,1,:] \
                        + self.w_obj_neighbours * distance_weighted_path[:,2,:] \
                        + self.w_data_neighbours * distance_weighted_path[:,3,:]
            elif self.n_neighbours == 3:
                self.w_obj_neighbours = (1-self.w_rootpath-self.w_children)
                context_emb = self.w_rootpath * distance_weighted_path[:,0,:] \
                        + self.w_children * distance_weighted_path[:,1,:] \
                        + self.w_obj_neighbours * distance_weighted_path[:,2,:]
            else:
                self.w_children = (1-self.w_rootpath)
                context_emb = self.w_rootpath * distance_weighted_path[:,0,:] \
                        + self.w_children * distance_weighted_path[:,1,:]

            contextual_node_emb = torch.cat((node_emb, context_emb), dim=1)
            output_node_emb = self.output(contextual_node_emb)
            results.append(output_node_emb)
        sim = self.cosine_sim_layer(results[0], results[1])
        return sim

def is_valid(test_onto, key):
    return tuple([el.split("#")[0] for el in key]) not in test_onto

def generate_data_neighbourless(elem_tuple):
    return [emb_indexer[elem] for elem in elem_tuple]

def embedify(seq):
    for item in seq:
        if isinstance(item, list):
            yield list(embedify(item))
        else:
            yield emb_indexer[item]

def generate_data(elem_tuple):
    return list(embedify([neighbours_dicts[elem] for elem in elem_tuple]))

def to_feature(inputs):
    inputs_lenpadded = [[[[path[:max_pathlen] + [0 for i in range(max_pathlen -len(path[:max_pathlen]))]
                                    for path in nbr_type[:max_paths]]
                                for nbr_type in ent[:max_types]]
                            for ent in elem]
                        for elem in inputs]
    inputs_pathpadded = [[[nbr_type + [[0 for j in range(max_pathlen)]
                             for i in range(max_paths - len(nbr_type))]
                            for nbr_type in ent] for ent in elem]
                        for elem in inputs_lenpadded]
    return inputs_pathpadded

def generate_input(elems, target):
    inputs, targets, nodes = [], [], []
    global direct_inputs, direct_targets
    for elem in list(elems):
        try:
            inputs.append(generate_data(elem))
            nodes.append(generate_data_neighbourless(elem))
            targets.append(target)
        except:
            direct_inputs.append(generate_data_neighbourless(elem))
            direct_targets.append(target)
    return inputs, nodes, targets

def count_non_unk(elem):
    return len([l for l in elem if l!="<UNK>"])

torch.set_default_dtype(torch.float64)

torch.manual_seed(0)
np.random.seed(0)

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
