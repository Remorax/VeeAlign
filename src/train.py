import configparser, logging, random, sys, pickle
import numpy as np
from collections import OrderedDict
from math import ceil
import torch
from torch import nn
from torch import optim
import torch.nn.functional as F
from copy import deepcopy
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
        src = train_folder + doc.getElementsByTagName('Ontology')[0].getAttribute("rdf:about").split("/")[-1].rsplit(".", 1)[0] + ".owl"
        targ = train_folder + doc.getElementsByTagName('Ontology')[1].getAttribute("rdf:about").split("/")[-1].rsplit(".", 1)[0] + ".owl"
        ontologies_in_alignment.append((src, targ))
        alignments.extend([(a.getAttribute('rdf:resource'), b.getAttribute('rdf:resource')) for (a,b) in ls])
    return alignments

prefix_path = "/".join(os.path.dirname(os.path.abspath(__file__)).split("/")[:-1]) + "/"

print ("Prefix path: ", prefix_path)

# Read `config.ini` and initialize parameter values
config = configparser.ConfigParser()
config.read(prefix_path + 'src/config.ini')

# Initialize variables from config

quick_mode = str(config["General"]["quick_mode"])
language = str(config["General"]["language"])

K = int(config["General"]["K"])
ontology_split = str(config["General"]["ontology_split"]) == "True"
max_false_examples = int(config["General"]["max_false_examples"])

alignment_folder = prefix_path + "datasets/" + str(config["General"]["dataset"]) + "/alignments/"
train_folder = prefix_path + "datasets/" + str(config["General"]["dataset"]) + "/ontologies/"
cached_embeddings_path = prefix_path + str(config["Paths"]["embedding_cache_path"])
model_path = prefix_path + str(config["Paths"]["save_model_path"])

spellcheck = config["Preprocessing"]["has_spellcheck"] == "True"

max_paths = int(config["Parameters"]["max_paths"])
max_pathlen = int(config["Parameters"]["max_pathlen"])
bag_of_neighbours = config["Parameters"]["bag_of_neighbours"] == "True"
weighted_sum = config["Parameters"]["weighted_sum"] == "True"

lr = float(config["Hyperparameters"]["lr"])
num_epochs = int(config["Hyperparameters"]["num_epochs"])
weight_decay = float(config["Hyperparameters"]["weight_decay"])
batch_size = int(config["Hyperparameters"]["batch_size"])
validation_interval = int(config["Hyperparameters"]["validation_interval"])
patience = int(config["Hyperparameters"]["patience"])

reference_alignments = load_alignments(alignment_folder)
gt_mappings = [tuple([elem.split("/")[-1] for elem in el]) for el in reference_alignments]
gt_mappings = [tuple([el.split("#")[0].rsplit(".", 1)[0] +  "#" +  el.split("#")[1] for el in tup]) for tup in gt_mappings]
print ("Ontologies being aligned are: ", ontologies_in_alignment)

# Preprocessing and parsing input data for training
preprocessing = DataParser(ontologies_in_alignment, language, gt_mappings)
data_ent, data_prop, emb_indexer_new, emb_indexer_inv_new, emb_vals_new, neighbours_dicts_ent, neighbours_dicts_prop, max_types = preprocessing.process(spellcheck, bag_of_neighbours)

if os.path.isfile(cached_embeddings_path):
    print("Found cached embeddings...")
    emb_indexer_cached, emb_indexer_inv_cached, emb_vals_cached = pickle.load(open(cached_embeddings_path, "rb"))
else:
    emb_indexer_cached, emb_indexer_inv_cached, emb_vals_cached = {}, {}, []

emb_vals, emb_indexer, emb_indexer_inv = list(emb_vals_cached), dict(emb_indexer_cached), dict(emb_indexer_inv_cached)

s = set(emb_indexer.keys())
idx = len(emb_indexer_inv)
for term in emb_indexer_new:
    if term not in s:
        emb_indexer[term] = idx
        emb_indexer_inv[idx] = term
        emb_vals.append(emb_vals_new[emb_indexer_new[term]])
        idx += 1

direct_inputs, direct_targets = [], []
threshold_results = {}

def optimize_threshold(model):
    '''
    Function to optimise threshold on validation set.
    Calculates performance metrics (precision, recall, F1-score, F2-score, F0.5-score) for a
    range of thresholds, dictated by the range of scores output by the model, with step size 
    0.001 and updates `threshold_results` which is the relevant dictionary.
    '''
    global val_data_t_ent, val_data_f_ent, threshold_results, batch_size, direct_inputs, direct_targets
    all_results = OrderedDict()
    direct_inputs, direct_targets = [], []
    with torch.no_grad():
        all_pred = []
        
        np.random.shuffle(val_data_t_ent)
        np.random.shuffle(val_data_f_ent)

        np.random.shuffle(val_data_t_prop)
        np.random.shuffle(val_data_f_prop)

        # Create two sets of inputs: one for entities and one for properties
        inputs_pos, nodes_pos, targets_pos = generate_input(val_data_t_ent, 1, neighbours_dicts_ent)
        inputs_neg, nodes_neg, targets_neg = generate_input(val_data_f_ent, 0, neighbours_dicts_ent)
        inputs_pos_prop, nodes_pos_prop, targets_pos_prop = generate_input(val_data_t_prop, 1, neighbours_dicts_prop)
        inputs_neg_prop, nodes_neg_prop, targets_neg_prop = generate_input(val_data_f_prop, 0, neighbours_dicts_prop)

        inputs_all = list(inputs_pos) + list(inputs_neg)
        targets_all = list(targets_pos) + list(targets_neg)
        nodes_all = list(nodes_pos) + list(nodes_neg)
        
        all_inp = list(zip(inputs_all, targets_all, nodes_all))
        all_inp_shuffled = random.sample(all_inp, len(all_inp))
        inputs_all, targets_all, nodes_all = list(zip(*all_inp_shuffled))

        inputs_all_prop = list(inputs_pos_prop) + list(inputs_neg_prop)
        targets_all_prop = list(targets_pos_prop) + list(targets_neg_prop)
        nodes_all_prop = list(nodes_pos_prop) + list(nodes_neg_prop)
        
        all_inp_prop = list(zip(inputs_all_prop, targets_all_prop, nodes_all_prop))
        all_inp_shuffled_prop = random.sample(all_inp_prop, len(all_inp_prop))
        if all_inp_shuffled_prop:
            inputs_all_prop, targets_all_prop, nodes_all_prop = list(zip(*all_inp_shuffled_prop))
        else:
            inputs_all_prop, targets_all_prop, nodes_all_prop = [], [], []

        if len(inputs_all_prop) == 0:
            max_prop_len = 0
        else:
            max_prop_len = np.max([[[len(elem) for elem in prop] for prop in elem_pair] 
            for elem_pair in inputs_all_prop])

        batch_size = min(batch_size, len(inputs_all))
        num_batches = int(ceil(len(inputs_all)/batch_size))
        batch_size_prop = int(ceil(len(inputs_all_prop)/num_batches))
        
        for batch_idx in range(num_batches):
            batch_start = batch_idx * batch_size
            batch_end = (batch_idx+1) * batch_size
            batch_start_prop = batch_idx * batch_size_prop
            batch_end_prop = (batch_idx+1) * batch_size_prop

            inputs = np.array(to_feature(inputs_all[batch_start: batch_end]))
            targets = np.array(targets_all[batch_start: batch_end])
            nodes = np.array(nodes_all[batch_start: batch_end])

            inputs_prop = np.array(pad_prop(inputs_all_prop[batch_start_prop: batch_end_prop], max_prop_len))
            targets_prop = np.array(targets_all_prop[batch_start_prop: batch_end_prop])
            nodes_prop = np.array(nodes_all_prop[batch_start_prop: batch_end_prop])
            
            targets = np.concatenate((targets, targets_prop), axis=0)

            inp_elems = torch.LongTensor(inputs).to(device)
            node_elems = torch.LongTensor(nodes).to(device)
            targ_elems = torch.DoubleTensor(targets).to(device)

            inp_props = torch.LongTensor(inputs_prop).to(device)
            node_props = torch.LongTensor(nodes_prop).to(device)

            # Run model on entities and properties 
            outputs = model(node_elems, inp_elems, node_props, inp_props)
            outputs = [el.item() for el in outputs]
            targets = [True if el.item() else False for el in targets]

            for idx, pred_elem in enumerate(outputs):
                if idx < len(nodes):
                    ent1 = emb_indexer_inv[nodes[idx][0]]
                    ent2 = emb_indexer_inv[nodes[idx][1]]
                else:
                    ent1 = emb_indexer_inv[nodes_prop[idx-len(nodes)][0]]
                    ent2 = emb_indexer_inv[nodes_prop[idx-len(nodes)][1]]
                if (ent1, ent2) in all_results:
                    print ("Error: ", ent1, ent2, "already present")
                all_results[(ent1, ent2)] = (pred_elem, targets[idx])
        
        direct_targets = [True if el else False for el in direct_targets]
        
        print ("Len (direct inputs): ", len(direct_inputs))
        for idx, direct_input in enumerate(direct_inputs):
            ent1 = emb_indexer_inv[direct_input[0]]
            ent2 = emb_indexer_inv[direct_input[1]]
            sim = cos_sim(emb_vals[direct_input[0]], emb_vals[direct_input[1]])
            all_results[(ent1, ent2)] = (round(sim, 3), direct_targets[idx])
        
        # Low threshold is lowest value output by model and high threshold is the highest value
        low_threshold = round(np.min([el[0] for el in all_results.values()]) - 0.02, 3)
        high_threshold = round(np.max([el[0] for el in all_results.values()]) + 0.02, 3)
        threshold = low_threshold
        step = 0.001

        if not val_data_t_prop:
            val_data_t_tot = val_data_t_ent
        else:
            val_data_t_tot = [tuple(pair) for pair in np.concatenate((val_data_t_ent, val_data_t_prop), axis=0)]
        # Iterate over every threshold with step size of 0.001 and calculate all evaluation metrics
        while threshold < high_threshold:
            threshold = round(threshold, 3)
            res = []
            for i,key in enumerate(all_results):
                if all_results[key][0] > threshold:
                    res.append(key)
            s = set(res)
            fn_list = [(key, all_results[key][0]) for key in val_data_t_tot if key not in s]
            fp_list = [(elem, all_results[elem][0]) for elem in res if not all_results[elem][1]]
            tp_list = [(elem, all_results[elem][0]) for elem in res if all_results[elem][1]]
            
            tp, fn, fp = len(tp_list), len(fn_list), len(fp_list)
            exception = False
            
            try:
                precision = tp/(tp+fp)
                recall = tp/(tp+fn)
                f1score = 2 * precision * recall / (precision + recall)
                f2score = 5 * precision * recall / (4 * precision + recall)
                f0_5score = 1.25 * precision * recall / (0.25 * precision + recall)
            except:
                exception = True
                step = 0.001
                threshold += step
                continue

            if threshold in threshold_results:
                threshold_results[threshold].append([precision, recall, f1score, f2score, f0_5score])
            else:
                threshold_results[threshold] = [[precision, recall, f1score, f2score, f0_5score]]
            threshold += step

class VeeAlign(nn.Module):
    # Defines the VeeAlign Siamese Network model
    def __init__(self, emb_vals, threshold=0.9):
        super().__init__() 
        self.n_neighbours = max_types
        self.max_paths = max_paths
        self.max_pathlen = max_pathlen
        self.embedding_dim = np.array(emb_vals).shape[1]
        
        self.threshold = threshold

        self.name_embedding = nn.Embedding(len(emb_vals), self.embedding_dim)
        self.name_embedding.load_state_dict({'weight': torch.from_numpy(np.array(emb_vals))})
        self.name_embedding.weight.requires_grad = False
        
        self.cosine_sim_layer = nn.CosineSimilarity(dim=1)
        self.output = nn.Linear(2*self.embedding_dim, 300)
        
        self.v = nn.Parameter(torch.DoubleTensor([1/(self.max_pathlen) for i in range(self.max_pathlen)]))
        if self.n_neighbours == 4:
            self.w_rootpath = nn.Parameter(torch.DoubleTensor([0.25]))
            self.w_children = nn.Parameter(torch.DoubleTensor([0.25]))
            self.w_obj_neighbours = nn.Parameter(torch.DoubleTensor([0.25]))
        elif self.n_neighbours == 3:
            self.w_rootpath = nn.Parameter(torch.DoubleTensor([0.33]))
            self.w_children = nn.Parameter(torch.DoubleTensor([0.33]))
        else:
            self.w_rootpath = nn.Parameter(torch.DoubleTensor([0.5]))

        self.prop_weight = nn.Parameter(torch.DoubleTensor([0.33]))
        self.domain_weight = nn.Parameter(torch.DoubleTensor([0.33]))

    def masked_softmax(self, inp):
        # To softmax all non-zero tensor values
        inp = inp.double()
        mask = ((inp != 0).double() - 1) * 9999  # for -inf
        return (inp + mask).softmax(dim=-1)

    def forward(self, nodes, features, prop_nodes, prop_features):
        '''
        Arguments:
            - nodes: entities being compared. dim: (batch_size, 2)
            - features: demarcated neighbourhood features of these entities. 
                        dim: (batch_size, 2, max_types, max_paths, max_pathlen)
            - prop_nodes: properties being compared. dim: (batch_size, 2)
            - prop_features: features of properties being compared. Includes 
                        domain and range of these properties. 
                        dim: (batch_size, 2, 3, max_prop_len)
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
            
            if weighted_sum:
                # Calculate unified path representation as a weighted sum of all paths.
                path_weights = self.masked_softmax(path_weights)
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
            node_weights = self.masked_softmax(node_weights.squeeze(1).reshape(-1, self.n_neighbours, self.max_pathlen)) # dim: (batch_size, 4, max_pathlen)
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
        sim_ent = self.cosine_sim_layer(results[0], results[1])
        if prop_nodes.nelement() != 0:
            # Calculate prop sum
            aggregated_prop_sum = torch.sum(self.name_embedding(prop_features), dim=-2)
            sim_prop = self.prop_weight * self.cosine_sim_layer(aggregated_prop_sum[:,0,0], aggregated_prop_sum[:,1,0])
            sim_prop += self.domain_weight * self.cosine_sim_layer(aggregated_prop_sum[:,0,1], aggregated_prop_sum[:,1,1])
            sim_prop += (1-self.prop_weight-self.domain_weight) * self.cosine_sim_layer(aggregated_prop_sum[:,0,2], aggregated_prop_sum[:,1,2])

            return torch.cat((sim_ent, sim_prop))
        return sim_ent

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

def pad_prop(inputs, max_prop_len):
    inputs_padded = [[[elem + [0 for i in range(max_prop_len - len(elem))]
                         for elem in prop]
                    for prop in elem_pair]
                for elem_pair in inputs]
    return inputs_padded

def generate_data(elem_tuple, neighbours_dicts):
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

def generate_input(elems, target, neighbours_dicts):
    inputs, targets, nodes = [], [], []
    global direct_inputs, direct_targets
    for elem in list(elems):
        try:
            inputs.append(generate_data(elem, neighbours_dicts))
            nodes.append(generate_data_neighbourless(elem))
            targets.append(target)
        except KeyError as e:
            direct_inputs.append(generate_data_neighbourless(elem))
            direct_targets.append(target)
        except Exception as e:
            raise
    return inputs, nodes, targets

def tensorize_entities(ent_pos, ent_neg, neighbours_dicts_ent):
    inputs_pos_ent, nodes_pos_ent, targets_pos_ent = generate_input(ent_pos, 1, neighbours_dicts_ent)
    inputs_neg_ent, nodes_neg_ent, targets_neg_ent = generate_input(ent_neg, 0, neighbours_dicts_ent)
    
    inputs_all_ent = list(inputs_pos_ent) + list(inputs_neg_ent)
    targets_all_ent = list(targets_pos_ent) + list(targets_neg_ent)
    nodes_all_ent = list(nodes_pos_ent) + list(nodes_neg_ent)
    
    all_inp_ent = list(zip(inputs_all_ent, targets_all_ent, nodes_all_ent))
    all_inp_shuffled_ent = random.sample(all_inp_ent, len(all_inp_ent))
    inputs_all_ent, targets_all_ent, nodes_all_ent = list(zip(*all_inp_shuffled_ent))

    return inputs_all_ent, targets_all_ent, nodes_all_ent

def tensorize_properties(prop_pos, prop_neg, neighbours_dicts_prop):
    inputs_pos_prop, nodes_pos_prop, targets_pos_prop = generate_input(prop_pos, 1, neighbours_dicts_prop)
    inputs_neg_prop, nodes_neg_prop, targets_neg_prop = generate_input(prop_neg, 0, neighbours_dicts_prop)

    inputs_all_prop = list(inputs_pos_prop) + list(inputs_neg_prop)
    targets_all_prop = list(targets_pos_prop) + list(targets_neg_prop)
    nodes_all_prop = list(nodes_pos_prop) + list(nodes_neg_prop)

    if len(inputs_all_prop) == 0:
        max_prop_len = 0
    else:
        max_prop_len = np.max([[[len(elem) for elem in prop] for prop in elem_pair] 
        for elem_pair in inputs_all_prop])
    
    all_inp_prop = list(zip(inputs_all_prop, targets_all_prop, nodes_all_prop))
    all_inp_shuffled_prop = random.sample(all_inp_prop, len(all_inp_prop))
    if all_inp_shuffled_prop:
        inputs_all_prop, targets_all_prop, nodes_all_prop = list(zip(*all_inp_shuffled_prop))
    else:
        inputs_all_prop, targets_all_prop, nodes_all_prop = [], [], []
    return inputs_all_prop, targets_all_prop, nodes_all_prop, max_prop_len

def batch_step(inputs_all_ent, targets_all_ent, nodes_all_ent, inputs_all_prop, targets_all_prop, nodes_all_prop, max_prop_len, batch_idx, batch_size, batch_size_prop):
    global model, optimizer
    batch_start_ent = batch_idx * batch_size
    batch_end_ent = (batch_idx+1) * batch_size
    batch_start_prop = batch_idx * batch_size_prop
    batch_end_prop = (batch_idx+1) * batch_size_prop
    
    inputs_ent = np.array(to_feature(inputs_all_ent[batch_start_ent: batch_end_ent]))
    targets_ent = np.array(targets_all_ent[batch_start_ent: batch_end_ent])
    nodes_ent = np.array(nodes_all_ent[batch_start_ent: batch_end_ent])

    inputs_prop = np.array(pad_prop(inputs_all_prop[batch_start_prop: batch_end_prop], max_prop_len))
    targets_prop = np.array(targets_all_prop[batch_start_prop: batch_end_prop])
    nodes_prop = np.array(nodes_all_prop[batch_start_prop: batch_end_prop])
    
    targets = np.concatenate((targets_ent, targets_prop), axis=0)
    
    inp_elems = torch.LongTensor(inputs_ent).to(device)
    node_elems = torch.LongTensor(nodes_ent).to(device)
    targ_elems = torch.DoubleTensor(targets).to(device)

    inp_props = torch.LongTensor(inputs_prop).to(device)
    node_props = torch.LongTensor(nodes_prop).to(device)

    optimizer.zero_grad()
    outputs = model(node_elems, inp_elems, node_props, inp_props)
    return outputs, targ_elems

def count_non_unk(elem):
    return len([l for l in elem if l!="<UNK>"])

torch.set_default_dtype(torch.float64)

torch.manual_seed(0)
np.random.seed(0)
random.seed(0)

data_items = data_ent.items()
np.random.shuffle(list(data_items))
data_ent = OrderedDict(data_items)

data_items = data_prop.items()
np.random.shuffle(list(data_items))
data_prop = OrderedDict(data_items)

print ("Number of entity pairs:", len(data_ent))
print ("Number of property pairs:", len(data_prop))

torch.set_default_dtype(torch.float64)

index=0
ontologies_in_alignment = [tuple([elem.split("/")[-1].split(".")[0] for elem in pair]) for pair in ontologies_in_alignment]
if ontology_split:
    # We split on the ontology-pair level
    step = int(len(ontologies_in_alignment)/K)
    
    val_onto = ontologies_in_alignment[len(ontologies_in_alignment)-step+1:]
    train_data_ent = {elem: data_ent[elem] for elem in data_ent if tuple([el.split("#")[0] for el in elem]) not in val_onto}
    val_data_ent = {elem: data_ent[elem] for elem in data_ent if tuple([el.split("#")[0] for el in elem]) in val_onto}
    
    train_data_prop = {elem: data_prop[elem] for elem in data_prop if tuple([el.split("#")[0] for el in elem]) not in val_onto}
    val_data_prop = {elem: data_prop[elem] for elem in data_prop if tuple([el.split("#")[0] for el in elem]) in val_onto}
    
    train_data_t_ent = [key for key in train_data_ent if train_data_ent[key]]
    train_data_f_ent = [key for key in train_data_ent if not train_data_ent[key]]

    train_data_t_prop = [key for key in train_data_prop if train_data_prop[key]]
    train_data_f_prop = [key for key in train_data_prop if not train_data_prop[key]]

    val_data_t_ent = [key for key in val_data_ent if val_data_ent[key]]
    val_data_f_ent = [key for key in val_data_ent if not val_data_ent[key]]

    val_data_t_prop = [key for key in val_data_prop if val_data_prop[key]]
    val_data_f_prop = [key for key in val_data_prop if not val_data_prop[key]]

else:
    # We split on the mapping-pair level
    ratio = float(1/K)
    data_t_ent = {elem: data_ent[elem] for elem in data_ent if data_ent[elem]}
    data_f_ent = {elem: data_ent[elem] for elem in data_ent if not data_ent[elem]}

    data_t_prop = {elem: data_prop[elem] for elem in data_prop if data_prop[elem]}
    data_f_prop = {elem: data_prop[elem] for elem in data_prop if not data_prop[elem]}

    data_t_items_ent = list(data_t_ent.keys())
    data_f_items_ent = list(data_f_ent.keys())

    data_t_items_prop = list(data_t_prop.keys())
    data_f_items_prop = list(data_f_prop.keys())

    val_data_t_ent = data_t_items_ent[int((ratio*index)*len(data_t_ent)):int((ratio*index + ratio)*len(data_t_ent))]
    val_data_f_ent = data_f_items_ent[int((ratio*index)*len(data_f_ent)):int((ratio*index + ratio)*len(data_f_ent))]

    train_data_t_ent = data_t_items_ent[:int(ratio*index*len(data_t_ent))] + data_t_items_ent[int(ratio*(index+1)*len(data_t_ent)):]
    train_data_f_ent = data_f_items_ent[:int(ratio*index*len(data_f_ent))] + data_f_items_ent[int(ratio*(index+1)*len(data_f_ent)):]

    val_data_t_prop = data_t_items_prop[int((ratio*index)*len(data_t_prop)):int((ratio*index + ratio)*len(data_t_prop))]
    val_data_f_prop = data_f_items_prop[int((ratio*index)*len(data_f_prop)):int((ratio*index + ratio)*len(data_f_prop))]

    train_data_t_prop = data_t_items_prop[:int(ratio*index*len(data_t_prop))] + data_t_items_prop[int(ratio*(index+1)*len(data_t_prop)):]
    train_data_f_prop = data_f_items_prop[:int(ratio*index*len(data_f_prop))] + data_f_items_prop[int(ratio*(index+1)*len(data_f_prop)):]

np.random.shuffle(train_data_f_ent)
train_data_f_ent = train_data_f_ent[:max_false_examples]

np.random.shuffle(train_data_f_prop)
train_data_f_prop = train_data_f_prop[:max_false_examples]

# Oversampling to maintain 1:1 ratio between positives and negatives
train_data_t_ent = np.repeat(train_data_t_ent, ceil(len(train_data_f_ent)/len(train_data_t_ent)), axis=0)
train_data_t_ent = train_data_t_ent[:len(train_data_f_ent)].tolist()

val_data_f_ent = random.sample(val_data_f_ent,len(val_data_t_ent))
val_data_f_prop = random.sample(val_data_f_prop,len(val_data_t_prop))

if train_data_t_prop:
    train_data_t_prop = np.repeat(train_data_t_prop, ceil(len(train_data_f_prop)/len(train_data_t_prop)), axis=0)
    train_data_t_prop = train_data_t_prop[:len(train_data_f_prop)].tolist()

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

model = VeeAlign(emb_vals).to(device)

optimizer = optim.Adam(model.parameters(), lr=lr, weight_decay=weight_decay)

print ("Starting training...")

min_val_loss = 1000
bad_losses = []
best_model = None
training_complete = False
for epoch in range(num_epochs):
    if training_complete:
        print (f"Early stopping since validation hasn't improved for {patience} consecutive runs.")
        break
    inputs_all_ent, targets_all_ent, nodes_all_ent = tensorize_entities(train_data_t_ent, train_data_f_ent, neighbours_dicts_ent)
    inputs_all_prop, targets_all_prop, nodes_all_prop, max_prop_len = tensorize_properties(train_data_t_prop, train_data_f_prop, neighbours_dicts_prop)

    batch_size = min(batch_size, len(inputs_all_ent))
    num_batches = int(ceil(len(inputs_all_ent)/batch_size))
    batch_size_prop = int(ceil(len(inputs_all_prop)/num_batches))
    
    inputs_all_ent_val, targets_all_ent_val, nodes_all_ent_val = tensorize_entities(val_data_t_ent, val_data_f_ent, neighbours_dicts_ent)
    inputs_all_prop_val, targets_all_prop_val, nodes_all_prop_val, max_prop_len_val = tensorize_properties(val_data_t_prop, val_data_f_prop, neighbours_dicts_prop)

    batch_size_val = min(batch_size, len(inputs_all_ent_val))
    num_batches_val = int(ceil(len(inputs_all_ent_val)/batch_size_val))
    batch_size_prop_val = int(ceil(len(inputs_all_prop_val)/num_batches_val))

    for batch_idx in range(num_batches):
        outputs, targ_elems = batch_step(inputs_all_ent, targets_all_ent, nodes_all_ent, inputs_all_prop, targets_all_prop, nodes_all_prop, max_prop_len, batch_idx, batch_size, batch_size_prop)
        
        loss = F.mse_loss(outputs, targ_elems)
        loss.backward()
        optimizer.step()
        
        if batch_idx%validation_interval == 0:
            print ("Conducting validation...")
            model.eval()
            all_outputs, all_targets = torch.DoubleTensor().to(device), torch.DoubleTensor().to(device)
            for batch_idx_val in range(num_batches_val):
                outputs_val, targ_elems_val = batch_step(inputs_all_ent_val, targets_all_ent_val, nodes_all_ent_val, inputs_all_prop_val, targets_all_prop_val, nodes_all_prop_val, max_prop_len_val, batch_idx_val, batch_size_val, batch_size_prop_val)
                all_outputs = torch.cat((all_outputs, outputs_val))
                all_targets = torch.cat((all_targets, targ_elems_val))
            val_loss = F.mse_loss(all_outputs, all_targets)
            print (f"Validation loss @ Epoch {epoch} Idx {batch_idx} = {val_loss}")
            if val_loss < min_val_loss:
                min_val_loss = val_loss
                bad_losses = []
                
                print ("Optimizing threshold for best model...")
                optimize_threshold(model)

                threshold_results_mean = {el: np.mean(threshold_results[el], axis=0) for el in threshold_results}    
                threshold = max(threshold_results_mean.keys(), key=(lambda key: threshold_results_mean[key][2]))

                model.threshold = threshold
                print (f"Saving best checkpoint with val loss {val_loss}...")
                torch.save(model.state_dict(), model_path + "_best.pt")
            else:
                bad_losses.append(val_loss)
                if len(bad_losses) > patience:
                    training_complete = True
            model.train()

print ("Training complete!")

model.eval()

print ("Optimizing threshold for last model...")
optimize_threshold(model)

threshold_results_mean = {el: np.mean(threshold_results[el], axis=0) for el in threshold_results}    
threshold = max(threshold_results_mean.keys(), key=(lambda key: threshold_results_mean[key][2]))

model.threshold = threshold

torch.save(model.state_dict(), model_path + "_last.pt")

print ("Done. Saved last and best models model at {} and {}".format(model_path + "_last.pt", model_path + "_best.pt"))
