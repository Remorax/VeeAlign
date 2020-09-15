import configparser, logging, random
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

prefix_path = "/".join(os.path.dirname(os.path.abspath(__file__)).split("/")[:-1]) + "/"

print ("Prefix path: ", prefix_path)

# Initialize variables from config

language = str(config["General"]["Language"])
K = int(config["General"]["K"])
ontology_split = str(config["General"]["ontology_split"]) == "True"
max_false_examples = int(config["General"]["max_false_examples"])

alignment_folder = prefix_path + str(config["Paths"]["alignment_folder"])
train_folder = prefix_path + str(config["Paths"]["train_folder"])
model_path = prefix_path + str(config["Paths"]["model_path"])

spellcheck = config["Preprocessing"]["has_spellcheck"] == "True"

max_paths = int(config["Parameters"]["max_paths"])
max_pathlen = int(config["Parameters"]["max_pathlen"])
threshold = float(config["Parameters"]["threshold"])
bag_of_neighbours = config["Parameters"]["bag_of_neighbours"] == "True"
weighted_average = config["Parameters"]["weighted_average"] == "True"

lr = float(config["Hyperparameters"]["lr"])
num_epochs = int(config["Hyperparameters"]["num_epochs"])
weight_decay = float(config["Hyperparameters"]["weight_decay"])
batch_size = int(config["Hyperparameters"]["batch_size"])

reference_alignments = load_alignments(alignment_folder)
gt_mappings = [tuple([elem.split("/")[-1] for elem in el]) for el in reference_alignments]
gt_mappings = [tuple([el.split("#")[0].lower() +  "#" +  el.split("#")[1] for el in tup]) for tup in gt_mappings]
print ("Ontologies being aligned are: ", ontologies_in_alignment)

# Preprocessing and parsing input data for training
preprocessing = DataParser(ontologies_in_alignment, language, gt_mappings)
data, emb_indexer, emb_indexer_inv, emb_vals, neighbours_dicts, max_types = preprocessing.process(spellcheck, bag_of_neighbours)

all_fn, all_fp = [], []
final_results = []
direct_inputs, direct_targets = [], []
threshold_results = {}

def test():
    '''
    Function to obtain output of model on test set
    '''
    global index, test_data_t, test_data_f, direct_inputs, direct_targets, batch_size
    all_results = OrderedDict()    
    direct_inputs, direct_targets = [], []
    with torch.no_grad():
        all_pred = []
        
        np.random.shuffle(test_data_t)
        np.random.shuffle(test_data_f)

        inputs_pos, nodes_pos, targets_pos = generate_input(test_data_t, 1)
        inputs_neg, nodes_neg, targets_neg = generate_input(test_data_f, 0)

        inputs_all = list(inputs_pos) + list(inputs_neg)
        targets_all = list(targets_pos) + list(targets_neg)
        nodes_all = list(nodes_pos) + list(nodes_neg)
        
        print ("Inputs len", len(inputs_all), len(test_data_t), len(test_data_f)) 
        all_inp = list(zip(inputs_all, targets_all, nodes_all))
        all_inp_shuffled = random.sample(all_inp, len(all_inp))
        inputs_all, targets_all, nodes_all = list(zip(*all_inp_shuffled))

        batch_size = min(batch_size, len(inputs_all))
        num_batches = int(ceil(len(inputs_all)/batch_size))
        for batch_idx in range(num_batches):
            batch_start = batch_idx * batch_size
            batch_end = (batch_idx+1) * batch_size

            inputs = np.array(to_feature(inputs_all[batch_start: batch_end]))
            targets = np.array(targets_all[batch_start: batch_end])
            nodes = np.array(nodes_all[batch_start: batch_end])
 
            inp_elems = torch.LongTensor(inputs).to(device)
            node_elems = torch.LongTensor(nodes).to(device)
            targ_elems = torch.DoubleTensor(targets)

            outputs = model(node_elems, inp_elems)
            outputs = [el.item() for el in outputs]
            targets = [True if el.item() else False for el in targets]

            for idx, pred_elem in enumerate(outputs):
                ent1 = emb_indexer_inv[nodes[idx][0]]
                ent2 = emb_indexer_inv[nodes[idx][1]]
                if (ent1, ent2) in all_results:
                    print ("Error: ", ent1, ent2, "already present")
                all_results[(ent1, ent2)] = (round(pred_elem, 3), targets[idx])
 
        direct_targets = [True if el else False for el in direct_targets]
        
        print ("Len (direct inputs): ", len(direct_inputs))
        for idx, direct_input in enumerate(direct_inputs):
            ent1 = emb_indexer_inv[direct_input[0]]
            ent2 = emb_indexer_inv[direct_input[1]]
            sim = cos_sim(emb_vals[direct_input[0]], emb_vals[direct_input[1]])
            all_results[(ent1, ent2)] = (round(sim, 3), direct_targets[idx])
    return (index, test_data_t, all_results)

def optimize_threshold():
    '''
    Function to optimise threshold on validation set.
    Calculates performance metrics (precision, recall, F1-score, F2-score, F0.5-score) for a
    range of thresholds, dictated by the range of scores output by the model, with step size 
    0.001 and updates `threshold_results` which is the relevant dictionary.
    '''
    global val_data_t, val_data_f, threshold_results, batch_size, direct_inputs, direct_targets
    all_results = OrderedDict()
    direct_inputs, direct_targets = [], []
    with torch.no_grad():
        all_pred = []
        
        np.random.shuffle(val_data_t)
        np.random.shuffle(val_data_f)

        inputs_pos, nodes_pos, targets_pos = generate_input(val_data_t, 1)
        inputs_neg, nodes_neg, targets_neg = generate_input(val_data_f, 0)

        inputs_all = list(inputs_pos) + list(inputs_neg)
        targets_all = list(targets_pos) + list(targets_neg)
        nodes_all = list(nodes_pos) + list(nodes_neg)
        
        all_inp = list(zip(inputs_all, targets_all, nodes_all))
        all_inp_shuffled = random.sample(all_inp, len(all_inp))
        inputs_all, targets_all, nodes_all = list(zip(*all_inp_shuffled))

        batch_size = min(batch_size, len(inputs_all))
        num_batches = int(ceil(len(inputs_all)/batch_size))
        for batch_idx in range(num_batches):
            batch_start = batch_idx * batch_size
            batch_end = (batch_idx+1) * batch_size

            inputs = np.array(to_feature(inputs_all[batch_start: batch_end]))
            targets = np.array(targets_all[batch_start: batch_end])
            nodes = np.array(nodes_all[batch_start: batch_end])
            
            inp_elems = torch.LongTensor(inputs).to(device)
            node_elems = torch.LongTensor(nodes).to(device)
            targ_elems = torch.DoubleTensor(targets)

            # Run model on input
            outputs = model(node_elems, inp_elems)
            outputs = [el.item() for el in outputs]
            targets = [True if el.item() else False for el in targets]

            for idx, pred_elem in enumerate(outputs):
                ent1 = emb_indexer_inv[nodes[idx][0]]
                ent2 = emb_indexer_inv[nodes[idx][1]]
                all_results[(ent1, ent2)] = (round(pred_elem, 3), targets[idx])
        
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

        # Iterate over every threshold with step size of 0.001 and calculate all evaluation metrics
        while threshold < high_threshold:
            threshold = round(threshold, 3)
            res = []
            for i,key in enumerate(all_results):
                if all_results[key][0] > threshold:
                    res.append(key)
            fn_list = [(key, all_results[key][0]) for key in val_data_t if key not in set(res)]
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
        
def calculate_performance():
    all_metrics, all_fn, all_fp = [], [], []
    for (index, test_data_t, all_results) in final_results:
        res = []
        for i,key in enumerate(all_results):
            if all_results[key][0] > threshold:
                res.append(key)
        s = set(res)
        fn_list = [(key, all_results[key][0]) for key in test_data_t if key not in s]
        fp_list = [(elem, all_results[elem][0]) for elem in res if not all_results[elem][1]]
        tp_list = [(elem, all_results[elem][0]) for elem in res if all_results[elem][1]]
        tp, fn, fp = len(tp_list), len(fn_list), len(fp_list)
        
        try:
            precision = tp/(tp+fp)
            recall = tp/(tp+fn)
            f1score = 2 * precision * recall / (precision + recall)
            f2score = 5 * precision * recall / (4 * precision + recall)
            f0_5score = 1.25 * precision * recall / (0.25 * precision + recall)
        except Exception as e:
            print (e)
            continue
        if ontology_split:
            print ("Performance for ", ontologies_in_alignment[index+2: index+3], "is :", (precision, recall, f1score, f2score, f0_5score))
        else:
            print ("Performance for ", index, "th fold is :", (precision, recall, f1score, f2score, f0_5score))
        all_fn.extend(fn_list)
        all_fp.extend(fp_list)
        all_metrics.append((precision, recall, f1score, f2score, f0_5score))
    return all_metrics, all_fn, all_fp

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
random.seed(0)

data_items = data.items()
np.random.shuffle(list(data_items))
data = OrderedDict(data_items)

print ("Number of entities:", len(data))

if ontology_split:
    step = int(len(ontologies_in_alignment)/K)
    data_iter = list(range(0, len(ontologies_in_alignment), step))
else:
    data_iter = list(range(K))
ontologies_in_alignment = [tuple([elem.split("/")[-1].rsplit(".",1)[0].replace(".", "_").lower() for elem in pair]) for pair in ontologies_in_alignment]

for index in data_iter:
    print ("Starting sliding window evaluation...")
    print ("Step {}/{}".format(index, K))
    if ontology_split:
        # We split on the ontology-pair level
        test_onto = ontologies_in_alignment[index:index+step]
        
        train_data = {elem: data[elem] for elem in data if tuple([el.split("#")[0] for el in elem]) not in test_onto}

        val_onto = test_onto[:(step-1)]
        test_onto = test_onto[(step-1):]
        val_data = {elem: data[elem] for elem in data if tuple([el.split("#")[0] for el in elem]) in val_onto}
        test_data = {elem: data[elem] for elem in data if tuple([el.split("#")[0] for el in elem]) in test_onto}
        print ("Val onto: ", val_onto, "test_onto: ", test_onto)
        print (list(data_items)[:100])
        train_data_t = [key for key in train_data if train_data[key]]
        train_data_f = [key for key in train_data if not train_data[key]]

        val_data_t = [key for key in val_data if val_data[key]]
        val_data_f = [key for key in val_data if not val_data[key]]

        test_data_t = [key for key in test_data if test_data[key]]
        test_data_f = [key for key in test_data if not test_data[key]]
    else:
        # We split on the mapping-pair level
        ratio = float(1/K)
        data_t = {elem: data[elem] for elem in data if data[elem]}
        data_f = {elem: data[elem] for elem in data if not data[elem]}

        data_t_items = list(data_t.keys())
        data_f_items = list(data_f.keys())

        test_data_t = data_t_items[int((ratio*index)*len(data_t)):int((ratio*index + ratio)*len(data_t))]
        test_data_f = data_f_items[int((ratio*index)*len(data_f)):int((ratio*index + ratio)*len(data_f))]

        train_data_tt = data_t_items[:int(ratio*index*len(data_t))] + data_t_items[int(ratio*(index+1)*len(data_t)):]
        train_data_ff = data_f_items[:int(ratio*index*len(data_f))] + data_f_items[int(ratio*(index+1)*len(data_f)):]

        np.random.shuffle(train_data_tt)
        np.random.shuffle(train_data_ff)

        val_data_t = train_data_tt[:int((ratio/2)*len(data_t))]
        val_data_f = train_data_ff[:int((ratio/2)*len(data_f))]

        train_data_t = train_data_tt[int((ratio/2)*len(data_t)):]
        train_data_f = train_data_ff[int((ratio/2)*len(data_f)):]

    print ("Training size:", len(train_data_t) + len(train_data_f), "Testing size:", len(test_data_t) + len(test_data_f))

    np.random.shuffle(train_data_f)
    train_data_f = train_data_f[:max_false_examples]

    # Oversampling to maintain 1:1 ratio between positives and negatives
    train_data_t = np.repeat(train_data_t, ceil(len(train_data_f)/len(train_data_t)), axis=0)
    train_data_t = train_data_t[:len(train_data_f)].tolist()

    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    model = SiameseNetwork().to(device)

    optimizer = optim.Adam(model.parameters(), lr=lr, weight_decay=weight_decay)

    for epoch in range(1):
        inputs_pos, nodes_pos, targets_pos = generate_input(train_data_t, 1)
        inputs_neg, nodes_neg, targets_neg = generate_input(train_data_f, 0)
        inputs_all = list(inputs_pos) + list(inputs_neg)
        targets_all = list(targets_pos) + list(targets_neg)
        nodes_all = list(nodes_pos) + list(nodes_neg)
        
        all_inp = list(zip(inputs_all, targets_all, nodes_all))
        all_inp_shuffled = random.sample(all_inp, len(all_inp))
        inputs_all, targets_all, nodes_all = list(zip(*all_inp_shuffled[:10]))

        batch_size = min(batch_size, len(inputs_all))
        num_batches = int(ceil(len(inputs_all)/batch_size))

        for batch_idx in range(num_batches):
            batch_start = batch_idx * batch_size
            batch_end = (batch_idx+1) * batch_size
            
            inputs = np.array(to_feature(inputs_all[batch_start: batch_end]))
            targets = np.array(targets_all[batch_start: batch_end])
            nodes = np.array(nodes_all[batch_start: batch_end])
            
            inp_elems = torch.LongTensor(inputs).to(device)
            node_elems = torch.LongTensor(nodes).to(device)
            targ_elems = torch.DoubleTensor(targets).to(device)

            optimizer.zero_grad()
            outputs = model(node_elems, inp_elems)

            loss = F.mse_loss(outputs, targ_elems)
            loss.backward()
            optimizer.step()

            if batch_idx%5000 == 0:
                print ("Epoch: {} Idx: {} Loss: {}".format(epoch, batch_idx, loss.item()))

    model.eval()
    
    # Do validation to optimize classification threshold    
    optimize_threshold()

    # Obtain output of model on test set
    final_results.append(test())

threshold_results_mean = {el: np.mean(threshold_results[el], axis=0) for el in threshold_results}    
threshold = max(threshold_results_mean.keys(), key=(lambda key: threshold_results_mean[key][2]))

all_metrics, all_fn, all_fp = calculate_performance()

print ("Final Results: " + str(np.mean(all_metrics, axis=0)))
print ("Threshold: ", threshold)
