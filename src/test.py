import configparser, sys, logging, random
import numpy as np
from collections import OrderedDict
from math import ceil
import torch, pickle
from torch import nn
from torch import optim
import torch.nn.functional as F
from ontology import *
from data_preprocessing import *

ont_name1, ont_name2 = sys.argv[1], sys.argv[2]
if ont_name1.endswith("/"):
    ont_name1 = ont_name1[:-1]
if ont_name2.endswith("/"):
    ont_name2 = ont_name2[:-1]

prefix_path = "/".join(os.path.dirname(os.path.abspath(__file__)).split("/")[:-1]) + "/"

# Read `config.ini` and initialize parameter values
config = configparser.ConfigParser()
config.read(prefix_path + 'src/config.ini')

print("Prefix path: ", prefix_path)

# Initialize variables from config
quick_mode = str(config["General"]["quick_mode"])

model_path = prefix_path + str(config["Paths"]["load_model_path"])
output_path = prefix_path + str(config["Paths"]["output_folder"])
cached_embeddings_path = prefix_path + str(config["Paths"]["embedding_cache_path"])
spellcheck = config["Preprocessing"]["has_spellcheck"] == "True"

max_paths = int(config["Parameters"]["max_paths"])
max_pathlen = int(config["Parameters"]["max_pathlen"])
bag_of_neighbours = config["Parameters"]["bag_of_neighbours"] == "True"
weighted_sum = config["Parameters"]["weighted_sum"] == "True"

batch_size = int(config["Hyperparameters"]["batch_size"])

test_ontologies = [tuple([ont_name1, ont_name2])]

# Preprocessing and parsing input data for testing
preprocessing = DataParser(test_ontologies, quick_mode)
test_data_ent, test_data_prop, emb_indexer_new, emb_indexer_inv_new, emb_vals_new, neighbours_dicts_ent, neighbours_dicts_prop, max_types = preprocessing.process(spellcheck, bag_of_neighbours)

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

class VeeAlign(nn.Module):
    # Defines the VeeAlign Siamese Network model
    def __init__(self, emb_vals, threshold=0.9):
        super().__init__() 
        self.n_neighbours = max_types
        self.max_paths = max_paths
        self.max_pathlen = max_pathlen
        self.embedding_dim = np.array(emb_vals).shape[1]
        
        self.threshold = nn.Parameter(torch.DoubleTensor([threshold]))
        self.threshold.requires_grad = False

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

def count_non_unk(elem):
    return len([l for l in elem if l!="<UNK>"])

def generate_data_neighbourless(elem_tuple):
    return [emb_indexer[elem] for elem in elem_tuple]

def embedify(seq):
    for item in seq:
        if isinstance(item, list):
            yield list(embedify(item))
        else:
            yield emb_indexer[item]

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

def pad_prop(inputs):
    inputs_padded = [[[elem + [0 for i in range(max_prop_len - len(elem))]
                         for elem in prop]
                    for prop in elem_pair]
                for elem_pair in inputs]
    return inputs_padded

def generate_input(elems, neighbours_dicts):
    inputs, nodes = [], []
    global direct_inputs
    for elem in list(elems):
        try:
            inputs.append(generate_data(elem, neighbours_dicts))
            nodes.append(generate_data_neighbourless(elem))
        except KeyError as e:
            direct_inputs.append(generate_data_neighbourless(elem))
        except Exception as e:
            print (e)
            raise
    return inputs, nodes

def write_results():
    ont_name_parsed1 = Ontology(ont_name1).extract_ns()
    ont_name_parsed2 = Ontology(ont_name2).extract_ns()
    ont_name1_pre = ont_name1 if (ont_name1.startswith("http://") or ont_name1.startswith("https://")) else "file://" + ont_name1
    ont_name2_pre = ont_name2 if (ont_name2.startswith("http://") or ont_name2.startswith("https://")) else "file://" + ont_name2
    rdf = \
    """<?xml version='1.0' encoding='utf-8' standalone='no'?>
<rdf:RDF xmlns='http://knowledgeweb.semanticweb.org/heterogeneity/alignment#'
         xmlns:rdf='http://www.w3.org/1999/02/22-rdf-syntax-ns#'
         xmlns:xsd='http://www.w3.org/2001/XMLSchema#'
         xmlns:align='http://knowledgeweb.semanticweb.org/heterogeneity/alignment#'>
<Alignment>
  <xml>yes</xml>
  <level>0</level>
  <type>**</type>
  <onto1>
    <Ontology rdf:about="{}">
      <location>{}</location>
    </Ontology>
  </onto1>
  <onto2>
    <Ontology rdf:about="{}">
      <location>{}</location>
    </Ontology>
  </onto2>""".format(ont_name_parsed1.split("#")[0], ont_name1_pre, ont_name_parsed2.split("#")[0], ont_name2_pre)
    for (a,b,score) in final_list:
        mapping = """
  <map>
    <Cell>
      <entity1 rdf:resource='{}'/>
      <entity2 rdf:resource='{}'/>
      <relation>=</relation>
      <measure rdf:datatype='http://www.w3.org/2001/XMLSchema#float'>{}</measure>
    </Cell>
  </map>""".format(ont_name_parsed1 + "#".join(a.split("#")[1:]), ont_name_parsed2 + "#".join(b.split("#")[1:]), score)
        rdf += mapping
    rdf += """
</Alignment>
</rdf:RDF>"""
    return rdf

torch.set_default_dtype(torch.float64)

torch.manual_seed(0)
np.random.seed(0)
random.seed(0)

np.random.shuffle(test_data_ent)
np.random.shuffle(test_data_prop)

torch.set_default_dtype(torch.float64)

print ("Loading trained model....")
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

pretrained_dict = torch.load(model_path, map_location=torch.device(device))
pretrained_dict = {k: v for k, v in pretrained_dict.items() if k!="name_embedding.weight"}
max_types = len([key for key in pretrained_dict.keys() if key.startswith("w_")]) + 1

model = VeeAlign(emb_vals).to(device)
model_dict = model.state_dict()

model_dict.update(pretrained_dict)
model.load_state_dict(model_dict)

threshold = model.threshold.data.cpu().numpy()[0]

print ("Model loaded successfully!")

print ("Optimum Threshold: {}".format(threshold))

model.eval()

print ("Length of test data(ent): {} test data(prop):{}".format(len(test_data_ent), len(test_data_prop)))

all_results = OrderedDict()    
direct_inputs = []
with torch.no_grad():
    inputs_all_ent, nodes_all_ent = generate_input(test_data_ent, neighbours_dicts_ent)
    inputs_all_prop, nodes_all_prop = generate_input(test_data_prop, neighbours_dicts_prop)
     
    all_inp = list(zip(inputs_all_ent, nodes_all_ent))
    all_inp_shuffled = random.sample(all_inp, len(all_inp))
    inputs_all_ent, nodes_all_ent = list(zip(*all_inp_shuffled))

    if len(inputs_all_prop) == 0:
        max_prop_len = 0
    else:
        max_prop_len = np.max([[[len(elem) for elem in prop] for prop in elem_pair] 
        for elem_pair in inputs_all_prop])
    
    all_inp_prop = list(zip(inputs_all_prop, nodes_all_prop))
    all_inp_shuffled_prop = random.sample(all_inp_prop, len(all_inp_prop))
    if all_inp_shuffled_prop:
        inputs_all_prop, nodes_all_prop = list(zip(*all_inp_shuffled_prop))
    else:
        inputs_all_prop, nodes_all_prop = [], []

    print ("Max prop len: ", max_prop_len)
    batch_size = min(batch_size, len(inputs_all_ent))
    num_batches = int(ceil(len(inputs_all_ent)/batch_size))
    batch_size_prop = int(ceil(len(inputs_all_prop)/num_batches))

    print ("Num batches: {} Batch size (prop): {}".format(num_batches, batch_size_prop))
    for batch_idx in range(num_batches):
        batch_start = batch_idx * batch_size
        batch_end = (batch_idx+1) * batch_size
        batch_start_prop = batch_idx * batch_size_prop
        batch_end_prop = (batch_idx+1) * batch_size_prop

        inputs_ent = np.array(to_feature(inputs_all_ent[batch_start: batch_end]))
        nodes_ent = np.array(nodes_all_ent[batch_start: batch_end])

        inputs_prop = np.array(pad_prop(inputs_all_prop[batch_start_prop: batch_end_prop]))
        nodes_prop = np.array(nodes_all_prop[batch_start_prop: batch_end_prop])

        inp_ents = torch.LongTensor(inputs_ent).to(device)
        node_ents = torch.LongTensor(nodes_ent).to(device)
        inp_props = torch.LongTensor(inputs_prop).to(device)
        node_props = torch.LongTensor(nodes_prop).to(device)
        
        outputs = model(node_ents, inp_ents, node_props, inp_props)
        outputs = [el.item() for el in outputs]

        for idx, pred_elem in enumerate(outputs):
            if idx < len(nodes_ent):
                ent1 = emb_indexer_inv[nodes_ent[idx][0]]
                ent2 = emb_indexer_inv[nodes_ent[idx][1]]
            else:
                ent1 = emb_indexer_inv[nodes_prop[idx-len(nodes_ent)][0]]
                ent2 = emb_indexer_inv[nodes_prop[idx-len(nodes_ent)][1]]
            if (ent1, ent2) in all_results:
                print ("Error: ", ent1, ent2, "already present")
            all_results[(ent1, ent2)] = (round(pred_elem, 3), pred_elem>=threshold)
    
    print ("Len (direct inputs): ", len(direct_inputs))
    for idx, direct_input in enumerate(direct_inputs):
        ent1 = emb_indexer_inv[direct_input[0]]
        ent2 = emb_indexer_inv[direct_input[1]]
        sim = cos_sim(emb_vals[direct_input[0]], emb_vals[direct_input[1]])
        all_results[(ent1, ent2)] = (round(sim, 3), pred_elem>=threshold)
    
final_list = [(elem[0], elem[1], str(all_results[elem][0])) for elem in all_results if all_results[elem][1]]

ont_name_parsed1 = Ontology(ont_name1).extract_ns().split("/")[-1].split("#")[0].rsplit(".", 1)[0]
ont_name_parsed2 = Ontology(ont_name2).extract_ns().split("/")[-1].split("#")[0].rsplit(".", 1)[0]

f = ont_name_parsed1 + "-" + ont_name_parsed2 + ".rdf"

open(output_path + f, "w+").write(write_results())

print ("The final alignment file can be found below: ")
print ("file://" + output_path + f)
