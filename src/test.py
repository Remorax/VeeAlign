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

prefix_path = "/".join(os.path.dirname(os.path.abspath(__file__)).split("/")[:-1]) + "/"

print ("Prefix path: ", prefix_path)

# Initialize variables from config
model_path = prefix_path + str(config["Paths"]["model_path"])
output_path = prefix_path + str(config["Paths"]["output_folder"])
spellcheck = config["Preprocessing"]["has_spellcheck"] == "True"

max_neighbours = int(config["Parameters"]["max_neighbours"])
min_neighbours = int(config["Parameters"]["min_neighbours"])
threshold = float(config["Parameters"]["threshold"])
batch_size = int(config["Hyperparameters"]["batch_size"])

test_ontologies = [tuple([ont_name1, ont_name2])]

# Preprocessing and parsing input data for testing
preprocessing = DataParser(test_ontologies)
test_data, emb_indexer, emb_indexer_inv, emb_vals, neighbours_dicts = preprocessing.process(spellcheck, max_neighbours)

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

def count_non_unk(elem):
    return len([l for l in elem if l!="<UNK>"])

def generate_data_neighbourless(elem_tuple):
    op = np.array([emb_indexer[elem] for elem in elem_tuple])
    return op

def generate_data(elem_tuple):
    return np.array([[emb_indexer[el] for el in neighbours_dicts[elem.split("#")[0]][elem]] for elem in elem_tuple])

def generate_input(elems):
    inputs = []
    print ("Generating input data to model...")
    for elem in list(elems):
        try:
            inputs.append(generate_data(elem))
        except Exception as e:
            direct_inputs.append(generate_data_neighbourless(elem))
            continue
    return np.array(inputs)

def embed(inputs):
    return np.array([emb_vals[idx] for idx in inputs])

def write_results():
    ont_name_parsed1 = "http://" + ont_name1.split("/")[-1].split(".")[0]
    ont_name_parsed2 = "http://" + ont_name2.split("/")[-1].split(".")[0]
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
      </onto2>""".format(ont_name_parsed1, ont_name1, ont_name_parsed2, ont_name2)
    for (a,b,score) in final_list:
        mapping = """
        <map>
          <Cell>
            <entity1 rdf:resource='{}'/>
            <entity2 rdf:resource='{}'/>
            <relation>=</relation>
            <measure rdf:datatype='http://www.w3.org/2001/XMLSchema#float'>{}</measure>
          </Cell>
        </map>""".format(ont_name_parsed1 + "#" + a.split("#")[-1], ont_name_parsed2 + "#" + b.split("#")[-1], score)
        rdf += mapping
    rdf += """
    </Alignment>
    </rdf:RDF>
    """
    return rdf

neighbours_dicts = {ont: {el: neighbours_dicts[ont][el][:max_neighbours] for el in neighbours_dicts[ont]
       if count_non_unk(neighbours_dicts[ont][el]) > min_neighbours} for ont in neighbours_dicts}

np.random.shuffle(test_data)

torch.set_default_dtype(torch.float64)

print ("Loading trained model....")
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
model = SiameseNetwork().to(device)
model.load_state_dict(torch.load(model_path, map_location=torch.device(device)))

print ("Model loaded successfully!")

model.eval()

all_results = OrderedDict()
direct_inputs, direct_targets = [], []
with torch.no_grad():


    inputs_all = generate_input(test_data)
    
    indices_all = np.random.permutation(len(inputs_all))
    inputs_all = np.array(inputs_all)[indices_all]

    batch_size = min(batch_size, len(inputs_all))
    num_batches = int(ceil(len(inputs_all)/batch_size))
    for batch_idx in range(num_batches):
        batch_start = batch_idx * batch_size
        batch_end = (batch_idx+1) * batch_size

        inputs = inputs_all[batch_start: batch_end]
        inp = inputs.transpose(1,0,2)
        inputs = np.apply_along_axis(embed, 2, inputs)
        
        inp_elems = torch.DoubleTensor(inputs).to(device)

        outputs = model(inp_elems)
        outputs = [el.item() for el in outputs]

        for idx, pred_elem in enumerate(outputs):
            ent1 = emb_indexer_inv[inp[0][idx][0]]
            ent2 = emb_indexer_inv[inp[1][idx][0]]
            if (ent1, ent2) in all_results:
                print ("Error: ", ent1, ent2, "already present")
            all_results[(ent1, ent2)] = (pred_elem, pred_elem>=threshold)
    
    print ("Len (direct inputs): ", len(direct_inputs))
    for idx, direct_input in enumerate(direct_inputs):
        ent1 = emb_indexer_inv[direct_input[0]]
        ent2 = emb_indexer_inv[direct_input[1]]
        sim = cos_sim(emb_vals[direct_input[0]], emb_vals[direct_input[1]])
        all_results[(ent1, ent2)] = (sim, sim>=threshold)
    
final_list = [(elem[0], elem[1], str(round(all_results[elem][0], 3))) for elem in all_results if all_results[elem][1]]

f = "VeeAlign-" + ont_name1.split("/")[-1].split(".")[0] + "-" + ont_name2.split("/")[-1].split(".")[0] + ".rdf"

open(output_path + f, "w+").write(write_results())
