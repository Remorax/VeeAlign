from ontology import *
import os, itertools, requests, re, subprocess, tarfile
import tensorflow_hub as hub
import numpy as np
from scipy import spatial

# Returns cosine similarity of two vectors
def cos_sim(a,b):
    return 1 - spatial.distance.cosine(a, b)

class DataParser():
    """Data parsing class"""
    def __init__(self, ontologies_in_alignment, USE_folder, gt_mappings=None):
        self.ontologies_in_alignment = ontologies_in_alignment
        self.USE_folder = USE_folder
        self.gt_mappings = gt_mappings
        self.USE_link = "https://tfhub.dev/google/universal-sentence-encoder-large/5?tf-hub-format=compressed"
        self.stopwords = ["has"]
    
    def extractUSEEmbeddings(self, words):
        # Extracts USE embeddings
        try:
            embed = hub.KerasLayer(self.USE_folder)
        except Exception as e:
            print ("Downloading USE embeddings...")
            r = requests.get(self.USE_link)
            open("USE.tar.gz", "wb").write(r.content)
            
            tar = tarfile.open("USE.tar.gz", "r:gz")
            tar.extractall(path=self.USE_folder)
            tar.close()

            os.remove("USE.tar.gz")
            embed = hub.KerasLayer(self.USE_folder)
            pass
        word_embeddings = embed(words)
        return word_embeddings.numpy()

    def generate_mappings(self):
        # Combinatorial mapping of entities in ontology pair(s)
        all_mappings = []
        for l in self.ontologies_in_alignment:
            ont1 = Ontology(l[0])
            ont2 = Ontology(l[1])
            
            ent1 = ont1.get_entities()
            ent2 = ont2.get_entities()
            
            obj1 = ont1.get_object_properties()
            obj2 = ont2.get_object_properties()
            
            data1 = ont1.get_data_properties()
            data2 = ont2.get_data_properties()

            mappings = list(itertools.product(ent1, ent2)) + list(itertools.product(obj1, obj2)) + list(itertools.product(data1, data2))

            all_mappings.extend([(l[0] + "#" + el[0], l[1] + "#" + el[1]) for el in mappings])

        if self.gt_mappings:
            data = {mapping: False for mapping in all_mappings}
            for mapping in set(self.gt_mappings):
                data[mapping] = True
            return data
        return all_mappings

    def path_to_root(self, element, parents_dict):
        # Extracts the path to the root recursively, 
        # i.e. all the "ancestral" nodes that lie from current node to root node
        if element not in parents_dict or not parents_dict[element]:
            return []
        output = flatten([[e] + self.path_to_root(e, parents_dict) for e in parents_dict[element]])
        return output

    def construct_abbreviation_resolution_dict(self, all_mappings):
        # Constructs an abbrevation resolution dict
        print ("Constructing abbrevation resolution dict....")
        abbreviations_dict = {}
        final_dict = {}

        for mapping in all_mappings:
            mapping = tuple([el.split("#")[1] for el in mapping])
            is_abb = re.search("[A-Z][A-Z]+", mapping[0])
            if is_abb:
                abbreviation = "".join([el[0].upper() for el in mapping[1].split("_")])
                if is_abb.group() in abbreviation:
                    
                    start = abbreviation.find(is_abb.group())
                    end = start + len(is_abb.group())
                    fullform = "_".join(mapping[1].split("_")[start:end])
                    
                    rest_first = " ".join([el for el in mapping[0].replace(is_abb.group(), "").split("_") if el]).lower()
                    rest_second = " ".join(mapping[1].split("_")[:start] + mapping[1].split("_")[end:])
                    if is_abb.group() not in final_dict:
                        final_dict[is_abb.group()] = [(fullform, rest_first, rest_second)]
                    else:
                        final_dict[is_abb.group()].append((fullform, rest_first, rest_second))

            is_abb = re.search("[A-Z][A-Z]+", mapping[1])
            if is_abb:
                abbreviation = "".join([el[0].upper() for el in mapping[0].split("_")])
                
                if is_abb.group() in abbreviation:
                    start = abbreviation.find(is_abb.group())
                    end = start + len(is_abb.group())
                    fullform = "_".join(mapping[0].split("_")[start:end])

                    rest_first = " ".join([el for el in mapping[1].replace(is_abb.group(), "").split("_") if el]).lower()
                    rest_second = " ".join(mapping[0].split("_")[:start] + mapping[0].split("_")[end:])
                    if is_abb.group() not in final_dict:
                        final_dict[is_abb.group()] = [(fullform, rest_first, rest_second)]
                    else:
                        final_dict[is_abb.group()].append((fullform, rest_first, rest_second))

        keys = [el for el in list(set(flatten([flatten([tup[1:] for tup in final_dict[key]]) for key in final_dict]))) if el]
        abb_embeds = dict(zip(keys, self.extractUSEEmbeddings(keys)))

        scored_dict = {}
        for abbr in final_dict:
            sim_list = [(tup[0], tup[1], tup[2], cos_sim(abb_embeds[tup[1]], abb_embeds[tup[2]])) if tup[1] and tup[2]
                        else (tup[0], tup[1], tup[2], 0) for tup in final_dict[abbr]]
            scored_dict[abbr] = sorted(list(set(sim_list)), key=lambda x:x[-1], reverse=True)

        resolved_dict = {key: scored_dict[key][0] for key in scored_dict}
        filtered_dict = {key: " ".join(resolved_dict[key][0].split("_")) for key in resolved_dict if resolved_dict[key][-1] > 0.9}
        return filtered_dict

    def camel_case_split(self, identifier):
        # Splits camel case strings
        matches = re.finditer('.+?(?:(?<=[a-z])(?=[A-Z])|(?<=[A-Z])(?=[A-Z][a-z])|$)', identifier)
        return [m.group(0) for m in matches]

    def parse(self, word):
        return flatten([el.split("_") for el in self.camel_case_split(word)])

    def run_abbreviation_resolution(self, inp, filtered_dict):
        # Resolving abbreviations to full forms
        print ("Resolving abbreviations...")
        inp_resolved = []
        for concept in inp:
            for key in filtered_dict:
                concept = concept.replace(key, filtered_dict[key])
            final_list = []
            # Lowering case except in abbreviations
            for word in concept.split(" "):
                if not re.search("[A-Z][A-Z]+", word):
                    final_list.append(word.lower())
                else:
                    final_list.append(word)
            concept = " ".join(final_list)
            inp_resolved.append(concept)
        return inp_resolved
    
    def extract_keys(self):
        # Extracts all entities for which USE embeddings needs to be extracted
        extracted_elems = []

        for ont_name in list(set(flatten(self.ontologies_in_alignment))):
            ont = Ontology(ont_name)
            entities = ont.get_entities()
            props = ont.get_object_properties() + ont.get_data_properties()
            triples = list(set(flatten(ont.get_triples())))
            extracted_elems.extend([ont_name + "#" + elem for elem in entities + props + triples])

        extracted_elems = list(set(extracted_elems))
        inp = [" ".join(self.parse(word.split("#")[1])) for word in extracted_elems]
        print ("Total number of extracted unique classes and properties from entire RA set: ", len(extracted_elems))

        extracted_elems = ["<UNK>"] + extracted_elems

        return inp, extracted_elems


    def run_spellcheck(self, inp):
        # Spelling checker and corrector
        print ("Running spellcheck...")
        url = "https://montanaflynn-spellcheck.p.rapidapi.com/check/"

        headers = {
            'x-rapidapi-host': "montanaflynn-spellcheck.p.rapidapi.com",
            'x-rapidapi-key': "9965b01207msh06291e57d6f2c55p1a6a16jsn0fb016da4a62"
            }

        inp_spellchecked = []
        for concept in inp:
            querystring = {"text": concept}
            response = requests.request("GET", url, headers=headers, params=querystring).json()
            if response["suggestion"] != concept:
                resolved = str(concept)
                final_list = []
                for word in concept.split(" "):
                    if not re.search("[A-Z][A-Z]+", concept):
                        final_list.append(word.lower())
                    else:
                        final_list.append(word)
                resolved = " ".join(final_list)
                for word in response["corrections"]:
                    if not re.search("[A-Z][A-Z]+", concept):
                        resolved = resolved.replace(word.lower(), response["corrections"][word][0].lower())
                        
                
                print ("Corrected {} to {}".format(concept, resolved))
                inp_spellchecked.append(resolved)
            else:
                inp_spellchecked.append(concept)

        return inp_spellchecked


    def remove_stopwords(self, inp):
        # Remove high frequency stopwords
        inp_filtered = []
        for elem in inp:
            words = " ".join([word for word in elem.split() if word not in self.stopwords])
            words = words.replace("-", " ")
            inp_filtered.append(words)
        return inp_filtered

    def extract_embeddings(self, inp, extracted_elems):
        # Creates embeddings to index dict, word to index dict etc
        embeds = np.array(self.extractUSEEmbeddings(inp))
        embeds = np.array([np.zeros(embeds.shape[1],)] + list(embeds))
        embeddings = dict(zip(extracted_elems, embeds))

        emb_vals = list(embeddings.values())
        emb_indexer = {key: i for i, key in enumerate(list(embeddings.keys()))}
        emb_indexer_inv = {i: key for i, key in enumerate(list(embeddings.keys()))}

        return emb_vals, emb_indexer, emb_indexer_inv

    def get_one_hop_neighbours(self, ont):
        ont_obj = Ontology(ont)
        triples = ont_obj.get_triples()
        entities = [(a,b) for (a,b,c) in triples]
        neighbours_dict = {elem: [elem] for elem in list(set(flatten(entities)))}
        for e1, e2 in entities:
            neighbours_dict[e1].append(e2)
            neighbours_dict[e2].append(e1)
        
        rootpath_dict = ont_obj.parents_dict
        rootpath_dict = {elem: self.path_to_root(elem, rootpath_dict) for elem in rootpath_dict}
        ont = ont.split("/")[-1].split(".")[0]

        for entity in neighbours_dict:
            if entity in rootpath_dict and len(rootpath_dict[entity]) > 0:
                neighbours_dict[entity].extend(rootpath_dict[entity])
            else:
                continue
        neighbours_dict = {el: neighbours_dict[el][:1] + sorted(list(set(neighbours_dict[el][1:])))
                           for el in neighbours_dict}
        neighbours_dict = {ont + "#" + el: [ont + "#" + e for e in neighbours_dict[el]] for el in neighbours_dict}
        return neighbours_dict

    def construct_neighbour_dicts(self):
        neighbours_dicts = {ont.split("/")[-1].split(".")[0]: self.get_one_hop_neighbours(ont) 
                            for ont in list(set(flatten(self.ontologies_in_alignment)))}
        max_neighbours = np.max(flatten([[len(el[e]) for e in el] for el in neighbours_dicts.values()]))
        neighbours_lens = {ont: {key: len(neighbours_dicts[ont][key]) for key in neighbours_dicts[ont]}
                           for ont in neighbours_dicts}
        neighbours_dicts = {ont: {key: neighbours_dicts[ont][key] + ["<UNK>" for i in range(max_neighbours -len(neighbours_dicts[ont][key]))]
                      for key in neighbours_dicts[ont]} for ont in neighbours_dicts}
        return neighbours_dicts

    def process(self, spellcheck=False):
        all_mappings = self.generate_mappings()
        inp, extracted_elems = self.extract_keys()
        filtered_dict = self.construct_abbreviation_resolution_dict(all_mappings)
        inp_resolved = self.run_abbreviation_resolution(inp, filtered_dict)
        if spellcheck:
            inp_resolved = self.run_spellcheck(inp_resolved)
        inp_filtered = self.remove_stopwords(inp_resolved)
        emb_vals, emb_indexer, emb_indexer_inv = self.extract_embeddings(inp_filtered, extracted_elems)
        neighbours_dicts = self.construct_neighbour_dicts()

        return all_mappings, emb_indexer, emb_indexer_inv, emb_vals, neighbours_dicts
