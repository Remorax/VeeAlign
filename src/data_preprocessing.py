from ontology import *
import os, itertools, re, logging, requests, urllib
import tensorflow_text
import tensorflow_hub as hub
import numpy as np
from scipy import spatial
from copy import deepcopy

# Returns cosine similarity of two vectors
def cos_sim(a,b):
    return 1 - spatial.distance.cosine(a, b)

class DataParser():
    """Data parsing class"""
    def __init__(self, ontologies_in_alignment, language, gt_mappings=None):
        self.ontologies_in_alignment = ontologies_in_alignment
        self.gt_mappings = gt_mappings
        self.language = language
        if self.language == "en":
            self.USE_link = "https://tfhub.dev/google/universal-sentence-encoder-large/5?tf-hub-format=compressed"
        else:
            self.USE_link = "https://tfhub.dev/google/universal-sentence-encoder-multilingual/3?tf-hub-format=compressed"
        self.USE = hub.load(self.USE_link)
        self.stopwords = ["has"]
    
    def extractUSEEmbeddings(self, words):
        # Extracts USE embeddings
        
        word_embeddings = self.USE(words)
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

            pre1 = l[0].split("/")[-1].rsplit(".",1)[0].replace("-", "_")
            pre2 = l[1].split("/")[-1].rsplit(".",1)[0].replace("-", "_")
            all_mappings.extend([(pre1 + "#" + el[0], pre2 + "#" + el[1]) for el in mappings])

        if self.gt_mappings:
            s = set(all_mappings)
            data = {mapping: False for mapping in all_mappings}
            for mapping in set(self.gt_mappings):
                if mapping in s:
                    data[mapping] = True
                else:
                    mapping = tuple([el.replace(",-", "_") for el in mapping])
                    if mapping in s:
                        data[mapping] = True
                    else:
                        print ("Warning: {} given in alignments could not be found in source/target ontology.".format(mapping))
                        continue
            return data
        return all_mappings

    def path_to_root(self, elem, ont_mappings, curr = [], rootpath=[]):
        # Extracts the path to the root recursively, 
        # i.e. all the "ancestral" nodes that lie from current node to root node
        curr.append(elem)
        if elem not in ont_mappings or not ont_mappings[elem]:
            rootpath.append(curr)
            return
        for node in ont_mappings[elem]:
            curr_orig = deepcopy(curr)
            _ = self.path_to_root(node, ont_mappings, curr, rootpath)
            curr = curr_orig
        return rootpath

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
        print ("Results after abbreviation resolution: ", filtered_dict)
        return filtered_dict

    def camel_case_split(self, identifier):
        # Splits camel case strings
        matches = re.finditer('.+?(?:(?<=[a-z])(?=[A-Z])|(?<=[A-Z])(?=[A-Z][a-z])|$)', identifier)
        return [m.group(0) for m in matches]

    def parse(self, word):
        return " ".join(flatten([el.split("_") for el in self.camel_case_split(word)]))

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
        mapping_ont = {}

        for ont_name in list(set(flatten(self.ontologies_in_alignment))):
            ont = Ontology(ont_name)
            entities = ont.get_entities()
            props = ont.get_object_properties() + ont.get_data_properties()
            triples = list(set(flatten([(a,b,c) for (a,b,c,d) in ont.get_triples()])))
            ont_name_filt = ont_name.split("/")[-1].rsplit(".",1)[0].replace("-", "_")
            mapping_ont[ont_name_filt] = ont
            extracted_elems.extend([ont_name_filt + "#" + elem for elem in entities + props + triples])

        extracted_elems = list(set(extracted_elems))
        inp = []
        for word in extracted_elems:
            ont_name = word.split("#")[0]
            elem = word.split("#")[1]
            try:
                ff = mapping_ont[ont_name].mapping_dict[elem]
            except Exception as e:
                print (e)
                pass
            inp.append(self.parse(mapping_ont[ont_name].mapping_dict.get(elem, elem)))

        print ("Total number of extracted unique classes and properties from entire RA set: ", len(extracted_elems))

        extracted_elems = ["<UNK>"] + extracted_elems

        return inp, extracted_elems


    def run_spellcheck(self, inp):
        # Spelling checker and corrector
        print ("Running spellcheck...")

        url = "https://grammarbot.p.rapidapi.com/check"

        headers = {
            'x-rapidapi-host': "grammarbot.p.rapidapi.com",
            'x-rapidapi-key': "9965b01207msh06291e57d6f2c55p1a6a16jsn0fb016da4a62",
            'content-type': "application/x-www-form-urlencoded"
            }

        inp_spellchecked = []

        for concept in inp_resolved:
            payload = "language=en-US&text=" + urllib.parse.quote_plus(concept)
            response = requests.request("POST", url, data=payload, headers=headers).json()
            concept_corrected = str(concept)
            
            for elem in response["matches"]:
                start, end = elem["offset"], elem["offset"] + elem["length"]
                concept_corrected = concept_corrected[:start] + elem["replacements"][0]["value"] + concept_corrected[end:]
            
            if concept.lower() != concept_corrected.lower():
                print ("{} corrected to {}".format(concept, concept_corrected))
                inp_spellchecked.append(concept_corrected)
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

    def get_one_hop_neighbours(self, ont, bag_of_neighbours=False):
        ont_obj = Ontology(ont)
        triples = ont_obj.get_triples()
        entities = [(a,b) for (a,b,c,d) in triples]
        neighbours_dict = {elem: [[] for i in range(4)] for elem in list(set(flatten(entities)))}
        for (e1, e2, p, d) in triples:
            if e1==e2:
                continue
            if bag_of_neighbours:
                e1_path = e1
                e2_path = e2
            else:
                e1_path = [e1]
                e2_path = [e2]
            if d == "Object Property":
                neighbours_dict[e1][2].append(e2_path)
                neighbours_dict[e2][2].append(e1_path)
            elif d == "Datatype Property":
                neighbours_dict[e1][3].append(e2_path)
                neighbours_dict[e2][3].append(e1_path)
            elif d == "Subclass":
                neighbours_dict[e2][1].append(e1_path)
            else:
                print ("Error wrong value of d: ", d)
        
        rootpath_dict = ont_obj.parents_dict
        rootpath_dict_new = {}
        for elem in rootpath_dict:
            rootpath_dict_new[elem] = self.path_to_root(elem, rootpath_dict, [], [])
        ont = ont.split("/")[-1].rsplit(".",1)[0].replace("-", "_")

        for entity in neighbours_dict:
            if bag_of_neighbours:
                neighbours_dict[entity][1] = [neighbours_dict[entity][1]]
                neighbours_dict[entity][2] = [neighbours_dict[entity][2]]
                neighbours_dict[entity][3] = [neighbours_dict[entity][3]]
            if entity in rootpath_dict_new and len(rootpath_dict_new[entity]) > 0:
                neighbours_dict[entity][0].extend(rootpath_dict_new[entity])
            else:
                continue

        neighbours_dict = {ont + "#" + el: [[tuple([ont + "#" + node for node in path]) for path in nbr_type]
                                        for nbr_type in neighbours_dict[el]] 
                       for el in neighbours_dict}
        neighbours_dict = {el: [[list(path) for path in nbr_type] for nbr_type in neighbours_dict[el]]
                       for el in neighbours_dict}
        return neighbours_dict

    def construct_neighbour_dicts(self, bag_of_neighbours=False):
        neighbours_dicts = {}
        for ont in list(set(flatten(self.ontologies_in_alignment))):
            neighbours_dicts = {**neighbours_dicts, **self.get_one_hop_neighbours(ont, bag_of_neighbours)}
        max_types = np.max([len([nbr_type for nbr_type in elem if flatten(nbr_type)]) for elem in neighbours_dicts.values()])
        return neighbours_dicts, max_types

    def process(self, spellcheck=False, bag_of_neighbours=False):
        all_mappings = self.generate_mappings()
        inp, extracted_elems = self.extract_keys()
        if self.language=="en":
            filtered_dict = self.construct_abbreviation_resolution_dict(all_mappings)
            inp_resolved = self.run_abbreviation_resolution(inp, filtered_dict)
            if spellcheck:
                try:
                    inp_resolved = self.run_spellcheck(inp_resolved)
                except:
                    pass
            inp = self.remove_stopwords(inp_resolved)
        emb_vals, emb_indexer, emb_indexer_inv = self.extract_embeddings(inp, extracted_elems)
        neighbours_dicts, max_types = self.construct_neighbour_dicts(bag_of_neighbours)

        return all_mappings, emb_indexer, emb_indexer_inv, emb_vals, neighbours_dicts, max_types
