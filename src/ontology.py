import itertools
from xml.dom import minidom
from urllib.request import urlopen

flatten = lambda l: [item for sublist in l for item in sublist]

# Define class to parse ontology
class Ontology():
    def __init__(self, ontology):
        '''
        Instantiates an Ontology object. 
        Args:
            ontology: Path to ontology file
        Returns:
            Parsed Ontology object
        '''
        self.ontology = ontology
        if ontology.startswith("https://") or ontology.startswith("http://"):
            self.ontology_obj = minidom.parse(urlopen(ontology + "/"))
        else:
            self.ontology_obj = minidom.parse(ontology)
        self.root = self.ontology_obj.documentElement # root node
        
        self.languages = []
        self.construct_mapping_dict()
        # Detects language of ontology by taking max of all lang attributes in labels
        self.detect_language()
        
        # Dict that records the immediate "subclass_of" parent of any entity
        self.parents_dict = {}

        self.subclasses = self.parse_subclasses()
        self.object_properties = self.parse_object_properties()
        self.data_properties = self.parse_data_properties()
        self.triples = self.parse_triples()
        self.classes = self.parse_classes()        
    
    def construct_mapping_dict(self):
        '''
        Constructs ID to label mapping dict for ontologies where 
        entities are identified by IDs.
        '''
        elements = self.root.getElementsByTagName("owl:Class") + self.root.getElementsByTagName("owl:ObjectProperty") + self.root.getElementsByTagName("owl:DatatypeProperty")
        self.mapping_dict = {self.extract_ID(el, False): self.return_label(el) for el in elements if self.get_child_node(el, "rdfs:label")}
        self.mapping_dict_inv = {self.mapping_dict[key]: key for key in self.mapping_dict}
        return
    
    def return_label(self, el):
        '''
        Returns label of an element, and also detects language of the label
        '''
        label_element = self.get_child_node(el, "rdfs:label")[0]
        lang = label_element.getAttribute("xml:lang")
        if lang:
            self.languages.append(lang)
        return label_element.firstChild.nodeValue

    def detect_language(self):
        if self.languages:
            self.language = max(set(self.languages), key=self.languages.count)
        else:
            self.language = "en"

    def get_child_node(self, element, tag):
        '''
        Returns child node with a specific tag name given DOM element 
        Args:
            element: DOM parent element
            tag: Name of tag of child
        Returns:
            DOM child element 
        '''
        return [e for e in element._get_childNodes() if type(e)==minidom.Element and e._get_tagName() == tag]
        
    def has_attribute_value(self, element, attribute, value):
        '''
        Checks whether DOM element has attribute with a particular value
        Args:
            element: DOM parent element
            attribute: Attribute ID
            value: Required value of attribute
        Returns:
            boolean
        '''
        return True if element.getAttribute(attribute).split("#")[-1] == value else False
    
    def get_subclass_triples(self, check_coded=False):
        '''
        Returns subclass triples
        '''
        subclasses = self.get_subclasses(check_coded)
        return [(b,a,c,d) for (a,b,c,d) in subclasses]
    
    def parse_triples(self, union_flag=0, subclass_of=True, check_coded=False):
        '''
        Parses ontology to obtain object property, data property and subclass triples 
        Args:
            union_flag: 0, if classes containing union of n classes are to be denoted 
            as a single class or n separate classes, else 1
            subclass_of: Determines if subclass triples should be returned or not
            check_coded: Determines if element should be queried for its label while extracting ID
        Returns:
            list of 4-tuples of the form (a,b,c,d) where d is the type of property c is.
        '''
        obj_props = [(prop, "Object Property") for prop in self.object_properties]
        data_props = [(prop, "Datatype Property") for prop in self.data_properties]
        props = obj_props + data_props
        all_triples = []
        for prop, prop_type in props:
            domain_children = self.get_child_node(prop, "rdfs:domain")
            range_children = self.get_child_node(prop, "rdfs:range")
            domain_prop = self.filter_null([self.extract_ID(el) for el in domain_children])
            range_prop = self.filter_null([self.extract_ID(el) for el in range_children])
            if not domain_children or not range_children:
                # Either domain is not present or range is not present
                continue
            if not domain_prop:
                domain_prop = self.filter_null([self.extract_ID(el) for el in domain_children[0].getElementsByTagName("owl:Class")])
            if not range_prop:
                range_prop = self.filter_null([self.extract_ID(el) for el in range_children[0].getElementsByTagName("owl:Class")])
            if domain_prop and range_prop:
                if union_flag == 0:
                    all_triples.extend([(el[0], el[1], self.extract_ID(prop), prop_type) for el in list(itertools.product(domain_prop, range_prop))])
                else:
                    all_triples.append(("###".join(domain_prop), "###".join(range_prop), self.extract_ID(prop), prop_type))
        if subclass_of:
            all_triples.extend(self.get_subclass_triples(check_coded))
        return list(set(all_triples))
    
    def get_triples(self, union_flag=0, subclass_of=True, check_coded=False):
        '''
        Wrapper on top of parse_triples
        '''
        return self.parse_triples(union_flag, subclass_of, check_coded)

    def parse_subclasses(self):
        '''
        Parses ontology to obtain subclass triples
        Returns:
            list of subclass triples of the form (a,b,c,d)
        '''
        subclasses = self.root.getElementsByTagName("rdfs:subClassOf")
        subclass_pairs = []
        for el in subclasses:
            inline_subclasses = self.extract_ID(el)
            if inline_subclasses:
                # Subclass of with inline IDs
                subclass_pairs.append((el, el.parentNode, "subclass_of", "Subclass"))
            else:
                level1_class = self.get_child_node(el, "owl:Class")
                if not level1_class:
                    # Subclass of relations with owl Restrictions
                    restriction = el.getElementsByTagName("owl:Restriction")
                    if not restriction:
                        continue
                    prop = self.get_child_node(restriction[0], "owl:onProperty")
                    some_vals = self.get_child_node(restriction[0], "owl:someValuesFrom")
                    
                    if not prop or not some_vals:
                        continue
                    try:
                        if self.extract_ID(prop[0]) and self.extract_ID(some_vals[0]):
                            subclass_pairs.append((el.parentNode, some_vals[0], self.extract_ID(prop[0]), "Object Property"))
                        elif self.extract_ID(prop[0]) and not self.extract_ID(some_vals[0]):
                            class_vals = self.get_child_node(some_vals[0], "owl:Class")
                            subclass_pairs.append((el.parentNode, class_vals[0], self.extract_ID(prop[0]), "Object Property"))
                        elif not self.extract_ID(prop[0]) and self.extract_ID(some_vals[0]):
                            prop_vals = self.get_child_node(prop[0], "owl:ObjectProperty")
                            subclass_pairs.append((el.parentNode, some_vals[0], self.extract_ID(prop_vals[0]), "Object Property"))
                        else:
                            prop_vals = self.get_child_node(prop[0], "owl:ObjectProperty")
                            class_vals = self.get_child_node(some_vals[0], "owl:Class")
                            subclass_pairs.append((el.parentNode, class_vals[0], self.extract_ID(prop_vals[0]), "Object Property"))
                    except Exception as e:
                        try:
                            if not self.extract_ID(prop[0]) and self.extract_ID(some_vals[0]):
                                prop_vals = self.get_child_node(prop[0], "owl:DatatypeProperty")
                                subclass_pairs.append((el.parentNode, some_vals[0], self.extract_ID(prop_vals[0]), "Datatype Property"))
                            elif not self.extract_ID(prop[0]) and not self.extract_ID(some_vals[0]):
                                prop_vals = self.get_child_node(prop[0], "owl:DatatypeProperty")
                                class_vals = self.get_child_node(some_vals[0], "owl:Class")
                                subclass_pairs.append((el.parentNode, class_vals[0], self.extract_ID(prop_vals[0]), "Datatype Property"))
                        except Exception as e:
                            continue
                else:
                    if self.extract_ID(level1_class[0]):
                        # Subclass label under a level 1 tag
                        subclass_pairs.append((level1_class[0], el.parentNode, "subclass_of", "Subclass"))
                    else:
                        continue
        return subclass_pairs
        
    def get_subclasses(self, check_coded=False):
        '''
        Extracts entity ID from parsed subclass triples
        '''
        subclasses = [(self.extract_ID(a, not check_coded), self.extract_ID(b, not check_coded), c, d) for (a,b,c,d) in self.subclasses]
        self.parents_dict = {}
        for (a,b,c,d) in subclasses:
            if c == "subclass_of" and a!="Thing" and b!="Thing":
                if b not in self.parents_dict:
                    self.parents_dict[b] = [a]
                else:
                    self.parents_dict[b].append(a)
        return [el for el in subclasses if el[0] and el[1] and el[2] and el[0]!="Thing" and el[1]!="Thing"]
    
    def filter_null(self, data):
        return [el for el in data if el]
    
    def extract_ns(self):
        '''
        Extracts namespace of an ontology
        '''
        ns = self.ontology_obj.getElementsByTagName("rdf:RDF")[0].getAttribute("xmlns")
        if ns[-1] == "#":
            return ns
        return self.ontology_obj.doctype.entities.item(0).firstChild.nodeValue

    def extract_ID(self, element, check_coded = True):
        '''
        Returns ID for a parsed DOM element. In ontologies where classes are represented by 
        numerical IDs, it returns the label (stored in mapping_dict)
        '''
        element_id = element.getAttribute("rdf:ID") or element.getAttribute("rdf:resource") or element.getAttribute("rdf:about")
        element_id = element_id.split("#")[-1].split(";")[-1]
        if len(list(filter(str.isdigit, element_id))) >= 3 and "_" in element_id and check_coded:
            return self.mapping_dict[element_id]
        return element_id.replace("UNDEFINED_", "").replace("DO_", "")

    def parse_classes(self):
        '''
        Parse all entities, including domain and range entities in ontology
        '''
        class_elems = [self.extract_ID(el) for el in self.root.getElementsByTagName("owl:Class")]
        subclass_classes = list(set(flatten([el[:-2] for el in self.triples])))
        return list(set(self.filter_null(class_elems + subclass_classes)))
    
    def get_classes(self):
        return self.classes
    
    def get_entities(self):
        '''
        Parse only classes
        '''
        entities = [self.extract_ID(el) for el in self.root.getElementsByTagName("owl:Class")]
        return list(set(self.filter_null(entities)))

    def parse_data_properties(self):
        '''
        Parse all datatype properties, including functional and inverse functional datatype properties
        '''
        data_properties = [el for el in self.get_child_node(self.root, 'owl:DatatypeProperty')]
        fn_data_properties = [el for el in self.get_child_node(self.root, 'owl:FunctionalProperty') if el]
        fn_data_properties = [el for el in fn_data_properties if type(el)==minidom.Element and 
            [el for el in self.get_child_node(el, "rdf:type") if 
             self.has_attribute_value(el, "rdf:resource", "DatatypeProperty")]]
        inv_fn_data_properties = [el for el in self.get_child_node(self.root, 'owl:InverseFunctionalProperty') if el]
        inv_fn_data_properties = [el for el in inv_fn_data_properties if type(el)==minidom.Element and 
            [el for el in self.get_child_node(el, "rdf:type") if 
             self.has_attribute_value(el, "rdf:resource", "DatatypeProperty")]]
        return data_properties + fn_data_properties + inv_fn_data_properties
        
    def parse_object_properties(self):
        '''
        Parse all object properties, including functional and inverse functional object properties
        '''
        obj_properties = [el for el in self.get_child_node(self.root, 'owl:ObjectProperty')]
        fn_obj_properties = [el for el in self.get_child_node(self.root, 'owl:FunctionalProperty') if el]
        fn_obj_properties = [el for el in fn_obj_properties if type(el)==minidom.Element and 
            [el for el in self.get_child_node(el, "rdf:type") if 
             self.has_attribute_value(el, "rdf:resource", "ObjectProperty")]]
        inv_fn_obj_properties = [el for el in self.get_child_node(self.root, 'owl:InverseFunctionalProperty') if el]
        inv_fn_obj_properties = [el for el in inv_fn_obj_properties if type(el)==minidom.Element and 
            [el for el in self.get_child_node(el, "rdf:type") if 
             self.has_attribute_value(el, "rdf:resource", "ObjectProperty")]]
        return obj_properties + fn_obj_properties + inv_fn_obj_properties
    
    def get_object_properties(self):
        '''
        Wrapper to obtain object properties and parse them to return property IDs 
        '''
        obj_props = [self.extract_ID(el) for el in self.object_properties]
        return list(set(self.filter_null(obj_props)))
    
    def get_data_properties(self):
        '''
        Wrapper to obtain data properties and parse them to return property IDs 
        '''
        data_props = [self.extract_ID(el) for el in self.data_properties]
        return list(set(self.filter_null(data_props)))
