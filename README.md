# VeeAlign

This repository contains the code for the Attention-based Ontology Alignment system, VeeAlign.

## Setup
Please run the following commands: 

1. `conda create --name py37 python=3.7`
2. `pip3 install -r requirements.txt`

## Datasets

Five datasets are made available with this repository, in the format required by the code:

1. [Conference Dataset](http://oaei.ontologymatching.org/2020/conference/index.html)
2. [Lebensmittel Dataset](http://dbs.uni-leipzig.de/file/mapping_lebensmittel.zip)
3. [Freizeit Dataset](http://dbs.uni-leipzig.de/file/mapping_freizeit.zip)
4. [Web directory Dataset](http://dbs.uni-leipzig.de/file/mapping_webdirectory.zip)
5. [Multifarm Dataset](https://www.irit.fr/recherches/MELODI/multifarm/dataset-2015-open.zip)

To add a new dataset:

1. Create a folder `<dataset_name>` inside the `datasets` directory. 
2. This directory should contain two subdirectories `ontologies` and `alignments`, which contain the ontologies being aligned and the alignments respectively.

There are no naming conventions wrt the naming of the ontology/alignment files, but it is assumed that the RDF alignment file contains `<onto1>` and `<onto2>` tags denoting the ontologies being aligned.

## Training & Testing

To train a model, run `python3 train.py`.

To test a model, run `python3 test.py <ontology1> <ontology2>`. 

Note that:

1. Both `train.py` and `test.py` will run with configurational parameters described in `config.ini`.
2. `train.py` will train on alignments specified in `config.ini`
3. `test.py` will load pretrained model specified in `config.ini`

 Details on the configurational parameters and how to set them for reproducibility are described [here](#configuration).

## Files

There are five main script files:
1. `src/train.py`: This is the script that needs to be run to train your model.
2. `src/test.py`: This is the script that needs to be run to test your model.
2. `src/data_preprocessing.py`: This script contains code to preprocess data for running.
3. `src/ontology.py`: This script contains code to parse the ontology.
4. `src/config.ini`: This file contains the configurational values for `main.py`, and can be adjusted to make the code run for `conference`, `lebensmittel`, `freizeit` and `web-directory`datasets. This is explained in further detail below.

We also provide the datasets used for experimentation, in the `datasets` folder (also included in the data appendix).

## Configuration

This section details in tabular format, the various configurational fields in `config.ini` and the best performing values for each of the datasets.

| Parameter          | Conference | Lebensmittel | Freizeit | Web-directory | Description                                                                                    |
|--------------------|------------|--------------|----------|---------------|------------------------------------------------------------------------------------------------|
| Name               | conference | lebensmittel | freizeit | web-directory | Name of dataset                                                                                |
| K                  | 7          | 5            | 5        | 5             | Value of K used in K-fold sliding window                                                       |
| ontology_split     | True       | False        | False    | False         | Split training data at ontology level (True) or on element level (False)                       |
| max_false_examples | 150000     | 150000       | 150000   | 150000        | Max number of false (dissimilar) examples to take while training                               |
| has_spellcheck     | True       | False        | False    | False         | Whether or not to use an English spelling checker while preprocessing.                         |
| max_paths          | 2          | 2            | 16       | 1             | Max number of paths to consider, per node                                                      |
| max_pathlen        | 26         | 1            | 1        | 3             | Max length of the path to consider                                                             |
| bag_of_neighbours  | True       | False        | False    | True          | Determines whether one-hop neighbours are treated as a bag of nodes, or path of length one     |
| weighted_sum       | True       | False        | False    | True          | Determines whether unified path representation is computed using weighted sum, or max pooling  |
| lr                 | 0.001      | 0.001        | 0.001    | 0.001         | Learning rate                                                                                  |
| num_epochs         | 50         | 50           | 50       | 50            | Number of epochs                                                                               |
| weight_decay       | 0.001      | 0.001        | 0.001    | 0.001         | Weight decay                                                                                   |
| batch_size         | 32         | 32           | 32       | 32            | Batch size                                                                                     |