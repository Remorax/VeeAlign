# Multifaceted Context Representation using Dual Attention for Ontology Alignment


This repository contains the code for our submission to AAAI 2021: Multifaceted Context Representation using Dual Attention for Ontology Alignment.

## Setup
Please run the following commands: 

1. `conda create --name py37 python=3.7`
2. `pip3 install -r requirements.txt`

## Files

There are two main files:
1. `src/main.py`: This is the only runnable script. It conducts K-fold "sliding window" evaluation, and can be run to reproduce the results of multi-facted two-step attention, as explained in the paper.
2. `src/config.ini`: This file contains the configurational values for `main.py`, and can be adjusted to make the code run for `conference`, `lebensmittel`, `freizeit` and `web-directory`datasets. This is explained in further detail below.

We also provide the datasets used for experimentation, in the `datasets` folder (also included in the data appendix).

## Configuration

This section details in tabular format, the various configurational fields in `config.ini` and the best performing values for each of the datasets.

| Parameter          | Conference | Lebensmittel | Freizeit | Web-directory | Description                                                                                                   |
|--------------------|------------|--------------|----------|---------------|---------------------------------------------------------------------------------------------------------------|
| Name               | conference | lebensmittel | freizeit | web-directory | Name of dataset                                                                                               |
| Language           | en         | de           | de       | de            | Language of dataset                                                                                           |
| K                  | 7          | 5            | 5        | 5             | Value of K used in K-fold sliding window                                                                      |
| ontology_split     | True       | False        | False    | False         | Determines if training data should be split on ontology level (True) or on element level (False)          |
| max_false_examples | 150000     | 150000       | 150000   | 150000        | Max number of false (dissimilar) examples to take while training                                          |
| has_spellcheck     | True       | False        | False    | False         | Whether or not to use an English spelling checker while preprocessing.                                    |
| max_paths          | 5          | 2            | 16       | 1             | Max number of paths to consider, per node                                                                     |
| max_pathlen        | 6         | 1            | 1        | 3             | Max length of the path to consider                                                                            |
| bag_of_neighbours  | False       | False        | False    | True          | Determines whether one-hop neighbours are treated as a bag of nodes, or path of length one (see paper)     |
| weighted_sum       | False       | False        | False    | True          | Determines whether unified path representation is computed using weighted sum, or max pooling (see paper) |
| lr                 | 0.001      | 0.001        | 0.001    | 0.001         | Learning rate                                                                                                 |
| num_epochs         | 50         | 50           | 50       | 50            | Number of epochs                                                                                              |
| weight_decay       | 0.001      | 0.001        | 0.001    | 0.001         | Weight decay                                                                                                  |
| batch_size         | 32         | 32           | 32       | 32            | Batch size                                                                                                    |