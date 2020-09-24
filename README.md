BANE 
============================================
[![PWC](https://img.shields.io/endpoint.svg?url=https://paperswithcode.com/badge/binarized-attributed-network-embedding/link-prediction-on-cora)](https://paperswithcode.com/sota/link-prediction-on-cora?p=binarized-attributed-network-embedding) ![License](https://img.shields.io/github/license/benedekrozemberczki/BANE.svg?color=blue&style=plastic) [![codebeat badge](https://codebeat.co/badges/3728636b-e766-429e-a453-4c4349e56d57)](https://codebeat.co/projects/github-com-benedekrozemberczki-bane-master)

<p align="justify">
An implementation of "Binarized Attributed Network Embedding". Attributed network embedding enables joint representation learning of node links and attributes. Existing attributed network embedding models are designed in continuous Euclidean spaces which often introduce data redundancy and impose challenges to storage and computation costs. To this end, we present a Binarized Attributed Network Embedding model (BANE for short) to learn binary node representation. Specifically, we define a new Weisfeiler-Lehman proximity matrix to capture data dependence between node links and attributes by aggregating the information of node attributes and links from neighboring nodes to a given target node in a layer-wise manner. Based on the Weisfeiler-Lehman proximity matrix, we formulate a new Weisfiler-Lehman matrix factorization learning function under the binary node representation constraint. The learning problem is a mixed integer optimization and an efficient cyclic coordinate descent (CCD) algorithm is used as the solution. Node classification and link prediction experiments on real-world datasets show that the proposed BANE model outperforms the state-of-the-art network embedding methods.
</p>
<div style="text-align:center"><img src ="bane.jpeg" ,width=720/></div>

The model is now also available in the package [Karate Club](https://github.com/benedekrozemberczki/karateclub).

This repository provides an implementation for BANE as described in the paper:

> **Binarized Attributed Network Embedding**
> YHong Yang, Shirui Pan, Peng Zhang, Ling Chen, Defu Lian, and Chengqi Zhang.
> ICDM, 2018.
> [[Paper]](https://www.researchgate.net/publication/328688614_Binarized_Attributed_Network_Embedding)

The reference dense MatLab implementation is available [[here]](https://github.com/ICDM2018-BANE/BANE).

### Requirements

The codebase is implemented in Python 3.5.2. package versions used for development are just below.
```
networkx          2.4
tqdm              4.28.1
numpy             1.15.4
pandas            0.23.4
texttable         1.5.0
scipy             1.1.0
argparse          1.1.0
sklearn           0.20.0
```
### Datasets
<p align="justify">
The code takes an input graph in a csv file. Every row indicates an edge between two nodes separated by a comma. The first row is a header. Nodes should be indexed starting with 0. Sample graphs for the `Twitch Brasilians` ,`Wikipedia Chameleons` and `Wikipedia Giraffes` are included in the  `input/` directory. </p>

The feature matrix can be stored two ways:
<p align="justify">
If the feature matrix is a **sparse binary** one it is stored as a json. Nodes are keys of the json and features are the values. For each node feature column ids are stored as elements of a list. The feature matrix is structured as:</p>

```javascript
{ 0: [0, 1, 38, 1968, 2000, 52727],
  1: [10000, 20, 3],
  2: [],
  ...
  n: [2018, 10000]}
```
<p align="justify">
If the feature matrix is **dense** it is assumed that it is stored as csv with comma separators. It has a header, the first column contains node identifiers and it is sorted by these identifers. It should look like this:</p>

| **NODE ID**| **Feature 1** | **Feature 2** | **Feature 3** | **Feature 4** |
| --- | --- | --- | --- |--- |
| 0 | 3 |0 |1.37 |1 |
| 1 | 1 |1 |2.54 |-11 |
| 2 | 2 |0 |1.08 |-12 |
| 3 | 1 |1 |1.22 |-4 |
| ... | ... |... |... |... |
| n | 5 |0 |2.47 |21 |


### Options
<p align="justify">
Learning of the embedding is handled by the `src/main.py` script which provides the following command line arguments.</p>

#### Input and output options

```
  --edge-path    STR     Input graph path.           Default is `input/ptbr_edges.csv`.
  --feature-path STR     Input Features path.        Default is `input/ptbr_features.json`.
  --output-path  STR     Embedding path.             Default is `output/ptbr_bane.csv`.
```

#### Model options

```
  --features               STR         Structure of the feature matrix.       Default is `sparse`. 
  --dimensions             INT         Number of embeding dimensions.         Default is 48.
  --order                  INT         Order of adjacency matrix powers.      Default is 1.
  --binarization-rounds    INT         Number of power interations.           Default is 10.
  --approximation-rounds   INT         Number of CDC interations.             Default is 5.
  --alpha                  FLOAT       Regularization parameter.              Default is 0.7.
  --gamma                  FLOAT       Weisfeiler-Lehman mixing parameter.    Default is 0.1.  
```

### Examples
<p align="justify">
The following commands learn a graph embedding and write the embedding to disk. The node representations are ordered by the ID.</p>
<p align="justify">
Creating a BANE embedding of the default dataset with the default hyperparameter settings. Saving the embedding at the default path.</p>

```sh
$ python src/main.py
```
Creating a BANE embedding of the default dataset with 128 dimensions and approximation order 1.

```sh
$ python src/main.py --dimensions 128 --order 1
```

Creating a BANE embedding of the default dataset with asymmetric mixing.

```sh
$ python src/main.py --gamma 0.1
```

Creating an embedding of an other dense structured dataset the `Wikipedia Giraffes`. Saving the output in a custom folder.

```sh
$ python src/main.py --edge-path input/giraffe_edges.csv --feature-path input/giraffe_features.csv --output-path output/giraffe_bane.csv --features dense
```
