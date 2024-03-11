# GRAPHMONSTER

Graphmonster will take an edgelist, create a graph, and remove noise from it. It then uses Node2Vec to find node embeddings, and umap (or t-sne) for reducing dimensionality. In the resulting visualised graph, nodes are coloured by community (Infomap algorithm) and sized by degree.

## First step: Remove noise and train embeddings

```
python train.py <parameters>
```

### Parameters
_Input_

`-f`, `--file`, name of edgelist file with one space-separated edge per line (e.g. `0 2` or `pig owl`), default = edgelist.txt

_Clustering_

`-k`, `--keep`, max number of communities to keep, default = 16.

_Node2Vec_

`-l`, `--length`, walk length, default = 16. 

`-n`, `--num`, number of walks per node, default = 10.

`-w`, `--win`, window for model, default = 10.

`-p`, `--pparam`, p-parameter (return), default = 1.

`-q`, `--qparam`, q-parameter (inout), default = 1.

----

`--tsne`, set this flag to use t-sne instead of umap.

## Second step: Dimensionality reduction and initial plot

```
python reduce.py <parameters>
```

_umap (default)_

`-n`, `--nneigh`, similar to perplexity, recommended 5-50 (must be >1), default = 10.

`-m`, `--mind`, minimum distance, emphasise local structure (low) or even distribution (high), recommended 0.001-0.5, default = 1.

`--mtrc`, metric (default = euclidean)

_t-sne_
`--tsne`, set this flag to use t-sne instead of umap

`-x`, `--perp`, perplexity for t-sne, recommended 5-50, default = 20.

`-i`, `--iters`, number of iterations, default = 600 (min 250).

---

#### Labelling communities

When running `gm.py`, the files `commlabels.txt` and `gm.csv` will be created. The community info in the csv can be used, depending on the character of your data, to look up more info about the nodes as input to setting the labels in the txt.


---

#### Revisualise function

```
python gm_rv.py <parameters>
```

After running the graph creation script (`gm.py`), the revisualisation script (`gm_rv.py`) can be run iteratively to tweak the visualisation. This may include editing `commlabels.txt`, and alterations of the following parameters:

`-s`, `--size`, nodesize will be (log degree * size), default = 60.

`-a`, `--alpha`, alpha opacity of nodes, default = 0.6.

`-d`, `--dark`, darkmode graph

`--svg`, set this flag to also create an svg file



Files for the visualisation(`gm.svg` and `gm.pdf`) will be created.

---

### Prerequisites

Run the following command to install package dependencies:

```
pip install -r requirements.txt
```
Newer versions might work, but here is a working setup as of 2024:
- networkx 2.7.1
- node2vec 0.4.6
- scikit-learn 1.0.2
- umap-learn 0.5.5
- seaborn 0.13.2
- infomap 2.7.1
