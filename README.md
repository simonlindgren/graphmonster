# GRAPHMONSTER

Graphmonster will take an edgelist, create a graph, and remove noise from it. It then uses Node2Vec to find node embeddings, and umap (or t-sne) for reducing dimensionality. In the resulting visualised graph, nodes are coloured by community (Infomap algorithm) and sized by degree.

### Usage

```
python gm.py <parameters>
```
Files for the visualisation(`gm.svg` and `gm.pdf`) will be created.

### Parameters

`-f`, `--file`, name of edgelist file with one space-separated edge per line (e.g. `0 2` or `pig owl`), default = edgelist.txt

`--tsne`, set this flag to use t-sne instead of umap.

_Clustering_

`-k`, `--keep`, max number of communities to keep, default = 16.

_Node2Vec_

`-l`, `--length`, walk length, default = 16. 

`-n`, `--num`, number of walks per node, default = 10.

`-w`, `--win`, window for model, default = 10.

`-p`, `--pparam`, p-parameter (return), default = 1.

`-q`, `--qparam`, q-parameter (inout), default = 1.

_umap_

`-n`, `--nneigh`, similar to perplexity, recommended 5-50 (must be >1), default = 10.

`-m`, `-- mind`, minimum distance, emphasise local structure (low) or even distribution (high), recommended 0.001-0.5, default = 1.

`--mtrc`, metric (default = euclidean)

_t-sne_

`-x`, `--perp`, perplexity for t-sne, recommended 5-50, default = 20.

`-i`, `--iters`, number of iterations, default = 600 (min 250).

---

#### Twittergrab function
If your edgelist consists of Twitter user id numbers, you can use the twittergrab function. After running graphmonster (which will output a file named `gm.csv`), run:

```
python gm_tg.py
```

Twittergrabber will read the csv, call up the Twitter API, and get profile data on the top ten users (by degree) in each cluster. This information will be saved in `community-identification.txt`.

A valid set of Twitter api credentials must be provided in `credentials.py`.

---

#### Revisualise function

```
python gm_rv.py <parameters>
```

After running the graph creation script (`gm.py`), and possibly the twittergrab script (`gm_tg.py`), the revisualisation script (`gm_rv.py`) can be run iteratively to tweak the visualisation. This may include editing `commlabels.txt`, and alterations of the following parameters:

`-s`, `--size`, nodesize will be (log degree * size), default = 60.

`-a`, `--alpha`, alpha opacity of nodes, default = 0.6.

`-d`, `--dark`, darkmode graph

`--svg`, set this flag to also create an svg file

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

---

### Notes

**May 4, 2021** Had some troubles getting it to run. Found that having an updated version of `umap-learn` installed (incl. `numba` and `llvmlite`) was important.
