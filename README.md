# On topological descriptors for graph products
[Mattie Ji](https://github.com/maroon-scorch) | [Amauri H. Souza](https://www.amauriholanda.org) |  [Vikas Garg](https://www.aalto.fi/en/people/vikas-kumar-garg)

This is the official repo for the paper [On topological descriptors for graph products](https://neurips.cc/virtual/2025/loc/san-diego/poster/117706) (NeurIPS 2025).

In our work, we study how persistent homology (PH) and Euler characteristics (EC) perform on box product of graphs with respect to some filtration. In particular, we establish theoretical bounds on the expressivity of PH and EC on graph products. In particular, we provide algorithms to compute the PH diagrams of the product of vertex- and edge-level filtrations on the graph product. The algorithms are **Theorem 4** and **Theorem 5** respectively in the paper, which are implemented in the folder [notebooks](notebooks).

## Citation

```bibtex
@inproceedings{
ji2025on,
title={On topological descriptors for graph products},
author={Mattie Ji and Amauri H. Souza and Vikas K. Garg},
booktitle={The Thirty-ninth Annual Conference on Neural Information Processing Systems (NeurIPS)},
year={2025},
url={https://openreview.net/forum?id=VCORb5Fw8e}
}
```


## Requirements

### Implementation and Runtime
For the folders [notebooks](notebooks) and [runtime](runtime):

```
google              NA
gudhi               3.11.0
matplotlib          3.10.0
networkx            3.5
numpy               2.0.2
torch               2.8.0+cu126
```
The jupyter notebooks were made in the setting:
```
-----
IPython             7.34.0
jupyter_client      7.4.9
jupyter_core        5.9.1
notebook            6.5.7
-----
Python 3.12.12
Linux-6.6.105+-x86_64-with-glibc2.35
-----
```

## Graph Classification

For the real-world graph classification experiments, we first preprocess the datasets augmented with persistence tuples, e.g.:
```
precompute_data.py \
    --dataset NCI1 \
    --filtration_type full_prod \
    --filtration_fn degree \
    --folder pre_processed
```

Above, ```full_prod``` refers to computing the Cartesian product of the graphs followed by the computation of their persistence diagrams.
We also support ```vertex_prod```, which denotes the vertex-level product filtration (as described in Theorem 4 of the paper).

Then, we use the ```main.py``` script to train persistence-augmented GNNs. For example:
``` 
python main.py \
    --dataset NCI1 \
    --filtration_type full_prod \
    --gnn_depth 2 \
    --seed 42 \
    --gnn gin \
    --logdir results \
    --max_epochs 500
```

## Acknowledgments
Some of the routines in this repository were modified from the code in [RePHINE](https://github.com/Aalto-QuML/RePHINE).