# Orchid 🌸 – Ollivier-Ricci Curvature for Hypergraphs: A Unified Framework

<table>
    <tr>
        <td>
            This repository provides a Julia library and a command-line interface that implements the <i>Ollivier-Ricci Curvature for Hypergraphs in Data</i> (Orchid) Framework. <br/><br/>
            This project is based on the research paper <a href="https://doi.org/10.48550/arXiv.2210.12048">Ollivier-Ricci Curvature for Hypergraphs: A Unified Framework</a>, published at ICLR 2023. <br/><br/>
            The full reproducibility package, including the data that can be shared, is available on <a href="https://doi.org/10.5281/zenodo.7624573">Zenodo</a>. <br/><br/>
            If you find this repository helpful, please consider citing our paper!
        </td>
    <td>
        <img
  src="/orchid_thumbnail.png"
  alt="Orchid Thumbnail"
  style="display: inline-block; margin: 0 auto; width: 300px">
        </td>    
</tr>
</table>

```bibtex
@inproceedings{coupette2023orchid,
    title     = {Ollivier-Ricci Curvature for Hypergraphs: A Unified Framework},
    author    = {Corinna Coupette and Sebastian Dalleiger and Bastian Rieck},
    booktitle = {The Eleventh International Conference on Learning Representations (ICLR)},
    year      = {2023},
    url       = {https://openreview.net/forum?id=sPCKNl5qDps},
    doi       = {10.48550/arXiv.2210.12048}
}
```

## Installation

### Julia Library

To install the Orchid Julia library:
```julia-repl
julia> using Pkg
julia> Pkg.add(url="https://github.com/aidos-lab/orchid.git")
```
Alternatively, we can install Orchid from the command line:
```sh
julia -e 'using Pkg; Pkg.add(url="https://github.com/aidos-lab/orchid.git")'
```

### Command-Line Interface
To use the command-line interface, we additionally need `bin/orchid.jl` and its dependencies.
```sh
git clone https://github.com/aidos-lab/orchid
julia -e 'using Pkg; Pkg.add(path="./orchid"); Pkg.add.(["ArgParse", "JSON", "Glob", "CodecZlib"])'
```

## Usage

### Julia REPL

Assuming the hypergraph resides in variable `X`:

```julia-repl
julia> using Orchid
julia> hypergraph_curvatures(DispersionUnweightedStar, AggregationMax, X, 0.01)

help?> Orchid.hypergraph_curvatures
```

To inspect the results:

    hypergraph_curvatures

### Arguments
- `disperser`: Dispersion (options: DisperseUnweightedClique, DisperseWeightedClique, or DisperseUnweightedStar – Orchid paper: μ)
- `aggregator`: Aggregation (options: AggregateMean, AggregateMax, or (AggregateMean, AggregateMax) – Orchid paper: AGG)
- `input`: Incidence-matrix or edge-list encoding of the hypergraph
- `alpha`: Self-dispersion weight (smoothing parameter corresponding to the laziness of the random walk – Orchid paper: α)
- `cost`: Cost computation strategy (options: CostOndemand^, CostMatrix)

    ^ useful for very large hyper graphs.

### Command-Line Interface 

To use the command-line interface:  

```sh
chmod +x bin/orchid.jl
bin/orchid.jl --help
bin/orchid.jl --aggregation mean --dispersion WeightedClique -i data/toy.ihg.tsv -o results/toy.orc.json 
bin/orchid.jl --aggregation max --dispersion UnweightedStar --alpha 0.1 -i data/toys.chg.tsv -o results/toys.orc.json
```

The first execution might take some time.

### Bash Scripts

For convenience, we provide bash scripts to perform the curvature computations in the configurations reported in the ICLR paper for the shareable datasets used in the paper as well as (for illustration) for tiny toy data. 
Both scripts compute curvatures with alpha in {0.0,0.1,0.2,0.3,0.4,0.5} for all combinations of dispersion and aggregation:

- `reproduce.sh`: Computation for `{dblp,ndc-ai,ndc-pc}.ihg.tsv` and `{dblp-v,mus,sha,stex,syn_hcm,syn_hcm-hsbm,syn_hnmp,syn_hsbm}.chg.tsv`; results are stored to `results` folder as gzip-compressed JSON files.
- `reproduce_toy.sh`: Computation for `toy.ihg.tsv` and `toys.chg.tsv`; results are stored to `results` folder as uncompressed JSON files.

Note that `reproduce.sh`, when run as-is, will consume considerable computational resources. 
The easiest way to restrict computation to smaller datasets or some parts of our configuration space is to redefine some of the arrays at the top of the script.

## Experiments

To evaluate our curvature results, we require additional python packages.
We recommend installing these into a virtual environment, the classic option being [venv](https://docs.python.org/3/library/venv.html).
```sh
pip install -r experiments/requirements.txt
```

For our clustering, MMD, and kPCA experiments on collections of hypergraphs, we first compute their curvatures.
```sh
bin/orchid.jl --aggregation mean --dispersion WeightedClique -i data/syn_hcm-hsbm.chg.tsv.gz -o results/syn_hcm-hsbm.orc.json.gz 
```
Then, we evaluate the collection of curvatures using the tools in `experiments/`.
```sh
python experiments/graph-clustering.py -k 2 -i results/syn_hcm-hsbm.orc.json.gz -o gc/syn_hcm-hsbm.gc.json.gz 
python experiments/kpca.py -k 2 -i results/syn_hcm-hsbm.orc.json.gz -o kpca/syn_hcm-hsbm.kpca.json.gz 
python experiments/mmd.py -i results/syn_hcm-hsbm.orc.json.gz -o mmd/syn_hcm-hsbm.mmd.json.gz 
```

For our node-clustering experiments with individual hypergraphs, we proceed similarly, now computing curvatures before we cluster the nodes.
```sh
bin/orchid.jl --aggregation mean --dispersion WeightedClique -i data/dblp.ihg.tsv.gz -o results/dblp.orc.json.gz
python experiments/node-clustering.py -k 2 -i results/dblp.orc.json.gz -o nc/dblp.nc.json.gz
```

To produce the files containing the competing local features, which can be input to the experiment scripts in place of the curvature files:

```sh
python experiments/features.py -i data/sha.chg.tsv.gz -o features/sha.chg.json.gz
```

## Data Formats used by the Command-Line Interface

### Inputs

The data underlying our experiments are provided in a concise tsv format which allows us, inter alia, to store an entire hypergraph collection in *one* file.  
The files encoding *individual hypergraphs* end with `ihg.tsv[.gz]`.  
The files encoding *collections of hypergraphs* end with `chg.tsv[.gz]`.  
Nodes are assumed to be consecutive, *one-indexed* integers.

#### Individual hypergraphs (ihg): {name}.ihg.tsv.gz

Each row is a hyperedge, with the identifiers of nodes occurring in the hyperedge separated by `\t` characters.

Example (`data/toy.ihg.tsv`):
```sh
1   2   3   4   5
2   3
5   7   3   6
```

#### Collections of hypergraphs (chg): {name}.chg.tsv.gz

Just like the format for individual hypergraphs, 
except that now the *first* identifier in each row identifies the hypergraph to which the hyperedge belongs.

Example (`data/toys.chg.tsv`): 
```sh
2   1   2   3   4   5
2   2   3
2   5   7   3   6
0   1   2   4
0   1   3   5
0   1   4   6
0   6   4   2   5
```

Note that Orchid will treat the hypergraphs in the order in which their unique identifiers appear in the input, so in the example above, the hypergraph with ID 2 will occur before the hypergraph with ID 0 in the results.  
The example also illustrates that we do not assume the hypergraph identifiers to be one-indexed or consecutive.

### Outputs

Curvature files are (optionally: gzip-compressed) JSON files of the form:

```sh
[
  {
    "node_curvature_neighborhood":[...],
    "directional_curvature":[
      [...i values...],
      [...j values...],
      [...k values...]
    ],
    "node_curvature_edges":[...],
    "edge_curvature":[...],
    "aggregation":"Orchid.AggregateMax",
    "dispersion":"UnweightedStar",
    "input":"../data/toys.chg.tsv",
    "alpha":0.1
  },
  {
    ...
  }
]
```

That is, we provide a list of JSON objects, one for each input hypergraph.  
If the input is an individual hypergraph, the list will just have one entry.  
If the input is a collection of hypergraphs, the list will contain the hypergraphs in the order they were found in the input file. 

## Disclaimer

We refactored the entire code base and introduced the {ihg,chg}.tsv[.gz] data format after ICLR 2023. 
The material results are the same, but there might be small deviations in details.

## Contributing

Contributions to Orchid are welcome.  
If you find any issues or have suggestions for improvements, please open an issue or submit a pull request in the GitHub repository: https://github.com/aidos-lab/orchid
