## Project moved at [cantinulab/Circe](Https://github.com/cantinilab/Circe)
<p align="center">
  <picture>
    <source media="(prefers-color-scheme: dark)" srcset="https://raw.githubusercontent.com/r-trimbour/atacnet/main/logo_dark_theme.svg" width="600">
    <source media="(prefers-color-scheme: light)" srcset="https://raw.githubusercontent.com/r-trimbour/atacnet/main/logo.svg" width="600">
    <img alt="Circe logo" src="https://raw.githubusercontent.com/r-trimbour/atacnet/main/logo.svg" width="600">
  </picture>
</p>

-----------------
# CIRCE: Cis-regulatory interactions between chromatin regions
[![Unit_Tests](https://github.com/r-trimbour/atacnet/actions/workflows/codecov.yaml/badge.svg)](https://github.com/r-trimbour/atacnet/actions/workflows/codecov.yaml)
[![codecov](https://codecov.io/gh/r-trimbour/atacnet/graph/badge.svg?token=0OIFAP28D7)](https://codecov.io/gh/r-trimbour/atacnet)
[![PyPI version](https://img.shields.io/pypi/v/atacnet?color=blue)](https://img.shields.io/pypi/v/atacnet)
[![Downloads](https://static.pepy.tech/badge/atacnet/month)](https://pepy.tech/project/atacnet)


## Description
This repo contains a python package for inferring **co-accessibility networks from single-cell ATAC-seq data**, using [skggm](https://www.github.com/skggm/skggm) for the graphical lasso and [scanpy](https://www.github.com/theislab/scanpy) for data processing.

It is based on the pipeline and hypotheses presented in the manuscript "Cicero Predicts cis-Regulatory DNA Interactions from Single-Cell Chromatin Accessibility Data" by Pliner et al. (2018). This R package [Cicero](https://cole-trapnell-lab.github.io/cicero-release/) is available [here](https://www.github.com/cole-trapnell-lab/cicero-release).

Results may slitghly vary between both packages, notably due to the different implementations of graphical lasso. 
<br> Currently, scores are very close when applied to the same metacells, computed from Cicero's methodology. (cf comparison plots below). It should run significantly faster than Cicero _(e.g.: running time of 5 sec instead of 17 min for the dataset 2)_.

_If you have any suggestion, don't hesitate ! This package is still a work in progress :)_


## Installation
The package can be installed using pip:

```
pip install atacnet
```

 and from github
```
pip install "git+https://github.com/r-trimbour/atacnet.git"
```

## Minimal example
```
import anndata as ad
import atacnet as an

atac = ad.read_h5ad('atac_data.h5ad')
an.add_region_infos(atac)
an.compute_atac_network(atac)
an.extract_atac_links(atac)
```

## Comparison to Cicero R package
<br> *On the same metacells obtained from Cicero code.*

All tests can be found in the [AtacNet benchmark repo](https://github.com/r-trimbour/atacnet_benchmark)

### Toy dataset 1 (fake data): 
- Pearson correlation coefficient: 0.999126
- Spearman correlation coefficient: 0.99838
<img src="https://raw.githubusercontent.com/r-trimbour/atacnet/main/Figures/correlation_toy_dataset1.png" align="center" width="480"/>

### Real dataset 2 (subsample of neurips PBMC)
- Pearson correlation coefficient: 0.999958
- Spearman correlation coefficient: 0.999911
<img src="https://github.com/r-trimbour/atacnet/raw/main/Figures/correlation_real_dataset2.png" align="center" width="480"/>

Performance on real dataset 2:
- Runtime: ~100x faster
- Memory usage: ~5x less
<img src="https://github.com/r-trimbour/atacnet/raw/main/Figures/perf_real_dataset2.png" align="center" width="480"/>

### Coming:

- _**Calculate metacells !**_
- _Add stats on similarity on large datasets._
- _Add stats on runtime, memory usage._
- This package can be run on multiple cores. _Currently working to speed up the mutlithreding use._
- _Fix seed for reproducibility._

## Usage
It is currently developped to work with AnnData objects. Check Example1.ipynb for a simple usage example.

