# HTracker

Reproducible material for **A Joint Inversion-Segmentation approach to Assisted Seismic Interpretation -
Ravasi M., Birnie, C.** available as [Arxiv Preprint](https://arxiv.org/abs/2102.03860).

## Project structure
This repository is organized as follows:
* **htracker**:   python library containing routines for joint inversion-segmentation and tracking of horizons 
* **notebooks**:  set of jupyter notebooks reproducing figures in the paper

## Getting started
To ensure reproducibility of the results, we suggest using the `environment.yml` file when creating an environment.
Provided that Anaconda (or Miniconda) is avaiable in your system, to create a new environment simply run:

```
conda env create -f environment.yml
```

to create an environment called `htracker`.