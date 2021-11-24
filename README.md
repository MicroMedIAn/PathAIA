<div align="center">

# PathAIA


**Simple digital pathology analysis tools.**

---

<p align="center">
  <a href="#basic-usage">Basic Usage</a> •
  <a href="#advanced-features">Advanced features</a> •
  <a href="https://linktothedoc.com">Docs</a> •
  <a href="#community">Community</a> •
  <a href="#license">License</a>
</p>

<!-- DO NOT ADD CONDA DOWNLOADS... README CHANGES MUST BE APPROVED BY EDEN OR WILL -->
[![PyPI - Python Version](https://img.shields.io/pypi/pyversions/pathaia)](https://pypi.org/project/pathaia/)
[![PyPI Status](https://badge.fury.io/py/pathaia.svg)](https://badge.fury.io/py/pathaia)
[![PyPI Status](https://pepy.tech/badge/pathaia)](https://pepy.tech/project/pathaia)
[![codecov](https://codecov.io/gh/ArnaudAbreu/PathAIA/branch/master/graph/badge.svg?token=SE4ZX0LXN6)](https://codecov.io/gh/ArnaudAbreu/PathAIA)
[![Documentation Status](https://readthedocs.org/projects/pathaia/badge/?version=latest)](https://pathaia.readthedocs.io/en/latest/?badge=latest)
[![license](https://img.shields.io/badge/License-GPLv3-blue.svg)](https://github.com/ArnaudABreu/PathAIA/blob/master/LICENSE)

</div>


---

## PathAIA aims to standardize and automate most of WSI analysis in digital pathology
If you feel like you keep rewriting the same code over and over again when working on Whole Slide Images and you wish there were a nicely integrated library to automate all this, you came to the right place. With PathAIA we aim to create a fast, high level and modular library to work on WSI at scale in order to perform image analysis or to create a well rounded dataset for your machine learning project.

---

## Basic Usage

### Step 0: Install

Simple installation from PyPI
```bash
pip install pathaia
```

### Step 1: Import pathaia's patch extraction tool

```python
from pathaia.patches import patchify_folder_hierarchically
```

### Step 2: Define your extraction parameters
You can extract at multiple pyramid levels with a hierarchical structure between patches of different levels. You can control pretty much every extraction parameter you like, from patch size to interval between patches or filters to chose which patch to extract. You can also decide whether you want to save patches as png or just extract coordinates in csv.

```python
infolder = "/path/to/input/slide/folder"
outfolder = "/path/to/output/patches/folder"
top_level = 5
low_level = 0
psize = 224
interval = {"x": 224, "y": 224}
silent = list(range(low_level, top_level+1))
extensions = [".svs"]
recurse = False
slide_filters = ["full"]
verbose = 2
```
With these parameters you will find all svs slides that are directly in `infolder` and extract patch coordinates from levels 0 to 5 with a hierarchical structure. No png image will be stored as `silent` lists all levels. Patches will be contiguous with size 224 and will only be extracted from tissue zone that are determined by filtering on slide thumbnails. With `verbose=2` thumbnails of extracted areas are also stored on disk.

### Step 3: Extract !

```python
patchify_folder_hierarchically(
    infolder,
    outfolder,
    top_level,
    low_level,
    psize,
    interval,
    silent=silent,
    extensions=extensions,
    recurse=recurse,
    slide_filters=slide_filters,
    verbose=verbose,
)
```
Output csv will look like :
|         id         |       parent      |        level       |   x  |   y  |  dx |  dy |
|:-----------:|:-------------:|:------------:|:----:|:----:|:---:|:---:|
|  Patch identifier  | Parent identifier | int (0, max level) |  int |  int | int | int |
|         #1         |        None       |          2         |   0  |   0  | 996 | 996 |
|        #1#1        |         #1        |          1         |   0  |   0  | 448 | 448 |
|       #1#1#1       |        #1#1       |          0         |   0  |   0  | 224 | 224 |
|       #1#1#2       |        #1#1       |          0         |   0  |  224 | 224 | 224 |
|         ...        |         ...       |         ...        |  ... |  ... | ... | ... |

## Advanced features
You can use more advanced features to work on slides, most notably using your custom filters. Check [documentation](https://linktothedoc.com) for more info.

---

## Community

The lightning community is maintained by 4 core contributors from [Institut Universitaire du Cancer de Toulouse - Oncopole](https://www.iuct-oncopole.fr/):
* [Arnaud Abreu](https://github.com/ArnaudAbreu)
* [Pilar Ortega](https://github.com/pilarOrtega)
* [Robin Schwob](https://github.com/schwobr)
* [Kevin Cortacero](https://github.com/KevinCortacero)

### Asking for help
If you have any questions please:
1. [Read the docs](https://pathaia.readthedocs.io/en/latest/index.html).
2. [Check existing issues](https://github.com/ArnaudAbreu/PathAIA/issues), or [add a new issue](https://github.com/ArnaudAbreu/PathAIA/issues/new)

## License

Please observe the GNU GPL 3.0 license that is listed in this repository.

## BibTeX
If you want to cite the framework feel free to use this.

```bibtex
@article{pathaia2021,
  title={PathAIA},
  author={Abreu, A and .al},
  journal={GitHub. Note: https://github.com/ArnaudAbreu/PathAIA},
  volume={3},
  year={2021}
}
```
