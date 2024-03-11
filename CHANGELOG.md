# Changelog

All notable changes to this project will be documented in this file. The format is based on [Keep a Changelog](https://keepachangelog.com/zh-CN/1.0.0/).

Version format: `{MAJOR}.{MINOR}.{PATCH}` , which is based on [Semantic Versioning](https://semver.org/lang/zh-CN/), check it out for more information.

- MAJOR: Version when you make incompatible API changes. (Refactoring, Adding extra capabilities. This is not a fix. )
- MINOR: Version when you add functionality in a backwards compatible manner. (Fixing, correcting an existing functionality.)
- PATCH:  Version when you make backwards compatible bug fixes. (Programmer error. Bug.)

## [Unreleased]
## [0.1.0]-8-13-2021

### Fixed

- Fixed typos in `samples_genetator.py`.

## [0.0.1]-8-12-2021

### Added

- Added `CHANGELOG.md`.
- Added mannual and introduction files. (see `./docs`).
- Added `samples_genetator.py`.

### Removed

- Removed changes.md.

## [Released]

## 2.6.2 3-10-2020

* **M**: remove degree-wise graph op support
* **M**: integrate graph op with new dropout
* **N**: add convergence variance metric
* **F**: minor fixes as well as improvements

## 2.6.1 12-17-2020

* **F**: minor fix for multi-run parallel training

## 2.6.0 12-3-2020

* **F**: fix obsolete dataset partition logic for data preprocessing
* **N**: add graph (`feature_ver = "2"`) / kekulized SMILES (`feature_ver = "1"`) output for data preprocessing
* **N**: add support of graph input for model_4v4's batch data loader
* **N**: add support for training with graph input
* **M**: remove support of dropout value setting via config; from now on only model structure parameters are stored in config
* **M**: change how dropout is executed in model_4v4

## 2.4.3 11-27-2020

* **F**: minor fix for multi run training with `-srcfolder` 

## 2.4.2 11-26-2020

* **F**: minor fix for multi fold training with `-srcfolder` 

## 2.4.1 11-16-2020

* **M**: Major modification: format of data set (train/test) changed to `dict` type. Old `list` format is still supported for back compatibility, but new features added from now on won't be guaranteed to run on old formatted data set. Support for old `list` format will be discontinued in the next major release.
* **M**: For version controlling, model weights are now saved along with model config as well. For differentiation purpose, model files are now saved with `.pt` extension. Old formatted model files are still supported for back compatibility.
* **M**: ligand_based_VS_data_preprocessing.py: refactored heavily, refer to `readme.md` for the new interface.
* **M**: Train: replace `AuROC` subplot with `score level enrichment curves` subplot in log summary plot.
* **N**: Train: add score level relative enrichment metric for model performance evaluation.
* **N**: Train: add `mrun` arg to support bundle training and comparative re-running.
* **N**: Train: add `group_by_scaffold` arg to support scaffold-wise splitting / training.
* **N**: Train: add `srcfolder` arg for convenient comparative re-run with the same train/test sets as well as multi-fold indexing.
* **N**: Add `version` attribute to `CONFIG` class, from now on each `config` instance must have this attribute set explicitly for version controlling purpose.
* **N**: Train: add scaffold overlap analysis summary for trainset & testset.
* **R**: Train: remove `shuffle_trainset` arg, dataset will always be shuffled by default for every training mode.
* **R**: Train: remove support of hyperparameter setting via `config` except for model structure related ones. All training related hyperparameters must be set via input args from now on.
* **R**: Remove `weight_decay` attribute from all `config`s, this training related parameter is kept as unexposed in source code of `ligand_based_VS_train.py`.
* **R**: Remove support of customized `weighting_method` setting.


## 1.19.7 11-3-2020

* **N**: alpha: add score level enrichment metric; in the summary plot, previous "AUC" subplot is replaced with the new score level enrichment subplot

