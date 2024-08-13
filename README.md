# High-precision rod-shaped bacterial cell length measurement based on Omnipose

<!-- PROJECT SHIELDS -->

[![Contributors][contributors-shield]][contributors-url]
[![Forks][forks-shield]][forks-url]
[![Stargazers][stars-shield]][stars-url]
[![Issues][issues-shield]][issues-url]
[![MIT License][license-shield]][license-url]

<!-- PROJECT LOGO -->
<br />

## Abstract
Cell length is one of  important parameters of cell morphology, which is of great significance for understanding cell structure, function,and physiological status. This paper uses the Omnipose method to segment rod-shaped bacterial cells, determine their skeleton and measure the length of rod-shaped bacterial cells. Compared with other algorithms, this algorithm proposed in this paper has smaller errors, achieving higher accuracy and stability in measuring bacterial cell length. The method described in the paper can quickly and automatically batch process a large number of cell images, and calculate the length of cells with high accuracy. This algorithm  also provides a convenient, efficient, and reliable method for the study and experimentations of cell morphology.

## Installation


1. Clone this repo, and we'll call the directory that you cloned as ${FAIRMOT_ROOT}
```sh
https://github.com/chengkf/cell.git
```
2. Install dependencies. We use python 3.10 and pytorch >= 1.7.0
```sh
conda create -n yourenvs
conda activate yourenvs
pip install torch==1.13.1+cu117 torchvision==0.14.1+cu117 torchaudio==0.13.1 --extra-index-url https://download.pytorch.org/whl/cu117
cd ${FAIRMOT_ROOT}
pip install omnipose
pip install -r requirements.txt
```

## run
```sh
python3 app.py
```

## Licensing

For detailed information, please refer to [LICENSE](https://github.com/chengkf/cell/blob/master/LICENSE)。

## Acknowledgement


- [omnipose](https://github.com/kevinjohncutler/omnipose)



<!-- links -->
[your-project-path]:chengkf/cell
[contributors-shield]: https://img.shields.io/github/contributors/chengkf/cell.svg?style=flat-square
[contributors-url]: https://github.com/chengkf/cell/graphs/contributors
[forks-shield]: https://img.shields.io/github/forks/chengkf/cell.svg?style=flat-square
[forks-url]: https://github.com/chengkf/cell/network/members
[stars-shield]: https://img.shields.io/github/stars/chengkf/cell.svg?style=flat-square
[stars-url]: https://github.com/chengkf/cell/stargazers

[issues-shield]: https://img.shields.io/github/issues/chengkf/cell.svg?style=flat-square
[issues-url]: https://img.shields.io/github/issues/chengkf/cell.svg
[license-shield]: https://img.shields.io/github/license/chengkf/cell.svg?style=flat-square
[license-url]: https://github.com/chengkf/cell/blob/master/LICENSE


