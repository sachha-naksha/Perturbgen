 [![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
 ![python](https://img.shields.io/badge/Python-3.10-brightgreen)


# PerturbGen Foundation model for dynamic cellular states

## 1. Usage

First, clone the repo and change to the project directory.

```shell
git clone https://github.com/Lotfollahi-lab/Perturbgen.git
```

Install Poetry (one-time):
(wanna know what is poetry? have a look at https://python-poetry.org)
```shell
curl -sSL https://install.python-poetry.org | python3 -
```
Optional: alternative way to install poetry using pipx (https://pipx.pypa.io/stable/installation/)
```shell
pipx install poetry
```

Create/install the environment and dependencies:
```shell
cd Perturbgen
poetry env use python3.11
poetry install
```

Activate the enviroment
```shell
source "$(poetry env info -p)/bin/activate"
```

The project contains some jupyter notebooks, which were converted to python files
due to better handling in the repository.
These files end with `_nb.py` and can be converted back to a `.ipynb` file with
`jupytext`:

```shell
jupytext --to ipynb --execute <your_file>_nb.py
```
## Examples

For usage, see:
- [Preprocessing and data curation](docs/examples/01_preprocessing_and_curation.ipynb)
- [Tokenizing transcriptomes and training the model](docs/examples/02_tokenize_and_train.ipynb)
- [In silico perturbation and analyses](docs/examples/03_analysis.ipynb)

The model can be run ALSO by following the three different steps:
1. Data preprocessing, tokenisation
2. Training the masking model
3. Training the count decoder model
4. Load checkpoint and generate predictions
The executable scripts for each step are located in the `T_perturb/cytomeister/batch_job_script` directory.


## Citation

If you use our repository or code in your research, please cite us:

```

```
