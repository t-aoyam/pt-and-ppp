# pt-and-ppp

⚠️ **The repo is still under construction - please check back later for a complete runnable repo (expected to be done before ACL 2025)** ⚠️

This is a repo for a paper "Language Model Grow Less Humanlike beyond Phase Transition" (ACL, 2025). I'm still cleaning up the repo structure, dependencies, etc., but it should be ready before ACL 2025!

## Setting up the Environment

```
$ conda create -n pt-ppp python=3.8
$ conda activate pt-ppp
$ pip install -r requirements.txt
```

## Repo Structure

```
pt-and-ppp/
│  README.md
│  .gitignore
│
├─ src/                  
│   ├─ data/             # lightweight data-loading helpers
│   │    ├─ configs/     # .json configs for training
│   │    └─ .../         # all other result files
│   │
│   ├─ models/           # model classes
│   ├─ training/         # training loops, trainer classes, etc.
│   ├─ evaluation/       # metrics, analysis utilities
│   └─ utils/
│
├─ notebooks/
│   └─ figures.ipynb     # all figures in the paper can be generated here
│
├─ scripts/              # .sh scripts
│   ├─ train.sh
│   └─ evaluate.sh
│
├─ models/               # all pytorch models
│   └─ model-name
│        └─ checkpoint-xxx/
│
└─ data/                 # data for pretraining, evaluation, human eye-tracking/self-paced reading data, etc.
    ├─ README.md         # where to download, expected hashes, etc.
    ├─ configs/     # .json configs for training
    └─ .../         # all other result files
```
## How to Run the Code

### Creating Data

See [`data/README.md`](https://github.com/t-aoyam/pt-and-ppp/tree/main/data/README.md) for downloading, generating, and preprocessing data.

### Training the Model

See `scripts/training_sample.sh` for how to train a model with and without syntax/copying regularization.

### Evaluating the Model

See `scripts/evaluation_sample.sh` for how to get LM surprisal, SAS and induction head detection, etc.

For $\Delta LL$, you need to (1) obtain by-word LM surprisal, (2) add suprisal values back to the reading time data, and (3) run regression analyses in R.

```
python -m evaluation.get_surprisal.py -m [model_dir]  # get surprisal
python -m src.data.hatch.py -m [model_dir]  # put surprisal values back into the reading time data
Rscript src/evaluation/by_head_analysis.R "$model" NULL NULL  # run regression analyses and obtain delta log-likelihood
```

## Citation

If you use the code in this repo, please cite our paper (official ACL bibtex is coming soon):
```
@misc{aoyama2025languagemodelsgrowhumanlike,
      title={Language Models Grow Less Humanlike beyond Phase Transition}, 
      author={Tatsuya Aoyama and Ethan Wilcox},
      year={2025},
      eprint={2502.18802},
      archivePrefix={arXiv},
      primaryClass={cs.CL},
      url={https://arxiv.org/abs/2502.18802}, 
}
```
