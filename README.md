# SelfDecode-phasing

## running

running:
```bash
python phasing_using_pyro_HMM.py
```
loads the data, trains the `model_31`, then phases (and imputes) the `test` dataset, and finally measures the accuracy of phasing and imputation, without dumping the phased dataset.

Loading of data is handled with this script `/read_genomic_datasets.py`, everything else is right in the `phasing_using_pyro_HMM.py` file.

some key dependencies:
```
jax                                0.2.26
jaxlib                             0.1.75
numpy                              1.20.3
numpydoc                           1.1.0
numpyro                            0.8.0
pyro-api                           0.1.2
pyro-ppl                           1.7.0
torch                              1.10.0+cpu
```

## Accuracy
terrible.

With different hyperparameters on small sequences of length around 1000-5000, with around 160 full-sequenced training samples and 40 test chip sequences, i've seen accuracy of phasing going to 70% by concordance (around 50% is no accuracy at all). For some different hyperparameters, accuracy of imputation has been going up to 30% (by concordance) considering only sites which contain at least one of two ALT calls in the validation (true) dataset.

This is only so called plain "Factorial HMM with two hidden states" without any additional features.

## BACKLOG
 - remove unused and redundant code from original examples, e.g.:
    - branchings depending on unused keys like `--jit`, `--raftery-parameterization`, `--hidden-dim`
    - remove `lengths` variable that's used alongside `sequences`/`full_sequences`/`sparse_sequences`, which is completely redundant in our case where all sequences are of the same length
