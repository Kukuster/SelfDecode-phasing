# SelfDecode-phasing



## TODO:
 - remove unused and redundant code from original examples, e.g.:
    - brachings depending on unused keys like `--jit`, `--raftery-parameterization`, `--hidden-dim`
    - remove `lengths` variable that's used alongside `sequences`/`full_sequences`/`sparse_sequences`, which is completely redundant in our case where all sequences are of the same length
