from typing import Literal, Union
#import jax.numpy as jnp
import numpy as np
import torch

from pd import pd_read_vcf



DATA_DIR="E:\\Nikita\\dev_data\\"

num_of_sites = 14635 # btw
test_dataset_filepath  = f"{DATA_DIR}400_random_samples_BP0-1000000.80-test-samples_EUROFINS-masked_unimputed-unphased_int-data.tsv.gz"
train_dataset_filepath = f"{DATA_DIR}400_random_samples_BP0-1000000.320-ref-samples.vcf.gz"
valid_dataset_filepath = f"{DATA_DIR}400_random_samples_BP0-1000000.80-test-samples.vcf.gz"




def read_genomic_datasets(
    test_samples:    Union[int, None] = None,
    train_samples:   Union[int, None] = None,
    sequence_length: Union[int, None] = None,
):
    df_test  = pd_read_vcf( test_dataset_filepath)
    df_train = pd_read_vcf(train_dataset_filepath)
    df_valid = pd_read_vcf(valid_dataset_filepath)

    test  = np.array(df_test [df_test .columns[9:]], dtype=np.int32).transpose()
    train = np.array(df_train[df_train.columns[9:]], dtype=object  ).transpose()
    valid = np.array(df_valid[df_valid.columns[9:]], dtype=object  ).transpose()

    # print("test",  test.shape)
    # print("train", train.shape)
    # print("valid", valid.shape)


    """
    Trimming
    """
    if sequence_length:
        test  = test [ :test_samples, :sequence_length] if test_samples  else test [:, :sequence_length]
        valid = valid[ :test_samples, :sequence_length] if test_samples  else valid[:, :sequence_length]
        train = train[:train_samples, :sequence_length] if train_samples else train[:, :sequence_length]
    else:
        test  = test [ :test_samples, :] if test_samples  else test [:, :]
        valid = valid[ :test_samples, :] if test_samples  else valid[:, :]
        train = train[:train_samples, :] if train_samples else train[:, :]


    """
    Wrapping each element in a list.
    E.g. shape before: (17, 23)
         shape after:  (17, 23, 1)
    """
    test  = np.expand_dims(test,  axis=len(test .shape)).astype(np.int32)
    train = np.expand_dims(train, axis=len(train.shape)).astype(object)
    valid = np.expand_dims(valid, axis=len(valid.shape)).astype(object)



    train_translated = np.zeros(list(train.shape)[:-1] + [2], dtype=np.int32)
    valid_translated = np.zeros(list(valid.shape)[:-1] + [2], dtype=np.int32)

    #TODO: vectorize this, probably using np.select()
    # see: https://stackoverflow.com/questions/67665705/numpy-equivalent-of-pandas-replace-dictionary-mapping
    for name, dataset_calls, dataset in (
        # ("test",  calls_test,  test),
        ("train", train, train_translated),
        ("valid", valid, valid_translated)
    ):
        for i, sample_half in enumerate(dataset):
            for j, call in enumerate(sample_half):
                if   dataset_calls[i][j][0] == "0|0":
                    dataset[i][j] = (0,0)
                elif dataset_calls[i][j][0] == "0|1":
                    dataset[i][j] = (0,1)
                elif dataset_calls[i][j][0] == "1|0":
                    dataset[i][j] = (1,0)
                elif dataset_calls[i][j][0] == "1|1":
                    dataset[i][j] = (1,1)
                else:
                    raise ValueError("'train' and 'valid' datasets should only contain genotype values, i.e. '0|0', '0|1', '1|0', or '1|1'")

    # print("test",  test.shape)
    # print("train", train.shape)
    # print("valid", valid.shape)

    # exit(0)
    data = {
        "test": {
            "sequence_lengths": torch.tensor([test.shape[1] for i in test]).to(torch.long),
            "sequences": torch.tensor(test).to(torch.long),
        },
        "train": {
            "sequence_lengths": torch.tensor([train_translated.shape[1] for i in train_translated]).to(torch.long),
            "sequences": torch.tensor(train_translated).to(torch.long),
        },
        "valid": {
            "sequence_lengths": torch.tensor([valid_translated.shape[1] for i in valid_translated]).to(torch.long),
            "sequences": torch.tensor(valid_translated).to(torch.long),
        },
    }


    assert len(data["test"]["sequence_lengths"]) == len(data["test"]["sequences"])

    return data


if __name__ == "__main__":
    datasets = read_genomic_datasets()
    test_dataset  = datasets['test'] ['sequences']
    train_dataset = datasets['train']['sequences']
    valid_dataset = datasets['valid']['sequences']

