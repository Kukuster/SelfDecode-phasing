import numpy as np

from lib.pd import pd_read_vcf

"""
Have to rewrite this using bcftools (and tabix for indexing input datasets)
So the function receives two inputs: full dataset of test samples, and list of sites.

Currently this works under the assumption:
 - both FULL and CHIP datasets are harmonized in the same way
"""


FULL_DATASET_OF_TEST_SAMPLES = "/home/ubuntu/files/400_random_samples_BP0-1000000.80-test-samples.vcf.gz"
CHIP_DATASET = "/home/ubuntu/files/400_random_samples_BP0-1000000_EUROFINS.vcf.gz"

OUTPUT_DATASET = "/home/ubuntu/files/400_random_samples_BP0-1000000.80-test-samples_EUROFINS-masked_unimputed-unphased_int-data.tsv.gz"


"""
Prepares test dataset from:
1) A full dataset of test samples
2) A vcf with only sites that are needed in the output file

1. Masks given first vcf file with sites that are present in the second vcf file
    by setting all that aren't present in both to `np.nan`
2. Translates genotype and NaNs to whatever vocabulary defined in `.replace({})`.
    For "unphasing", translation should map "0|1" and "1|0" to the same number.
"""
def prepare_test_dataset():

    df = pd_read_vcf(FULL_DATASET_OF_TEST_SAMPLES)
    df_CHIP = pd_read_vcf(CHIP_DATASET)


    df['key'] = df['#CHROM'].astype(str) + ':' + df['POS'].astype(str) + '.' + df['REF'] + '.' + df['ALT']
    df_CHIP['key'] = df_CHIP['#CHROM'].astype(str) + ':' + df_CHIP['POS'].astype(str) + '.' + df_CHIP['REF'] + '.' + df_CHIP['ALT']

    df.set_index('key', inplace=True)
    df_CHIP.set_index('key', inplace=True)


    samples = df.columns[9:] # in VCF the first 9 columns are data about the SNP, and all other columns are samples
    df_result = df.copy()
    for col in samples:
        df_result[col] = df_CHIP[col]
        df_result[col].replace({
            np.nan: '0',
            '0|0': '1',
            '1|0': '2',
            '0|1': '2',
            '1|1': '3'
        }, inplace=True)
        df_result[col].astype(str)

    df_result.to_csv(OUTPUT_DATASET, index=False, sep="\t", line_terminator="\n")


if __name__ == "__main__":
    prepare_test_dataset()
