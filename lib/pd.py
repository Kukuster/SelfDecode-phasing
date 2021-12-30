import io
import gzip

import magic
import pandas as pd


"""
Wrapper around pd.read_csv() that reads table from vcf, vcf.gz, tsv, or tsv.gz
with an arbitrary length of a vcf header
"""
def pd_read_vcf(filepath: str):

    mime: str = magic.from_file(filepath, mime=True)
    if mime == 'application/gzip' or mime == 'application/x-gzip':
        filepath_o_gz: io.RawIOBase = gzip.open(filepath, 'r')  # type: ignore # GzipFile and RawIOBase _are_ in fact compatible
        filepath_o = io.TextIOWrapper(io.BufferedReader(filepath_o_gz))
    elif mime == 'text/plain':
        filepath_o = open(filepath, 'r')
    else:
        raise ValueError(f"ERROR: input file is of unsupported type: \"{mime}\". Only plain text and gzip/x-gzip/bgzip formats are supported")

    file_line = filepath_o.readline()
    num_of_vcf_header_lines = 0
    while file_line.startswith('##'):
        num_of_vcf_header_lines += 1
        file_line = filepath_o.readline()

    filepath_o.close()

    return pd.read_csv(filepath, header=num_of_vcf_header_lines, sep="\t")

