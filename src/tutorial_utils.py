import kipoiseq
from kipoiseq import Interval
import pyfaidx
import pickle
import numpy as np
import os
import torch
import argparse

def parser_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--bins', type=int, default=600)
    parser.add_argument('--crop', type=int, default=50)
    parser.add_argument('--embed_dim', default=960, type=int)
    args = parser.parse_args([])
    return args

def get_args():
    args = parser_args()
    return args

def one_hot_encode(sequence):
    return kipoiseq.transforms.functional.one_hot_dna(sequence).astype(np.float32)
def pad_seq_matrix(matrix, pad_left, pad_right, pad_len=300):
    # add flanking region to each sample
    dmatrix = np.concatenate((pad_left, matrix[:, :, -pad_len:]), axis=0)[:-1, :, :]
    umatrix = np.concatenate((matrix[:, :, :pad_len], pad_right), axis=0)[1:, :, :]
    return np.concatenate((dmatrix, matrix, umatrix), axis=2)

def pad_signal_matrix(matrix,pad_dnase_left,pad_dnase_right, pad_len=300):
    dmatrix = np.vstack((pad_dnase_left, matrix[:, -pad_len:]))[:-1, :]
    umatrix = np.vstack((matrix[:, :pad_len], pad_dnase_right))[1:, :]
    return np.hstack((dmatrix, matrix, umatrix))


def prepare_input(
    fasta_extractor,
    chrom, start, end,
    atac_data
):
    """
    Generate the inputs to the model
    Args:
        fasta_extractor (kipoiseq.extractors.FastaStringExtractor): kipoiseq.extractors.FastaStringExtractor object
        chrom (str), start (int), end (int): Specify a genomic region (chromosome, start genomic position, and end genomic position),
                and the size of the genomic region should be divisible by 1000.
        dnase (ndarray): a numpy array representing DNase-seq signals in the same chromosome

    Returns:
        torch.Tensor: a torch tensor (N x 5 x 1600) representing input genomic sequences and DNase-seq, where N represents the number of
                1kb sequences in the input genomic region
    """
    assert end - start == 500000

    start = start - 50000
    end = end + 50000

    if isinstance(chrom, str):
        try:
            chrom_atac = int(chrom.replace('chr',''))
        except Exception:
            chrom_atac = chrom.replace('chr','')
    else:
        chrom_atac = chrom

    if start>=end:
        raise ValueError('the start of genomic region should be small than the end.')
    if start < 300 or end+300 > atac_data[chrom_atac].shape[-1]:
        raise ValueError('please leave enough flanking region.')

    target_interval = kipoiseq.Interval(chrom, start, end)
    sequence_one_hot = one_hot_encode(fasta_extractor.extract(target_interval))
    sequence_matrix = sequence_one_hot.reshape(-1, 1000, 4).swapaxes(1, 2)

    pad_interval = kipoiseq.Interval(chrom, start - 300, start)
    seq_pad_left = np.expand_dims(one_hot_encode(fasta_extractor.extract(pad_interval)).swapaxes(0, 1), 0)
    pad_interval = kipoiseq.Interval(chrom, end, end + 300)
    seq_pad_right = np.expand_dims(one_hot_encode(fasta_extractor.extract(pad_interval)).swapaxes(0, 1), 0)
    seq_input = pad_seq_matrix(sequence_matrix, seq_pad_left, seq_pad_right)


    pad_atac_left=atac_data[chrom_atac][:,start-300:start].toarray().squeeze(0)
    pad_atac_right = atac_data[chrom_atac][:,end:end+300].toarray().squeeze(0)
    center_atac = atac_data[chrom_atac][:,start:end].toarray().squeeze(0).reshape(-1, 1000)

    atac_input = np.expand_dims(pad_signal_matrix(center_atac ,pad_atac_left,pad_atac_right), 1)

    inputs = torch.tensor(np.concatenate((seq_input, atac_input), axis=1)).float()
    return inputs.unsqueeze(0)


def extract_outputs(outputs):
    rep_1d, rep_2d, _, _ = outputs
    return rep_1d.cpu().detach().numpy(), rep_2d.cpu().detach().numpy()


class FastaStringExtractor:
    def __init__(self, fasta_file):
        self.fasta = pyfaidx.Fasta(fasta_file)
        self._chromosome_sizes = {k: len(v) for k, v in self.fasta.items()}

    def extract(self, interval: Interval, **kwargs) -> str:
        # Truncate interval if it extends beyond the chromosome lengths.
        chromosome_length = self._chromosome_sizes[interval.chrom]
        trimmed_interval = Interval(interval.chrom,
                                    max(interval.start, 0),
                                    min(interval.end, chromosome_length),
                                    )
        # pyfaidx wants a 1-based interval
        sequence = str(self.fasta.get_seq(trimmed_interval.chrom,
                                          trimmed_interval.start + 1,
                                          trimmed_interval.stop).seq).upper()
        # Fill truncated values with N's.
        pad_upstream = 'N' * max(-interval.start, 0)
        pad_downstream = 'N' * max(interval.end - chromosome_length, 0)
        return pad_upstream + sequence + pad_downstream

    def close(self):
        return self.fasta.close()