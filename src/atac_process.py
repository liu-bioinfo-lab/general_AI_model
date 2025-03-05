import pyBigWig
import numpy as np
import pickle
from scipy.sparse import csr_matrix
from optparse import OptionParser

def DNase_processing():
    usage = 'usage: %prog [options] <bigWig_file>'
    parser = OptionParser(usage)
    (options, args) = parser.parse_args()
    if len(args) != 1:
        parser.error('Please justify the DNase-seq bigwig file path')
    else:
        dnase_file = args[0]
    bw = pyBigWig.open(dnase_file)
    signals = {}
    for chrom, length in bw.chroms().items():
        try:
            if chrom == 'chrX':
                chr = 'X'
            else:
                chr = int(chrom[3:])
        except Exception:
            continue
        temp = np.zeros(length)
        intervals = bw.intervals(chrom)
        for interval in intervals:
            temp[interval[0]:interval[1]] = interval[2]
            
        seq_length = length // 1000 * 1000
        signals[chr] = csr_matrix(temp[:seq_length]).astype('float16')

    with open(dnase_file.replace('bigWig', 'pickle'), 'wb') as file:
        pickle.dump(signals, file)

if __name__=='__main__':
    DNase_processing()
