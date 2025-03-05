import os,hicstraw, sys
from scipy.sparse import csr_matrix,save_npz
import numpy as np

resolution=1000
genome_lens = [248956422, 242193529, 198295559, 190214555, 181538259, 170805979, 159345973,
                145138636, 138394717, 133797422, 135086622, 133275309, 114364328, 107043718,
                101991189, 90338345, 83257441, 80373285, 58617616, 64444167, 46709983, 50818468, 156040895]
genome_lens=np.array(genome_lens)//resolution

chrs = [str(i) for i in range(1, 23)] + ['X']
cl=sys.argv[1]
norm=sys.argv[2]
hicfile=sys.argv[3]
output_path=sys.argv[4]

if len(sys.argv) != 5:
    raise ValueError('miss arguments')

if norm not in ['KR','OE']:
    raise ValueError('incorrect normalization')

# hicpath='/scratch/drjieliu_root/drjieliu/zhenhaoz/intacthic/'

def write_file():
    for chr in chrs:
        whic_file=os.path.join(output_path,cl+'_chr%s_%s_1kb.txt' % (chr,norm))
        f = open( whic_file, 'w')
        if norm=='KR':
            result = hicstraw.straw('observed', 'SCALE',hicfile, 'chr'+chr, 'chr'+chr,
                                     'BP', resolution)
        elif norm=='OE':
            result = hicstraw.straw('oe', 'NONE', hicfile, 'chr'+chr, 'chr'+chr,
                                    'BP', resolution)
        for i in range(len(result)):
            f.write("{0}\t{1}\t{2}\n".format(result[i].binX, result[i].binY, result[i].counts))
        f.close()

def txttomatrix(txt_file,resolution):
    rows=[]
    cols=[]
    data=[]
    with open(txt_file,'r') as f:
        for line in f:
            contents=line.strip().split('\t')
            bin1=int(contents[0])//resolution
            bin2 = int(contents[1]) // resolution
            if np.abs(bin2-bin1)>1000:
                continue
            value=float(contents[2])
            rows.append(bin1)
            cols.append(bin2)
            data.append(value)
    return np.array(rows),np.array(cols),np.array(data)

def write_to_mtx():
    for i in range(len(chrs)):
        print(cl,i,norm)
        length = genome_lens[i]

        txtfile=os.path.join(output_path,cl+'_chr%s_%s_1kb.txt' % (chrs[i],norm))

        row, col, data = txttomatrix(txtfile, resolution)
        temp = csr_matrix((data, (row, col)), shape=(length, length))

        npz_file=os.path.join(output_path,cl+'_chr%s_%s_1kb.npz' % (chrs[i],norm))
        save_npz(npz_file, temp)

if not os.path.exists(output_path):
    os.mkdir(output_path)
write_file()
write_to_mtx()