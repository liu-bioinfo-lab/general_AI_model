from torch.utils.data import DataLoader, Dataset, random_split, ConcatDataset
import os, torch, random
import pandas as pd
import numpy as np
from scipy.sparse import load_npz
import torch.optim as optim
import sys, inspect, datetime, argparse
import gc
currentdir = os.path.dirname(os.path.abspath(inspect.getfile(inspect.currentframe())))
parentdir = os.path.dirname(currentdir)
sys.path.insert(0, parentdir)
from curriculum.lora_prompt_model_1 import build_model
import pickle, time
from erna.util import load_ref_genome,load_dnase,pad_signal_matrix


def parser_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--bins', type=int, default=600)
    parser.add_argument('--crop', type=int, default=50)
    parser.add_argument('--embed_dim', default=960, type=int)
    parser.add_argument('--epochs', default=20, type=int)
    parser.add_argument('--accum_iter', default=2, type=int)
    parser.add_argument('--lr', default=1e-5, type=float)
    parser.add_argument('--atac_block', default=True, action='store_false')
    parser.add_argument('--lora_r_pretrain', default=0, type=float)
    parser.add_argument('--lora_r_pretrain_1', default=0, type=float)
    parser.add_argument('--lora_trunk_r', default=0, type=float)
    parser.add_argument('--lora_head_epi_r', default=0, type=float)
    parser.add_argument('--lora_head_rna_r', default=0, type=float)
    parser.add_argument('--lora_head_erna_r', default=0, type=float)
    parser.add_argument('--lora_head_microc_r', default=0, type=float)
    parser.add_argument('--lora_head_hic_r', default=0, type=float)
    parser.add_argument('-l', '--logits_type', type=str, default='dilate')
    parser.add_argument('-p', '--prefix', type=str, default='')
    parser.add_argument('-t', '--tissue', type=str, default='')
    parser.add_argument('--prompt', default=False, action='store_true')
    parser.add_argument('--external', default=True, action='store_false')
    parser.add_argument('-s', '--include_sign_prediction', default=False, action='store_true')
    args = parser.parse_args()
    return args


def get_args():
    args = parser_args()
    return args
def load_dnase_1(dnase_seq):
    dnase_seq = np.expand_dims(pad_signal_matrix(dnase_seq.astype('float32').reshape(-1, 1000)), axis=1)
    return torch.tensor(dnase_seq)

device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')

args = get_args()
model = build_model(args)
model.to(device)
model.load_state_dict(torch.load('../curriculum/models/ddp_scale_final_2_full.pt'))
model.eval()


atac_gtex_path='/nfs/turbo/umms-drjieliu/proj/EPCOT_eqtl/ATAC/arrays/'

target_tissue = args.tissue

data = np.loadtxt('eqtl_data/' + target_tissue + '.txt', dtype='str')

with open('genes.pickle', 'rb') as f:
    gene_annotation = pickle.load(f)
ordered_genes = sorted(list(gene_annotation.keys()))

tmpgeneTSS = np.loadtxt('ensemblTSS.txt', dtype='str')
geneTSS_dic = {tmpgeneTSS[i, 0]: int(tmpgeneTSS[i, 1]) for i in range(tmpgeneTSS.shape[0])}

geneids = data[:, 2].astype('int')
tss_loc = np.array([geneTSS_dic[ordered_genes[gid]] for gid in geneids])

variant_loc = data[:, 7].astype('int')

chroms_var=data[:, 3].astype('str')
chroms_var=np.where(chroms_var=='23','X',chroms_var)

ref_data = {}
atac_data = {}
chroms = [i for i in range(1, 23)] + ['X']
for chr in chroms:
    ref_data[chr] = load_ref_genome(chr)
    tmpatac = np.load(atac_gtex_path + target_tissue + '_atac_%s.npy' % chr)
    atac_data[chr] = load_dnase_1(tmpatac)


def load_data( chrom,inp_s, inp_e):
    if chrom != 'X':
        chrom = int(chrom)

    input = torch.cat((ref_data[chrom][inp_s:inp_e], atac_data[chrom][inp_s:inp_e]), dim=1).unsqueeze(0).to(device)
    return input


all_modals = ['epi', 'rna', 'bru', 'microc', 'hic', 'intacthic', 'external_tf', 'tt', 'groseq', 'grocap', 'proseq',
                  'netcage', 'starr']
pred_modals = ['epi', 'rna', 'bru', 'microc', 'intacthic', 'tt', 'groseq', 'grocap', 'netcage']
outputs_v = {}
outputs_t= {}

for step,vidx in enumerate(tss_loc):
    tss_bin=tss_loc[step]//1000
    var_bin=variant_loc[step]//1000
    start_bin= (tss_bin+var_bin)//2-299
    end_bin = (tss_bin + var_bin) // 2 + 301
    chrom_bin=chroms_var[step]
    valid_input = load_data(chrom_bin,start_bin,end_bin)

    with torch.no_grad():
        # output, external_output = model(inputs)
        reps,twod_reps,output, external_output =model(valid_input,return_rep=True)
    reps=reps.detach().cpu().numpy()
    reps=reps[:,args.crop:-args.crop,:]
    twod_reps = twod_reps.detach().cpu().numpy()
    twod_reps = twod_reps[:, args.crop:-args.crop,args.crop:-args.crop, :]

    tmps=[o.detach().cpu().numpy() for o in (output+external_output)]
    predictions = dict(zip(all_modals, tmps))

    varbin1=var_bin-start_bin-50
    tssbin1=tss_bin-start_bin-50


    reps=reps[:,np.array([varbin1,tssbin1]),:]
    twod_reps = twod_reps[:, varbin1, tssbin1, :]
    if step==0:
        output_reps=reps
        output_twodreps=twod_reps
        for modx in pred_modals:
            if modx not in ['microc', 'intacthic']:
                outputs_v[modx]=predictions[modx][:, varbin1, :]
                outputs_t[modx]=predictions[modx][:, tssbin1, :]
            else:
                outputs_v[modx]=predictions[modx][:, tssbin1, varbin1, :]
                outputs_t[modx]=predictions[modx][:, tssbin1, varbin1 :]
    else:
        output_reps=np.vstack((output_reps,reps))
        output_twodreps = np.vstack((output_twodreps,twod_reps))

        for modx in pred_modals:
            if modx not in ['microc', 'intacthic']:
                outputs_v[modx]=np.vstack((outputs_v[modx],predictions[modx][:, varbin1, :]))
                outputs_t[modx]=np.vstack((outputs_t[modx],predictions[modx][:, tssbin1, :]))
            else:
                outputs_v[modx] = np.vstack((outputs_v[modx], predictions[modx][:, tssbin1, varbin1, :]))
                outputs_t[modx] = np.vstack((outputs_t[modx], predictions[modx][:, tssbin1, varbin1, :]))



outs={'enh':outputs_v,'tss':outputs_t}
with open(target_tissue+'_preds.pickle', 'wb') as f:
    pickle.dump(outs, f)
np.save(target_tissue+'_reps.npy',output_reps)
np.save(target_tissue+'_2dreps.npy', output_twodreps)