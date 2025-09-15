import os, sys
import inspect

currentdir = os.path.dirname(os.path.abspath(inspect.getfile(inspect.currentframe())))
parentdir = os.path.dirname(currentdir)
sys.path.insert(0, parentdir)
import argparse, torch,re
import numpy as np
from src.util import prepare_train_full_data,prepare_external_data,load_rna_strand,load_ref_genome
import torchvision.transforms as T
from layers import EpiMseLoss,HiCMseLoss
from src.model import build_model
import torch.optim as optim
import time, pickle
from scipy.stats import pearsonr, spearmanr
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data import DataLoader, Dataset
from torch.utils.data.distributed import DistributedSampler
import datetime
from copy import deepcopy
import torch.nn.functional as F
import torch.nn as nn

def parser_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--bins', type=int, default=600)
    parser.add_argument('--crop', type=int, default=50)
    parser.add_argument('--embed_dim', default=960, type=int)
    parser.add_argument('--epochs', default=20, type=int)
    parser.add_argument('--accum_iter', default=2, type=int)
    parser.add_argument('--lr', default=1e-5, type=float)
    parser.add_argument('--batchsize', type=int, default=1)
    parser.add_argument('--atac_block', default=True, action='store_false')
    parser.add_argument('--full', default=False, action='store_true')
    parser.add_argument('--lora_r_pretrain', default=0, type=float)
    parser.add_argument('--lora_r_pretrain_1', default=0, type=float)
    parser.add_argument('--lora_trunk_r', default=0, type=float)
    parser.add_argument('--lora_head_epi_r', default=0, type=float)
    parser.add_argument('--lora_head_rna_r', default=0, type=float)
    parser.add_argument('--lora_head_erna_r', default=0, type=float)
    parser.add_argument('--lora_head_microc_r', default=0, type=float)
    parser.add_argument('-l', '--logits_type', type=str, default='dilate')
    parser.add_argument('-p', '--prefix', type=str, default='')
    parser.add_argument('--prompt', default=False, action='store_true')
    parser.add_argument('--teacher', default=True, action='store_false')
    parser.add_argument('--external', default=True, action='store_false')
    parser.add_argument('--fixT', default=False, action='store_true')
    parser.add_argument('-o','--out',type=str,default='')
    args = parser.parse_args()
    return args
def get_args():
    args = parser_args()
    return args


args = get_args()
try:
    device = xm.xla_device()
    print(f"XLA device detected: {device}")
except Exception as e:
    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
    print(f"No XLA device detected. Error: {e}")

model = build_model(args)
model.load_state_dict(torch.load(model_path, map_location='cpu'),strict=True)
model.eval()
model.to(device)
test_cells=['K562']
def prepare_test_atac(cls,types='bulk'):
    atac_data = {}
    ref_data = {}
    chroms = [10,21]
    print(types)
    for cl in cls:
        atac_data[cl] = {}
        with open('data/%s_atac.pickle'%cl, 'rb') as f:
            atacseq = pickle.load(f)
    for chr in chroms:
        ref_data[chr] = load_ref_genome(chr)

    return atac_data, ref_data

atac_data, ref_data = prepare_test_atac(test_cells,types='bulk')

input_locs = np.load('src/input_region_nodup_noX.npy')

def test_indices(input_locs):
    # test_chrs=[2,10,21]
    test_chrs=[10,21]
    test_indices=[]
    for chrom in test_chrs:
        test_indices.append(np.where(input_locs[:,0]==chrom)[0])
    test_indices=np.concatenate(test_indices)
    print(test_indices.shape)
    return test_indices

test_idx = test_indices(input_locs)
print(input_locs.shape,test_idx.shape)
region_index = np.load('src/index_region_nodup_noX.npy')


def load_data(lidx, cl):
    chrom, s, e = input_locs[lidx]
    input = torch.cat((ref_data[chrom][s:e], atac_data[cl][chrom][s:e]), dim=1).unsqueeze(0).to(device)
    return input

for cl in train_cells:
    print(cl)
    # Flat dict: one key per modality (no "pred"/"targ")
    outputs = {
        'epi': [], 'erna': [], 'pro': [], 'groseq': [], 'proseq': [],
        'netcage': [], 'cage': [], 'intacthic': [], 'rna': [], 'tt': []
    }

    for vidx in test_idx:
        with torch.no_grad():
            # Still use load_data to know which modalities exist (None checks).
            valid_input = load_data(vidx, cl)

            out, external_out = model(valid_input, use_prompt=False)

            # EPI
            if valid_epi is not None:
                pred_epi = out[0][:, :, :245].cpu().data.detach().numpy()[:, :, (temp_lmasks[cl] > 0)].astype('float16')
                outputs['epi'].append(pred_epi)

            # CAGE (initiation)
            if valid_cage is not None:
                outputs['cage'].append(out[1].cpu().data.detach().numpy())

            # RNA (steady-state)
            if valid_rna is not None:
                outputs['rna'].append(out[1][:, :, 1].cpu().data.detach().numpy())

            # ERNA
            if valid_erna is not None:
                outputs['erna'].append(out[2].cpu().data.detach().numpy())

            # Intact Hi-C
            if valid_intacthic is not None:
                outputs['intacthic'].append(out[5].cpu().data.detach().numpy().astype('float16'))

            # PRO-cap
            if valid_procap is not None:
                outputs['pro'].append(external_out[4].cpu().data.detach().numpy())

            # PRO-seq
            if valid_proseq is not None:
                outputs['proseq'].append(external_out[4].cpu().data.detach().numpy())

            # GRO-seq
            if valid_groseq is not None:
                outputs['groseq'].append(external_out[2].cpu().data.detach().numpy())

            # NET-CAGE
            if valid_netcage is not None:
                outputs['netcage'].append(external_out[5].cpu().data.detach().numpy())

            # TT-seq
            if valid_tt is not None:
                outputs['tt'].append(external_out[1].cpu().data.detach().numpy())

    # Concatenate per modality if we collected any predictions; otherwise set to None
    for mod in list(outputs.keys()):
        if len(outputs[mod]) > 0:
            outputs[mod] = np.concatenate(outputs[mod], axis=0)
        else:
            outputs[mod] = None

    os.makedirs('results', exist_ok=True)
    with open(f'results/pred_only_{cl}.pickle', 'wb') as f:
        pickle.dump(outputs, f)
