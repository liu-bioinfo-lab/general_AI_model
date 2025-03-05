import os, sys
import inspect

currentdir = os.path.dirname(os.path.abspath(inspect.getfile(inspect.currentframe())))
parentdir = os.path.dirname(currentdir)
sys.path.insert(0, parentdir)
import argparse, torch,re
import numpy as np
from src.util import prepare_train_full_data,prepare_external_data,load_rna_strand
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
    parser.add_argument('--full', default=False, action='store_true')
    parser.add_argument('-p', '--prefix', type=str, default='')
    parser.add_argument('-o','--out',type=str,default='')
    args = parser.parse_args()
    return args
def get_args():
    args = parser_args()
    return args


def split_dataset(seed=24):

    input_locs = np.load('data/input_region_dup_250_noX.npy')
    dataset_size = input_locs.shape[0]
    indices = np.arange(dataset_size)
    valid_split = int(np.floor(dataset_size * 0.8))
    test_split = int(np.floor(dataset_size * 0.9))
    np.random.seed(seed)
    np.random.shuffle(indices)
    train_indices, valid_indices= indices[:valid_split], indices[test_split:]
    return train_indices, valid_indices


def log_clip(x, coptype):
    if coptype == 'microc' or coptype == 'intacthic':
        return np.clip(np.log2(x), -2, 10)
    elif coptype == 'hic':
        return np.clip(x, 0, 24)
    elif coptype == 'chiapet':
        return np.log2(x + 1)
    else:
        return x

class l1_loss_smooth(nn.Module):
    def __init__(self, beta=1.0):
        super(l1_loss_smooth, self).__init__()
        self.beta=beta
    def forward(self,input, target):

        n = torch.abs(input - target)
        cond = n < self.beta
        loss = torch.where(cond, 0.5 * n ** 2 / self.beta, n - 0.5 * self.beta)
        return (self.beta/2)*loss.mean()

def get_criterion(modal):

    if modal =='microc' or modal=='rna' or modal== 'bru' or modal=='intacthic' or modal=='rna_strand':
        criterion = torch.nn.MSELoss()
    elif modal=='hic':
        criterion = HiCMseLoss(alpha=1)
    else:
        criterion=EpiMseLoss(alpha=0.1)
    return criterion


def cal_loss(preds,targets,cl,lmask=None):
    loss=0
    modals=['epi','rna','bru','microc','hic','intacthic','rna_strand']
    alphas=[0.5, 1, 1, 3, 1,1,1]
    for i,x in enumerate(modals):
        if targets[i] is not None:
            criterion=get_criterion(x)
            if i:
                if x=='rna':
                    if cl =='HCT116' or cl=='CD8_T':
                        idx_m=np.array([1,2])
                    elif cl=='A549':
                        idx_m=np.array([0,2])
                    elif cl=='HFFc6' or cl=='B':
                        idx_m = np.array([1])
                    else:
                        idx_m=np.array([0,1,2])
                    loss += alphas[i]*criterion(preds[i][:,:,idx_m], targets[i])
                else:
                    loss+= alphas[i]*criterion(preds[i],targets[i])
            else:
                loss += alphas[i]*criterion(preds[i], targets[i],lmask)
    return loss


def cal_entex_loss_wt(preds,targets,lmask=None):
    loss = 0
    modals = ['epi', 'rna', 'bru', 'microc', 'hic','intacthic','rna_strand']
    for i,x in enumerate(modals):
        if targets[i] is not None:
            criterion=get_criterion(x)
            if x=='epi':
                loss += criterion(preds[i], targets[i], lmask)
            elif x=='rna':
                loss += criterion(preds[i][:, :, 1:2], targets[i])
            elif x == 'bru' or x == 'microc':
                continue
            else:
                loss += criterion(preds[i],targets[i])
    return loss


def cal_external_loss(preds,targets,T_targets=None,lmask=None,self_training=False,beta=0.5):
    loss = 0
    modals = ['external_tf', 'tt', 'groseq', 'grocap', 'proseq','netcage','starr']
    alphas=[1,3,2,3,3,2,3]
    for i, x in enumerate(modals):
        if targets[i] is not None and not self_training:
            if x=='external_tf':
                criterion = EpiMseLoss(alpha=0.002)
                tmp=criterion(preds[i], targets[i], lmask)
                loss += tmp
            else:
                tmp=F.mse_loss(preds[i], targets[i])
                loss += alphas[i]*tmp

        elif self_training:
            loss+=beta*F.mse_loss(preds[i], T_targets[i])
    return loss


def cop_target(x, gau_sigma=0., crop=10, coptype='hic'):
    temp = x.copy()
    np.fill_diagonal(temp, 0)
    temp = torch.tensor(log_clip(temp.T + x, coptype=coptype)).unsqueeze(0)
    if gau_sigma:
        t = T.GaussianBlur(kernel_size=5, sigma=gau_sigma)
        temp = t(temp)
    return temp[:,crop:-crop, crop:-crop]


# Currently used
def init_distributed():
    # Initializes the distributed backend which will take care of sychronizing nodes/GPUs
    dist_url = "env://"  # default
    # only works with torch.distributed.launch // torch.run
    rank = int(os.environ["RANK"])
    world_size = int(os.environ['WORLD_SIZE'])
    local_rank = int(os.environ['LOCAL_RANK'])
    print("World Size:", world_size, flush=True)
    print("Local Rank:", local_rank, flush=True)
    dist.init_process_group(
        backend="nccl",
        init_method=dist_url,
        world_size=world_size,
        rank=rank,
        timeout=datetime.timedelta(seconds=5400))
    # this will make all .cuda() calls work properly
    torch.cuda.set_device(local_rank)
    # synchronizes all the threads to reach this point before moving on
    dist.barrier()
    # setup_for_distributed(rank == 0)


class indiceDataset(Dataset):
    def __init__(self,indices):
        self.loc_indices=indices
        self.num=self.loc_indices.shape[0]
    def __getitem__(self, index):
        return self.loc_indices[index]
    def __len__(self):
        return self.num

def dna_sequence_to_one_hot(sequence):
    seq_array = np.array(list(sequence))
    one_hot_matrix = np.zeros((4, len(sequence)), dtype=int)
    nucleotides = ['A', 'C', 'G', 'T']
    for i, nucleotide in enumerate(nucleotides):
        one_hot_matrix[i] = (seq_array == nucleotide).astype(int)
    return one_hot_matrix

def pad_seq_matrix(matrix, pad_left,pad_right,pad_len=300):
    dmatrix = np.concatenate((pad_left[np.newaxis,:,:], matrix[:, :, -pad_len:]), axis=0)[:-1, :, :]
    umatrix = np.concatenate((matrix[:, :, :pad_len], pad_right[np.newaxis,:,:]), axis=0)[1:, :, :]
    return np.concatenate((dmatrix, matrix, umatrix), axis=2)

def main(gpu, args):
    torch.cuda.set_device(gpu)
    rank = gpu



    train_cells = ['GM12878', 'K562', 'HepG2', 'MCF-7', 'H1', 'HFFc6', 'A549', 'HCT116', 'IMR-90', 'CD8_T', 'B']
    cell_dict = {'epi': ['GM12878', 'K562', 'HepG2', 'MCF-7', 'H1', 'A549', 'HCT116', 'IMR-90', 'CD8_T'],
                 'rna': ['HepG2', 'K562', 'GM12878', 'MCF-7', 'H1', 'A549', 'HCT116', 'IMR-90', 'HFFc6', 'CD8_T', 'B'],
                 'bru': ['GM12878', 'K562', 'HepG2', 'MCF-7', 'HCT116', 'IMR-90'],
                 'microc': ['H1', 'HFFc6'], 'hic': ['GM12878', 'K562', 'HepG2', 'H1', 'HFFc6'],
                 'intacthic': ['GM12878', 'K562', 'HepG2', 'MCF-7', 'HCT116', 'IMR-90', 'CD8_T', 'B'],
                 'rna_strand': ['HepG2', 'K562', 'GM12878', 'MCF-7', 'H1', 'HCT116', 'IMR-90', 'HFFc6', 'CD8_T', 'B']}
    task_list = {'GM12878': [1, 1, 1, 0, 1, 1, 1], 'K562': [1, 1, 1, 0, 1, 1, 1], 'HepG2': [1, 1, 1, 0, 1, 1, 1],
                 'MCF-7': [1, 1, 1, 0, 0, 1, 1],
                 'H1': [1, 1, 0, 1, 1, 0, 1], 'HFFc6': [0, 1, 0, 1, 1, 0, 1], 'A549': [1, 1, 0, 0, 0, 0, 0],
                 'HCT116': [1, 1, 1, 0, 0, 1, 1],
                 'IMR-90': [1, 1, 1, 0, 0, 1, 1], 'CD8_T': [1, 1, 0, 0, 0, 1, 1], 'B': [0, 1, 0, 0, 0, 1, 1]}

    external_cell_dict = {'external_tf': ['GM12878', 'K562', 'HepG2', 'H1'],
                          'tt': ['K562', 'MCF-7'], 'groseq': ['GM12878', 'K562', 'HepG2', 'H1', 'A549', 'IMR-90'],
                          'grocap': ['GM12878', 'K562'], 'proseq': ['K562'],
                          'netcage': ['GM12878', 'K562', 'HepG2', 'MCF-7'],
                          'starr': ['K562']}


    train_tissues = ['gastrocnemius_medialis_f53', 'transverse_colon_f53','spleen_f51','stomach_f53',
                     'adrenal_gland_m54','gs_f53','tibial_nerve_f51','hlv_f51']
    tissue_dict = {'epi': train_tissues, 'rna': ['gastrocnemius_medialis_f53', 'transverse_colon_f53','spleen_f51',
                                                 'adrenal_gland_m54','gs_f53','tibial_nerve_f51','hlv_f51'],
                   'bru': [], 'microc': [], 'hic': [],
                   'intacthic':['gastrocnemius_medialis_f53','stomach_f53'],
                   'rna_strand':['gastrocnemius_medialis_f53', 'transverse_colon_f53',
                                 'adrenal_gland_m54','gs_f53','tibial_nerve_f51','hlv_f51']}

    task_list_tissues = {}
    for tissue in train_tissues:
        task_list_tissues[tissue] = [1, 1, 0, 0, 0,0,0]
        if tissue in tissue_dict['intacthic']:
            task_list_tissues[tissue] = [1, 1, 0, 0, 0,1,0]
        if tissue in tissue_dict['rna_strand']:
            task_list_tissues[tissue] = [1, 1, 0, 0, 0,1,1]

    all_modals=['epi', 'rna', 'bru', 'microc', 'hic','intacthic','rna_strand','external_tf', 'tt', 'groseq', 'grocap', 'proseq','netcage','starr']


    model = build_model(args)
    # model.load_state_dict(torch.load('models/ddp_scale_final_2_full.pt', map_location='cpu'), strict=False)

    model.cuda(gpu)
    model = DDP(model, find_unused_parameters = True,device_ids=[gpu])
    model.train()
    if rank==0:
        for n, p in model.named_parameters():
            print(n,p.requires_grad)

    optimizer = optim.AdamW(filter(lambda p: p.requires_grad, model.parameters()), lr=args.lr,
                            weight_decay=1e-6)
    if rank == 0:
        print(sum([p.numel() if p.requires_grad else 0 for n, p in model.named_parameters()]))
        print('load data start', flush=True)

    bulk_atac_data, ref_data, epi_data, rna_data,bru_data, microc_data,microc_KR_data, hic_data, ctcf_chiapet, polr2_chiapet,intacthic_oe_data,intacthic_kr_data\
        = prepare_train_full_data(cell_dict=deepcopy(cell_dict),tissue_dict=deepcopy(tissue_dict),train_cells=train_cells)
    rna_strand_data={}
    for cl in cell_dict['rna_strand']+tissue_dict['rna_strand']:
        rna_strand_data[cl]=load_rna_strand(cl)

    with open('/nfs/turbo/umms-drjieliu/usr/zzh/mutimodal_epcot/atac_bw/'
              'sc/train_scatac_merge.pickle', 'rb') as f:
        scatac_data = pickle.load(f)


    external_tf_data, tt_data, groseq_data, grocap_data, proseq_data,netcage_data,starr_data=\
        prepare_external_data(external_cell_dict)
    if rank == 0:
        print('load data finished', flush=True)

    if cell_dict:
        tmp = deepcopy(cell_dict)
        for k in tmp.keys():
            cell_dict[k] += tissue_dict[k]
    else:
        cell_dict.update(tissue_dict)
    task_list.update(task_list_tissues)
    if rank == 0:
        print(task_list)
        print(cell_dict)
        print(external_cell_dict)

    with open('data/epi_label_masks.pickle', 'rb') as f:
        temp_lmasks = pickle.load(f)
    for cl in (train_cells+train_tissues):
        if cl in temp_lmasks.keys():
            temp_lmasks[cl] = temp_lmasks[cl].reshape(1, -1)
        else:
            temp_lmasks[cl] = None

    with open('data/external_tfs_lmask.pickle', 'rb') as f:
        external_lmasks = pickle.load(f)
    for cl in train_cells:
        if cl in external_lmasks.keys():
            external_lmasks[cl] = external_lmasks[cl].reshape(1, -1)
        else:
            external_lmasks[cl] = None


    input_locs = np.load('data/input_region_dup_250_noX.npy')
    region_index = np.load('data/index_region_dup_250_noX.npy')

    if rank == 0:
        print("Size:", np.shape(input_locs))

    def load_data(lidx, cl,use_sc=False):
        chrom, s, e = input_locs[lidx]
        if use_sc:
            tmp_atac = torch.tensor(scatac_data[cl][chrom][s:e].astype(np.float32).toarray()).unsqueeze(1).float()
        else:
            tmp_atac=torch.tensor(bulk_atac_data[cl][chrom][s:e].astype(np.float32).toarray()).unsqueeze(1).float()

        input = torch.cat((ref_data[chrom][s:e], tmp_atac), dim=1).unsqueeze(0).cuda(gpu)

        if cl in microc_data.keys():
            tmp = microc_data[cl][chrom][s:e, s:e].astype(np.float32).toarray()
            microc_oe_label = cop_target(tmp, gau_sigma=0.8, crop=args.crop, coptype='microc').unsqueeze(-1)
            microc_oe_label = microc_oe_label.float()

            tmp = microc_KR_data[cl][chrom][s:e, s:e].astype(np.float32).toarray()
            microc_KR_label = cop_target(tmp, gau_sigma=0.8, crop=args.crop, coptype='microc').unsqueeze(-1)
            microc_KR_label = microc_KR_label.float()
            microc_label=torch.cat((microc_oe_label,microc_KR_label),dim=-1).cuda(gpu)
        else:
            microc_label = None

        if cl in intacthic_oe_data.keys():
            tmp = intacthic_oe_data[cl][chrom][s:e, s:e].astype(np.float32).toarray()
            intacthic_oe_label = cop_target(tmp, gau_sigma=1, crop=args.crop, coptype='intacthic').unsqueeze(-1)
            intacthic_oe_label = intacthic_oe_label.float()

            tmp = intacthic_kr_data[cl][chrom][s:e, s:e].astype(np.float32).toarray()
            intacthic_KR_label = cop_target(tmp, gau_sigma=1, crop=args.crop, coptype='intacthic').unsqueeze(-1)
            intacthic_KR_label = intacthic_KR_label.float()
            intacthic_label=torch.cat((intacthic_oe_label,intacthic_KR_label),dim=-1).cuda(gpu)
        else:
            intacthic_label = None

        if cl in hic_data.keys():  # Hi-C Labels
            ss, ee = s // 5, e // 5
            if 'K562' in cl or 'HepG2' in cl:
                hic_label = cop_target(hic_data[cl][chrom][ss:ee, ss:ee].toarray(), gau_sigma=0.4, crop=args.crop // 5,
                                  coptype='hic').unsqueeze(-1)
            elif 'gastrocnemius_medialis' in cl or 'transverse_colon' in cl:
                hic_label = cop_target(hic_data[cl][chrom][ss:ee, ss:ee].toarray(), gau_sigma=0.6, crop=args.crop // 5,
                                  coptype='hic').unsqueeze(-1)
            else:
                ctcfm = cop_target(ctcf_chiapet[cl][chrom][ss:ee, ss:ee].toarray(), gau_sigma=1, crop=args.crop//5,
                                   coptype='chiapet').unsqueeze(-1)
                polr2m = cop_target(polr2_chiapet[cl][chrom][ss:ee, ss:ee].toarray(), gau_sigma=1, crop=args.crop//5,
                                    coptype='chiapet').unsqueeze(-1)
                hicd = cop_target(hic_data[cl][chrom][ss:ee, ss:ee].toarray(), gau_sigma=0.4, crop=args.crop // 5,
                                  coptype='hic').unsqueeze(-1)
                hic_label = torch.cat((ctcfm, polr2m, hicd), dim=-1)
            hic_label = hic_label.float().cuda(gpu)
        else:
            hic_label = None

        rna_strand_label= rna_strand_data[cl][chrom][:,s+args.crop:e-args.crop].T.unsqueeze(0).float().cuda(gpu) if cl in rna_strand_data.keys() else None

        rnaidx=region_index[lidx]
        rna_label = (rna_data[cl][rnaidx:rnaidx + 1, args.crop:-args.crop,:]).float().cuda(gpu) if cl in rna_data.keys() else None
        bru_label= (bru_data[cl][rnaidx:rnaidx + 1, args.crop:-args.crop,:]).float().cuda(gpu) if cl in bru_data.keys() else None

        epi_label = (epi_data[cl][chrom][s + args.crop:e - args.crop, :]).float().unsqueeze(0).cuda(gpu) if cl in epi_data.keys() else None
        if args.external:
            external_tf_label=(external_tf_data[cl][chrom][s + args.crop:e - args.crop, :]).float()\
                .unsqueeze(0).cuda(gpu) if cl in external_tf_data.keys() else None
            tt_label = (tt_data[cl][rnaidx:rnaidx + 1, args.crop:-args.crop, :]).float().cuda(gpu) if cl in tt_data.keys() else None
            groseq_label = (groseq_data[cl][rnaidx:rnaidx + 1, args.crop:-args.crop, :]).float().cuda(
                gpu) if cl in groseq_data.keys() else None
            grocap_label = (grocap_data[cl][rnaidx:rnaidx + 1, args.crop:-args.crop, :]).float().cuda(
                gpu) if cl in grocap_data.keys() else None
            proseq_label = (proseq_data[cl][rnaidx:rnaidx + 1, args.crop:-args.crop, :]).float().cuda(
                gpu) if cl in proseq_data.keys() else None
            netcage_label = (netcage_data[cl][rnaidx:rnaidx + 1, args.crop:-args.crop, :]).float().cuda(
                gpu) if cl in netcage_data.keys() else None
            starr_label = (starr_data[cl][rnaidx:rnaidx + 1, args.crop:-args.crop, :]).float().cuda(
                gpu) if cl in starr_data.keys() else None
            return input, [epi_label, rna_label,bru_label, microc_label, hic_label,intacthic_label,rna_strand_label],\
                   [external_tf_label,tt_label,groseq_label,grocap_label,proseq_label,netcage_label,starr_label]
        return input, [epi_label, rna_label,bru_label, microc_label, hic_label,intacthic_label,rna_strand_label],None


    def prepare_dataloader(dataset: Dataset):
        return DataLoader(
            dataset,
            batch_size=args.batchsize,
            pin_memory=True,
            sampler=DistributedSampler(dataset)
        )

    train_indices, valid_indices= split_dataset()
    train_loader=prepare_dataloader(indiceDataset(train_indices))
    valid_loader = prepare_dataloader(indiceDataset(valid_indices))


    def model_valid(model):
        if rank == 0:
            print("validation step", flush=True)
        preds, targets = {}, {}
        for cl in (train_cells+train_tissues):
            preds[cl] = {mod:[] for mod in all_modals}
            targets[cl] = {mod:[] for mod in all_modals}

        for step, idx_x in enumerate(valid_loader):
            vidx = idx_x.int().item()
            for cl in (train_cells+train_tissues):
                if cl in ['GM12878','K562','HepG2','gastrocnemius_medialis_f53', 'transverse_colon_f53']:
                    valid_input, [epi_label, rna_label, bru_label, microc_label, hic_label,
                                  intacthic_label,rna_strand_label], external_label = load_data(vidx, cl,use_sc=True)
                else:
                    valid_input, [epi_label, rna_label, bru_label, microc_label, hic_label,intacthic_label,rna_strand_label],external_label = load_data(vidx, cl)
                with torch.no_grad():
                    output,external_out = model(valid_input, task_list=task_list[cl])
                    if epi_label is not None:
                        preds[cl]['epi'].append(
                            output[0][:, :, temp_lmasks[cl].squeeze() > 0].cpu().data.detach().numpy().astype('float16'))
                        targets[cl]['epi'].append(
                            epi_label.cpu().data.detach().numpy().astype('float16'))
                    if rna_label is not None:
                        if cl in train_tissues or cl =='HFFc6' or cl=='B':
                            rna_task_idx = np.array([1])
                        elif cl=='A549':
                            rna_task_idx=np.array([0,2])
                        elif cl=='HCT116' or cl=='CD8_T':
                            rna_task_idx=np.array([1,2])
                        else:
                            rna_task_idx = np.array([0,1, 2])
                        preds[cl]['rna'].append(output[1].cpu().data.detach().numpy()[:,:,rna_task_idx])
                        targets[cl]['rna'].append(rna_label.cpu().data.detach().numpy())
                    if bru_label is not None:
                        preds[cl]['bru'].append(output[2].cpu().data.detach().numpy())
                        targets[cl]['bru'].append(bru_label.cpu().data.detach().numpy())

                    if microc_label is not None:
                        tmps = np.nanmean(
                            [pearsonr(output[3].cpu().data.detach().numpy()[:, :, :, i].flatten().squeeze(),
                                      microc_label.cpu().data.detach().numpy()[:, :, :, i].flatten().squeeze())[0]
                             for i in range(2)])
                        preds[cl]['microc'].append(tmps)
                    if hic_label is not None:
                        if cl == 'K562' or cl == 'HepG2' or 'gastrocnemius_medialis' in cl or 'transverse_colon' in cl:
                            tmps=pearsonr(output[4].cpu().data.detach().numpy()[:, :, :, 2:].flatten().squeeze(),
                                          hic_label.cpu().data.detach().numpy().flatten().squeeze())[0]
                        else:
                            tmps = np.nanmean(
                                [pearsonr(output[4].cpu().data.detach().numpy()[:, :, :, i].flatten().squeeze(),
                                          hic_label.cpu().data.detach().numpy()[:, :, :, i].flatten().squeeze())[0]
                                 for i in range(3)])
                        preds[cl]['hic'].append(tmps)
                    if intacthic_label is not None:
                        tmps = np.nanmean(
                            [pearsonr(output[5].cpu().data.detach().numpy()[:, :, :, i].flatten().squeeze(),
                                      intacthic_label.cpu().data.detach().numpy()[:, :, :, i].flatten().squeeze())[0]
                             for i in range(2)])
                        preds[cl]['intacthic'].append(tmps)
                    if rna_strand_label is not None:
                        preds[cl]['rna_strand'].append(output[6].cpu().data.detach().numpy())
                        targets[cl]['rna_strand'].append(rna_strand_label.cpu().data.detach().numpy())

                    if args.external:
                        if external_label[0] is not None:
                            preds[cl]['external_tf'].append(
                                external_out[0][:, :, external_lmasks[cl].squeeze() > 0].cpu().data.detach().numpy().astype('float16'))
                            targets[cl]['external_tf'].append(external_label[0].cpu().data.detach().numpy().astype('float16'))
                        if external_label[1] is not None:
                            preds[cl]['tt'].append(external_out[1].cpu().data.detach().numpy())
                            targets[cl]['tt'].append(external_label[1].cpu().data.detach().numpy())
                        if external_label[2] is not None:
                            preds[cl]['groseq'].append(external_out[2].cpu().data.detach().numpy())
                            targets[cl]['groseq'].append(external_label[2].cpu().data.detach().numpy())
                        if external_label[3] is not None:
                            preds[cl]['grocap'].append(external_out[3].cpu().data.detach().numpy())
                            targets[cl]['grocap'].append(external_label[3].cpu().data.detach().numpy())
                        if external_label[4] is not None:
                            preds[cl]['proseq'].append(external_out[4].cpu().data.detach().numpy())
                            targets[cl]['proseq'].append(external_label[4].cpu().data.detach().numpy())

                        if external_label[5] is not None:
                            preds[cl]['netcage'].append(external_out[5].cpu().data.detach().numpy())
                            targets[cl]['netcage'].append(external_label[5].cpu().data.detach().numpy())
                        if external_label[6] is not None:
                            preds[cl]['starr'].append(external_out[6].cpu().data.detach().numpy())
                            targets[cl]['starr'].append(external_label[6].cpu().data.detach().numpy())



        scores={mod:[] for mod in all_modals}

        copy_cell_dict=deepcopy(cell_dict)
        copy_cell_dict.update(external_cell_dict)
        for task in copy_cell_dict.keys():
            for cl in copy_cell_dict[task]:
                if task in ['bru','rna','rna_strand','tt', 'groseq', 'grocap', 'proseq','netcage','starr']:
                    tmp_task_target=np.concatenate(targets[cl][task], axis=0)
                    tmp_task_pred=np.concatenate(preds[cl][task], axis=0)
                    tmprnascore = np.mean([pearsonr( tmp_task_pred[:, :, i].flatten(),tmp_task_target[:, :, i].flatten())[0]
                                            for i in range(tmp_task_pred.shape[-1])])
                    scores[task].append(tmprnascore)
                elif task in ['epi','external_tf']:
                    tmp_score = []
                    tmp_preds, tmp_targets = np.concatenate(preds[cl][task], axis=0), np.concatenate(targets[cl][task],
                                                                                                     axis=0)
                    for ti in range(tmp_preds.shape[-1]):
                        tmp_score.append(pearsonr(tmp_preds[:, :, ti].flatten(), tmp_targets[:, :, ti].flatten())[0])
                    scores[task].append(np.mean(tmp_score))
                else:
                    scores[task].append(np.nanmean(preds[cl][task]))
        return scores


    best_score = 0
    for epoch in range(args.epochs):
        train_loader.sampler.set_epoch(epoch)
        train_loss = 0
        model.train()
        dist.barrier()
        max_steps= train_indices.shape[0]// int(os.environ['WORLD_SIZE'])+1 \
            if train_indices.shape[0]% int(os.environ['WORLD_SIZE']) else train_indices.shape[0]// int(os.environ['WORLD_SIZE'])
        if rank == 0:
            print(max_steps)
        for step, idx_x in enumerate(train_loader):
            tidx = idx_x.int().item()
            tts = time.time()

            for cl in train_cells:
                train_input, train_target,external_target = load_data(tidx, cl)
                output,external_out = model(train_input, task_list=task_list[cl])
                loss = cal_loss(output, train_target,cl, temp_lmasks[cl])
                if args.external:
                    external_loss=cal_external_loss(preds=external_out, targets=external_target,lmask=external_lmasks[cl])
                    loss+= external_loss
                loss = loss / (args.accum_iter * len(train_cells))
                loss.backward()
                train_loss += loss.detach()
            for cl in ['GM12878','K562','HepG2']:
                train_input, train_target, external_target = load_data(tidx, cl,use_sc=True)
                output, external_out = model(train_input, task_list=task_list[cl])
                loss = cal_loss(output, train_target, cl, temp_lmasks[cl])
                if args.external:
                    external_loss = cal_external_loss(preds=external_out, targets=external_target,
                                                      lmask=external_lmasks[cl])
                    loss += external_loss
                loss = loss / (args.accum_iter * 4)
                loss.backward()
                train_loss += loss.detach()


            for cl in train_tissues:
                train_input, train_target,external_target = load_data(tidx, cl)
                output,external_out = model(train_input)
                loss=cal_entex_loss_wt(output, train_target,lmask= temp_lmasks[cl])
                loss = loss / (args.accum_iter * len(train_tissues))
                loss.backward()
                train_loss += loss.detach()
            for cl in ['transverse_colon_f53','gastrocnemius_medialis_f53']:
                train_input, train_target, external_target = load_data(tidx, cl,use_sc=True)
                output, external_out = model(train_input)
                loss = cal_entex_loss_wt(output, train_target, lmask=temp_lmasks[cl])
                loss = loss / (args.accum_iter * 2)
                loss.backward()
                train_loss += loss.detach()

            if ((step + 1) % args.accum_iter == 0) or (step + 1 == max_steps):
                optimizer.step()
                optimizer.zero_grad()
            if rank==0 and step % 1000 == 0:
                with open(args.out+'ddp_log_%s_full.txt'%args.prefix, 'a') as f:
                    f.write('Epoch: %s, step: %s, train_loss: %s\n' % (epoch, step, train_loss))

        dist.barrier()
        if rank == 0:
            print("Finished epoch", flush=True)


        model.eval()
        print('start validation....')
        valid_scores = model_valid(model)

        cur_score = [valid_scores[t] for t in all_modals]
        outputs = [None for _ in range(int(os.environ['WORLD_SIZE']))]
        dist.all_gather_object(outputs, cur_score)

        if rank==0:
            def list_mean(x):
                return np.array(x).mean(0)
            gather_scores={}

            finalscore = 0
            for i,x in enumerate(all_modals):
                gather_scores[x]=list_mean([outputs[pi][i] for pi in range(int(os.environ['WORLD_SIZE']))])
                finalscore+=gather_scores[x].sum()
            print(gather_scores,finalscore)

            with open(args.out+'ddp_log_%s_full.txt'%args.prefix, 'a') as f:
                for t in all_modals:
                    f.write('Task: %s, score: %s\n' % (t, ','.join(str(round(s, 3)) for s in gather_scores[t])))



            if finalscore > best_score:
                best_score=finalscore
                torch.save(model.module.state_dict(),os.path.join(args.out,'%s.pt'%args.prefix))






if __name__ == "__main__":
    args = get_args()
    # setup(args)
    init_distributed()
    main(int(os.environ['LOCAL_RANK']), args)