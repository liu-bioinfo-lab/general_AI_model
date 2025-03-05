from torch.utils.data import DataLoader, Dataset,random_split,ConcatDataset
import os,torch,random
import numpy as np
from scipy.sparse import load_npz
import torch.optim as optim
import sys,inspect,datetime,argparse
currentdir = os.path.dirname(os.path.abspath(inspect.getfile(inspect.currentframe())))
parentdir = os.path.dirname(currentdir)
sys.path.insert(0, parentdir)

from layers import build_global_model

import torch.nn as nn
import pickle
from einops import rearrange


def parser_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--bins', type=int, default=600)
    parser.add_argument('--crop', type=int, default=50)
    parser.add_argument('--embed_dim', default=960, type=int)
    parser.add_argument('--epochs', default=20, type=int)
    parser.add_argument('--accum_iter', default=2, type=int)
    parser.add_argument('--lr', default=1e-4, type=float)
    parser.add_argument('--atac_block', default=True, action='store_false')
    parser.add_argument('-l', '--logits_type', type=str, default='dilate')
    parser.add_argument('--prompt', default=False, action='store_true')
    parser.add_argument('--external', default=True, action='store_false')
    parser.add_argument('-c', '--cl', type=str)
    args = parser.parse_args()
    return args
def get_args():
    args = parser_args()
    return args



class QtlDataset(Dataset):
    def __init__(self,cell,cell_idx):
        self.atac_dir='/nfs/turbo/umms-drjieliu/proj/EPCOT_eqtl/'
        self.sample_dir='eqtl_data/'

        self.samples=self.load_samples(cell,cell_idx)
        self.num=self.samples.shape[0]

    def load_samples(self,cell,cell_idx):
        data=np.loadtxt(self.sample_dir+cell+'.txt',dtype='str')

        remaining_data=np.delete(data,[8,9],axis=1).astype('int')
        cell_vector = np.full((remaining_data.shape[0], 1), cell_idx)
        return np.concatenate((cell_vector, remaining_data), axis=1)

    def one_hot_encode(self,seq, mapping={'A': 0, 'C': 1, 'G': 2, 'T': 3}):
        one_hot = torch.zeros(len(seq), len(mapping), dtype=torch.int)
        for i, nucleotide in enumerate(seq):
            one_hot[i, mapping[nucleotide]] = 1
        return one_hot.T

    def __getitem__(self, index):
        return self.samples[index]

    def __len__(self):
        return self.num






def build_dataloaders(cell):
    train_tissues = ['uterus', 'colon_sigmoid', 'adrenal_gland', 'pancreas',
             'esophagus_mucosa', 'spleen', 'ovary', 'heart_left_ventricle',
             'artery_coronary', 'colon_transverse', 'artery_tibial', 'stomach',
             'thyroid','small_intestine','lung','esophagus_muscularis','artery_aorta',
             'esophagus_gej','nerve_tibial','skin_sun_exposed',
             'skin_not_sun_exposed','liver','LCL','adipose_subcutaneous','fibroblasts','muscle','CD4_T-cell_naive',
                'CD8_T-cell_naive', 'NK-cell_naive', 'monocyte_naive',
            'B-cell_naive','Treg_memory', 'Treg_naive','sensory_neuron','macrophage_IFNg',
            'macrophage_naive', 'macrophage_IFNg+Salmonella', 'macrophage_Salmonella','iPSC']
    cell_idx=train_tissues.index(cell)
    torch.manual_seed(24)
    cell_dataset=QtlDataset(cell,cell_idx)
    print(cell,cell_idx,len(cell_dataset))

    test_loader = DataLoader(
            cell_dataset,
            batch_size=1,
            shuffle=False
        )
    return test_loader


def generate_input_region(gene_start,gene_end,qtl_loc,strand,tss_loc):
    tss_loc_bin=tss_loc//1000
    qtl_loc_bin = qtl_loc // 1000
    middle_bin = (tss_loc_bin + qtl_loc_bin) // 2
    input_start = (middle_bin - 299) * 1000
    input_end = (middle_bin + 301) * 1000

    qtl_idx=(qtl_loc-input_start)//1000
    tss_idx=(tss_loc-input_start)//1000
    return input_start,input_end,qtl_idx,tss_idx


def set_dropout_to_p(model,p):
    for child_name, child in model.named_children():
        if isinstance(child, nn.Dropout):
            setattr(model, child_name, nn.Dropout(p=p))
        elif list(child.children()):
            set_dropout_to_p(child,p)

def cal_rep_diff(alt_rep,ref_rep, tss_idx,snp_idx=None):
    p_diff = alt_rep - ref_rep
    if snp_idx is not None:
        p_diff =torch.cat((p_diff[:, snp_idx, :], p_diff[:, tss_idx, :]),dim=-1)
    else:
        p_diff=p_diff[:,tss_idx,:]
    return p_diff


def main():
    train_tissues = ['uterus', 'colon_sigmoid', 'adrenal_gland', 'pancreas',
                     'esophagus_mucosa', 'spleen', 'ovary', 'heart_left_ventricle',
                     'artery_coronary', 'colon_transverse', 'artery_tibial', 'stomach',
                     'thyroid', 'small_intestine', 'lung', 'esophagus_muscularis', 'artery_aorta',
                     'esophagus_gej', 'nerve_tibial', 'skin_sun_exposed',
                     'skin_not_sun_exposed', 'liver', 'LCL', 'adipose_subcutaneous', 'fibroblasts','muscle',
                     'CD4_T-cell_naive','CD8_T-cell_naive',
                     'NK-cell_naive', 'monocyte_naive', 'B-cell_naive','Treg_memory', 'Treg_naive','sensory_neuron',
                     'macrophage_IFNg','macrophage_naive', 'macrophage_IFNg+Salmonella', 'macrophage_Salmonella','iPSC']


    args = get_args()
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


    global_model=build_global_model()
    global_model.load_state_dict(torch.load('../curriculum/models/ddp_rna_strand_2_full.pt', map_location='cpu'),strict=False)
    global_model.to(device)
    global_model.eval()


    test_loader=build_dataloaders(cell=args.cl)


    reference_sequence = {}
    for chromosome in ([i for i in range(1, 23)] + ['X']):
        ref_path = '/nfs/turbo/umms-drjieliu/usr/zzh/KGbert/3D/data/ref_genome/'
        ref_file = os.path.join(ref_path, 'chr%s.npz' % chromosome)
        if chromosome=='X':
            reference_sequence[23] = load_npz(ref_file).toarray()
        else:
            reference_sequence[chromosome] = load_npz(ref_file).toarray()

    with open('genes.pickle', 'rb') as f:
        gene_annotation = pickle.load(f)
    ordered_genes = sorted(list(gene_annotation.keys()))

    tmpgeneTSS = np.loadtxt('ensemblTSS.txt', dtype='str')
    geneTSS_dic = {tmpgeneTSS[i, 0]: int(tmpgeneTSS[i, 1]) for i in range(tmpgeneTSS.shape[0])}


    def load_atacseq_data(cell_idx, chrom):
        if chrom == 23:
            chrom = 'X'
        atac_dir = '/nfs/turbo/umms-drjieliu/proj/EPCOT_eqtl/ATAC/'
        assert cell_idx==train_tissues.index(args.cl)
        cell = train_tissues[cell_idx]
        assert cell==args.cl
        atac_seq_data = np.load(atac_dir + 'arrays/' + cell + '_atac_' + str(chrom) + '.npy', mmap_mode='r')
        return atac_seq_data



    def generate_input(genome_seq, genome_seq_pad_left, genome_seq_pad_right, atac_seq, atac_seq_pad_left,
                       atac_seq_pad_right):
        pad_left = np.expand_dims(np.vstack((genome_seq_pad_left, np.expand_dims(atac_seq_pad_left, 0))), 0)
        pad_right = np.expand_dims(np.vstack((genome_seq_pad_right, np.expand_dims(atac_seq_pad_right, 0))), 0)
        center = np.vstack((genome_seq, np.expand_dims(atac_seq, 0)))
        center = rearrange(center, 'n (b l)-> b n l', l=1000)
        dmatrix = np.concatenate((pad_left, center[:, :, -300:]), axis=0)[:-1, :, :]
        umatrix = np.concatenate((center[:, :, :300], pad_right), axis=0)[1:, :, :]
        return np.concatenate((dmatrix, center, umatrix), axis=2)

    def prepare_global_model_inputs(chrom, input_start, input_end, cell_idx):

        atac_seq_data = load_atacseq_data(cell_idx, chrom)

        atac_seq_for_ref = np.array(atac_seq_data[input_start:input_end])
        atac_seq_pad_left = np.array(atac_seq_data[input_start - 300:input_start])
        atac_seq_pad_right = np.array(atac_seq_data[input_end:input_end + 300])

        ref_genome_seq = reference_sequence[chrom][:, input_start: input_end]
        ref_seq_pad_left = reference_sequence[chrom][:, input_start - 300:input_start]
        ref_seq_pad_right = reference_sequence[chrom][:, input_end:input_end + 300]
        ref_input_data = generate_input(ref_genome_seq, ref_seq_pad_left, ref_seq_pad_right, atac_seq_for_ref,
                                        atac_seq_pad_left, atac_seq_pad_right)
        return torch.tensor(ref_input_data).float().unsqueeze(0)

    def model_inputs_outputs(input_data,tss_loc):
        input_sample_list = input_data
        # dist_data=[]
        cell_idx, label, qtl_idx, gene_idx, chrom, gene_start, gene_end, strand, \
                qtl_loc, sign_target = [tmp_x.item() for tmp_x in input_sample_list[0]]

        global_input_start,global_input_end,qtl_bin_idx,tss_bin_idx= \
            generate_input_region(gene_start,gene_end,qtl_loc,strand,tss_loc)
        global_input=prepare_global_model_inputs(chrom, global_input_start,global_input_end, cell_idx).to(device)
        # dist_data.append(np.abs(qtl_bin_idx-tss_bin_idx))
        with torch.no_grad():
            twoDrep,seq_reps,x_microc,x_intacthic=global_model(global_input)
            assert twoDrep.shape[1]==seq_reps.shape[1] and seq_reps.shape[1]==600
            assert x_microc.shape[1]==600 and x_intacthic.shape[1]==600

            assert qtl_bin_idx>=51 and qtl_bin_idx<=498+50
            assert tss_bin_idx >= 51 and tss_bin_idx <= 498 + 50
            print(qtl_bin_idx,tss_bin_idx)
            # tmp_2drep = twoDrep.cpu().data.detach().squeeze()[qtl_bin_idx, tss_bin_idx, :].unsqueeze(0)
            tmp_seqrep=seq_reps.cpu().data.detach().squeeze()[[qtl_bin_idx, tss_bin_idx],:]
            tmp_2drep=twoDrep.cpu().data.detach().squeeze()[qtl_bin_idx,tss_bin_idx-2:tss_bin_idx+3,:].unsqueeze(0)

            tmp_microc = x_microc.cpu().data.detach().squeeze()[qtl_bin_idx,tss_bin_idx-2:tss_bin_idx+3,:].unsqueeze(0)
            tmp_intacthic = x_intacthic.cpu().data.detach().squeeze()[qtl_bin_idx,tss_bin_idx-2:tss_bin_idx+3,:].unsqueeze(0)
        return tmp_2drep.numpy().astype('float32'),tmp_seqrep.numpy().astype('float32'),tmp_microc.numpy().astype('float32'),tmp_intacthic.numpy().astype('float32')


    rep_dic_2d = {}
    rep_dic_seq = {}

    for step, testing_data in enumerate(test_loader):
        input_sample_list = testing_data

        cell_idx, label, qtl_idx, gene_idx, chrom, gene_start, gene_end, strand, \
        qtl_loc, sign_target = [tmp_x.item() for tmp_x in input_sample_list[0]]

        tss_loc=geneTSS_dic[ordered_genes[gene_idx]]


        tss_loc_bin = tss_loc // 1000
        qtl_loc_bin = qtl_loc // 1000
        key_pos=str(chrom)+'_'+str(qtl_loc_bin)+'_'+str(tss_loc_bin)
        if key_pos not in rep_dic_2d.keys():
            val_2d,val_seq,s_microc,s_hic=model_inputs_outputs(testing_data,tss_loc)
            rep_dic_2d[key_pos]=val_2d
            rep_dic_seq[key_pos] = val_seq

    with open('rep_data/'+args.cl+'_seq_rep.pickle','wb') as f:
        pickle.dump(rep_dic_seq,f)
    with open('rep_data/new_'+args.cl+'_twod_rep.pickle','wb') as f:
        pickle.dump(rep_dic_2d,f)







if __name__ == "__main__":


    main()

