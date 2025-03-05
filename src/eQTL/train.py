from torch.utils.data import DataLoader, Dataset,random_split,ConcatDataset,Subset
import os,torch,random
import numpy as np
from scipy.sparse import load_npz
import torch.optim as optim
import sys,inspect,datetime,argparse
currentdir = os.path.dirname(os.path.abspath(inspect.getfile(inspect.currentframe())))
parentdir = os.path.dirname(currentdir)
sys.path.insert(0, parentdir)
from layers import build_global_model,build_local_model
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data.distributed import DistributedSampler
import torch.nn as nn
import pickle,time
from einops import rearrange
from sklearn.metrics import average_precision_score,roc_auc_score
from torch.nn.utils.rnn import pad_sequence
from torch.optim.lr_scheduler import LambdaLR



def parser_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--bins', type=int, default=600)
    parser.add_argument('--crop', type=int, default=50)
    parser.add_argument('--embed_dim', default=960, type=int)
    parser.add_argument('--epochs', default=5, type=int)
    parser.add_argument('--lr', default=5e-5, type=float)
    parser.add_argument('--atac_block', default=True, action='store_false')
    parser.add_argument('-l', '--logits_type', type=str, default='dilate')
    parser.add_argument('--prompt', default=False, action='store_true')
    parser.add_argument('--external', default=True, action='store_false')
    parser.add_argument('-s','--seed',type=int, default=1024)
    parser.add_argument('-w','--warm_up_lr', default=False, action='store_true')
    parser.add_argument('-p','--prefix', default='', type=str)
    parser.add_argument('-t', '--twod', default=True, action='store_false')
    parser.add_argument( '--indel', default=False, action='store_true')
    parser.add_argument('-d', '--fold', default=0,type=int)
    parser.add_argument('-f', '--fix', default=False, action='store_true')
    parser.add_argument('-v', '--var', default=True, action='store_false')
    parser.add_argument('-m', '--var_model', default='enf')
    args = parser.parse_args()
    return args
def get_args():
    args = parser_args()
    return args



def remove_padded_values(tensor, pad_value=24):
    filtered_tensor = tensor[:, torch.all(tensor!=pad_value,dim=0)]
    return filtered_tensor.numpy()
class QtlDataset(Dataset):
    def __init__(self,cell,cell_idx,total_length=200):
        with open('genes.pickle', 'rb') as f:
            gene_annotation = pickle.load(f)
        self.ordered_genes = sorted(list(gene_annotation.keys()))

        tmpgeneTSS = np.loadtxt('ensemblTSS.txt', dtype='str')
        self.geneTSS_dic = {tmpgeneTSS[i, 0]: int(tmpgeneTSS[i, 1]) for i in range(tmpgeneTSS.shape[0])}

        self.atac_dir='/nfs/turbo/umms-drjieliu/proj/EPCOT_eqtl/'

        self.sample_dir = 'eqtl_data/'

        self.samples,self.ref_alleles,self.alt_alleles,self.var_score=self.load_samples(cell,cell_idx)
        self.num=self.samples.shape[0]
        self.total_length=total_length

        with open('genes.pickle', 'rb') as f:
            self.gene_info = pickle.load(f)
        genetype_indices=self.gene_type_embed(self.samples[:,3])
        self.samples=np.concatenate((self.samples,genetype_indices),axis=1)

    def gene_type_embed(self,x,genetype_dict={'protein_coding':0,'lncRNA':1}):
        geneids = sorted(list(self.gene_info.keys()))
        genetype_ids=[genetype_dict[self.gene_info[geneids[t]][-1]]
                      if genetype_dict.get(self.gene_info[geneids[t]][-1]) is not None else 2 for t in x]
        return np.array(genetype_ids).reshape(-1,1)


    def pad_input(self,x,padding_value=24,length=10):
        x_pad_length=length-x.shape[1]
        pad_data= padding_value*torch.ones((x.shape[0],x_pad_length,x.shape[-1]))

        return torch.cat((x,pad_data),dim=1)

    def load_samples(self,cell,cell_idx):

        data = np.loadtxt(self.sample_dir  + cell + '.txt', dtype='str')


        ref_allele_columns=data[:,8]
        alt_allele_columns=data[:,9]

        ref_allele_columns=[self.one_hot_encode(seq) for seq in ref_allele_columns]
        alt_allele_columns = [self.one_hot_encode(seq) for seq in alt_allele_columns]

        ref_data_padded = pad_sequence(ref_allele_columns, batch_first=True, padding_value=24)
        alt_data_padded = pad_sequence(alt_allele_columns, batch_first=True, padding_value=24)
        ref_data_padded=self.pad_input(ref_data_padded,length=10)
        alt_data_padded = self.pad_input(alt_data_padded,length=10)

        remaining_data=np.delete(data,[8,9],axis=1).astype('int')
        cell_vector = np.full((remaining_data.shape[0], 1), cell_idx)

        if args.var_model=='enf':
            variant_score=np.load('all_enf_data/'+cell+'_pred_diff_all_1.npy')
            print(cell, data.shape, variant_score.shape)
        elif args.var_model=='borzoi':
            variant_score = np.load('borzoi_data/variant_scores/' + cell + '_diff_l2_1.npy')
            print(cell, data.shape, variant_score.shape)
        else:
            print('not using variant scores',cell, data.shape)
            variant_score=None

        return np.concatenate((cell_vector, remaining_data), axis=1),ref_data_padded,alt_data_padded,variant_score

    def one_hot_encode(self,seq, mapping={'A': 0, 'C': 1, 'G': 2, 'T': 3}):
        # Create a one-hot encoded matrix for the sequence
        one_hot = torch.zeros(len(seq), len(mapping), dtype=torch.int)
        for i, nucleotide in enumerate(seq):
            one_hot[i, mapping[nucleotide]] = 1
        return one_hot

    def __getitem__(self, index):
        return self.samples[index],self.ref_alleles[index],self.alt_alleles[index],self.var_score[index]

    def __len__(self):
        return self.num


class QtlDataset_1(Dataset):
    def __init__(self,total_length=200):
        self.samples,self.ref_alleles,self.alt_alleles=self.load_samples()
        self.num=self.samples.shape[0]
        self.total_length=total_length

        with open('genes.pickle', 'rb') as f:
            self.gene_info = pickle.load(f)
        genetype_indices=self.gene_type_embed(self.samples[:,3])
        self.samples=np.concatenate((self.samples,genetype_indices),axis=1)
        self.var_score=torch.zeros_like(self.ref_alleles)

    def gene_type_embed(self,x,genetype_dict={'protein_coding':0,'lncRNA':1}):
        geneids = sorted(list(self.gene_info.keys()))
        genetype_ids=[genetype_dict[self.gene_info[geneids[t]][-1]]
                      if genetype_dict.get(self.gene_info[geneids[t]][-1]) is not None else 2 for t in x]
        return np.array(genetype_ids).reshape(-1,1)

    def pad_input(self,x,padding_value=24,length=10):
        x_pad_length=length-x.shape[1]
        pad_data= padding_value*torch.ones((x.shape[0],x_pad_length,x.shape[-1]))
        return torch.cat((x,pad_data),dim=1)

    def load_samples(self):
        data = np.loadtxt('evalset.txt', dtype='str')

        ref_allele_columns=data[:,9]
        alt_allele_columns=data[:,10]

        ref_allele_columns=[self.one_hot_encode(seq) for seq in ref_allele_columns]
        alt_allele_columns = [self.one_hot_encode(seq) for seq in alt_allele_columns]

        ref_data_padded = pad_sequence(ref_allele_columns, batch_first=True, padding_value=24)
        alt_data_padded = pad_sequence(alt_allele_columns, batch_first=True, padding_value=24)
        ref_data_padded=self.pad_input(ref_data_padded,length=10)
        alt_data_padded = self.pad_input(alt_data_padded,length=10)
        remaining_data=np.delete(data,[9,10,11,13,14],axis=1).astype('int')
        return remaining_data,ref_data_padded,alt_data_padded

    def one_hot_encode(self,seq, mapping={'A': 0, 'C': 1, 'G': 2, 'T': 3}):
        one_hot = torch.zeros(len(seq), len(mapping), dtype=torch.int)
        for i, nucleotide in enumerate(seq):
            one_hot[i, mapping[nucleotide]] = 1
        return one_hot

    def __getitem__(self, index):
        return self.samples[index],self.ref_alleles[index],self.alt_alleles[index],self.var_score[index]

    def __len__(self):
        return self.num



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

def build_dataloaders(cells):
    train_datasets=[]
    valid_loaders={}
    test_loaders={}

    torch.manual_seed(24)
    for cell_idx,cell in enumerate(cells):
        if cell in ['uterus', 'colon_sigmoid', 'adrenal_gland', 'pancreas',
             'esophagus_mucosa', 'spleen', 'ovary', 'heart_left_ventricle',
             'artery_coronary', 'colon_transverse', 'artery_tibial', 'stomach',
             'thyroid','lung','esophagus_muscularis','artery_aorta',
             'esophagus_gej','nerve_tibial','skin_sun_exposed',
             'skin_not_sun_exposed','liver','LCL','adipose_subcutaneous', 'fibroblasts','muscle']:

            cell_dataset = QtlDataset(cell, cell_idx)
            np.random.seed(24)
            index = np.arange(len(cell_dataset))
            np.random.shuffle(index)

            test_start,test_end= int(args.fold*0.2 * len(cell_dataset)), int((args.fold+1)*0.2 * len(cell_dataset))
            train_i, test_i = np.concatenate((index[:test_start],index[test_end:])), index[test_start:test_end]
            train_dataset = Subset(cell_dataset, indices=train_i)
            valid_dataset = Subset(cell_dataset, indices=test_i)
            print(cell, len(train_dataset), len(valid_dataset))
            train_datasets.append(train_dataset)

            valid_loaders[cell]=DataLoader(
                valid_dataset,
                batch_size=16,
                pin_memory=True,
                shuffle=False,
                sampler=DistributedSampler(valid_dataset,shuffle=False)
            )
        else:
            test_dataset=QtlDataset(cell,cell_idx)
            print(cell,len(test_dataset))
            test_loaders[cell] = DataLoader(
                test_dataset,
                batch_size=16,
                pin_memory=True,
                shuffle=False,
                sampler=DistributedSampler(test_dataset,shuffle=False)
            )
    train_datasets=ConcatDataset(train_datasets)
    train_loader= DataLoader(
            train_datasets,
            batch_size=16,
            pin_memory=True,
            sampler=DistributedSampler(train_datasets)
        )
    return train_loader,test_loaders,valid_loaders,len(train_datasets)


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

def copy_param(model1, model2):
    for param_m1, param_m2 in zip(model1.parameters(), model2.parameters()):
        param_m2.data.copy_(param_m1.data)



def lr_lambda(epoch):
    if epoch < 5:
        return 1
    else:
        return 0.2

def main(gpu, args):
    torch.cuda.set_device(gpu)
    rank = gpu



    seed_value=args.seed+rank
    torch.manual_seed(seed_value)
    torch.cuda.manual_seed_all(seed_value)

    train_tissues = ['CD4_T-cell_naive','CD8_T-cell_naive', 'NK-cell_naive', 'monocyte_naive',
                   'B-cell_naive','Treg_memory', 'Treg_naive','sensory_neuron','macrophage_IFNg',
                   'macrophage_naive', 'macrophage_IFNg+Salmonella', 'macrophage_Salmonella','iPSC',
                     'uterus', 'colon_sigmoid', 'adrenal_gland', 'pancreas',
             'esophagus_mucosa', 'spleen', 'ovary', 'heart_left_ventricle',
             'artery_coronary', 'colon_transverse', 'artery_tibial', 'stomach',
             'thyroid','lung','esophagus_muscularis','artery_aorta',
             'esophagus_gej','nerve_tibial','skin_sun_exposed',
             'skin_not_sun_exposed','liver','LCL','adipose_subcutaneous', 'fibroblasts','muscle']

    tissue_twod_rep_dic={}
    tissue_seq_rep_dic = {}

    for tissue in train_tissues:

        with open('rep_data/'+tissue+'_twod_rep.pickle','rb') as f:
            tissue_twod_rep_dic[tissue]=pickle.load(f)
        with open('rep_data/'+tissue+'_seq_rep.pickle','rb') as f:
            tissue_seq_rep_dic[tissue]=pickle.load(f)


    model=build_local_model(args)

    if not args.fix:
        model.load_state_dict(torch.load('models/2d_base_%s_%s_%s.pt'%(args.var,args.var_model,args.fold), map_location='cpu'), strict=False)
    model.cuda(gpu)
    if int(os.environ['WORLD_SIZE'])>1:
        model = torch.nn.SyncBatchNorm.convert_sync_batchnorm(model)
    model = DDP(model,find_unused_parameters = True,  device_ids=[gpu])
    model.train()

    global_model = build_global_model()
    global_model.load_state_dict(torch.load('../curriculum/models/ddp_rna_strand_2_full.pt', map_location='cpu'),
                                 strict=False)
    global_model.cuda(gpu)
    global_model = DDP(global_model, find_unused_parameters=False, device_ids=[gpu])
    global_model.eval()

    if rank == 0:
        for n, p in model.named_parameters():
            if args.fix:
                if 'pretrain_model.' in n or 'attention_pool.' in n:
                    p.requires_grad = False
            print(n, p.requires_grad)

    if args.fix:
        args.epochs=5
        args.prefix='base'
        optimizer = optim.Adam(filter(lambda p: p.requires_grad, model.parameters()), lr=args.lr)
    else:
        args.epochs = 3
        # set_dropout_to_p(model, p=0.01)
        args.prefix = 'ft'
        optimizer = optim.Adam(filter(lambda p: p.requires_grad, model.parameters()), lr=args.lr)





    train_loader,test_loaders,valid_loaders,training_dataset_size=build_dataloaders(train_tissues)


    if rank==0:
        print('Training dataset size: ', training_dataset_size)

    reference_sequence = {}
    for chromosome in ([i for i in range(1, 23)] + ['X']):
        # ref_path = '/nfs/turbo/umms-drjieliu/usr/zzh/KGbert/3D/data/ref_genome/'
        ref_path = '../refSeq/hg38/'
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
        cell = train_tissues[cell_idx]
        atac_seq_data = np.load(atac_dir + 'arrays/' + cell + '_atac_' + str(chrom) + '.npy', mmap_mode='r')
        return atac_seq_data



    def prepare_local_model_inputs(chrom, input_start, input_end, cell_idx, ref=None, alt=None, snp_loc=None):
        atac_seq_data = load_atacseq_data(cell_idx, chrom)

        atac_seq_for_ref=np.expand_dims(np.array(atac_seq_data[input_start-300:input_end+300]),axis=0)
        genome_seq_for_ref=reference_sequence[chrom][:, input_start-300: input_end+300].copy()
        input_for_ref=torch.tensor(np.concatenate((genome_seq_for_ref,atac_seq_for_ref.copy()),axis=0)).unsqueeze(0).float()

        if ref is None:
            return input_for_ref

        length_of_ref_alleles, length_of_alt_alleles = ref.shape[1], alt.shape[1]
        assert np.array_equal(genome_seq_for_ref[:, snp_loc - 1-input_start+300:snp_loc-1-input_start+300+length_of_ref_alleles], ref)

        genome_seq_for_alt=np.concatenate([
            reference_sequence[chrom][:, input_start-300:snp_loc - 1].copy(),
            alt,
            reference_sequence[chrom][:,snp_loc-1+length_of_ref_alleles:
                                        input_start+1300+(length_of_ref_alleles-length_of_alt_alleles)].copy()
        ],axis=1)

        value_for_alt_allele=atac_seq_for_ref[0, snp_loc - 1-input_start+300]
        atac_for_alt_allele=np.array([value_for_alt_allele for _ in range(length_of_alt_alleles)])

        atac_seq_for_alt=np.concatenate([
            np.array(atac_seq_data[input_start-300:snp_loc - 1]).copy(),
            atac_for_alt_allele,
            np.array(atac_seq_data[snp_loc-1+length_of_ref_alleles:
                                        input_start+1300+(length_of_ref_alleles-length_of_alt_alleles)]).copy()
        ])

        input_for_alt = torch.tensor(np.concatenate(
            (genome_seq_for_alt, np.expand_dims(atac_seq_for_alt,axis=0)), axis=0)).unsqueeze(0).float()

        return input_for_ref,input_for_alt

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

    def model_inputs_outputs(input_data):
        input_sample_list, ref_alleles, alt_alleles,var_scores = input_data
        local_ref_data_input,local_alt_data_input,targets=[],[],[]
        twod_rep_input=[]
        # seq_rep_input=[]
        dist_data=[]
        tss_data_input=[]
        gene_type_data=[]
        inside_gene=[]
        for i in range(input_sample_list.shape[0]):
            ref_allele, alt_allele = ref_alleles[i].T, alt_alleles[i].T
            ref_allele, alt_allele = remove_padded_values(ref_allele, pad_value=24), remove_padded_values(alt_allele,
                                                                                                          pad_value=24)

            cell_idx, label, _, gene_idx, chrom, gene_start, gene_end, strand, \
                qtl_loc, _,genetype_idx = [tmp_x.item() for tmp_x in input_sample_list[i]]

            inside_gene.append(1 if qtl_loc<gene_end and qtl_loc>gene_start else 0)

            label = torch.tensor([label]).unsqueeze(0).float()
            local_input_start, local_input_end = qtl_loc-500,qtl_loc+500
            local_ref_input, local_alt_input = prepare_local_model_inputs(chrom, local_input_start, local_input_end, cell_idx, ref_allele, alt_allele, qtl_loc)

            tss_loc=geneTSS_dic[ordered_genes[gene_idx]]
            tss_input, tss_end = tss_loc-500,tss_loc+500

            ref_tss_input=[]
            for genome_pos in range(tss_input-1000,tss_input+2000,1000):
                ref_tss_input.append(prepare_local_model_inputs(chrom, genome_pos, genome_pos+1000, cell_idx))
            ref_tss_input=torch.cat(ref_tss_input).unsqueeze(0)
            tss_data_input.append(ref_tss_input)

            global_input_start, global_input_end, qtl_bin_idx, tss_bin_idx = \
                generate_input_region(gene_start, gene_end,qtl_loc, strand,tss_loc)

            tss_loc_bin = tss_loc // 1000
            qtl_loc_bin = qtl_loc // 1000
            dist_data.append(np.abs(tss_loc - qtl_loc))
            if tissue_twod_rep_dic[train_tissues[cell_idx]].get(str(chrom) + '_' + str(qtl_loc_bin) + '_' + str(tss_loc_bin)) is not None:
                twod_rep = tissue_twod_rep_dic[train_tissues[cell_idx]][str(chrom) + '_' + str(qtl_loc_bin) + '_' + str(tss_loc_bin)]

                twod_rep_input.append(twod_rep)
            else:
                global_input = prepare_global_model_inputs(chrom, global_input_start, global_input_end, cell_idx).cuda(gpu)
                with torch.no_grad():
                    twoDrep, seq_reps, x_microc, x_intacthic = global_model(global_input)
                    twod_rep_input.append(twoDrep.cpu().data.detach()[:, qtl_bin_idx, tss_bin_idx, :].numpy())


            gene_type_data.append(genetype_idx)


            local_ref_data_input.append(local_ref_input)
            local_alt_data_input.append(local_alt_input)
            targets.append(label)
        local_ref_data_input=torch.cat(local_ref_data_input,dim=0).cuda(gpu)
        local_alt_data_input = torch.cat(local_alt_data_input, dim=0).cuda(gpu)
        twod_rep_input= torch.tensor(np.concatenate(twod_rep_input)).float().cuda(gpu)

        tss_data_input = torch.cat(tss_data_input, dim=0).cuda(gpu)
        targets=torch.cat(targets, dim=0).cuda(gpu)
        dist_data=torch.tensor(np.array(dist_data)).unsqueeze(1).float().cuda(gpu)
        gene_type_data=np.array(gene_type_data)
        inside_gene=np.array(inside_gene)
        return local_ref_data_input,local_alt_data_input,twod_rep_input,None,targets,dist_data,\
               tss_data_input,gene_type_data,inside_gene,var_scores.float().cuda(gpu)


    best_score=0
    # set_dropout_to_p(model, p=0.01)
    for epoch in range(args.epochs):
        train_loader.sampler.set_epoch(epoch)
        train_loss = 0

        criterion = nn.BCEWithLogitsLoss()
        model.train()

        dist.barrier()
        max_steps = training_dataset_size // int(os.environ['WORLD_SIZE']) + 1 \
            if training_dataset_size % int(os.environ['WORLD_SIZE']) else training_dataset_size// int(
            os.environ['WORLD_SIZE'])
        if rank==0:
            print('max steps:',max_steps)

        for step, training_data in enumerate(train_loader):


            tts = time.time()

            ref_local_input, alt_local_input, twodrep_input, seq_rep_input, label, dist_data, tss_data_input, \
            gene_type_data, inside_gene, var_scores = model_inputs_outputs(training_data)
            if args.twod:
                if not args.var:
                    var_scores=None
                qtl_logit= model(alt_local_input, ref_local_input, x_tss=tss_data_input,
                                              x_2d_rep=twodrep_input,
                                              x_seq_rep=seq_rep_input, gene_type=gene_type_data,
                                              inside_gene=inside_gene, dist=dist_data, x_var=var_scores)
            else:
                qtl_logit = model(alt_local_input, ref_local_input, x_tss=tss_data_input, x_2d_rep=None,dist=None)

            loss = criterion(qtl_logit, label)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            train_loss += loss.detach()

            if rank == 0:
                print(step,np.round(time.time()-tts,2),np.round(loss.item(),3))
            if rank==0 and step % 1000 == 0:
                current_lr = optimizer.param_groups[0]['lr']
                with open('log_%s_%s%s_%s.txt'%(args.prefix,args.twod,'_indel' if args.indel else '', args.fold), 'a') as f:
                    f.write('Epoch: %s, step: %s, loss: %s, train_loss: %s, lr: %s\n' % (epoch, step,np.round(loss.item(),3) ,train_loss,current_lr))

        dist.barrier()
        if rank == 0:
            print("Finished epoch", flush=True)

        model.eval()
        print('start validation....')

        if rank == 0:
            print("validation step", flush=True)

        preds, targets = {}, {}
        distance={}
        for cl in ['uterus', 'colon_sigmoid', 'adrenal_gland', 'pancreas',
             'esophagus_mucosa', 'spleen', 'ovary', 'heart_left_ventricle',
             'artery_coronary', 'colon_transverse', 'artery_tibial', 'stomach',
             'thyroid','lung','esophagus_muscularis','artery_aorta',
             'esophagus_gej','nerve_tibial','skin_sun_exposed',
             'skin_not_sun_exposed','liver','LCL','adipose_subcutaneous', 'fibroblasts','muscle']:
            preds[cl], targets[cl],distance[cl]= [], [],[]
            for step, valid_data in enumerate(valid_loaders[cl]):
                ref_local_input, alt_local_input, twodrep_input,seq_rep_input, label, dist_data, tss_data_input,\
                gene_type_data,inside_gene, var_scores=model_inputs_outputs(valid_data)

                with torch.no_grad():
                    if args.twod:
                        if not args.var:
                            var_scores = None
                        qtl_logit = model(alt_local_input, ref_local_input, x_tss=tss_data_input,
                                                      x_2d_rep=twodrep_input,
                                                      x_seq_rep=seq_rep_input, gene_type=gene_type_data,
                                                      inside_gene=inside_gene, dist=dist_data, x_var=var_scores,
                                        )
                    else:
                        qtl_logit = model(alt_local_input, ref_local_input, x_tss=tss_data_input, x_2d_rep=None,
                                          dist=None)
                    qtl_prob = torch.sigmoid(qtl_logit)

                preds[cl].append(qtl_prob.cpu().data.detach().numpy())
                targets[cl].append(label.cpu().data.detach().numpy())
                distance[cl].append(np.where(dist_data.squeeze(1).cpu().data.detach().numpy()>3000,1,0))

        for cl in ['CD4_T-cell_naive','CD8_T-cell_naive', 'NK-cell_naive', 'monocyte_naive',
                   'B-cell_naive','Treg_memory', 'Treg_naive','sensory_neuron','macrophage_IFNg',
                   'macrophage_naive', 'macrophage_IFNg+Salmonella', 'macrophage_Salmonella','iPSC']:
            preds[cl], targets[cl],distance[cl]= [], [],[]
            for step, testing_data in enumerate(test_loaders[cl]):
                ref_local_input, alt_local_input, twodrep_input,seq_rep_input, label,dist_data,tss_data_input,\
                gene_type_data,inside_gene,var_scores= model_inputs_outputs(testing_data)

                with torch.no_grad():
                    if args.twod:
                        if not args.var:
                            var_scores = None
                        qtl_logit= model(alt_local_input, ref_local_input, x_tss=tss_data_input,
                                                      x_2d_rep=twodrep_input,
                                                      x_seq_rep=seq_rep_input, gene_type=gene_type_data,
                                                      inside_gene=inside_gene, dist=dist_data, x_var=var_scores,
                                        )
                    else:
                        qtl_logit= model(alt_local_input, ref_local_input, x_tss=tss_data_input, x_2d_rep=None,
                                          dist=None)
                    qtl_prob=torch.sigmoid(qtl_logit)


                preds[cl].append(qtl_prob.cpu().data.detach().numpy())
                targets[cl].append(label.cpu().data.detach().numpy())
                distance[cl].append(np.where(dist_data.squeeze(1).cpu().data.detach().numpy()> 3000, 1, 0))
        predict_vals=[np.concatenate(preds[cl],axis=0).squeeze()  for cl in train_tissues]
        target_vals=[np.concatenate(targets[cl],axis=0).squeeze()  for cl in train_tissues]
        dist_vals=[np.concatenate(distance[cl])  for cl in train_tissues]


        if int(os.environ['WORLD_SIZE'])==1:
            eval_gtexcells = ['uterus', 'colon_sigmoid', 'adrenal_gland', 'pancreas',
                              'esophagus_mucosa', 'spleen', 'ovary', 'heart_left_ventricle',
                              'artery_coronary', 'colon_transverse', 'artery_tibial', 'stomach',
                              'thyroid', 'lung', 'esophagus_muscularis', 'artery_aorta',
                              'esophagus_gej', 'nerve_tibial', 'skin_sun_exposed',
                              'skin_not_sun_exposed', 'liver', 'LCL', 'adipose_subcutaneous', 'fibroblasts', 'muscle']
            all_var_ids = set()
            for cell in eval_gtexcells:

                data = np.loadtxt('eqtl_data/' + cell + '.txt', dtype='str')
                index = np.arange(data.shape[0])
                np.random.seed(24)
                np.random.shuffle(index)

                test_start,test_end = int(args.fold * 0.2 * data.shape[0]), int(
                    (args.fold + 1) * 0.2 * data.shape[0])
                train_i = np.concatenate((index[:test_start], index[test_end:]))

                data = data[train_i]
                variants = [data[i, 3] + '_' + data[i, 7] for i in range(data.shape[0])]
                for var in variants:
                    all_var_ids.add(var)
            print(args.fold,len(all_var_ids))
            eval_gcells = ['CD4_T-cell_naive','CD8_T-cell_naive', 'NK-cell_naive', 'monocyte_naive',
                           'B-cell_naive', 'Treg_memory', 'Treg_naive', 'sensory_neuron', 'macrophage_IFNg',
                           'macrophage_naive', 'macrophage_IFNg+Salmonella', 'macrophage_Salmonella', 'iPSC']

            all_pred = np.concatenate([np.concatenate(preds[cl]) for cl in eval_gcells])
            all_targ = np.concatenate([np.concatenate(targets[cl]) for cl in eval_gcells])
            all_datas = []
            for cell_idx, cell in enumerate(eval_gcells):
                data = np.loadtxt('eqtl_data/' + cell + '.txt', dtype='str')
                all_datas.append(data)
            all_datas = np.vstack(all_datas)
            all_datas = np.vstack(all_datas)
            variants = [all_datas[i, 3] + '_' + all_datas[i, 7] for i in range(all_datas.shape[0])]
            indices = []
            for i, var in enumerate(variants):
                if var not in all_var_ids:
                    indices.append(i)
            indices = np.array(indices)
            roc = roc_auc_score(all_targ[indices], all_pred[indices])
            ap = average_precision_score(all_targ[indices], all_pred[indices])
            print('----',args.fold,roc, ap)


        outputs_p = [None for _ in range(int(os.environ['WORLD_SIZE']))]
        outputs_t = [None for _ in range(int(os.environ['WORLD_SIZE']))]
        outputs_dist=[None for _ in range(int(os.environ['WORLD_SIZE']))]

        dist.all_gather_object(outputs_p, predict_vals)
        dist.all_gather_object(outputs_t, target_vals)
        dist.all_gather_object(outputs_dist,dist_vals)

        if rank == 0:
            roc_scores,ap_scores=[],[]
            roc_scores_dist, ap_scores_dist = [], []
            predseq,targseq,distances=[],[],[]
            results={}
            for i, tissue in enumerate(train_tissues):
                results[tissue]={}
                tmp_pred_array=np.concatenate([outputs_p[pi][i] for pi in range(int(os.environ['WORLD_SIZE']))])
                tmp_targ_array = np.concatenate([outputs_t[pi][i] for pi in range(int(os.environ['WORLD_SIZE']))])
                tmp_dist=np.concatenate([outputs_dist[pi][i] for pi in range(int(os.environ['WORLD_SIZE']))])


                predseq.append(tmp_pred_array)
                targseq.append(tmp_targ_array)
                distances.append(tmp_dist)


                roc_scores.append(roc_auc_score(tmp_targ_array,tmp_pred_array))
                ap_scores.append(average_precision_score(tmp_targ_array,tmp_pred_array))

                roc_scores_dist.append(roc_auc_score(tmp_targ_array[tmp_dist>0], tmp_pred_array[tmp_dist>0]))
                ap_scores_dist.append(average_precision_score(tmp_targ_array[tmp_dist>0], tmp_pred_array[tmp_dist>0]))

                print(tissue, roc_auc_score(tmp_targ_array,tmp_pred_array),average_precision_score(tmp_targ_array,tmp_pred_array),
                      roc_auc_score(tmp_targ_array[tmp_dist>0],tmp_pred_array[tmp_dist>0]),average_precision_score(tmp_targ_array[tmp_dist>0],tmp_pred_array[tmp_dist>0]))

                results[tissue]['pred']=tmp_pred_array
                results[tissue]['targ'] = tmp_targ_array
                results[tissue]['dist'] = tmp_dist

            predseq1,predseq2=np.concatenate(predseq[:13]),np.concatenate(predseq[13:])
            targseq1,targseq2=np.concatenate(targseq[:13]),np.concatenate(targseq[13:])
            distances1,distances2=np.concatenate(distances[:13]),np.concatenate(distances[13:])

            allroc_2 = roc_auc_score(targseq2, predseq2)
            allap_2 = average_precision_score(targseq2, predseq2)
            allroc_dist_2 = roc_auc_score(targseq2[distances2 > 0], predseq2[distances2 > 0])
            allap_dist_2 = average_precision_score(targseq2[distances2 > 0], predseq2[distances2 > 0])

            allroc_1=roc_auc_score(targseq1,predseq1)
            allap_1=average_precision_score(targseq1,predseq1)
            allroc_dist_1=roc_auc_score(targseq1[distances1>0],predseq1[distances1>0])
            allap_dist_1=average_precision_score(targseq1[distances1>0],predseq1[distances1>0])


            with open('log_%s_%s%s_%s.txt'%(args.prefix,args.twod,'_indel' if args.indel else '', args.fold), 'a') as f:
                f.write('eval: %s, %s, dist: %s, %s  \n'%(np.round(allroc_2,3),np.round(allap_2 ,3),
                                                          np.round(allroc_dist_2 ,3),np.round(allap_dist_2 ,3)))

            with open('log_%s_%s%s_%s.txt' % (args.prefix, args.twod, '_indel' if args.indel else '', args.fold), 'a') as f:
                f.write('test: %s, %s, dist: %s, %s  \n' % (np.round(allroc_1, 3), np.round(allap_1, 3),
                                                            np.round(allroc_dist_1, 3), np.round(allap_dist_1, 3)))


            with open('log_%s_%s%s_%s.txt'%(args.prefix,args.twod,'_indel' if args.indel else '', args.fold), 'a') as f:
                f.write('averge auroc: %s, average AP: %s \n'%(np.round(np.mean(roc_scores[13:]),3),np.round(np.mean(ap_scores[13:]),3)))

            with open('log_%s_%s%s_%s.txt'%(args.prefix,args.twod,'_indel' if args.indel else '', args.fold), 'a') as f:
                f.write('dist auroc: %s, dist AP: %s \n'%(np.round(np.mean(roc_scores_dist[13:]),3),np.round(np.mean(ap_scores_dist[13:]),3)))

            with open('log_%s_%s%s_%s.txt'%(args.prefix,args.twod,'_indel' if args.indel else '', args.fold), 'a') as f:
                f.write('averge auroc: %s, average AP: %s \n'%(np.round(np.mean(roc_scores[:13]),3),np.round(np.mean(ap_scores[:13]),3)))

            with open('log_%s_%s%s_%s.txt'%(args.prefix,args.twod,'_indel' if args.indel else '', args.fold), 'a') as f:
                f.write('dist auroc: %s, dist AP: %s \n'%(np.round(np.mean(roc_scores_dist[:13]),3),np.round(np.mean(ap_scores_dist[:13]),3)))




            cur_score=allap_2
            if cur_score > best_score:
                with open('log_%s_%s%s_%s.txt'%(args.prefix,args.twod,'_indel' if args.indel else '', args.fold), 'a') as f:
                    f.write('save model \n')
                best_score=cur_score
                if args.twod:
                    torch.save(model.module.state_dict(), 'models/2d_%s%s_%s_%s_%s.pt' % (args.prefix,'_indel' if args.indel else '',args.var,args.var_model,args.fold))
                    with open('results/2d_%s%s_%s_%s_%s.pickle' % (args.prefix,'_indel' if args.indel else '',args.var, args.var_model,args.fold),'wb') as f:
                        pickle.dump(results,f)
                else:
                    torch.save(model.module.state_dict(), 'models/local_model_%s.pt'%args.prefix)
                    with open('results/local_%s%s.pickle' % (args.prefix,'_indel' if args.indel else ''),'wb') as f:
                        pickle.dump(results,f)






if __name__ == "__main__":
    args = get_args()
    # setup(args)
    init_distributed()
    main(int(os.environ['LOCAL_RANK']), args)