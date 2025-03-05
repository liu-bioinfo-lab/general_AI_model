import pickle,h5py
import numpy as np
from scipy.sparse import load_npz,csr_matrix
import torch
import os,gdown
from scipy.stats import zscore
from copy import deepcopy

def value_clip(x,vmax=36):
    return x.clip(-2,vmax)
def normalize_rna(x,alpha=95,scale=2):
    percentile_v=np.percentile(x[x>0],alpha)
    return scale*torch.tensor(np.arcsinh(x / percentile_v)).float().unsqueeze(-1)

def pad_signal_matrix(matrix, pad_len=300):
    paddings = np.zeros(pad_len).astype('float32')
    dmatrix = np.vstack((paddings, matrix[:, -pad_len:]))[:-1, :]
    umatrix = np.vstack((matrix[:, :pad_len], paddings))[1:, :]
    return np.hstack((dmatrix, matrix, umatrix))
def pad_seq_matrix(matrix, pad_len=300):
    paddings = np.zeros((1, 4, pad_len)).astype('int8')
    dmatrix = np.concatenate((paddings, matrix[:, :, -pad_len:]), axis=0)[:-1, :, :]
    umatrix = np.concatenate((matrix[:, :, :pad_len], paddings), axis=0)[1:, :, :]
    return np.concatenate((dmatrix, matrix, umatrix), axis=2)

def load_ref_genome(chr):
    ref_path = 'refSeq/hg38/'
    ref_file = os.path.join(ref_path, 'chr%s.npz' % chr)
    ref_file= os.path.abspath(ref_file)
    ref_gen_data = load_npz(ref_file).toarray().reshape(4, -1, 1000).swapaxes(0, 1)
    return torch.tensor(pad_seq_matrix(ref_gen_data))



def load_dnase(dnase_seq):
    dnase_seq = np.expand_dims(pad_signal_matrix(dnase_seq.astype('float32').toarray().reshape(-1, 1000)), axis=1)
    return torch.tensor(dnase_seq)

def load_bru(cl):
    bru_path='modality_data/'
    bruseq_file =bru_path+'%s_bru_seq_cov.h5'%cl
    with h5py.File(bruseq_file, 'r') as hf:
        bruseq_data = normalize_rna(np.array(hf['targets']).astype('float32'))
    bruuvseq_file = bru_path+'%s_bruuv_seq_cov.h5'%cl
    with h5py.File(bruuvseq_file, 'r') as hf:
        bruuvseq_data = normalize_rna(np.array(hf['targets']).astype('float32'))
    bruchase_file = bru_path+'%s_bruchase_seq_cov.h5'%cl
    with h5py.File(bruchase_file, 'r') as hf:
        bruchase_data = normalize_rna(np.array(hf['targets']).astype('float32'))

    bru_data=torch.cat((bruseq_data,bruuvseq_data,bruchase_data),dim=-1)
    return bru_data

def load_rna(cl):
    data_list=[]
    cage_path='modality_data/'
    rna_path='modality_data/'

    cageseq_file ='%s_cage_seq_cov.h5'%cl
    if os.path.isfile(cage_path+cageseq_file):
        with h5py.File(cage_path+cageseq_file, 'r') as hf:
            cageseq_data = normalize_rna(np.array(hf['targets']).astype('float32'))
        data_list.append(cageseq_data)

    trnaseq_file='%s_trna_seq_cov.h5'%cl
    if os.path.isfile(rna_path+trnaseq_file):
        with h5py.File(rna_path+trnaseq_file, 'r') as hf:
            trnaseq_data = normalize_rna(np.array(hf['targets']).astype('float32'))
        data_list.append(trnaseq_data)

    prnaseq_file = '%s_prna_seq_cov.h5' % cl
    if os.path.isfile(rna_path + prnaseq_file):
        with h5py.File(rna_path + prnaseq_file, 'r') as hf:
            prnaseq_data = normalize_rna(np.array(hf['targets']).astype('float32'))
        data_list.append(prnaseq_data)
    rna_data=torch.cat(data_list,dim=-1)
    print(cl,rna_data.shape)
    return rna_data

def load_entex_rna(cl):
    entex_trna_path='modality_data/'

    trnaseq_file = entex_trna_path+'%s_trna_seq_cov.h5' % cl
    with h5py.File(trnaseq_file, 'r') as hf:
        trnaseq_data = normalize_rna(np.array(hf['targets']).astype('float32'))
    return trnaseq_data



def load_epi(cl,lmask):
    epigenome_path='/nfs/turbo/umms-drjieliu/usr/zzh/scEPCOT/pretrain/data/'

    with open(epigenome_path+'pval_labels_%s_247.pickle' % cl, 'rb') as f:
        tmp_label = pickle.load(f)
    for i in range(1, 23):
        indices=np.where(lmask > 0)[0]
        tmp_label[i] = (tmp_label[i].toarray().T)[:, indices]
        tmp_label[i] = zscore(tmp_label[i], axis=0)
        tmp_label[i] = torch.tensor(value_clip(tmp_label[i])).to(torch.float16)
    return tmp_label

def load_microc(cl,chr):
    microc_path = '/nfs/turbo/umms-drjieliu/usr/zzh/KGbert/3D/OE_matrix/'
    microc_oe = load_npz(microc_path + '%s_Micro-C/chr%s_1kb.npz' % (cl, chr)).astype('float16')
    microc_kr=load_npz(microc_path + '%s_Micro-C/chr%s_KR_1kb.npz' % (cl, chr)).astype('float16')

    return microc_oe,microc_kr

def load_hic(cl,chr):
    hic_path = '/nfs/turbo/umms-drjieliu/usr/zzh/KGbert/3D/OE_matrix/'
    hic = load_npz(hic_path + '%s/chr%s_5kb_1Mb.npz' % (cl, chr))
    return hic

def load_intacthic(cl,chr):
    intacthic_path = '/nfs/turbo/umms-drjieliu/usr/zzh/mutimodal_epcot/intact-hic/'
    intacthic_oe = load_npz(intacthic_path + '%s/chr%s_OE_1kb.npz' % (cl, chr)).astype('float16')
    intacthic_kr=load_npz(intacthic_path + '%s/chr%s_KR_1kb.npz' % (cl, chr)).astype('float16')
    return intacthic_oe,intacthic_kr


def load_chiapet(cl,target,chr):
    hic_path = '/nfs/turbo/umms-drjieliu/usr/zzh/KGbert/3D/OE_matrix/'
    hic = load_npz(hic_path + '%s_%s/chr%s_5kb.npz' % (cl, target,chr))
    return hic




def prepare_train_full_data(cell_dict,train_cells,tissue_dict=None):
    assert cell_dict is not None or tissue_dict is not None

    with open('data/epi_label_masks.pickle', 'rb') as f:
        temp_lmasks = pickle.load(f)
    ref_data = {}

    epi_data = {}
    rna_data={}
    bru_data={}

    microc_data, microc_kr_data = {},{}
    intacthic_oe_data,intacthic_kr_data={},{}
    hic_data,ctcf_chiapet,polr2_chiapet  = {},{},{}


    chroms = [i for i in range(1, 23)]
    for chr in chroms:
        ref_data[chr] = load_ref_genome(chr)

    atac_data = {}
    if cell_dict is not None:
        with open('/nfs/turbo/umms-drjieliu/usr/zzh/mutimodal_epcot/atac_bw/train_atac_merge_fp16.pickle', 'rb') as f:
            atac_data=pickle.load(f)

    if tissue_dict is not None:
        with open('/nfs/turbo/umms-drjieliu/usr/zzh/mutimodal_epcot/atac_bw/train_atac_merge_entex_fp16_2.pickle', 'rb') as f:
            atac_t_data=pickle.load(f)
        atac_data.update(atac_t_data)

    if cell_dict is not None:
        for cl in cell_dict['rna']:
            rna_data[cl] = load_rna(cl)
    else:
        cell_dict = {}
    if tissue_dict is not None:
        for cl in tissue_dict['rna']:
            rna_data[cl] = load_entex_rna(cl)
        if cell_dict:
            tmp = deepcopy(cell_dict)
            for k in tmp.keys():
                cell_dict[k] += tissue_dict[k]
        else:
            cell_dict.update(tissue_dict)

    for cl in cell_dict['epi']:
        epi_data[cl] = load_epi(cl,temp_lmasks[cl])

    for cl in cell_dict['bru']:
        bru_data[cl] = load_bru(cl)

    for cl in cell_dict['microc']:
        microc_data[cl] = {}
        microc_kr_data[cl]={}
        for chr in chroms:
            microc_data[cl][chr],microc_kr_data[cl][chr] = load_microc(cl, chr)

    for cl in cell_dict['intacthic']:
        intacthic_oe_data[cl] = {}
        intacthic_kr_data[cl]={}
        for chr in chroms:
            intacthic_oe_data[cl][chr],intacthic_kr_data[cl][chr] = load_intacthic(cl, chr)

    for cl in cell_dict['hic']:
        ctcf_chiapet[cl] = {}
        polr2_chiapet[cl] = {}
        hic_data[cl]={}
        if cl=='K562' or cl=='HepG2' or 'gastrocnemius_medialis' in cl or 'transverse_colon' in cl:
            for chr in chroms:
                hic_data[cl][chr] = load_hic(cl, chr)
        else:
            for chr in chroms:
                ctcf_chiapet[cl][chr] = load_chiapet(cl, 'CTCF', chr)
                polr2_chiapet[cl][chr] = load_chiapet(cl, 'POLR2', chr)
                hic_data[cl][chr] = load_hic(cl, chr)
    return atac_data,ref_data,epi_data,rna_data,bru_data,microc_data,microc_kr_data,\
           hic_data, ctcf_chiapet, polr2_chiapet,intacthic_oe_data,intacthic_kr_data

def load_groseq(cl):
    groseq_path='modality_data/'

    groseq_fwd_file = groseq_path+'%s_groseq_fwd_seq_cov.h5' % cl
    with h5py.File(groseq_fwd_file, 'r') as hf:
        groseq_fwd_data = normalize_rna(np.array(hf['targets']).astype('float32'))

    groseq_rev_file = groseq_path+'%s_groseq_rev_seq_cov.h5' % cl
    with h5py.File(groseq_rev_file, 'r') as hf:
        groseq_rev_data = normalize_rna(np.array(hf['targets']).astype('float32'))
    return torch.cat((groseq_fwd_data,groseq_rev_data),dim=-1)

def load_grocap(cl):
    groseq_path = 'modality_data/'

    grocap_fwd_file = groseq_path+'fwd_%s_grocap_seq_cov.h5' % cl
    with h5py.File(grocap_fwd_file, 'r') as hf:
        grocap_fwd_data = normalize_rna(np.array(hf['targets']).astype('float32'))

    grocap_rev_file = groseq_path+'rev_%s_grocap_seq_cov.h5' % cl
    with h5py.File(grocap_rev_file, 'r') as hf:
        grocap_rev_data = normalize_rna(np.array(hf['targets']).astype('float32'))

    grocap_tap_fwd_file = groseq_path+'fwd_%s_grocap_wTAP_seq_cov.h5' % cl
    with h5py.File(grocap_tap_fwd_file, 'r') as hf:
        grocap_tap_fwd_data = normalize_rna(np.array(hf['targets']).astype('float32'))

    grocap_tap_rev_file = groseq_path+'rev_%s_grocap_wTAP_seq_cov.h5' % cl
    with h5py.File(grocap_tap_rev_file, 'r') as hf:
        grocap_tap_rev_data = normalize_rna(np.array(hf['targets']).astype('float32'))

    return torch.cat((grocap_fwd_data,grocap_rev_data,grocap_tap_fwd_data,grocap_tap_rev_data),dim=-1)

def load_ttseq(cl):
    tt_path='modality_data/'
    TT_fwd_file = tt_path+'%s_TT_fwd_seq_cov.h5' % cl
    with h5py.File(TT_fwd_file, 'r') as hf:
        TT_fwd_data = normalize_rna(np.array(hf['targets']).astype('float32'))

    TT_rev_file = tt_path+'%s_TT_rev_seq_cov.h5' % cl
    with h5py.File(TT_rev_file, 'r') as hf:
        TT_rev_data = normalize_rna(np.array(hf['targets']).astype('float32'))
    return torch.cat((TT_fwd_data, TT_rev_data),dim=-1)

def load_netcage(cl):
    netcage_path='modality_data/'
    netcage_fwd_file = netcage_path+'%s_netcage_fwd_seq_cov.h5' % cl
    with h5py.File(netcage_fwd_file, 'r') as hf:
        netcage_fwd_data = normalize_rna(np.array(hf['targets']).astype('float32'))

    netcage_rev_file = netcage_path+'%s_netcage_rev_seq_cov.h5' % cl
    with h5py.File(netcage_rev_file, 'r') as hf:
        netcage_rev_data = normalize_rna(np.array(hf['targets']).astype('float32'))
    return torch.cat((netcage_fwd_data, netcage_rev_data),dim=-1)



def load_proseq(cl):
    data_list = []
    pro_path='modality_data/'

    proseq_fwd_file = '%s_pro_fwd_seq_cov.h5' % cl
    proseq_rev_file = '%s_pro_rev_seq_cov.h5' % cl
    if os.path.isfile(pro_path+proseq_fwd_file):
        with h5py.File(pro_path+proseq_fwd_file, 'r') as hf:
            proseq_fwd_data = normalize_rna(np.array(hf['targets']).astype('float32'))
        with h5py.File(pro_path+proseq_rev_file, 'r') as hf:
            proseq_rev_data = normalize_rna(np.array(hf['targets']).astype('float32'))
        data_list.append(proseq_fwd_data)
        data_list.append(proseq_rev_data)

    procap_file = '%s_procap_seq_cov.h5' % cl
    if os.path.isfile(pro_path + procap_file ):
        with h5py.File(pro_path + procap_file, 'r') as hf:
            procap_data = normalize_rna(np.array(hf['targets']).astype('float32'))
        data_list.append(procap_data)

    prodata=torch.cat(data_list,dim=-1)
    print(cl,prodata.shape)
    return prodata

def load_external_tfs(cl):
    with open('/nfs/turbo/umms-drjieliu/usr/zzh/scEPCOT/generalizability/pval_unseen_%s_external.pickle' % cl, 'rb') as f:
        tmp_label = pickle.load(f)
    for i in range(1, 23):
        tmp_label[i] = zscore(tmp_label[i].T, axis=0)
        tmp_label[i] = torch.tensor(value_clip(tmp_label[i])).to(torch.float16)
    return tmp_label

def load_external_starr(cl):
    starr_file = '/nfs/turbo/umms-drjieliu/proj/STARR-seq/data/%s_starr_seq_cov.h5' % cl
    with h5py.File(starr_file, 'r') as hf:
        starr_data = normalize_rna(np.array(hf['targets']).astype('float32'),alpha=98)
    return starr_data


def load_rna_strand(cl,alpha=2):
    input_data={}
    with open('modality_data/norm_%s_fwd_1kb.pickle'%cl,'rb') as f:
        fwd_data=pickle.load(f)
    with open('modality_data/norm_%s_rev_1kb.pickle'%cl,'rb') as f:
        rev_data=pickle.load(f)
    for chrom in range(1,23):
        input_data[chrom]= alpha*torch.tensor(np.vstack((fwd_data[chrom],rev_data[chrom])))
    return input_data

def prepare_external_data(cell_dict):
    external_tf_data={}
    tt_data,groseq_data,grocap_data,proseq_data,starr_data,netcage_data={}, {}, {}, {},{},{}

    for cl in cell_dict['external_tf']:
        external_tf_data[cl] = load_external_tfs(cl)
    for cl in cell_dict['tt']:
        tt_data[cl] = load_ttseq(cl)
    for cl in cell_dict['groseq']:
        groseq_data[cl]=load_groseq(cl)
    for cl in cell_dict['grocap']:
        grocap_data[cl]=load_grocap(cl)
    for cl in cell_dict['proseq']:
        proseq_data[cl]=load_proseq(cl)

    for cl in cell_dict['netcage']:
        netcage_data[cl]=load_netcage(cl)
    for cl in cell_dict['starr']:
        starr_data[cl]=load_external_starr(cl)

    return external_tf_data,tt_data,groseq_data,grocap_data,proseq_data,netcage_data,starr_data


def download_refseq_hg38():
    if not os.path.exists('refSeq'):
        os.mkdir('refSeq')
    gdown.download('https://drive.google.com/uc?id=1iqOKKNFwjl9hMZovxxhG-t_Y1MZw45R0',
                   output='refSeq/hg38.tar.gz')
    os.system('tar -xvf refSeq/hg38.tar.gz -C refSeq/')
    os.remove('refSeq/hg38.tar.gz')

def merge_atac(cells: list):
    atac_data={}
    for cl in cells:
        atac_data[cl]={}
        with open('%s_atac.pickle'%cl, 'rb') as f:
            atacseq = pickle.load(f)

        for chr in range(1, 23):
            tmp=load_dnase(atacseq[chr])
            atac_data[cl][chr]=csr_matrix(tmp.squeeze()).astype('float16')
    print(atac_data.keys())
    with open('data/train_scatac_merge.pickle','wb') as f:
        pickle.dump(atac_data,f)
