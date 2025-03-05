import os, torch, random
import pandas as pd
import numpy as np
from scipy.sparse import load_npz
import argparse
from ..model import build_model
from ..util import load_dnase,load_ref_genome
import pickle, time
import gdown


def parser_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--bins', type=int, default=600)
    parser.add_argument('--crop', type=int, default=50)
    parser.add_argument('--embed_dim', default=960, type=int)
    parser.add_argument('-d','--download_model', default=False, action='store_ture')
    args = parser.parse_args()
    return args


def get_args():
    args = parser_args()
    return args


def download_models():
    if not os.path.exists('models'):
        os.mkdir('models')
    gdown.download('https://drive.google.com/uc?id=1SlPT9jHpaj5JgPZ_3uD0z6sXM28faCwT',
                   output='models/general_woK562.pt')

def main():
    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
    args = get_args()

    if args.download_model:
        download_models()
    checkpoint_path ='models/general_woK562.pt'
    model = build_model(args)
    model.load_state_dict(torch.load(checkpoint_path, map_location='cpu'),strict=False)
    model.to(device)
    model.eval()


    chr_lens = [248956422, 242193529, 198295559, 190214555, 181538259, 170805979, 159345973, 145138636, 138394717,
                133797422, 135086622, 133275309, 114364328, 107043718, 101991189, 90338345, 83257441, 80373285,
                58617616, 64444167, 46709983, 50818468, 156040895]
    chrs = ['chr' + str(i) for i in range(1, 23)] + ['chrX']

    df = pd.read_csv('ENCODE-E2G_Predictions.tsv', sep='\t')
    df['TSS_bin'] = df['TSS_from_universe'] // 1000
    df['enh_bin'] = ((df['chromStart'] + df['chromEnd']) / 2) // 1000
    df['distance'] = np.abs(df['TSS_from_universe'] - (df['chromStart'] + df['chromEnd']) / 2)
    df['distance'] = np.abs(df['TSS_bin'] - df['enh_bin'])
    df = df[df['distance'] < 500]
    df = df.reset_index(drop=True)
    df['mid'] = (df['TSS_bin'] + df['enh_bin']) // 2
    df['seqStart_bin'] = df['mid'] - 249
    df['seqEnd_bin'] = df['mid'] + 251
    category_multiplier = dict(zip(chrs, chr_lens))
    df['chrom_len'] = df['chrom'].map(category_multiplier) // 1000
    df = df[df['seqStart_bin'] - 50 > 0]
    df = df[df['seqEnd_bin'] + 50 < df['chrom_len']]
    df = df.reset_index(drop=True)

    print(df)
    with open('../data/K562_atac.pickle', 'rb') as f:
        atacseq = pickle.load(f)

    ref_data={}
    atac_data={}
    for chrom in ([i for i in range(1,23)]+['X']):
        ref_data['chr'+(str(chrom))]=load_ref_genome(chrom)
        atac_data['chr'+(str(chrom))]=load_dnase(atacseq[chrom])

    all_modals = ['epi', 'rna', 'bru', 'microc', 'hic', 'intacthic','rna_strand' ,'external_tf', 'tt', 'groseq', 'grocap', 'proseq',
                  'netcage', 'starr']
    pred_modals = ['epi', 'rna', 'bru', 'microc', 'intacthic', 'tt', 'groseq', 'grocap', 'netcage']
    outputs_e = {}
    outputs_t= {}

    for i in range(len(df)):
        chrom = df.iloc[i]['chrom']
        inputstart_bin=df.loc[i, 'seqStart_bin'].astype('int64')-50
        inputend_bin = df.loc[i, 'seqEnd_bin'].astype('int64') + 50

        input_seq=ref_data[chrom][inputstart_bin:inputend_bin]
        input_atac = atac_data[chrom][inputstart_bin:inputend_bin]
        inputs=torch.cat((input_seq,input_atac),dim=1).unsqueeze(0).float().to(device)

        with torch.no_grad():
            reps,twod_reps,output, external_output =model(inputs,return_rep=True)
        reps=reps.detach().cpu().numpy()
        reps=reps[:,args.crop:-args.crop,:]
        twod_reps = twod_reps.detach().cpu().numpy()
        twod_reps = twod_reps[:, args.crop:-args.crop,args.crop:-args.crop, :]

        tmps=[o.detach().cpu().numpy() for o in (output+external_output)]
        predictions = dict(zip(all_modals, tmps))

        enh_bin= df.loc[i, 'enh_bin'].astype('int64')- df.loc[i, 'seqStart_bin'].astype('int64')
        tss_bin = df.loc[i, 'TSS_bin'].astype('int64') - df.loc[i, 'seqStart_bin'].astype('int64')

        reps=reps[:,np.array([enh_bin,tss_bin]),:]
        twod_reps = twod_reps[:, enh_bin, tss_bin, :]
        if i==0:
            output_reps=reps
            output_twodreps=twod_reps
            for modx in pred_modals:
                if modx not in ['microc', 'intacthic']:
                    outputs_e[modx]=predictions[modx][:, enh_bin, :]
                    outputs_t[modx]=predictions[modx][:, tss_bin, :]
                else:
                    outputs_e[modx]=predictions[modx][:, tss_bin, enh_bin, :]
                    outputs_t[modx]=predictions[modx][:, tss_bin, enh_bin, :]
        else:
            output_reps=np.vstack((output_reps,reps))
            output_twodreps = np.vstack((output_twodreps,twod_reps))

            for modx in pred_modals:
                if modx not in ['microc', 'intacthic']:
                    outputs_e[modx]=np.vstack((outputs_e[modx],predictions[modx][:, enh_bin, :]))
                    outputs_t[modx]=np.vstack((outputs_t[modx],predictions[modx][:, tss_bin, :]))
                else:
                    outputs_e[modx] = np.vstack((outputs_e[modx], predictions[modx][:, tss_bin, enh_bin, :]))
                    outputs_t[modx] = np.vstack((outputs_t[modx], predictions[modx][:, tss_bin, enh_bin, :]))

        print(i,chrom,output_reps.shape,output_twodreps.shape, outputs_e['epi'].shape, outputs_e['microc'].shape)



    outs={'enh':outputs_e,'tss':outputs_t}
    with open('data/re2g_wok562_pred.pickle', 'wb') as f:
        pickle.dump(outs, f)
    np.save('data/re2g_1dreps.npy',output_reps)
    np.save('data/re2g_2dreps.npy', output_twodreps)

if __name__=='__main__':
    main()

