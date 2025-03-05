#### scripts for training MPRAnn and ResNet on lentimpra within-cell type prediction

from torch.utils.data import DataLoader, Dataset,Subset
import os,torch,random
import numpy as np
from scipy.sparse import load_npz
import torch.optim as optim
import sys,inspect,datetime,argparse
currentdir = os.path.dirname(os.path.abspath(inspect.getfile(inspect.currentframe())))
parentdir = os.path.dirname(currentdir)
sys.path.insert(0, parentdir)
import pandas as pd
from scipy.stats import pearsonr,spearmanr
import torch.nn as nn
import pickle,time
from torch.optim.lr_scheduler import LambdaLR



def parser_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--bins', type=int, default=600)
    parser.add_argument('--crop', type=int, default=50)
    parser.add_argument('--embed_dim', default=960, type=int)
    parser.add_argument('--epochs', default=20, type=int)
    parser.add_argument('--lr', default=1e-3, type=float)
    parser.add_argument('--atac_block', default=True, action='store_false')
    parser.add_argument('-l', '--logits_type', type=str, default='dilate')
    parser.add_argument('--prompt', default=False, action='store_true')
    parser.add_argument('--external', default=True, action='store_false')
    parser.add_argument('-s','--seed',type=int, default=1024)
    parser.add_argument('-q','--seq_only', default=False, action='store_true')
    parser.add_argument('-c','--cell', default='K562',type=str)
    parser.add_argument('-m','--model',choices=['resnet','mpra'],default='resnet')
    parser.add_argument('-t', '--twod', default=True, action='store_false')
    parser.add_argument('-d', '--indel', default=False, action='store_true')
    parser.add_argument('-f', '--fix', default=False, action='store_true')
    args = parser.parse_args()
    return args
def get_args():
    args = parser_args()
    return args

class MPRADataset(Dataset):
    def __init__(self,cells):
        datas=[]
        for idx,cell in enumerate(cells):
            data=pd.read_csv('data/'+cell+'.csv')
            data=data.iloc[:,2:]
            data['chr.hg38']=data['chr.hg38'].str.replace('chr', '')
            # .replace('X', '23')
            data['chr.hg38']= pd.to_numeric(data['chr.hg38'], errors='coerce')
            data = data.dropna(subset=['chr.hg38'])
            data=data.to_numpy().astype('float')
            cell_idx=np.full((data.shape[0],1),idx)
            data=np.concatenate((cell_idx,data),axis=1)
            datas.append(data)
        self.data=np.concatenate(datas)
        print(self.data)
        self.num=self.data.shape[0]

    def __getitem__(self, index):
        return self.data[index]

    def __len__(self):
        return self.num


def build_dataloaders(train_cells):
    cell_dataset = MPRADataset(train_cells)
    np.random.seed(24)
    index = np.arange(len(cell_dataset))
    np.random.shuffle(index)

    train_i,valid_i, test_i = index[:int(0.7 * len(cell_dataset))],index[int(0.7 * len(cell_dataset)):int(0.8 * len(cell_dataset))],\
                              index[int(0.8 * len(cell_dataset)):]
    train_dataset = Subset(cell_dataset, indices=train_i)
    valid_dataset = Subset(cell_dataset, indices=valid_i)
    test_dataset=Subset(cell_dataset, indices=test_i)
    print(len(train_dataset), len(valid_dataset),len(test_dataset))
    train_loader= DataLoader(
            train_dataset,
            batch_size=32,
            shuffle=True
        )

    valid_loader= DataLoader(
        valid_dataset,
        batch_size=32,
        shuffle=False,
    )

    test_loader = DataLoader(
        test_dataset,
        batch_size=32,
        shuffle=False,
    )
    return train_loader,valid_loader,test_loader


class ExponentialActivation(torch.nn.Module):
    def forward(self, input):
        return torch.exp(input)

class Convblock(nn.Module):
    def __init__(self, in_channel, kernel_size, dilate_size):
        super().__init__()
        self.conv = nn.Sequential(
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Conv1d(in_channel, in_channel,
                kernel_size, padding=self.pad(kernel_size, dilate_size),
                    dilation=dilate_size),
            nn.BatchNorm1d(in_channel)
        )
        self.ds=dilate_size
    def pad(self,kernelsize, dialte_size):
        return (kernelsize - 1) * dialte_size // 2
    def forward(self,x):
        return self.conv(x)
class Residual(nn.Module):
    def __init__(self, fn):
        super().__init__()
        self.fn = fn
    def forward(self, x, **kwargs):
        return self.fn(x, **kwargs) + x
class ResNet(nn.Module):
    def __init__(self,seq_only=False):
        super(ResNet, self).__init__()
        self.seq_only=seq_only
        if seq_only:
            self.conv_1=nn.Sequential(
                nn.Conv1d(4,196,kernel_size=15,padding=7),
                nn.BatchNorm1d(196),
                ExponentialActivation(),
                nn.Dropout(0.2),
                nn.Conv1d(196,196,kernel_size=3,padding=1),
                nn.BatchNorm1d(196)
            )
        else:
            self.conv_1 = nn.Sequential(
                nn.Conv1d(5, 196, kernel_size=15, padding=7),
                nn.BatchNorm1d(196),
                nn.ReLU(),
                nn.Dropout(0.2),
                nn.Conv1d(196, 196, kernel_size=3, padding=1),
                nn.BatchNorm1d(196)
            )
        dilate_convs = []
        for i in range(5):
            dilate_convs.append(
                Convblock(196,kernel_size=3,dilate_size=2**i)
            )
        self.res_conv=Residual(
            nn.Sequential(*dilate_convs)
        )

        self.conv_2=nn.Sequential(
            nn.MaxPool1d(10),
            nn.Dropout(0.2),
            nn.Conv1d(196,256,kernel_size=7,padding=3),
            nn.BatchNorm1d(256),
            nn.ReLU(),
            nn.MaxPool1d(5,padding=2),
            nn.Dropout(0.2)
        )
        self.Linear=nn.Sequential(
            nn.Linear(5*256,512),
            nn.BatchNorm1d(512),
            nn.ReLU(),
            nn.Dropout(0.5),
            nn.Linear(512, 256),
            nn.BatchNorm1d(256),
            nn.ReLU(),
            nn.Dropout(0.5),
            nn.Linear(256,1)
        )
    def rc(self, x):
        if not self.seq_only:
            x1 = x.clone()
            dna, atac = x1[:, :4, :], x1[:, 4:, :]
        else:
            dna=x.clone()
        reversed_dna = dna.flip(dims=[2])
        complement_mapping = torch.tensor([3, 2, 1, 0])
        reverse_complement_dna = reversed_dna[:, complement_mapping, :]
        if not self.seq_only:
            reverse_atac = atac.flip(dims=[2])
            return torch.cat([reverse_complement_dna, reverse_atac], dim=1)
        return reverse_complement_dna
    def forward(self,x):
        rx=self.rc(x)

        x=self.conv_1(x)
        x=self.res_conv(x)
        x=self.conv_2(x)
        x = torch.flatten(x, start_dim=-2, end_dim=-1)
        x=self.Linear(x)

        rx = self.conv_1(rx)
        rx = self.res_conv(rx)
        rx = self.conv_2(rx)
        rx = torch.flatten(rx, start_dim=-2, end_dim=-1)
        rx = self.Linear(rx)
        return torch.cat([x,rx],dim=1)

class MPRANN(nn.Module):
    def __init__(self,seq_only=False):
        super(MPRANN, self).__init__()
        self.seq_only=seq_only
        if seq_only:
            self.conv_1=nn.Sequential(
                nn.Conv1d(4,250,kernel_size=7),
                nn.ReLU(),
                nn.BatchNorm1d(250),
            )
        else:
            self.conv_1 = nn.Sequential(
                nn.Conv1d(5, 250, kernel_size=7),
                nn.ReLU(),
                nn.BatchNorm1d(250),
            )

        self.conv_2=nn.Sequential(
            nn.Conv1d(250, 250, kernel_size=8),
            nn.Softmax(dim=1),
            nn.BatchNorm1d(250),
            nn.MaxPool1d(2),
            nn.Dropout(0.1),
            nn.Conv1d(250,250,kernel_size=3),
            nn.Softmax(dim=1),
            nn.BatchNorm1d(250),
            nn.Conv1d(250, 100, kernel_size=2),
            nn.Softmax(dim=1),
            nn.BatchNorm1d(100),
            nn.Dropout(0.1),
        )
        self.Linear=nn.Sequential(
            nn.Linear(10500,300),
            nn.Sigmoid(),
            nn.Dropout(0.3),
            nn.Linear(300, 200),
            nn.Sigmoid(),
            nn.Linear(200,1)
        )
    def rc(self, x):
        if not self.seq_only:
            x1 = x.clone()
            dna, atac = x1[:, :4, :], x1[:, 4:, :]
        else:
            dna=x.clone()
        reversed_dna = dna.flip(dims=[2])
        complement_mapping = torch.tensor([3, 2, 1, 0])
        reverse_complement_dna = reversed_dna[:, complement_mapping, :]
        if not self.seq_only:
            reverse_atac = atac.flip(dims=[2])
            return torch.cat([reverse_complement_dna, reverse_atac], dim=1)
        return reverse_complement_dna
    def forward(self,x):
        rx=self.rc(x)
        x=self.conv_1(x)
        x=self.conv_2(x)
        x = torch.flatten(x, start_dim=-2, end_dim=-1)
        # print(x.shape)
        x=self.Linear(x)

        rx = self.conv_1(rx)
        rx = self.conv_2(rx)
        rx = torch.flatten(rx, start_dim=-2, end_dim=-1)
        rx = self.Linear(rx)
        return torch.cat([x,rx],dim=1)

def init_weights(m):
    if isinstance(m, nn.Linear) or isinstance(m, nn.Conv1d):
        nn.init.normal_(m.weight, mean=0.0, std=0.005)
        if m.bias is not None:
            nn.init.constant_(m.bias, 0)

def lr_lambda(epoch):
    if epoch < 10:
        return 1
    else:
        return 0.2
def main():

    args = get_args()
    torch.manual_seed(args.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(args.seed)
    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
    if args.model=='resnet':
        model = ResNet(seq_only=args.seq_only)
        if args.seq_only:
            model.apply(init_weights)
    else:
        args.epochs = 40
        model = MPRANN(seq_only=args.seq_only)

    model.to(device)
    model.train()
    criterion = nn.MSELoss()
    for n, p in model.named_parameters():
        print(n, p.requires_grad)


    optimizer = optim.Adam(filter(lambda p: p.requires_grad, model.parameters()), lr=args.lr)
    scheduler = LambdaLR(optimizer, lr_lambda)

    train_cells = [args.cell]
    train_loader, valid_loader,test_loader = build_dataloaders(train_cells=train_cells)

    reference_sequence = {}

    if not args.seq_only:
        atac_seq_data = {cell: {} for cell in train_cells}
        tmp_atac_data = {}
        for cell in train_cells:
            with open('../data/' + cell + '_atac.pickle', 'rb') as f:
                tmp_atac_data[cell] = pickle.load(f)

    for chromosome in [i for i in range(1, 23)]:
        # ref_path = '/nfs/turbo/umms-drjieliu/usr/zzh/KGbert/3D/data/ref_genome/'
        ref_path = '../refSeq/hg38/'
        ref_file = os.path.join(ref_path, 'chr%s.npz' % chromosome)
        if chromosome == 'X':
            reference_sequence[23] = load_npz(ref_file).toarray()
        else:
            reference_sequence[chromosome] = load_npz(ref_file).toarray()
        if not args.seq_only:
            for cell in train_cells:
                atac_seq_data[cell][chromosome] = tmp_atac_data[cell][chromosome].toarray()



    def model_inputs_outputs(input_data,cell=None):
        data_input,targets = [], []
        for i in range(input_data.shape[0]):
            cl_idx, label_f, label_r, chrom, loc_s, loc_e = [tmp_x.item() for tmp_x in input_data[i]]
            label = torch.tensor([label_f,label_r]).unsqueeze(0).float()


            chrom=int(chrom)

            mid_point=(int(loc_s)+int(loc_e))//2
            input_s, input_e = mid_point - 115, mid_point + 115

            seq=reference_sequence[chrom][:,input_s:input_e]
            if not args.seq_only:
                if cell is None:
                    cell=train_cells[int(cl_idx)]
                atac = atac_seq_data[cell][chrom][:, input_s:input_e]
                seq = np.concatenate([seq, atac])

            input_array=torch.tensor(seq).unsqueeze(0).float()
            data_input.append(input_array)
            targets.append(label)

        data_input = torch.cat(data_input, dim=0).to(device)
        targets = torch.cat(targets, dim=0).to(device)
        return data_input, targets

    best_score = 0
    for epoch in range(args.epochs):
        train_loss = 0
        model.train()

        for step, training_data in enumerate(train_loader):

            tts = time.time()

            data_input,label=model_inputs_outputs(training_data)
            pred_act=model(data_input)
            loss = criterion(pred_act, label)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            train_loss += loss.detach()

        if args.model=='resnet':
            scheduler.step()

        model.eval()
        print("validation step")

        preds, targets = [], []
        for step, valid_data in enumerate(valid_loader):
            data_input, label = model_inputs_outputs(valid_data)
            with torch.no_grad():
                pred_act = model(data_input)
            preds.append(pred_act.cpu().data.detach().numpy())
            targets.append(label.cpu().data.detach().numpy())

        predict_vals = np.concatenate(preds, axis=0).squeeze().flatten()
        target_vals = np.concatenate(targets, axis=0).squeeze().flatten()

        pr_score = pearsonr(predict_vals, target_vals)[0]
        spm_score = spearmanr(predict_vals, target_vals)[0]
        print('----', pr_score,spm_score)
        cur_score = pr_score
        with open('log_mpra_base_%s.txt'%train_cells[0], 'a') as f:
            f.write('epoch: %s, pr: %s, spm:%s, %s, %s \n' % (epoch, pr_score, spm_score,args.model,args.seq_only))
        if cur_score > best_score:
            with open('log_mpra_base_%s.txt'%train_cells[0], 'a') as f:
                f.write('save model \n')
            best_score = cur_score
            if args.seq_only:
                torch.save(model.state_dict(), 'models/mpra_base_%s_seq_%s.pt'%(args.model,train_cells[0]))
            else:
                torch.save(model.state_dict(), 'models/mpra_base_%s_%s.pt'%(args.model,train_cells[0]))

        preds, targets = [], []
        for step, testing_data in enumerate(test_loader):
            data_input, label = model_inputs_outputs(testing_data)
            with torch.no_grad():
                pred_act = model(data_input)
            preds.append(pred_act.cpu().data.detach().numpy())
            targets.append(label.cpu().data.detach().numpy())
        predict_vals = np.concatenate(preds, axis=0).squeeze().flatten()
        target_vals = np.concatenate(targets, axis=0).squeeze().flatten()

        pr_score=pearsonr(predict_vals,target_vals)[0]
        spm_score = spearmanr(predict_vals, target_vals)[0]
        print('***',pr_score)
        with open('log_mpra_base_%s.txt'%train_cells[0], 'a') as f:
            f.write('cell: %s, pr: %s, spm:%s, %s, %s \n' % (train_cells[0],pr_score,spm_score,args.model,args.seq_only))




if __name__ == "__main__":
    main()

