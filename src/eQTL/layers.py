import math

import torch
import torch.nn as nn
import os,sys
import inspect
currentdir = os.path.dirname(os.path.abspath(inspect.getfile(inspect.currentframe())))
parentdir = os.path.dirname(currentdir)
sys.path.insert(0, parentdir)
from src.model import build_model,Enformer,pairdilate_cop,cop_attn_head,build_backbone,TrunkModel,Tranmodel
from einops import rearrange,repeat
from einops.layers.torch import Rearrange
from torch import einsum
import torch.nn.functional as F
import numpy as np


class AttentionPool(nn.Module):
    def __init__(self, dim,nhead=1):
        super().__init__()
        self.pool_fn = Rearrange('b (n p) d-> b n p d', n=1)
        self.pool_fn_attn=nn.Sequential(
            Rearrange('b l (h d) -> b h l d',h=nhead),
            Rearrange('b h (n l) d -> b h n l d', n=1)
        )

        self.to_attn_logits = nn.Parameter(torch.cat([torch.eye(dim) for _ in range(nhead)],1))
        self.project=nn.Linear(nhead*dim,dim)
    def forward(self, x):
        attn_logits = einsum('b n d, d e -> b n e', x, self.to_attn_logits)
        x = self.pool_fn(x)
        logits = self.pool_fn_attn(attn_logits)
        attn = logits.softmax(dim = -2)
        xattn= einsum('b h n l d, b n l d -> b h n l d',attn,x).sum(dim = -2).squeeze(2)
        xattn=rearrange(xattn,'b h d -> b (h d)')

        return self.project(xattn)



class CNNmodel(nn.Module):
    def __init__(self,fix,dp=0.005):
        super(CNNmodel, self).__init__()
        self.seq_conv_block1=nn.Sequential(
            nn.Conv1d(4, 640, kernel_size=12),
            nn.ReLU(inplace=True),
            nn.Conv1d(640, 640, kernel_size=8),
            nn.ReLU(inplace=True),
            nn.MaxPool1d(kernel_size=4, stride=4),
            nn.BatchNorm1d(640),
            # nn.GroupNorm(16,640)
        )

        self.atac_block1=nn.Sequential(
            nn.Conv1d(1, 64, kernel_size=12),
            nn.ReLU(inplace=True),
            nn.Conv1d(64,64, kernel_size=8),
            nn.ReLU(inplace=True),
            nn.MaxPool1d(kernel_size=4, stride=4),
            nn.BatchNorm1d(64),
            # nn.GroupNorm(8, 64),
            Rearrange('b c n -> b n c'),
        )

        self.atac_lstm1= nn.LSTM(input_size=64, hidden_size=32, num_layers=1,
                    batch_first=True,bidirectional=True)

        self.atac_block1_project=nn.Sequential(
            Rearrange('b n c -> b c n'),
            nn.Conv1d(64, 640, kernel_size=1)
        )

        self.seq_conv_block2 = nn.Sequential(
            nn.Conv1d(640, 1024, kernel_size=8),
            nn.ReLU(inplace=True),
            nn.Conv1d(1024, 1024, kernel_size=8),
            nn.ReLU(inplace=True),
            nn.MaxPool1d(kernel_size=4, stride=4),
            nn.BatchNorm1d(1024),
        )

        self.atac_block2 = nn.Sequential(
            nn.Conv1d(640, 128, kernel_size=1),
            nn.Conv1d(128, 128, kernel_size=8),
            nn.ReLU(inplace=True),
            nn.Conv1d(128, 128, kernel_size=8),
            nn.ReLU(inplace=True),
            nn.MaxPool1d(kernel_size=4, stride=4),
            nn.BatchNorm1d(128),
            Rearrange('b c n -> b n c'),
        )
        self.atac_lstm2 = nn.LSTM(input_size=128, hidden_size=64, num_layers=1,
                                  batch_first=True, bidirectional=True)

        self.atac_block2_project = nn.Sequential(
            Rearrange('b n c -> b c n'),
            nn.Conv1d(128, 1024, kernel_size=1)
        )


        self.seq_conv_block3 = nn.Sequential(
            nn.Conv1d(1024, 1536, kernel_size=4),
            nn.ReLU(inplace=True),
            nn.Conv1d(1536, 1536, kernel_size=4),
            nn.ReLU(inplace=True),
            nn.BatchNorm1d(1536),
        )

        self.fix=fix
        self.dp=dp
        self.num_channels = 1536

    def forward(self, x):
        x_seq,x_atac=x[:,:4,:],x[:,4:,:]

        x_seq_1 = self.seq_conv_block1(x_seq)
        x_atac_1=self.atac_block1(x_atac)

        x_atac_1, (h_n,h_c)=self.atac_lstm1(x_atac_1)
        x_atac_1=self.atac_block1_project(x_atac_1)
        x_seq_1+=x_atac_1

        x_map_1=x_seq_1

        if not self.fix:
            x_seq_1=F.dropout(x_seq_1,self.dp,training=self.training)


        x_seq_2 = self.seq_conv_block2(x_seq_1)
        x_atac_2=self.atac_block2(x_atac_1)

        x_atac_2, (h_n,h_c) = self.atac_lstm2(x_atac_2)
        x_seq_2 += self.atac_block2_project(x_atac_2)


        x_map_2=x_seq_2
        if not self.fix:
            x_seq_2 = F.dropout(x_seq_2, self.dp, training=self.training)
        out = self.seq_conv_block3(x_seq_2)

        return x_map_1,x_map_2,out



class TranmodelLocal(nn.Module):
    def __init__(self, backbone,embed_dim):
        super().__init__()
        self.backbone = backbone
        self.input_proj_new = nn.Conv1d(backbone.num_channels, embed_dim, kernel_size=1)
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=embed_dim,
            nhead=4,
            dim_feedforward=2 * embed_dim,
            batch_first=True,
            dropout=0
        )
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=1)
    def forward(self, input):
        # input=rearrange(input,'b n c l -> (b n) c l')
        x_map_1,x_map_2,src = self.backbone(input)
        src=self.input_proj_new(src)
        src = src.permute(0, 2, 1)
        src = self.transformer(src)
        return x_map_1,x_map_2,src

class ClassifyModel(nn.Module):
    def __init__(self, pretrain_model, embed_dim,
                 mid_dim=96,var_model='enf'):
        super().__init__()
        self.pretrain_model = pretrain_model
        self.attention_pool = AttentionPool(embed_dim, nhead=4)

        self.mlp_1 = nn.Conv1d(89, 1, kernel_size=1)
        self.qtl_local_conv = nn.Conv1d(89, 1, kernel_size=1)

        self.qtl_local_linear = nn.Sequential(
            nn.Linear(embed_dim, 2 * mid_dim),
            nn.BatchNorm1d(2*mid_dim),
            nn.ReLU(),
            nn.Dropout(0.05),
            nn.Linear(2 * mid_dim, mid_dim)
        )

        self.tss_conv_1=nn.Conv1d(3, 1, kernel_size=1)
        self.tss_linear = nn.Linear(embed_dim, mid_dim)

        self.project_1 = nn.Sequential(
            nn.Conv1d(640, 320, kernel_size=1),
            nn.BatchNorm1d(320),
            nn.ReLU(),
            nn.Dropout(0.01),
            nn.Conv1d(320, 128, kernel_size=1),
        )

        self.project_2 = nn.Sequential(
            nn.Conv1d(1024, 648, kernel_size=1),
            nn.BatchNorm1d(648),
            nn.ReLU(),
            nn.Dropout(0.01),
            nn.Conv1d(648, 128, kernel_size=1),
        )

        self.embed_linear = nn.Sequential(
            nn.Linear(embed_dim+256, 256),
            # nn.BatchNorm1d(256),
            nn.ReLU(),
            nn.Dropout(0.01),
            nn.Linear(256, mid_dim)
        )
        self.max_dist = 10
        self.distance_embed = nn.Parameter(torch.zeros((self.max_dist + 1, mid_dim)))
        self.gene_type_embed = nn.Parameter(torch.zeros(3, mid_dim))
        self.inside_gene_embed = nn.Parameter(torch.zeros(2, mid_dim))

        self.linear_tss_qtl = nn.Sequential(
            nn.Linear(mid_dim *2, mid_dim),
            # nn.BatchNorm1d(mid_dim),
            nn.ReLU(),
            nn.Linear(mid_dim, mid_dim)
        )

        self.gate_linear = nn.Sequential(
            nn.Linear(mid_dim * 2+1, mid_dim),
            nn.ReLU(),
            nn.Dropout(0.01),
            nn.Linear(mid_dim, mid_dim)
        )
        self.rep_linear = nn.Sequential(
            nn.Linear(mid_dim * 2+1, 128),
            nn.ReLU(),
            nn.Dropout(0.01),
            nn.Linear(128, mid_dim)
        )
        if var_model == 'enf':
            self.linear_variant_score = nn.Sequential(
                # nn.Linear(5314, 256),
                nn.Linear(7612, 256),
                nn.BatchNorm1d(256),
                nn.ReLU(),
                nn.Dropout(0.05),
                nn.Linear(256, mid_dim // 2),
                nn.ReLU(),
                nn.Linear(mid_dim // 2, mid_dim)
            )
        elif var_model == 'borzoi':
            self.linear_variant_score = nn.Sequential(
                nn.Linear(7612, 512),
                nn.BatchNorm1d(512),
                nn.ReLU(),
                nn.Dropout(0.05),
                nn.Linear(512, mid_dim // 2),
                nn.ReLU(),
                nn.Linear(mid_dim // 2, mid_dim)
            )
        else:
            raise ValueError('wrong variant score source')

        self.dist_linear=nn.Sequential(
            nn.Linear(1,1),
            nn.ReLU()
        )

        self.var_mapping=nn.Linear(mid_dim*2,mid_dim)
        self.classifier = nn.Linear(mid_dim, 1)
        self.local_classifier = nn.Linear(mid_dim, 1)

    def forward(self, x1, x2, x_tss=None, x_2d_rep=None,x_seq_rep=None,
                dist=None, gene_type=None, inside_gene=None,x_var=None):
        x1_fea1, x1_fea2, x1 = self.pretrain_model(x1)
        x1_fea1 = self.project_1(x1_fea1)
        x1_fea2 = self.project_2(x1_fea2)

        x2_fea1, x2_fea2, x2 = self.pretrain_model(x2)
        x2_fea1 = self.project_1(x2_fea1)
        x2_fea2 = self.project_2(x2_fea2)

        x_diff_1 = (x1_fea1 - x2_fea1).sum(-1)
        x_diff_2 = (x1_fea2 - x2_fea2).sum(-1)
        x_diff = self.mlp_1(x1 - x2).squeeze(1)
        x_diff = self.embed_linear(torch.cat((x_diff_1, x_diff_2, x_diff), dim=1))

        if x_var is not None:
            x_var=self.linear_variant_score(x_var)
            x_diff=self.var_mapping(torch.cat([x_diff,x_var],dim=1))

        if x_2d_rep is None:
            x = self.local_classifier(x_diff)
            return x

        x_tss = rearrange(x_tss, 'b n c l -> (b n) c l')
        _, _, x_tss = self.pretrain_model(x_tss)
        x_tss=self.attention_pool(x_tss)

        x_tss=rearrange(x_tss,'(b n) l -> b n l',n=3)
        x_tss = self.tss_linear(self.tss_conv_1(x_tss).squeeze(1))

        x_qtl_local = self.qtl_local_linear(self.qtl_local_conv(x2).squeeze(1))

        if gene_type is not None:
            x_gt = self.gene_type_embed[gene_type]
            x_tss += x_gt
        if inside_gene is not None:
            x_qtl_local += self.inside_gene_embed[inside_gene]
        x_prompt=self.linear_tss_qtl(torch.cat((x_tss, x_qtl_local), dim=1))


        x_dist=self.dist_linear(dist)
        x_2d_rep = torch.cat((x_2d_rep, x_prompt,x_dist), dim=1)
        alpha = torch.sigmoid(self.gate_linear(x_2d_rep))

        x_2d_rep = self.rep_linear(x_2d_rep)
        x = x_2d_rep + alpha * x_diff
        x = self.classifier(x)
        return x


class PredictionHeads(nn.Module):
    def __init__(
        self,
        pretrain_model,
        embed_dim,
        hidden_dim=256,
        crop=50,
        prompt=False
    ):
        super().__init__()
        self.crop=crop
        self.expert = pretrain_model

        dilate_cnn_cop=pairdilate_cop(embed_dim=hidden_dim,in_channel=96,dilate_rate=4,
                                          kernel_size=5,prompt=prompt)
        self.prediction_head_dilate_cop = cop_attn_head(embed_dim, dilate_cnn_cop, hidden_dim, in_dim=96,
                                                        microc_crop=crop,
                                                        hic_crop=crop // 5, prompt=prompt)

    def forward(self,x):
        x_rep=self.expert(x,use_prompt=False)
        _,_, x_microc_attn_logits,_ =self.prediction_head_dilate_cop(x_rep,use_prompt=False)
        return x_microc_attn_logits



def build_local_model(args):
    backbone = CNNmodel(fix=args.fix)
    # backbone = seq_CNN()
    pretrain_model = TranmodelLocal(
            backbone=backbone,
            embed_dim=960,
    )
    model=ClassifyModel(
        pretrain_model=pretrain_model,
        embed_dim=960,
        var_model=args.var_model
    )

    pretrained_path = '../curriculum/models/ddp_rna_strand_2_full.pt'
    model_dict = model.state_dict()
    pretrained_dict = torch.load(pretrained_path, map_location='cpu')
    pretrained_dict = {k.replace('expert.', ''): v for k, v in pretrained_dict.items() if
                       k.replace('expert.', '') in model_dict}
    model_dict.update(pretrained_dict)
    model.load_state_dict(model_dict)
    return model

def build_global_model():
    backbone = build_backbone()
    pretrain_model = Tranmodel(
            backbone=backbone,
            embed_dim=960,
    )

    trunk = TrunkModel(
        pretrain_model,
        embed_dim=960,
        bins=600,
    )
    model = PredictionHeads(
        pretrain_model=trunk,
        embed_dim=960,
    )

    return model
