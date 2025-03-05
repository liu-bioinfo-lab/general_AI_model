import torch,math
from torch import nn, Tensor,einsum
from einops import rearrange,repeat
from einops.layers.torch import Rearrange
import torch.nn.functional as F

def exists(val):
    return val is not None

def default(val, d):
    return val if exists(val) else d

def exponential_linspace_int(start, end, num, divisible_by = 1):
    def _round(x):
        return int(round(x / divisible_by) * divisible_by)

    base = math.exp(math.log(end / start) / (num - 1))
    return [_round(start * base**i) for i in range(num)]

def log(t, eps = 1e-20):
    return torch.log(t.clamp(min = eps))

def get_positional_features_exponential(positions, features, seq_len, min_half_life = 3.):
    max_range = math.log(seq_len) / math.log(2.)
    half_life = 2 ** torch.linspace(min_half_life, max_range, features, device = positions.device)
    half_life = half_life[None, ...]
    positions = positions.abs()[..., None]
    return torch.exp(-math.log(2.) / half_life * positions)

def get_positional_features_central_mask(positions, features, seq_len):
    center_widths = 2 ** torch.arange(1, features + 1, device = positions.device).float()
    center_widths = center_widths - 1
    return (center_widths[None, ...] > positions.abs()[..., None]).float()

def gamma_pdf(x, concentration, rate):
    log_unnormalized_prob = torch.xlogy(concentration - 1., x) - rate * x
    log_normalization = (torch.lgamma(concentration) - concentration * torch.log(rate))
    return torch.exp(log_unnormalized_prob - log_normalization)

def get_positional_features_gamma(positions, features, seq_len, stddev = None, start_mean = None, eps = 1e-8):
    if not exists(stddev):
        stddev = seq_len / (2 * features)
    if not exists(start_mean):
        start_mean = seq_len / features
    mean = torch.linspace(start_mean, seq_len, features, device = positions.device)
    mean = mean[None, ...]
    concentration = (mean / stddev) ** 2
    rate = mean / stddev ** 2
    probabilities = gamma_pdf(positions.float().abs()[..., None], concentration, rate)
    probabilities = probabilities + eps
    outputs = probabilities / torch.amax(probabilities)
    return outputs

def get_positional_embed(seq_len, feature_size, device):
    distances = torch.arange(-seq_len + 1, seq_len, device = device)

    feature_functions = [
        get_positional_features_exponential,
        get_positional_features_central_mask,
        get_positional_features_gamma
    ]

    num_components = len(feature_functions) * 2

    if (feature_size % num_components) != 0:
        raise ValueError(f'feature size is not divisible by number of components ({num_components})')

    num_basis_per_class = feature_size // num_components

    embeddings = []
    for fn in feature_functions:
        embeddings.append(fn(distances, num_basis_per_class, seq_len))

    embeddings = torch.cat(embeddings, dim = -1)
    embeddings = torch.cat((embeddings, torch.sign(distances)[..., None] * embeddings), dim = -1)
    return embeddings

def relative_shift(x):
    to_pad = torch.zeros_like(x[..., :1])
    x = torch.cat((to_pad, x), dim = -1)
    _, h, t1, t2 = x.shape
    x = x.reshape(-1, h, t2, t1)
    x = x[:, :, 1:, :]
    x = x.reshape(-1, h, t1, t2 - 1)
    return x[..., :((t2 + 1) // 2)]


class Residual(nn.Module):
    def __init__(self, fn):
        super().__init__()
        self.fn = fn

    def forward(self, x, **kwargs):
        return self.fn(x, **kwargs) + x

class GELU(nn.Module):
    def forward(self, x):
        return torch.sigmoid(1.702 * x) * x


class Attention_REL(nn.Module):
    def __init__(
        self,dim,num_rel_pos_features,
        heads = 8,
        dim_key = 64,
        dim_value = 64,
        dropout = 0.,
        pos_dropout = 0.,
    ):
        super().__init__()
        self.scale = dim_key ** -0.5
        self.heads = heads

        self.to_q = nn.Linear(dim, dim_key * heads, bias = False)
        self.to_k = nn.Linear(dim, dim_key * heads, bias = False)
        self.to_v = nn.Linear(dim, dim_value * heads, bias = False)

        self.to_out = nn.Linear(dim_value * heads, dim)
        nn.init.zeros_(self.to_out.weight)
        nn.init.zeros_(self.to_out.bias)


        self.num_rel_pos_features = num_rel_pos_features

        self.to_rel_k = nn.Linear(num_rel_pos_features, dim_key * heads, bias = False)
        self.rel_content_bias = nn.Parameter(torch.randn(1, heads, 1, dim_key))
        self.rel_pos_bias = nn.Parameter(torch.randn(1, heads, 1, dim_key))

        self.pos_dropout = nn.Dropout(pos_dropout)
        self.attn_dropout = nn.Dropout(dropout)

    def forward(self, x,twod_logits=None):
        n, h, device = x.shape[-2], self.heads, x.device

        q = self.to_q(x)
        k = self.to_k(x)
        v = self.to_v(x)
        q, k, v = map(lambda t: rearrange(t, 'b n (h d) -> b h n d', h = h), (q, k, v))

        q = q * self.scale

        content_logits = einsum('b h i d, b h j d -> b h i j', q + self.rel_content_bias, k)

        positions = get_positional_embed(n, self.num_rel_pos_features, device)
        positions = self.pos_dropout(positions)
        rel_k = self.to_rel_k(positions)

        rel_k = rearrange(rel_k, 'n (h d) -> h n d', h = h)
        rel_logits = einsum('b h i d, h j d -> b h i j', q + self.rel_pos_bias, rel_k)
        rel_logits = relative_shift(rel_logits)


        if exists(twod_logits):
            logits = content_logits + rel_logits + twod_logits
        else:
            logits = content_logits + rel_logits
        attn = logits.softmax(dim = -1)
        attn = self.attn_dropout(attn)

        out = einsum('b h i j, b h j d -> b h i d', attn, v)
        out = rearrange(out, 'b h n d -> b n (h d)')
        return self.to_out(out)

class Enformer(nn.Module):
    def __init__(
        self,
        dim = 512,
        depth = 4,
        heads = 6,
        attn_dim_key = 64,
        dropout_rate = 0.,
        attn_dropout = 0,
        pos_dropout = 0.,
    ):
        super().__init__()
        self.dim = dim
        transformer = []
        for _ in range(depth):
            transformer.append(nn.Sequential(
                Residual(nn.Sequential(
                    nn.LayerNorm(dim),
                    Attention_REL(
                        dim,
                        heads = heads,
                        dim_key = attn_dim_key,
                        dim_value = dim // heads,
                        dropout = attn_dropout,
                        pos_dropout = pos_dropout,
                        num_rel_pos_features = dim // heads,
                    ),
                    nn.Dropout(dropout_rate)
                )),
                Residual(nn.Sequential(
                    nn.LayerNorm(dim),
                    nn.Linear(dim, dim * 2),
                    nn.Dropout(dropout_rate),
                    nn.ReLU(),
                    nn.Linear(dim * 2, dim),
                    nn.Dropout(dropout_rate)
                ))
            ))

        self.transformer = nn.Sequential(
            *transformer
        )
    def forward(self,x):
        x = self.transformer(x)
        return x

class Convblock(nn.Module):
    def __init__(self,in_channel,kernel_size,dilate_size,dropout=0,group_num=16):
        super().__init__()
        self.conv=nn.Sequential(
            nn.Conv2d(
                in_channel, in_channel,
                kernel_size, padding=self.pad(kernel_size,1)),
            nn.GroupNorm(group_num, in_channel),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Conv2d(
                in_channel, in_channel,
                kernel_size, padding=self.pad(kernel_size, dilate_size),
                dilation=dilate_size),
        )
    def pad(self,kernelsize, dialte_size):
        return (kernelsize - 1) * dialte_size // 2
    def symmetric(self,x):
        return (x + x.permute(0,1,3,2)) / 2
    def forward(self,x):
        identity=x
        out=self.conv(x)
        x=out+identity
        x=self.symmetric(x)
        return F.relu(x)

class dilated_tower(nn.Module):
    def __init__(self,embed_dim,in_channel=64,kernel_size=7,dilate_rate=5,if_pool=False):
        super().__init__()
        dilate_convs=[]
        for i in range(dilate_rate+1):
            dilate_convs.append(
                Convblock(in_channel,kernel_size=kernel_size,dilate_size=2**i))

        if not if_pool:
            self.cnn = nn.Sequential(
                Rearrange('b l n d -> b d l n'),
                nn.Conv2d(embed_dim, in_channel, kernel_size=1),
                *dilate_convs,
                nn.Conv2d(in_channel, in_channel, kernel_size=1),
                Rearrange('b d l n -> b l n d'),
            )
        else:
            self.cnn = nn.Sequential(
                Rearrange('b l n d -> b d l n'),
                nn.Conv2d(embed_dim, in_channel, kernel_size=1),
                Convblock(in_channel, kernel_size=15, dilate_size=1),
                nn.AvgPool2d(5, stride=5),
                nn.ReLU(),
                nn.Conv2d(in_channel,in_channel, kernel_size=1),
                *dilate_convs,
                nn.Conv2d(in_channel, in_channel, kernel_size=1),
                Rearrange('b d l n -> b l n d'),
            )

    def forward(self,x,crop):
        x=self.cnn(x)
        x=x[:,crop:-crop,crop:-crop,:]
        return x

class atac_CNN(nn.Module):
    def __init__(self,):
        super(atac_CNN, self).__init__()
        self.seq_conv_block1=nn.Sequential(
            nn.Conv1d(4, 640, kernel_size=12),
            nn.ReLU(inplace=True),
            nn.Conv1d(640, 640, kernel_size=8),
            nn.ReLU(inplace=True),
            nn.MaxPool1d(kernel_size=4, stride=4),
            nn.BatchNorm1d(640,track_running_stats=False),
        )

        self.atac_block1=nn.Sequential(
            nn.Conv1d(1, 64, kernel_size=12),
            nn.ReLU(inplace=True),
            nn.Conv1d(64,64, kernel_size=8),
            nn.ReLU(inplace=True),
            nn.MaxPool1d(kernel_size=4, stride=4),
            nn.BatchNorm1d(64,track_running_stats=False),
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
            nn.BatchNorm1d(1024,track_running_stats=False),
        )

        self.atac_block2 = nn.Sequential(
            nn.Conv1d(640, 128, kernel_size=1),
            nn.Conv1d(128, 128, kernel_size=8),
            nn.ReLU(inplace=True),
            nn.Conv1d(128, 128, kernel_size=8),
            nn.ReLU(inplace=True),
            nn.MaxPool1d(kernel_size=4, stride=4),
            nn.BatchNorm1d(128,track_running_stats=False),
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
            nn.BatchNorm1d(1536,track_running_stats=False),
        )

        self.num_channels = 1536

    def forward(self, x):
        x_seq,x_atac=x[:,:4,:],x[:,4:,:]

        x_seq_1 = self.seq_conv_block1(x_seq)
        x_atac_1=self.atac_block1(x_atac)

        x_atac_1, (h_n,h_c)=self.atac_lstm1(x_atac_1)
        x_atac_1=self.atac_block1_project(x_atac_1)
        x_seq_1+=x_atac_1
        x_seq_1=F.dropout(x_seq_1,0.05,training=self.training)


        x_seq_2 = self.seq_conv_block2(x_seq_1)
        x_atac_2=self.atac_block2(x_atac_1)

        x_atac_2, (h_n,h_c) = self.atac_lstm2(x_atac_2)
        x_seq_2 += self.atac_block2_project(x_atac_2)
        x_seq_2 = F.dropout(x_seq_2, 0.05, training=self.training)

        out = self.seq_conv_block3(x_seq_2)
        return out


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


def init_zero_(layer):
    nn.init.constant_(layer.weight, 0.)
    if exists(layer.bias):
        nn.init.constant_(layer.bias, 0.)



class Symmetrize(nn.Module):
    def __init__(self, sym_type=0):
        super().__init__()
        self.sym_type=sym_type

    def forward(self,x):
        if self.sym_type:
            return (x + rearrange(x, 'b i j d -> b j i d')) * 0.5
        else:
            return (x + rearrange(x, 'b d i j -> b d j i')) * 0.5



class pairdilate_cop(nn.Module):
    def __init__(self,embed_dim,in_channel=80,dilate_rate=5,kernel_size=7):
        super().__init__()
        dilate_convs = []
        for i in range(dilate_rate + 1):
            dilate_convs.append(
                Convblock(in_channel, kernel_size=kernel_size, dilate_size=2 ** i))
        dilate_convs.append(Convblock(in_channel, kernel_size=5, dilate_size=1))

        self.Conv= nn.Sequential(
            Rearrange('b l n d -> b d l n'),
            nn.Conv2d(embed_dim, in_channel, kernel_size=1),
            Convblock(in_channel, kernel_size=9, dilate_size=1),
            Convblock(in_channel, kernel_size=9, dilate_size=1),
        )

        self.Down=nn.Sequential(
            nn.AvgPool2d(5, stride=5),
            Convblock(in_channel, kernel_size=7, dilate_size=1),
            Convblock(in_channel, kernel_size=5, dilate_size=1),
            nn.Conv2d(in_channel,in_channel, kernel_size=1),
        )
        self.DConv_layers=nn.Sequential(
            *dilate_convs,
        )

        self.Up=nn.Sequential(
            nn.ConvTranspose2d(in_channel,in_channel,kernel_size=5,stride=5),
            Symmetrize(sym_type=0),
            Convblock(in_channel,kernel_size=9,dilate_size=1),
            Convblock(in_channel, kernel_size=9, dilate_size=1),
            nn.Conv2d(in_channel, 2 * in_channel,kernel_size=1),
            nn.Dropout(0.),
            nn.ReLU(),
            nn.Conv2d(2*in_channel,in_channel,kernel_size=1),
        )

    def forward(self,x):
        x=self.Conv(x)
        x_down=self.Down(x)
        x_down=self.DConv_layers(x_down)
        x=x+self.Up(x_down)
        return x.permute(0,2,3,1), x_down.permute(0,2,3,1)


class merge_seq_cop_layer(nn.Module):
    def __init__(self,embed_dim,hidden_dim,heads=4):
        super().__init__()
        self.norm=nn.LayerNorm(embed_dim)
        self.attn1= Attention_REL(
                        embed_dim,
                        heads = heads,
                        dim_key = 32,
                        dim_value = embed_dim // heads,
                        dropout = 0,
                        pos_dropout = 0.,
                        num_rel_pos_features = embed_dim // heads,
                    )

        self.ffn=Residual(nn.Sequential(
                    nn.Dropout(0.),
                    nn.LayerNorm(embed_dim),
                    nn.Linear(embed_dim, embed_dim * 2),
                    nn.Dropout(0.),
                    nn.ReLU(),
                    nn.Linear(embed_dim * 2, embed_dim),
                    nn.Dropout(0.)
                )
            )
        self.project_attn=nn.Linear(hidden_dim, heads)
        init_zero_(self.project_attn)


    def forward(self,x,attn_logits):
        attn_logits=self.project_attn(attn_logits)
        attn_logits = rearrange(attn_logits, 'b l w h -> b h l w')
        x=self.norm(x)
        x=x+self.attn1(x,attn_logits)
        x=self.ffn(x)
        return x


class Tranmodel(nn.Module):
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
        input=rearrange(input,'b n c l -> (b n) c l')
        src = self.backbone(input)
        src=self.input_proj_new(src)
        src = src.permute(0, 2, 1)
        src = self.transformer(src)
        return src

class TrunkModel(nn.Module):
    def __init__(self,pretrain_model,embed_dim,bins):
        super().__init__()
        self.pretrain_model = pretrain_model
        self.bins=bins
        self.attention_pool = AttentionPool(embed_dim,nhead=4)
        self.project=nn.Sequential(
            Rearrange('b c n -> b c n'),
            nn.Conv1d(embed_dim, embed_dim, kernel_size=15, padding=7,groups=embed_dim),
            nn.InstanceNorm1d(embed_dim, affine=True),
            nn.Conv1d(embed_dim, embed_dim, kernel_size=1),
            nn.ReLU(inplace=True),
            nn.Dropout(0.),
            nn.Conv1d(embed_dim, embed_dim, kernel_size=9, padding=4),
            nn.InstanceNorm1d(embed_dim, affine=True),
            nn.ReLU(inplace=True),
            nn.Dropout(0.),
        )
        self.transformer1 = Enformer(dim=embed_dim, depth=2, heads=8)
        self.transformer2 = Enformer(dim=embed_dim, depth=2, heads=8)
        self.transformer3 = Enformer(dim=embed_dim, depth=2, heads=8)

    def forward(self, x):
        x=self.pretrain_model(x)
        x = self.attention_pool(x)
        x = rearrange(x,'(b n) c -> b c n', n=self.bins)
        x = self.project(x)
        x= rearrange(x,'b c n -> b n c')
        x = self.transformer1(x)
        x = self.transformer2(x)
        x = self.transformer3(x)
        return x



class cop_attn_head(nn.Module):
    def __init__(
        self,
        embed_dim,
        pairattn_cop,
        hidden_dim=512,
        in_dim=128,
        microc_crop=50,
        hic_crop=10,
        nhic_task=3,
        nmicroc_task=2,
    ):
        super().__init__()
        self.project = nn.Sequential(
            nn.Linear(embed_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
        )
        self.pairattn_cop = pairattn_cop
        self.prediction_head_hic = nn.Linear(in_dim,nhic_task)
        self.prediction_head_microc = nn.Linear(in_dim, nmicroc_task)

        self.prediction_head_intacthic=nn.Sequential(
            nn.Linear(in_dim,in_dim//2),
            nn.ReLU(),
            nn.Linear(in_dim // 2,2),
        )
        self.microc_crop=microc_crop
        self.hic_crop=hic_crop


    def output_head(self, x):
        bins=x.shape[1]
        x1 = torch.tile(x.unsqueeze(1), (1, bins, 1, 1))
        x2 = x1.permute(0, 2, 1, 3)
        mean_out = (x1 + x2) / 2
        dot_out = (x1 * x2)/math.sqrt(x.shape[-1])
        return mean_out + dot_out

    def forward(self,x):
        x_cop=self.project(x)
        x_cop = self.output_head(x_cop)
        x_micro_attn,x_hic_attn = self.pairattn_cop(x_cop)
        x_microc_out = self.prediction_head_microc(x_micro_attn[:,self.microc_crop:-self.microc_crop,self.microc_crop:-self.microc_crop,:])

        x_intacthic_out=self.prediction_head_intacthic(x_micro_attn[:,self.microc_crop:-self.microc_crop,self.microc_crop:-self.microc_crop,:])
        x_hic_out=self.prediction_head_hic(x_hic_attn[:,self.hic_crop:-self.hic_crop,self.hic_crop:-self.hic_crop,:])
        return x_microc_out,x_intacthic_out,x_micro_attn,x_hic_out

class PredictionHeads(nn.Module):
    def __init__(
        self,
        pretrain_model,
        embed_dim,
        hidden_dim=256,
        crop=50,
    ):
        super().__init__()
        self.crop=crop
        self.expert = pretrain_model

        dilate_cnn_cop=pairdilate_cop(embed_dim=hidden_dim,in_channel=96,dilate_rate=4, kernel_size=5)
        self.prediction_head_dilate_cop = cop_attn_head(embed_dim, dilate_cnn_cop, hidden_dim, in_dim=96,
                                                        microc_crop=crop, hic_crop=crop // 5)

        self.adaptor_merge_layer = merge_seq_cop_layer(embed_dim=embed_dim, hidden_dim=96)

        self.prediction_heads_rna = nn.Sequential(
            nn.Linear(embed_dim, 128),
            nn.ReLU(),
            nn.Linear(128,3),
            nn.Softplus()
        )

        self.prediction_heads_erna = nn.Sequential(
            nn.Linear(embed_dim, 128),
            nn.ReLU(),
            nn.Linear(128, 3),
            nn.Softplus()
        )

        self.prediction_heads_epi =nn.Sequential(
            nn.Linear(embed_dim, 256),
            nn.ReLU(),
            nn.Linear(256,247)
        )
        self.prediction_heads_rna_strand = nn.Sequential(
            nn.Linear(embed_dim, 128),
            nn.ReLU(),
            nn.Linear(128, 2),
            nn.Softplus()
        )

        self.external_pred_head_tfs=nn.Sequential(
            nn.Linear(embed_dim, 720),
            nn.ReLU(),
            nn.Linear(720,708),
        )

        self.external_pred_head_seqs = nn.Sequential(
            nn.Linear(embed_dim, 128),
            nn.ReLU(),
            nn.Linear(128, 11),
        )

        self.external_pred_head_netcage = nn.Sequential(
            nn.Linear(embed_dim, 128),
            nn.ReLU(),
            nn.Linear(128, 2),
        )
        self.external_pred_head_starr= nn.Sequential(
            nn.Linear(embed_dim, 32),
            nn.ReLU(),
            nn.Linear(32, 1),
        )


    def forward(
            self,
            x,
            task_list: list =[1,1,1,1,1,1,1],
            return_rep=False
    ):
        if len(task_list) != 7:
            raise ValueError("Task length error")

        x_rep=self.expert(x)
        x_microc,x_intacthic, x_microc_attn_logits,x_hic =self.prediction_head_dilate_cop(x_rep)
        if not task_list[4]: x_hic = None
        if not task_list[3]: x_microc=None
        if not task_list[5]: x_intacthic=None

        x_rep_update=self.adaptor_merge_layer(x_rep,x_microc_attn_logits)

        x_epi = self.prediction_heads_epi(x_rep_update[:, self.crop:-self.crop, :]) if task_list[0] else None
        x_rna = self.prediction_heads_rna(x_rep_update[:, self.crop:-self.crop, :]) if task_list[1] else None

        x_erna=self.prediction_heads_erna(x_rep_update[:, self.crop:-self.crop, :]) if task_list[2] else None
        x_rna_strand=self.prediction_heads_rna_strand(x_rep_update[:, self.crop:-self.crop, :]) if task_list[6] else None

        x_external_tfs=self.external_pred_head_tfs(x_rep_update[:, self.crop:-self.crop, :])
        x_external_seqs = self.external_pred_head_seqs(x_rep_update[:, self.crop:-self.crop, :])
        x_tt,x_groseq,x_grocap,x_proseq=x_external_seqs.split((2,2,4,3),dim=-1)
        x_netcage=self.external_pred_head_netcage(x_rep_update[:, self.crop:-self.crop, :])
        x_starr = self.external_pred_head_starr(x_rep_update[:, self.crop:-self.crop, :])

        outs=[x_epi,x_rna, x_erna,x_microc,x_hic,x_intacthic,x_rna_strand]
        external_outs=[x_external_tfs,x_tt,x_groseq,x_grocap,x_proseq,x_netcage,x_starr]

        if return_rep == True:
            return x_rep_update, x_microc_attn_logits, outs, external_outs
        return outs, external_outs





def build_backbone():
    model=atac_CNN()
    return model

def build_model(args):
    backbone = build_backbone()
    pretrain_model = Tranmodel(
            backbone=backbone,
            embed_dim=args.embed_dim,
    )

    trunk = TrunkModel(
        pretrain_model,
        embed_dim=args.embed_dim,
        bins=args.bins,
    )
    model = PredictionHeads(
        pretrain_model=trunk,
        embed_dim=args.embed_dim,
        crop=args.crop
    )

    return model
