import torch 
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader, RandomSampler
import copy
import sklearn
from sklearn import metrics
from sklearn.metrics import roc_auc_score, precision_recall_curve, auc, precision_score, recall_score, accuracy_score, f1_score
from base_utils import get_new_cols, drop_useless_cols, load_data, check_filename, write_file
import pickle
import re

""" Create a pytorch dataset for multimodal modeling -- input: clin_data (clin x), expr_data (expr x), mut_data (mut x), labels (y), device """      
class MultiModalDataset(Dataset):
    def __init__(self, clin_data, expr_data, mut_data, labels, device):
        super(MultiModalDataset).__init__()
        self.clin_data = torch.tensor(clin_data.to_numpy(), dtype=torch.float32).to(device)
        self.expr_data = torch.tensor(expr_data.to_numpy(), dtype=torch.float32).to(device)
        self.mut_data = torch.tensor(mut_data.to_numpy(), dtype=torch.float32).to(device)
        self.labels = torch.tensor(labels.to_numpy(), dtype=torch.long).to(device)
    def __getitem__(self, idx):
        return self.clin_data[idx], self.expr_data[idx], self.mut_data[idx], self.labels[idx]
    def __len__(self):
        return len(self.labels)

""" Sets random seed for reproducibility across random, numpy, torch (CPU and GPU) """
def set_seed(seed, deterministic=False):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    if deterministic: # if True, forces PyTorch to use deterministic algorithms (slower, more reproducible)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False

def generate_seeds(n=10, seed=42):
    random.seed(seed)
    return random.sample(range(1, 10_000), n)

def cycle_seeds(seed_list, idx, set_now=True, deterministic=False):
    current_seed = seed_list[idx % len(seed_list)]
    if set_now: set_seed(current_seed, deterministic=deterministic)
    return current_seed

def calc_metrics(y_true, y_out, y_pred, plot_metrics=False):
    prc_prec, prc_rec, _ = precision_recall_curve(y_true, y_out)
    fpr, tpr, _ = roc_curve(y_true, y_out)
    tn, fp, fn, tp = confusion_matrix(y_true, y_pred, labels=[0, 1]).ravel()
    results = {'auc': roc_auc_score(y_true, y_out), 'auprc': auc(prc_rec, prc_prec), 'acc': accuracy_score(y_true, y_pred), 'prec': precision_score(y_true, y_pred, zero_division=0), 'rec': recall_score(y_true, y_pred), 'spec': recall_score(y_true, y_pred, pos_label=0), 'f1': f1_score(y_true, y_pred), 'tn': tn, 'fp': fp, 'fn': fn, 'tp': tp}
    if plot_metrics: results.update({'prc_prec': prc_prec.tolist(), 'prc_rec': prc_rec.tolist(), 'fpr': fpr.tolist(), 'tpr': tpr.tolist()})
    return results

def earlystop_checkpoint(model, tparams, v_val, best_val, no_improve, chkpt_fn):
    if (tparams.es['metric'] == 'auc' and v_val > best_val) or (tparams.es['metric'] == 'loss' and v_val < best_val):
        torch.save(model.state_dict(), f"results/weights/{chkpt_fn}.pth")
        return v_val, 0
    return best_val, no_improve + 1

class ModelParams:
    def __init__(self, align_lr_to_clr=True, **kwargs):
        if kwargs.get('_from_json', False): return  
        def _update(key, default):
            return {**default, **kwargs.get(key, {})} 
        self.clin = _update('clin', {'n': [128, 128, 64], 'dropout': [0, 0.2, 0.25], 'act': 'relu'})
        self.expr = _update('expr', {'n': [128, 128, 64], 'dropout': [0, 0.2, 0.5], 'act': 'relu'})
        self.mut = _update('mut', {'n': [128, 128, 64], 'dropout': [0, 0.2, 0.5], 'act': 'relu'})
        self.final = _update('final', {'n': [64, 32], 'dropout': [0.2, 0.3], 'act': 'relu'})
        self.att = _update('att', {'self': {'use': True, 'dropout': [0,0,0]}, 'cross': {'use': True, 'dropout': [0,0,0]}, 'num_heads': 2})
        self.norm = _update('norm', {'add_and_norm': False, 'combined': True})
        self.config = _update('config', {'opt': 'adam', 'lr': 0.0005, 'lr_kwargs': {}})
        self.clr = _update('clr', {'base_lr': 0.0003, 'max_lr': self.config['lr'], 'kwargs': {'step_size_up': 36, 'mode': 'triangular', 'base_momentum': 0.8, 'max_momentum': 0.9}})
        self.fusion = _update('fusion', {'type': 'LQA', 'cross_embed_red': 'avg', 'dropout': 0})
        if align_lr_to_clr and self.config['lr'] != self.clr['max_lr']:
            self.config['lr'] = self.clr['max_lr']
    def select_features(self, data, expr_all=False, mut_all=False):
        self.clin_features = data.cols['clin']
        self.expr_features = data.features['expr'] # same as data.cols['expr']
        self.mut_features = data.features['mut'] # same as data.cols['mut']
        self.features = [self.clin_features, self.expr_features, self.mut_features]
    def save_params(self, fn, path):
        param_dict = {'clin': self.clin, 'expr': self.expr, 'mut': self.mut, 'final': self.final, 'att': self.att, 'norm': self.norm, 'config': self.config, 'clr': self.clr, 'clin_features': self.clin_features, 'expr_features': self.expr_features, 'mut_features': self.mut_features, 'features': self.features}
        write_file('json', fn, param_dict, path)
    @classmethod
    def load_params(cls, fn, path):
        param_dict = load_data('json', fn, path)
        obj = cls(_from_json=True)
        for key, value in param_dict.items():
            setattr(obj, key, value)
        return obj

class TrainParams: 
    def __init__(self, epochs=20, bs=16, class_weights=None, use_gene_reg=True, gene_reg_weight=1e-4, max_grad_norm=5.0, get_val_att=False, restore_best=False, **kwargs):
        attribs = {'epochs':epochs, 'bs':bs, 'use_gene_reg':use_gene_reg, 'gene_reg_weight':gene_reg_weight, 'max_grad_norm':max_grad_norm, 'class_weights':class_weights, 'get_val_att':get_val_att, 'restore_best':restore_best}
        for name, value in attribs.items(): setattr(self, name, value)
        self.es = kwargs.get('es', {'use': True, 'patience': 3, 'metric': 'auc'})
        self.pos_weight = kwargs.get('pos_weight', None)
        self.coral_weight = kwargs.get('coral_weight', 1.0)
        self.use_coral = kwargs.get('use_coral', False)

""" Gene selection sparsity regularization (L1) """
def get_gene_reg(model, device, use_gene_reg, gene_reg_weight=1e-4):
    if not use_gene_reg: return torch.tensor(0.0, device=device)
    return (torch.norm(model.sel_e.weights, p=1) + torch.norm(model.sel_m.weights, p=1)) * gene_reg_weight

def save_model_weights(mdl, fn):
    torch.save(mdl.state_dict(), f"results/weights/{fn}.pth")

def load_saved_model(fn, params, device, seed=None):
    #if seed is not None: set_seed(seed, deterministic=True)
    model = MultiModalNN(params).to(device)
    model.load_state_dict(torch.load(f"results/weights/{fn}.pth"))
    return model

""" CORAL loss between source and target feature tensors """
def coral_loss_fn(source, target):
    d = source.size(1)
    s_c = source - source.mean(dim=0, keepdim=True) # center features
    s_cov = (s_c.T @ s_c) / (source.size(0) - 1) # compute covariance
    t_c = target - target.mean(dim=0, keepdim=True)
    t_cov = (t_c.T @ t_c) / (target.size(0) - 1)
    return ((s_cov - t_cov).pow(2).sum()) / (4 * d * d) # Frobenius norm between covariance matrices

class GeneSelector(nn.Module):
    def __init__(self, input_dim):
        super().__init__()
        self.weights = nn.Parameter(torch.ones(input_dim))
    def forward(self, x):
        return x * torch.sigmoid(self.weights) # x * self.weights # Genes with |weight| > 2σ retained

class SelfAttention(nn.Module):
    def __init__(self, embed_dim, num_heads, dropout=0):
        super(SelfAttention, self).__init__()
        self.mha = nn.MultiheadAttention(embed_dim, num_heads, batch_first=True, dropout=dropout)
    def forward(self, x):
        seq = x.unsqueeze(1) # adds sequence length dim (batch_size, seq_len, input_dim)
        mha_out, mha_w = self.mha(seq, seq, seq)
        return mha_out.squeeze(1), mha_w.squeeze(1) # [B,D], [B,1,1]

class CrossAttention(nn.Module):
    def __init__(self, embed_dim, num_heads, dropout=0):
        super(CrossAttention, self).__init__()
        self.mha_qk = nn.MultiheadAttention(embed_dim, num_heads, batch_first=True, dropout=dropout)
        self.mha_kq = nn.MultiheadAttention(embed_dim, num_heads, batch_first=True, dropout=dropout)
    def forward(self, q, kv):
        seq_q = q.unsqueeze(1)
        seq_kv = kv.unsqueeze(1)
        out_q, w_q = self.mha_qk(seq_q, seq_kv, seq_kv)
        out_k, w_k = self.mha_qk(seq_kv, seq_q, seq_q)
        return out_q.squeeze(1), out_k.squeeze(1), w_q.squeeze(1), w_k.squeeze(1)

class LearnedQueryAttentionPooling(nn.Module):
    def __init__(self, input_dim, num_heads):
        super().__init__()
        self.input_dim = input_dim
        self.num_heads = num_heads
        self.head_dim = input_dim // num_heads
        assert self.head_dim * num_heads == input_dim, "input_dim must be divisible by num_heads"
        self.query = nn.Parameter(torch.randn(1, num_heads, 1, self.head_dim)) # learnable query vector, 1 per head
        self.key_proj = nn.Linear(input_dim, input_dim)
        self.value_proj = nn.Linear(input_dim, input_dim)
        self.out_proj = nn.Linear(input_dim, input_dim)
    def forward(self, x, return_attn_weights=False):  # x: [B, N, D] (N = number of inputs to pool)
        B, N, D = x.size()
        H, D_head = self.num_heads, self.head_dim
        k = self.key_proj(x).view(B, N, H, D_head).transpose(1, 2) # Project inputs to key and value: [B, N, D] → [B, H, N, D_head]
        v = self.value_proj(x).view(B, N, H, D_head).transpose(1, 2)
        q = self.query.expand(B, -1, -1, -1) # Expand learned query: [1, H, 1, D_head] → [B, H, 1, D_head]
        att_scores = (q @ k.transpose(-2, -1)) / (D_head ** 0.5) # Compute attention scores: [B, H, 1, N]
        att_weights = F.softmax(att_scores, dim=-1)
        attended = (att_weights @ v).squeeze(2) # Weighted sum of values: [B, H, 1, D_head] → [B, H, D_head]
        attended = attended.transpose(1, 2).contiguous().view(B, D) # Concatenate heads: [B, H, D_head] → [B, D]
        out = self.out_proj(attended)
        return [out, att_weights] if return_attn_weights else [out, None] 

class ModalityEncoder(nn.Module):    
    def __init__(self, input_dim, hidden_dims, dropout_rates, activation):
        super(ModalityEncoder, self).__init__()
        layers = []
        in_dim = input_dim
        for h_dim, p in zip(hidden_dims, dropout_rates):
            layers.extend([nn.Linear(in_dim, h_dim), nn.LayerNorm(h_dim)])
            act = {'leaky_relu': nn.LeakyReLU, 'relu': nn.ReLU, 'gelu': nn.GELU, 'silu': nn.SiLU}[activation]()
            if p == 0: layers.append(act)
            else: layers.extend([act, nn.Dropout(p)])
            in_dim = h_dim
        self.net = nn.Sequential(*layers)
    def forward(self, x):
        return self.net(x)

class OutputClassifier(nn.Module):
    def __init__(self, hidden_dims, dropout_rates, activation):
        super(OutputClassifier, self).__init__()
        self.net = self.get_final_layers(hidden_dims, dropout_rates, activation)
    def get_final_layers(self, hidden_dims, dropout_rates, activation):
        if len(hidden_dims) == 1:
            return nn.Linear(hidden_dims[0], 1)
        final_layers = []
        in_dim = hidden_dims[0]
        for h_dim, p in zip(hidden_dims, dropout_rates):
            final_layers.extend([nn.Linear(in_dim, h_dim), nn.LayerNorm(h_dim)])
            act = {'leaky_relu': nn.LeakyReLU, 'relu': nn.ReLU, 'gelu': nn.GELU, 'silu': nn.SiLU}[activation]()
            if p == 0: final_layers.append(act)
            else: final_layers.extend([act, nn.Dropout(p)])
            in_dim = h_dim
        final_layers.append(nn.Linear(in_dim, 1))
        return nn.Sequential(*final_layers)
    def forward(self, x):
        return self.net(x)

class FusionEmbedding(nn.Module):
    def __init__(self, input_dim, params, ftype, cross_embed_red=None, dropout=0.0):
        super(FusionEmbedding, self).__init__()
        self.ftype = ftype
        self.cross_embed_red = cross_embed_red
        self.params = params
        self.get_num_modalities() # attention sees this as seq length
        if self.ftype not in ['LQA', 'LQA_res']:
            raise ValueError("Invaid fusion type")
        input_dim = 2 * input_dim if params.fusion['cross_embed_red'] == 'concat' else input_dim
        self.pool = LearnedQueryAttentionPooling(input_dim=input_dim, num_heads=2)
        if self.ftype == 'LQA_res': # ^^ with residual fusion (gated projection)
            self.fusion_gate = nn.Linear(input_dim, input_dim)
            self.fusion_proj = nn.Linear(input_dim, input_dim)
    def get_num_modalities(self):
        if not self.params.att['cross']['use']: self.num_modalities = 3
        elif self.cross_embed_red is None: self.num_modalities = 6
        elif self.cross_embed_red == 'avg': self.num_modalities = 3
        else: print("Invalid input for cross_embed_red")
    def format_embeddings(self, embeddings):
        if not self.params.att['cross']['use'] or self.cross_embed_red is None: 
            return embeddings
        z_ce, z_ec, z_cm, z_mc, z_em, z_me = embeddings
        if self.cross_embed_red == 'avg':
            z_c = 0.5 * (z_ce + z_cm) # C attended to E and M
            z_e = 0.5 * (z_ec + z_em) # E attended to C and M
            z_m = 0.5 * (z_mc + z_me) # M attended to C and E
            return [z_c, z_e, z_m]
    def forward(self, embeddings, return_attn_weights=False):
        clean_embeddings = self.format_embeddings(embeddings)
        x_stack = torch.stack(clean_embeddings, dim=1) # [B, M, D]
        z, w = self.pool(x_stack, return_attn_weights)
        if self.ftype == 'LQA_res':
            gates = torch.sigmoid(self.fusion_gate(z))
            z = self.fusion_proj(gates * z + z)
        return z, w
        
