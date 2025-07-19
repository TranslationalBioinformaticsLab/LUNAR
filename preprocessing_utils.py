import pandas as pd
import numpy as np
import warnings
import pickle
import re
from sklearn.feature_selection import f_classif, mutual_info_classif, SelectFpr, VarianceThreshold
from sklearn.preprocessing import RobustScaler, StandardScaler, MinMaxScaler, MaxAbsScaler
from sklearn.utils import resample
from base_utils import get_new_cols, load_data, write_file, check_filename

def remove_rarely_mutated_genes(g, f=0.01):
    mutation_freq = (g.tcga_x_train[g.cols['mut']] > 0).mean()
    keep_genes = mutation_freq[mutation_freq >= f].index.tolist()
    print(f"{len(g.cols['mut'])-len(keep_genes)} mut removed (less than {round(f*100)}% mutated), {len(keep_genes)} mut remaining")
    g.cols['mut'] = keep_genes
    all_cols = [*g.cols['clin'], *g.cols['expr'], *g.cols['mut']]
    g.tcga_x_train, g.tcga_x_val, g.tcga_x_test = [df[all_cols].copy() for df in [g.tcga_x_train, g.tcga_x_val, g.tcga_x_test]]
    g.glass_x_train, g.glass_x_val, g.glass_x_test = [df[all_cols].copy() for df in [g.glass_x_train, g.glass_x_val, g.glass_x_test]]

def log1p_mutations(g):
    cols = g.cols['mut']
    for attr in ['tcga_x_train', 'tcga_x_val', 'tcga_x_test', 'glass_x_train', 'glass_x_val', 'glass_x_test']:
        df = getattr(g, attr)
        float_block = df[cols].astype(float).apply(np.log1p)
        df.loc[:, cols] = float_block 

def remove_nearly_constant(g, threshold=1e-8):
    vt = VarianceThreshold(threshold=threshold).set_output(transform='pandas')
    g.tcga_x_train = vt.fit_transform(g.tcga_x_train)
    g.tcga_x_val = vt.transform(g.tcga_x_val)
    g.tcga_x_test = vt.transform(g.tcga_x_test)
    g.glass_x_train = vt.transform(g.glass_x_train)
    g.glass_x_val = vt.transform(g.glass_x_val)
    g.glass_x_test = vt.transform(g.glass_x_test)

def apply_scaler(g, ctype='robust', etype='robust', mtype='robust'):
    for stype, cols in zip([ctype, etype, mtype], [['age'], g.cols['expr'], g.cols['mut']]):
        if stype is None: continue
        tcga_sc = {'minmax': MinMaxScaler, 'robust': RobustScaler, 'maxabs': MaxAbsScaler, 'standard': StandardScaler}[stype]()
        g.tcga_x_train[cols] = tcga_sc.fit_transform(g.tcga_x_train[cols])
        g.tcga_x_val[cols] = tcga_sc.transform(g.tcga_x_val[cols])
        g.tcga_x_test[cols] = tcga_sc.transform(g.tcga_x_test[cols])
        g.glass_x_train[cols] = tcga_sc.transform(g.glass_x_train[cols])
        g.glass_x_val[cols] = tcga_sc.transform(g.glass_x_val[cols])
        g.glass_x_test[cols] = tcga_sc.transform(g.glass_x_test[cols])

class COMBAT:
    def __init__(self, g):
        self.tcga_tss_train, self.tcga_tss_val, self.tcga_tss_test = [g.tcga_tss.loc[split.index].copy() for split in [g.tcga_x_train, g.tcga_x_val, g.tcga_x_test]]
        self.glass_tss_train, self.glass_tss_val, self.glass_tss_test = [g.glass_tss.loc[split.index].copy() for split in [g.glass_x_train, g.glass_x_val, g.glass_x_test]]
        for split in ['train', 'val', 'test']:
            for ds in ['tcga', 'glass']:
                x = getattr(g, f"{ds}_x_{split}")
                tss = g.tcga_tss.loc[x.index].copy() if ds == 'tcga' else g.glass_tss.loc[x.index].copy()
                assert x.index.equals(tss.index), f"{ds.upper()} {split} split batch labels do not align with expression/mutation index"
        
    def correct_modality(self, g, mod):
        # Step 1: Estimate batch-specific means and global mean from training data for mean-only ComBat correction.
        mod_cols = g.cols[mod]
        tcga_batch_means, tcga_global_mean = self.estimate_combat_means(g.tcga_x_train[mod_cols], self.tcga_tss_train)
        glass_batch_means, glass_global_mean = self.estimate_combat_means(g.glass_x_train[mod_cols], self.glass_tss_train)
        # Step 2: Apply correction
        g.tcga_x_train.loc[:, mod_cols] = self.apply_combat_correction(g.tcga_x_train[mod_cols], self.tcga_tss_train, tcga_batch_means, tcga_global_mean)
        g.tcga_x_val.loc[:, mod_cols] = self.apply_combat_correction(g.tcga_x_val[mod_cols], self.tcga_tss_val, tcga_batch_means, tcga_global_mean)
        g.tcga_x_test.loc[:, mod_cols] = self.apply_combat_correction(g.tcga_x_test[mod_cols], self.tcga_tss_test, tcga_batch_means, tcga_global_mean)
        g.glass_x_train.loc[:, mod_cols] = self.apply_combat_correction(g.glass_x_train[mod_cols], self.glass_tss_train, glass_batch_means, glass_global_mean)
        g.glass_x_val.loc[:, mod_cols] = self.apply_combat_correction(g.glass_x_val[mod_cols], self.glass_tss_val, glass_batch_means, glass_global_mean)
        g.glass_x_test.loc[:, mod_cols] = self.apply_combat_correction(g.glass_x_test[mod_cols], self.glass_tss_test, glass_batch_means, glass_global_mean)

    def estimate_combat_means(self, x_train_mod, batch_labels):
        batches = batch_labels.unique()
        batch_means = {}
        for b in batches:
            idx = batch_labels == b
            batch_data = x_train_mod.loc[idx].copy()
            batch_means[b] = batch_data.mean(axis=0)
        global_mean = x_train_mod.mean(axis=0)
        return batch_means, global_mean

    def apply_combat_correction(self, mod_df, batches, batch_means, global_mean):
        corrected = pd.DataFrame(index=mod_df.index, columns=mod_df.columns, dtype=float)
        for i in mod_df.index:
            b = batches.loc[i]
            if b in batch_means:
                corrected.loc[i, :] = mod_df.loc[i] - batch_means[b] + global_mean
            else:
                # Fallback: apply only global centering (no batch correction) if batch in test/val is not in train
                corrected.loc[i, :] = mod_df.loc[i] - global_mean + global_mean
                warnings.warn(f"Batch '{b}' in correction set not seen during training. Skipping batch adjustment.")
        return corrected.astype(float)

class CorrelationAnalysis:
    def __init__(self, action, glioma, corr_mx_fn, data_path='data/', threshold=0.95, calculate_pairs=True, corr_pairs_fn='', load_corr_pairs=False):
        self.data_path = data_path
        self.corr_mx = self._get_corr_mx(action, glioma, corr_mx_fn)
        if calculate_pairs: 
            self.corr_pairs = self._get_corr_pairs(threshold, corr_pairs_fn, load_corr_pairs)
            print(f"{len(self.corr_pairs)} highly correlated pairs")
        self.known_genes = self._get_known_genes_from_literature()

    def remove_correlated_genes(self, action, glioma, fn='', primary_method='variance', secondary_method='corr count', prioritize_known_genes=True, get_removal_set_only=False, clin_features_to_drop=[], save_removal=True):
        if action == 'load':
            with open(self.data_path + fn, 'rb') as f:
                self.remove = pickle.load(f)
            self._apply_removal(glioma)
            return
        self.remove = {}
        if prioritize_known_genes: 
            self._prioritize_known_genes(glioma)
        else: 
            self.corr_pairs_step1 = self.corr_pairs.copy()
            self.remove['step 1'] = {'clin': [], 'expr': [], 'mut': []}
        self.counts = {'primary': 0, 'secondary': 0}
        to_remove = set()
        for _, row in self.corr_pairs_step1.iterrows():
            var1, var2 = row['var1'], row['var2']
            if var1 in to_remove or var2 in to_remove: 
                continue
            drop_feature = self._select_drop_feature(glioma, var1, var2, primary_method, secondary_method)
            to_remove.add(drop_feature)
        expr_to_remove = get_new_cols(to_remove, glioma.cols['expr'])
        mut_to_remove = get_new_cols(to_remove, glioma.cols['mut'])
        self.remove['step 2'] = {'clin': clin_features_to_drop, 'expr': expr_to_remove, 'mut': mut_to_remove}
        if save_removal:
            with open(self.data_path + fn, 'wb') as f:
                pickle.dump(self.remove, f)
        if not get_removal_set_only:
            self._apply_removal(glioma)
            return

    def _apply_removal(self, glioma):
        glioma.cols_before_ca = {kind: glioma.cols[kind] for kind in ['clin', 'expr', 'mut']}
        glioma.cols = {
            kind: [f for f in glioma.cols_before_ca[kind] if f not in self.remove['step 1'][kind] + self.remove['step 2'][kind]]
            for kind in ['clin', 'expr', 'mut']
        }
        self.keep = glioma.cols['clin'] + glioma.cols['expr'] + glioma.cols['mut']
        glioma.tcga_x_train = glioma.tcga_x_train[self.keep].copy()
        glioma.tcga_x_val = glioma.tcga_x_val[self.keep].copy()
        glioma.tcga_x_test = glioma.tcga_x_test[self.keep].copy()
        glioma.glass_x_train = glioma.glass_x_train[self.keep].copy()
        glioma.glass_x_val = glioma.glass_x_val[self.keep].copy()
        glioma.glass_x_test = glioma.glass_x_test[self.keep].copy()
        print(f"features removed: {len(glioma.cols_before_ca['clin']) - len(glioma.cols['clin'])} clin, {len(glioma.cols_before_ca['expr']) - len(glioma.cols['expr'])} expr, {len(glioma.cols_before_ca['mut']) - len(glioma.cols['mut'])} mut")
        print(f"  final features: {len(glioma.cols['clin'])} clin, {len(glioma.cols['expr'])} expr, {len(glioma.cols['mut'])} mut\n")

    def _get_corr_mx(self, action, glioma, corr_mx_fn):
        if action == 'load': return load_data('pd', corr_mx_fn, self.data_path)
        corr_mx = glioma.tcga_x_train[glioma.cols['expr'] + glioma.cols['mut']].corr()
        corr_mx.to_pickle(self.data_path+check_filename(corr_mx_fn, 'pkl'))
        return corr_mx
        
    def _get_corr_pairs(self, threshold, corr_pairs_fn, load_corr_pairs):
        if load_corr_pairs: return load_data('pd', corr_pairs_fn, self.data_path)
        uppertri = self.corr_mx.where(np.triu(np.ones(self.corr_mx.shape), k=1).astype(bool)).abs()
        corr_pairs = uppertri.stack().reset_index()
        corr_pairs.columns = ['var1', 'var2', 'correlation']
        corr_pairs = corr_pairs.query("correlation >= @threshold")
        corr_pairs.to_pickle(self.data_path+check_filename(corr_pairs_fn, 'pkl'))
        return corr_pairs

    def _get_known_genes_from_literature(self):
        # https://pmc.ncbi.nlm.nih.gov/articles/PMC9427889/; https://pmc.ncbi.nlm.nih.gov/articles/PMC6407082/; https://pmc.ncbi.nlm.nih.gov/articles/PMC2818769/; https://pmc.ncbi.nlm.nih.gov/articles/PMC3910500/; https://pmc.ncbi.nlm.nih.gov/articles/PMC3443254/
        return sorted({'IDH1', 'IDH2', 'BRAF', 'CDKN2A', 'CDKN2B', 'ATRX', 'TERT', 'TP53', 'EGFR', 'KIAA1549', 'MGMT', 'MYB', 'MYBL1', 'YAP1', 'RELA', 'MYCN', 'SMARCB1', 'NF1', 'NF2', 'MAPK', 'PIK3R1', 'PIK3CA', 'RB1', 'PTEN', 'PDGFRA', 'ERBB2', 'CHI3L1', 'MET', 'CD44', 'MERTK', 'NES', 'CDK4', 'CCND2', 'NOTCH3', 'JAG1', 'LFNG', 'SMO', 'GAS1', 'GLI2', 'TRADD', 'RELB', 'TNFRSF1A', 'NKX2-2', 'OLIG2', 'CDKN1A', 'DCX', 'DLL3', 'ASCL1', 'TCF4', 'NEFL', 'GABRA1', 'SYT1', 'SLC12A5', 'LZTR1', 'SPTA1', 'GABRA6', 'KEL', 'CDK6', 'MDM2', 'SOX2', 'CCND1', 'CCNE2', 'QKI', 'TGFBR2', 'CIC', 'FUBP1'})
        
    def _prioritize_known_genes(self, glioma):
        # check highly correlated pairs for those involving known genes and remove any unknown genes correlated to known genes
        known_gene_pattern = f"^(?:{'|'.join(self.known_genes)})_"
        temp = self.corr_pairs[
            self.corr_pairs['var1'].str.contains(known_gene_pattern, case=False, na=False) |
            self.corr_pairs['var2'].str.contains(known_gene_pattern, case=False, na=False)
        ]
        to_remove = set()
        both_known_idxs = []
        for idx, row in temp.iterrows():
            var1_known = bool(re.match(known_gene_pattern, row['var1']))
            var2_known = bool(re.match(known_gene_pattern, row['var2']))
            if not var1_known:
                to_remove.add(row['var1'])
            elif not var2_known:
                to_remove.add(row['var2'])
            else:
                both_known_idxs.append(idx)
        self.both_known = temp.loc[both_known_idxs]
        self.corr_pairs_step1 = self.corr_pairs[~self.corr_pairs['var1'].isin(to_remove) & ~self.corr_pairs['var2'].isin(to_remove)].copy()
        self.remove['step 1'] = {'clin': [], 'expr': get_new_cols(to_remove, glioma.cols['expr']), 'mut': get_new_cols(to_remove, glioma.cols['mut'])}
        print(f"genes removed that were correlated with known genes: {len(to_remove)} ({len(self.corr_pairs) - len(self.corr_pairs_step1)} pairs resolved)")

    def _select_drop_feature(self, glioma, f1, f2, primary_method='variance', secondary_method='corr count'):
        # drop feature with lower variance, lower target correlation, higher num. of outliers, or higher num. of high correlations
        primary = self._calculate_feature_metrics(glioma, f1, f2, method=primary_method)
        if primary.loc[f1] != primary.loc[f2]: 
            self.counts['primary'] += 1
            return primary.idxmin() if primary_method in ['variance', 'target_corr'] else primary.idxmax()
        self.counts['secondary'] += 1
        secondary = self._calculate_feature_metrics(glioma, f1, f2, method=secondary_method)
        return secondary.idxmin() if secondary_method in ['variance', 'target_corr'] else secondary.idxmax()

    def _calculate_feature_metrics(self, glioma, f1, f2, method):
        if method == 'variance': return glioma.tcga_x_train[[f1, f2]].var()
        if method == 'target corr': return glioma.tcga_x_train[[f1, f2]].corrwith(glioma.tcga_y_train).abs()
        if method == 'outliers': return pd.Series(data={f1: self._count_outliers(glioma.tcga_x_train[f1]), f2: self._count_outliers(glioma.tcga_x_train[f2])})
        if method == 'corr count': return pd.Series(data={f1: len(self.corr_pairs.query("var1 == @f1 or var2 == @f1")), f2: len(self.corr_pairs.query("var1 == @f2 or var2 == @f2"))})

    def _count_outliers(self, s):
        q1, q3 = s.quantile(0.25), s.quantile(0.75)
        iqr = q3 - q1
        return ((s < (q1 - 1.5 * iqr)) | (s > (q3 + 1.5 * iqr))).sum()

class ExpressionStabilitySelector:
    def __init__(self, action, x=None, y=None, method='fpr', fn='', dp='data/', n_boots=100, fpr_alpha=0.05, mi_quantile=0.5, rs=42):             
        self.sel_freq = load_data('pd', fn, 'data/') if action=='load' else self.calc_freq(x, y, method, fn, n_boots, fpr_alpha, mi_quantile, rs)

    def calc_freq(self, x, y, method, fn, n_boots, fpr_alpha, mi_quantile, rs):  
        np.random.seed(rs)
        feature_counts = pd.Series(0, index=x.columns)
        for i in range(n_boots):
            x_boot, y_boot = resample(x, y, stratify=y, n_samples=len(y), replace=True, random_state=rs+i)
            if method == 'fpr':
                selector = SelectFpr(score_func=f_classif, alpha=fpr_alpha) # alpha = significance level (default 5%)
                selector.fit(x_boot, y_boot)
                selected = x_boot.columns[selector.get_support()]
            elif method == 'mi':
                scores = mutual_info_classif(x_boot, y_boot, random_state=rs+i)
                mi_thresh = np.quantile(scores, mi_quantile) # dynamic threshold per bootstrap
                selected = x_boot.columns[np.array(scores) >= mi_thresh]
            feature_counts[selected] += 1
        selection_freq = feature_counts / n_boots
        if fn != '': 
            write_file('pd', fn, selection_freq, dp)
        return selection_freq
    
    def select_by_threshold(self, stability_threshold=0.8):
        self.selection = self.sel_freq[self.sel_freq >= stability_threshold].index.tolist()
        print(f"{len(self.selection)} expr features selected")

    def apply_to_dataset(self, g):
        g.features = {'clin': g.cols['clin'], 'expr': self.selection, 'mut': g.cols['mut']}
        g.cols['expr'] = g.features['expr']
        all_cols = [*g.cols['clin'], *g.cols['expr'], *g.cols['mut']]
        g.tcga_x_train, g.tcga_x_val, g.tcga_x_test = [df[all_cols] for df in [g.tcga_x_train, g.tcga_x_val, g.tcga_x_test]]
        g.glass_x_train, g.glass_x_val, g.glass_x_test = [df[all_cols] for df in [g.glass_x_train, g.glass_x_val, g.glass_x_test]]
