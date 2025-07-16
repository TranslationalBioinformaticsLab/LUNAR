import pandas as pd
import numpy as np
from scipy import stats
import itertools
import json

def get_new_cols(incl_cols, ref_cols):
    return [c for c in incl_cols if c in ref_cols]

def check_filename(fn, extension):
    return f'{fn}.{extension}' if (len(fn) <= len(f'.{extension}')) or (fn[-len(f'.{extension}'):] != f'.{extension}') else fn

def drop_useless_cols(df, return_useless=False):
    useless = df.columns[(df.nunique() == 1)|(df.isna().all())]
    new_df = df.drop(columns=useless)
    return [new_df, useless] if return_useless else new_df

def load_data(kind, filename, data_path, clean=False, **kwargs):
    filename = check_filename(filename, {'list':'txt', 'json':'json', 'pd':'pkl', 'csv':'csv', 'raw txt':'txt', 'tsv':'tsv'}[kind])
    if kind in ['list', 'json']:
        with open(data_path+filename, 'r') as fin:
            return json.load(fin) if kind == 'json' else [line.strip() for line in fin.readlines()]
    elif kind == 'pd':
        return pd.read_pickle(data_path+filename)
    elif kind in ['csv', 'raw txt', 'tsv']:
        loaded = pd.read_csv(data_path+filename, sep=',' if kind == 'csv' else '\t', na_values=['[Not Available]','[NOT AVAILABLE]','[Not Applicable]','Not Available','NOT AVAILABLE','nan','[Discrepancy]','[Unknown]'], **kwargs)
        return drop_useless_cols(loaded) if clean else loaded

def write_file(kind, filename, data, data_path):
    filename = check_filename(filename, {'list':'txt', 'json':'json', 'pd':'pkl', 'csv':'csv'}[kind])
    if kind == 'list':
        with open(data_path+filename, 'w') as fout: fout.writelines('\n'.join(data))
    elif kind == 'json':
        with open(data_path+filename, 'w') as fout: json.dump(data, fout)
    elif kind == 'pd': 
        data.to_pickle(data_path+filename)
    elif kind == 'csv': 
        data.to_csv(data_path+filename)
