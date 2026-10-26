"""Phân tích thí điểm lượt 2 (gộp chuỗi tốn GPU của lượt 1). Chạy: python analyze_v2.py"""
import numpy as np, pandas as pd
pd.set_option('display.width', 250); pd.set_option('display.max_columns', 40)
v1 = pd.read_csv('g4_pilot_v1.csv'); v2 = pd.read_csv('g4_pilot_v2.csv')
EXP = {"gibbs", "sem", "gibbs_known", "hard_em"}
keys = ["dataset_id", "setting", "level", "seed", "method"]
k2 = set(map(tuple, v2[keys].astype(str).values))
old = v1[v1.method.isin(EXP)]
old = old[[tuple(r) not in k2 for r in old[keys].astype(str).values]]
d = pd.concat([v2, old], ignore_index=True)
short = {'credit-g': 'credit', 'diabetes': 'diab', 'blood-transfusion-service-center': 'blood',
         'banknote-authentication': 'bank', 'ilpd': 'ilpd', 'wdbc': 'wdbc'}
d['ds'] = d.dataset.map(short)
for col, pat in [('flips', r'flips_last=([0-9.]+)'), ('p0_acc', r'p0_acc=([0-9.]+)')]:
    d[col] = d.extra.str.extract(pat).astype(float)
tr = d.extra.str.extract(r'acc_trace=([0-9./]+)')[0]
d['acc_first'] = tr.str.split('/').str[0].astype(float)
d['acc_last'] = tr.str.split('/').str[-1].astype(float)
d.to_csv('g4_pilot_merged.csv', index=False)
PEER = {'pu_scar': ['nnpu', 'upu'], 'pu_sar': ['nnpu', 'upu'], 'llp': ['dllp', 'dllp_linear']}

def table(st, met, methods):
    x = d[d.setting == st]
    t = x.groupby(['level', 'ds', 'method'])[met].mean().unstack('method')
    return t[[m for m in methods if m in t]]

print("===== LLP AUC"); print(table('llp', 'auc', ['gibbs', 'gibbs_warm', 'gibbs_trueinit', 'sem_warm', 'hard_em', 'hard_em_warm', 'dllp', 'dllp_linear', 'naive', 'oracle']).round(3))
print("===== LLP logloss"); print(table('llp', 'logloss', ['gibbs', 'gibbs_warm', 'gibbs_trueinit', 'sem_warm', 'hard_em_warm', 'dllp', 'dllp_linear', 'naive', 'oracle']).round(3))
print("===== LLP chain accuracy vs truth (acc_last) and p0_acc")
print(d[(d.setting == 'llp') & d.method.isin(['gibbs_warm', 'gibbs_trueinit'])].groupby(['level', 'ds', 'method'])[['p0_acc', 'acc_first', 'acc_last', 'flips']].mean().unstack('method').round(3))
for st in ['pu_scar', 'pu_sar']:
    ms = ['gibbs', 'gibbs_warm', 'gibbs_trueinit', 'gibbs_known', 'gibbs_known_warm', 'sem_warm', 'hard_em', 'hard_em_warm', 'naive', 'nnpu', 'upu', 'oracle']
    for met in ['auc', 'logloss', 'logloss_pm']:
        print(f"===== {st} {met}"); print(table(st, met, ms).round(3))
    print(f"===== {st} chain acc")
    print(d[(d.setting == st) & d.method.isin(['gibbs_warm', 'gibbs_trueinit', 'gibbs_known_warm'])].groupby(['level', 'ds', 'method'])[['p0_acc', 'acc_first', 'acc_last', 'flips']].mean().unstack('method').round(3))
