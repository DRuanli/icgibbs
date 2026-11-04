"""V2 diagnosis (exploratory, 04/11/2026). Reads existing result files only; no new runs.
Usage: python -I v2_diagnosis.py <path to de-tai-moi>
Sections: A response-map fixed points (S2X); B Study 1 PLL q=0.6 chains; C Study 2 per-dataset gaps;
D Study 1 LLP by bag size; E sample-size curve."""
import sys, numpy as np, pandas as pd
B = sys.argv[1]
pd.set_option('display.width', 230); pd.set_option('display.max_columns', 40)

print('# A. Response map: fixed points of Phi(r) (S2X, Study 1 datasets, mean over 3 seeds)')
d = pd.read_csv(f'{B}/S2/results/s2x_summary.csv'); d = d[d.rho != 'u'].copy()
d['r'] = d.r.astype(float); d['phi'] = d.phi.astype(float)
rows = []
for (q, ds, b), g in d.groupby(['q', 'dataset', 'backend']):
    g = g.sort_values('r'); r = g.r.values; phi = g.phi.values; gain = phi - r
    sl = np.diff(phi) / np.diff(r); fps = []
    for i in range(len(r) - 1):
        if np.sign(gain[i]) != np.sign(gain[i + 1]):
            rs = r[i] - gain[i] * (r[i + 1] - r[i]) / (gain[i + 1] - gain[i])
            fps.append(f"{rs:.2f}{'s' if sl[i] < 1 else 'u'}")
    rows.append(dict(q=q, dataset=ds, C=g.n_classes.iloc[0], tfm=b.replace('_v2', ''), r_max=r.max().round(2),
                     min_gain=gain.min().round(3), gain_at_rmax=gain[-1].round(3), max_slope=sl.max().round(2),
                     fixed_points=' '.join(fps) or '-'))
A = pd.DataFrame(rows).sort_values(['q', 'dataset', 'tfm']); print(A.to_string(index=False))
print('\nmax slope of Phi by TFM (all cells): ', A.groupby('tfm').max_slope.agg(['min', 'median', 'max']).to_dict('index'))
print('cells with a negative-gain region: ', A[A.min_gain < 0].groupby('tfm').size().to_dict())

print('\n# B. Study 1 PLL q=0.6 (TabPFN v2): chain start/end accuracy of training labels vs test accuracy')
c = pd.read_csv(f'{B}/G8/results/pll_chain_diag.csv'); c = c[c.level == 0.6]
Bt = c.groupby('dataset').agg(C=('n_classes', 'first'), n_train=('n_train', 'first'), r_sweep1=('monitor_sweep1', 'mean'),
     r_last=('monitor_last', 'mean'), collapsed=('collapsed', 'sum'), gibbs=('gibbs_warm', 'mean'), naive=('naive', 'mean'),
     cc_tuned=('cc_tun', 'mean'), glws_tuned=('glwspll_tun', 'mean'), oracle=('oracle', 'mean')).round(3)
print(Bt.sort_values('C', ascending=False).to_string())

print('\n# C. Study 2 (TabICL v2): per-dataset gap to oracle and to the best baseline')
fin = pd.read_csv(f'{B}/S2/s2_final.csv'); fin = fin[fin.kept == True]
cen = pd.read_csv(f'{B}/S2/census/s2_census_candidates.csv')[['did', 'd']]
meta = fin.merge(cen, on='did', how='left'); meta['n'] = meta.n_after_dedup.astype(float)
meta.loc[meta.subsample.notna(), 'n'] = 10000
meta['n_train'] = (0.7 * meta.n).round(); meta['ctx_sub'] = meta.n_train > 2000
BL = {'llp8': ['dllp_def', 'dllp_tun', 'dllplin_def', 'dllplin_tun', 'easyllp_def', 'easyllp_tun', 'glws_def', 'glws_tun', 'llpvat_def', 'llpvat_tun'],
      'pll': ['cc_def', 'cc_tun', 'glwspll_def', 'glwspll_tun', 'proden_def', 'proden_tun', 'rc_def', 'rc_tun']}
for f, bl in [('s2_table_llp8.csv', BL['llp8']), ('s2_table_pll_0.3.csv', BL['pll']), ('s2_table_pll_0.6.csv', BL['pll'])]:
    t = pd.read_csv(f'{B}/S2/results/{f}').merge(meta[['name', 'd', 'n_train', 'ctx_sub']], left_on='dataset', right_on='name', how='left')
    t['best_bl'] = t[bl].max(axis=1); t['gibbs_m_best'] = t.gibbs_warm - t.best_bl
    t['gap_oracle'] = t.oracle - t.gibbs_warm; t['gibbs_m_hardem'] = t.gibbs_warm - t.hard_em_warm
    print(f'\n## {f}')
    print(t[['dataset', 'n_train', 'd', 'ctx_sub', 'gibbs_warm', 'best_bl', 'gibbs_m_best', 'oracle', 'gap_oracle', 'gibbs_m_hardem']]
          .sort_values('gap_oracle', ascending=False).round(3).to_string(index=False))
    print(f'mean gap to oracle {t.gap_oracle.mean():.3f}; datasets with gap >= 0.05: {(t.gap_oracle >= 0.05).sum()} of {len(t)}; '
          f'Gibbs below best baseline on {(t.gibbs_m_best < 0).sum()}')
    print('by ctx_sub:', t.groupby('ctx_sub').agg(n=('dataset', 'size'), gap=('gap_oracle', 'mean'), vs_best=('gibbs_m_best', 'mean')).round(3).to_dict('index'))

print('\n# D. Study 1 LLP (TabPFN v2) by bag size: Gibbs vs linear DLLP and vs best baseline')
for f in ['table_llp8.csv', 'table_llp_16.csv', 'table_llp_32.csv']:
    t = pd.read_csv(f'{B}/G8/results/{f}')
    bl = [x for x in t.columns if x.split('_')[0] in ('dllp', 'dllplin', 'easyllp', 'glws', 'llpvat')]
    t['best_bl'] = t[bl].max(axis=1)
    print(f'\n## {f}: mean Gibbs-linDLLP {(t.gibbs_warm - t.dllplin_def).mean():+.3f}; mean Gibbs-best {(t.gibbs_warm - t.best_bl).mean():+.3f}; '
          f'Gibbs below best on {(t.gibbs_warm < t.best_bl).sum()}/{len(t)}; mean gap to oracle {(t.oracle - t.gibbs_warm).mean():.3f}')
    t['g_m_best'] = t.gibbs_warm - t.best_bl
    print(t[['dataset', 'gibbs_warm', 'dllplin_def', 'best_bl', 'oracle', 'g_m_best']].sort_values('g_m_best').round(3).head(4).to_string(index=False))

print('\n# E. Study 1 nested sample-size curve (registered ablation, TabPFN v2): mean over datasets')
s = pd.read_csv(f'{B}/G8/results/size_curve.csv')
print(s.groupby(['setting', 'size'])[['gibbs_minus_comp', 'gibbs_minus_naive', 'n_train']].mean().round(3).to_string())

print('\n# F. Real partial labels (Study 1, TabICL v2): gap to oracle')
r = pd.read_csv(f'{B}/G8/results/rq5_acc.csv').set_index('method')
bl = [m for m in r.index if m not in ('gibbs_warm', 'oracle', 'naive', 'hard_em_warm', 'sem_warm')]
F = pd.DataFrame({'gibbs': r.loc['gibbs_warm'], 'best_baseline': r.loc[bl].max(), 'oracle': r.loc['oracle']})
F['gap_oracle'] = F.oracle - F.gibbs; F['gibbs_m_best'] = F.gibbs - F.best_baseline
print(F.round(3).to_string())

print('\n# G. Study 1 LLP bag 8: same datasets, TabPFN v2 vs TabICL v2 (registered TFM ablation)')
p = pd.read_csv(f'{B}/G8/results/table_llp8.csv')[['dataset', 'gibbs_warm', 'oracle']].rename(columns={'gibbs_warm': 'gibbs_tabpfn', 'oracle': 'oracle_tabpfn'})
i = pd.read_csv(f'{B}/G8/results/tabicl_llp_8.0.csv')[['dataset', 'gibbs_warm', 'oracle']].rename(columns={'gibbs_warm': 'gibbs_tabicl', 'oracle': 'oracle_tabicl'})
G = p.merge(i, on='dataset'); G['tabicl_m_tabpfn'] = G.gibbs_tabicl - G.gibbs_tabpfn
print(G.sort_values('tabicl_m_tabpfn', ascending=False).round(3).to_string(index=False))
