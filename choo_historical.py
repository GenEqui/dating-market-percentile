"""
(c) Choo-Siow (2006) validation across time, in the spirit of the paper's
1971/72 vs 1981/82 comparison.

The exact paper years need 1970/1980 Census + Vital Statistics microdata (IPUMS /
restricted). Here we reproduce the paper's METHOD and CENTRAL QUESTION -- did the
gains to marriage for YOUNG ADULTS fall over time? -- using the two freely
downloadable ACS 1-year PUMS years that span the widest gap: 2010 vs 2023
(CA+FL+NY+TX, PWGTP-weighted). Types = age band only (the paper's benchmark model).

Caveats vs the paper: married STOCK (not marriage flows), ACS (not decennial +
vital stats), 2010/2023 (not 1971/1982). It validates the estimator's ability to
detect the well-documented retreat from young-adult marriage.
"""
import glob, numpy as np, pandas as pd, choo_siow

SCRATCH = "data"   # local folder: 2010 PUMS in data/pums2010/, 2023 PUMS in data/pums/
BANDS = [(18,22),(23,27),(28,32),(33,37),(38,42),(43,49),(50,59),(60,70)]
NB = len(BANDS)
def band(a):
    for k,(lo,hi) in enumerate(BANDS):
        if a<=hi: return k
    return NB-1

YEARS = {
  2010: dict(files=[f"{SCRATCH}/pums2010/ss10p{s}.csv" for s in ("ca","fl","ny","tx")],
             rel="RELP", ref=0, spouse=1),
  2023: dict(files=[f"{SCRATCH}/pums/psam_p{s}.csv"    for s in ("06","12","36","48")],
             rel="RELSHIPP", ref=20, spouse=21),
}

def estimate_year(cfg):
    cols = ['SERIALNO','SEX','AGEP','MAR','PWGTP', cfg['rel']]
    frames=[]
    for fp in cfg['files']:
        d = pd.read_csv(fp, usecols=cols, dtype={'SERIALNO':str}, na_values=[''], low_memory=False)
        for c in ['SEX','AGEP','MAR','PWGTP', cfg['rel']]: d[c]=pd.to_numeric(d[c],errors='coerce')
        d = d[(d['AGEP']>=18)&(d['AGEP']<=70)&(d['PWGTP']>0)].copy()
        frames.append(d)
    df = pd.concat(frames, ignore_index=True)
    df['b'] = df['AGEP'].map(band).astype(int)

    single = df[df['MAR']!=1]
    def by(sub, sx):
        v=np.zeros(NB)
        s=sub[sub['SEX']==sx]
        for b,w in zip(s['b'],s['PWGTP']): v[b]+=w
        return v
    mu_i0, mu_0j = by(single,1), by(single,2)

    rel=cfg['rel']
    ref=df[df[rel]==cfg['ref']][['SERIALNO','SEX','b','PWGTP']].rename(columns={'SEX':'sr','b':'br','PWGTP':'wr'})
    sp =df[df[rel]==cfg['spouse']][['SERIALNO','SEX','b','PWGTP']].rename(columns={'SEX':'ss','b':'bs','PWGTP':'ws'})
    cpl=ref.merge(sp,on='SERIALNO'); cpl=cpl[cpl['sr']!=cpl['ss']]
    cpl['hb']=np.where(cpl['sr']==1,cpl['br'],cpl['bs']); cpl['wb']=np.where(cpl['sr']==2,cpl['br'],cpl['bs'])
    cpl['w']=(cpl['wr']+cpl['ws'])/2
    mu_ij=np.zeros((NB,NB))
    for h,wb,w in zip(cpl['hb'],cpl['wb'],cpl['w']): mu_ij[h,wb]+=w
    mu_ij=np.maximum(mu_ij,1.0)

    pi=choo_siow.estimate_gains(mu_ij,mu_i0,mu_0j)['pi']
    m=mu_i0+mu_ij.sum(1); f=mu_0j+mu_ij.sum(0)
    married_share_m = mu_ij.sum(1)/m   # 1 - single rate, men by band
    return dict(pi=pi, single_m=mu_i0/m, single_f=mu_0j/f, ncpl=int(len(cpl)))

res = {y: estimate_year(cfg) for y,cfg in YEARS.items()}
print(f"couples linked: 2010={res[2010]['ncpl']:,}  2023={res[2023]['ncpl']:,}\n")

lab=[f"{lo}-{hi}" for lo,hi in BANDS]
print("Gains to marriage pi_ij on the DIAGONAL (same-age-band pairing), by year:")
print(f"{'band':>8} {'pi 2010':>9} {'pi 2023':>9} {'change':>8}")
for b in range(NB):
    d=res[2023]['pi'][b,b]-res[2010]['pi'][b,b]
    star=' <-- young adults' if b in (0,1,2) else ''
    print(f"{lab[b]:>8} {res[2010]['pi'][b,b]:>9.3f} {res[2023]['pi'][b,b]:>9.3f} {d:>+8.3f}{star}")

ya = [0,1,2]  # 18-32
g10 = np.mean([res[2010]['pi'][b,b] for b in ya])
g23 = np.mean([res[2023]['pi'][b,b] for b in ya])
print(f"\nMean diagonal gains, young adults (18-32): 2010 {g10:.3f} -> 2023 {g23:.3f}  ({g23-g10:+.3f})")
print("Share ever-married-now among young adults (18-32), men:")
print(f"  2010 {1-np.mean([res[2010]['single_m'][b] for b in ya]):.3f}  ->  2023 {1-np.mean([res[2023]['single_m'][b] for b in ya]):.3f}")
print(f"  women: 2010 {1-np.mean([res[2010]['single_f'][b] for b in ya]):.3f}  ->  2023 {1-np.mean([res[2023]['single_f'][b] for b in ya]):.3f}")
print("\nPaper's finding (1971->1981): gains to marriage for young adults FELL substantially.")
print("If the 2010->2023 young-adult diagonal gains are negative, the method reproduces that direction.")
