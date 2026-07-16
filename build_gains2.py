"""
Choo-Siow (2006) on ACS 2023 PUMS, upgraded:
  (b) TYPES = age-band x SES-tercile  -> age-assortative matching is now endogenous.
  (a) REAL per-metro supply vectors    -> each US city gets its own age x SES x sex
      single/available population, mapped county -> PUMA via the Census crosswalk.

Estimation is national (16 states, PWGTP-weighted). pi is structural, so the app
holds it fixed and re-solves the equilibrium under each metro's ACTUAL supply
composition (sex ratio AND age/SES mix), not a scaled sex ratio.
"""
import json, glob, numpy as np, pandas as pd
import choo_siow

SCRATCH = "data"   # local folder holding ACS PUMS CSVs (in data/pums/) + tract_puma.txt (Census crosswalk)
FILES = sorted(glob.glob(f"{SCRATCH}/pums/psam_p*.csv"))

# ---- type structure: 6 age bands x 3 SES terciles = 18 types per sex ----
AGE_BANDS = [(18,25),(26,30),(31,35),(36,42),(43,52),(53,70)]   # inclusive
NB, NS = len(AGE_BANDS), 3
NT = NB*NS
def age_band(a):
    for k,(lo,hi) in enumerate(AGE_BANDS):
        if a<=hi: return k
    return NB-1

# app-consistent SES sub-scores (income + education), same mappings as index.html
INCOME_PTS=[(3000,2),(10300,10),(28000,25),(50200,50),(88700,75),(150000,90),(201000,95),(430000,99),(1000000,99.8)]
EDU_MAP={'none':6,'highschool':24,'some':44,'associate':55,'bachelor':72,'master':86,'professional':94}
def interp(x,pts): return np.interp(x,[p[0] for p in pts],[p[1] for p in pts])
def schl_sub(schl):
    s=schl.fillna(0).astype(float)
    key=np.select([s<=15,s.isin([16,17]),s.isin([18,19]),s==20,s==21,s==22,s>=23],
        ['none','highschool','some','associate','bachelor','master','professional'],default='highschool')
    return np.array([EDU_MAP[k] for k in key])
WM=np.array([.193,.050]); WM/=WM.sum()      # men's SES weights (income,edu)
WF=np.array([.063,.036]); WF/=WF.sum()      # women's
def ses_index(df,male):
    inc=interp(np.clip(df['PINCP'].values,0,None),INCOME_PTS); edu=schl_sub(df['SCHL'])
    W=WM if male else WF
    return W[0]*inc+W[1]*edu
def wq(vals,w,qs):
    o=np.argsort(vals); v=vals[o]; cw=np.cumsum(w[o])/w[o].sum()
    return v[np.searchsorted(cw,qs)]

# ---- metros -> county FIPS (STATEFP, COUNTYFP) ----
METRO_COUNTIES = {
 "San Jose / Silicon Valley, CA":[("06","085")],
 "San Francisco, CA":[("06","075"),("06","081")],
 "Seattle, WA":[("53","033")], "Denver, CO":[("08","031")],
 "Austin, TX":[("48","453")], "Portland, OR":[("41","051")],
 "San Diego, CA":[("06","073")], "Phoenix, AZ":[("04","013")],
 "Dallas, TX":[("48","113")], "Houston, TX":[("48","201")],
 "Los Angeles, CA":[("06","037")], "Minneapolis, MN":[("27","053")],
 "Chicago, IL":[("17","031")], "Miami, FL":[("12","086")],
 "Boston, MA":[("25","025"),("25","017")], "Philadelphia, PA":[("42","101")],
 "New York, NY":[("36","061"),("36","047"),("36","081"),("36","005"),("36","085")],
 "Washington, DC":[("11","001")], "Atlanta, GA":[("13","121"),("13","089")],
 "Baltimore, MD":[("24","510")], "Memphis, TN":[("47","157")],
}
# county -> set of PUMAs from the tract->PUMA crosswalk
cw=pd.read_csv(f"{SCRATCH}/tract_puma.txt",dtype=str)
cw.columns=[c.strip().lstrip('﻿') for c in cw.columns]
county_pumas={}
for _,r in cw.iterrows():
    county_pumas.setdefault((r['STATEFP'],r['COUNTYFP']),set()).add(r['PUMA5CE'])
METRO_PUMAS={m:(cs[0][0], set().union(*[county_pumas.get(c,set()) for c in cs])) for m,cs in METRO_COUNTIES.items()}

# ---- load all states; keep type + geo ----
cols=['SERIALNO','SEX','AGEP','MAR','SCHL','PINCP','RELSHIPP','PWGTP','PUMA']
frames=[]
for fp in FILES:
    st=fp.split('psam_p')[1][:2]                      # state FIPS from filename
    d=pd.read_csv(fp,usecols=cols,dtype={'SERIALNO':str,'PUMA':str},na_values=[''],low_memory=False)
    for c in ['SEX','AGEP','MAR','SCHL','PINCP','RELSHIPP','PWGTP']: d[c]=pd.to_numeric(d[c],errors='coerce')
    d=d[(d['AGEP']>=18)&(d['AGEP']<=70)&(d['PWGTP']>0)].copy()
    d['STFIP']=st; d['PUMA']=d['PUMA'].str.zfill(5)
    frames.append(d)
df=pd.concat(frames,ignore_index=True)
print(f"pooled adults 18-70: {len(df):,}  across {len(FILES)} states")

# SES tercile cutpoints per sex, then type = band*3 + tercile
men=df['SEX']==1; wom=df['SEX']==2
df.loc[men,'ses']=ses_index(df[men],True); df.loc[wom,'ses']=ses_index(df[wom],False)
em=wq(df.loc[men,'ses'].values,df.loc[men,'PWGTP'].values,np.array([1/3,2/3]))
ef=wq(df.loc[wom,'ses'].values,df.loc[wom,'PWGTP'].values,np.array([1/3,2/3]))
df['sbin']=np.where(df['SEX']==1,np.searchsorted(em,df['ses'].values),np.searchsorted(ef,df['ses'].values))
df['band']=df['AGEP'].map(age_band)
df['type']=(df['band']*NS+df['sbin']).astype(int)

# ---- national matching: singles + linked married couples ----
single=df[df['MAR']!=1]
def byType(sub,sexval):
    v=np.zeros(NT)
    s=sub[sub['SEX']==sexval]
    for t,w in zip(s['type'],s['PWGTP']): v[t]+=w
    return v
mu_i0=byType(single,1); mu_0j=byType(single,2)
ref=df[df['RELSHIPP']==20][['SERIALNO','SEX','type','PWGTP']].rename(columns={'SEX':'sr','type':'tr','PWGTP':'wr'})
sp =df[df['RELSHIPP']==21][['SERIALNO','SEX','type','PWGTP']].rename(columns={'SEX':'ss','type':'ts','PWGTP':'ws'})
cpl=ref.merge(sp,on='SERIALNO'); cpl=cpl[cpl['sr']!=cpl['ss']]
cpl['ht']=np.where(cpl['sr']==1,cpl['tr'],cpl['ts']); cpl['wt']=np.where(cpl['sr']==2,cpl['tr'],cpl['ts'])
cpl['w']=(cpl['wr']+cpl['ws'])/2
mu_ij=np.zeros((NT,NT))
for h,wt,w in zip(cpl['ht'],cpl['wt'],cpl['w']): mu_ij[h,wt]+=w
print(f"national married couples linked: {len(cpl):,}")
mu_ij=np.maximum(mu_ij,1.0)

est=choo_siow.estimate_gains(mu_ij,mu_i0,mu_0j); pi=est['pi']
m_nat=mu_i0+mu_ij.sum(1); f_nat=mu_0j+mu_ij.sum(0)
sol=choo_siow.solve_equilibrium(pi,m_nat,f_nat)
print(f"national equilibrium recover rel.err {np.max(np.abs(sol['mu_ij']-mu_ij))/mu_ij.max():.1e}")

# ---- per-metro available population by type (all adults, both sexes) ----
def metro_supply(name):
    st,pumas=METRO_PUMAS[name]
    sub=df[(df['STFIP']==st)&(df['PUMA'].isin(pumas))]
    return byType(sub,1).tolist(), byType(sub,2).tolist(), int(len(sub))
metros={}
for name in METRO_COUNTIES:
    m,f,n=metro_supply(name); metros[name]={'m':[round(x,1) for x in m],'f':[round(x,1) for x in f],'n':n}

out=dict(
  kind="agexses", nbands=NB, nses=NS, ntypes=NT,
  age_bands=AGE_BANDS,
  ses_edges_men=[round(float(x),2) for x in em], ses_edges_women=[round(float(x),2) for x in ef],
  weights_men=dict(income=float(WM[0]),education=float(WM[1])),
  weights_women=dict(income=float(WF[0]),education=float(WF[1])),
  pi=np.round(pi,4).tolist(),
  m_nat=[round(x,1) for x in m_nat], f_nat=[round(x,1) for x in f_nat],
  metros=metros,
  source="ACS 2023 1-year PUMS, 16 states, PWGTP-weighted; metros via county->PUMA crosswalk",
)
json.dump(out,open("gains_us2.json","w"))
print(f"wrote gains_us2.json  ({NT} types, {len(metros)} metros)")

# diagnostics: single rate by age band (SES-avg) — should trace an age curve
base=choo_siow.solve_equilibrium(pi,m_nat,f_nat)
mrate=base['mu_i0']/m_nat; frate=base['mu_0j']/f_nat
print("\nnational single-rate by age band (avg over SES):")
print(f"{'ageband':>10} {'men':>6} {'women':>6}")
for b,(lo,hi) in enumerate(AGE_BANDS):
    idx=[b*NS+s for s in range(NS)]
    print(f"{str(lo)+'-'+str(hi):>10} {mrate[idx].mean():>6.2f} {frate[idx].mean():>6.2f}")
print("\nmetro sample sizes:", {k:v['n'] for k,v in list(metros.items())[:6]})
