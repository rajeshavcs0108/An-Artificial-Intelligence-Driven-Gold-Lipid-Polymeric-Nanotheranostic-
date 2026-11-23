import sys, json, warnings
import os; S=os.path.dirname(os.path.dirname(os.path.abspath(__file__))); sys.path.insert(0,S+"/scripts")
import numpy as np, pandas as pd
import matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt
from matplotlib.patches import Patch, Rectangle
from scipy import stats
from sklearn.decomposition import PCA
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score, roc_auc_score, average_precision_score, roc_curve, precision_recall_curve
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import RepeatedStratifiedKFold
from lifelines import KaplanMeierFitter, CoxPHFitter
from lifelines.statistics import logrank_test, multivariate_logrank_test
from style_config import RNG_SEED, OKABE_ITO as C, DPI, apply_style
warnings.filterwarnings("ignore"); apply_style(); D=S+"/data_uvm"; F=S+"/figs_uvm"; import os; os.makedirs(F,exist_ok=True)
R={}; os.makedirs(S+"/out",exist_ok=True)
def bh(ps):
    ps=np.asarray(ps,float); n=len(ps); o=np.argsort(ps); q=np.empty(n); prev=1.0
    for rank,i in enumerate(o[::-1]): k=n-rank; prev=min(prev,ps[i]*n/k); q[i]=prev
    return q
NA=["[Not Available]","[Not Applicable]","[Unknown]","[Discrepancy]"]
pt=pd.read_csv(D+"/data_clinical_patient.txt",sep="\t",comment="#",na_values=NA); sm=pd.read_csv(D+"/data_clinical_sample.txt",sep="\t",comment="#",na_values=NA)
d=sm.merge(pt,on="PATIENT_ID").set_index("SAMPLE_ID"); N=len(d); R["n"]=N
an=pd.read_csv(S+"/data_uvm_pan/data_clinical_sample.txt",sep="\t",comment="#").set_index("SAMPLE_ID")[["ANEUPLOIDY_SCORE"]]; d=d.join(an)
d["OS"]=d.OS_STATUS.str.startswith("1").astype(int); d["DFS"]=d.DFS_STATUS.str.startswith("1").astype(float).where(d.DFS_STATUS.notna())
R["clin"]={"male":int((d.SEX=="Male").sum()),"female":int((d.SEX=="Female").sum()),"age_median":float(d.AGE.median()),"age_min":float(d.AGE.min()),"age_max":float(d.AGE.max()),
 "age_q1":float(d.AGE.quantile(.25)),"age_q3":float(d.AGE.quantile(.75)),"os_events":int(d.OS.sum()),"os_median_fu":float(d.OS_MONTHS.median()),"os_max":float(d.OS_MONTHS.max()),
 "dfs_n":int(d.DFS.notna().sum()),"dfs_events":int(d.DFS.sum()),"thick_median":float(d.TUMOR_THICKNESS.median()),"diam_median":float(pd.to_numeric(d.TUMOR_BASAL_DIAMETER,errors="coerce").median()),
 "tmb_median":float(d.TMB_NONSYNONYMOUS.median()),"hist":d.HISTOLOGICAL_DIAGNOSIS.value_counts().to_dict(),"cancer_type":d.CANCER_TYPE_DETAILED.value_counts().to_dict(),
 "stage":d.AJCC_PATHOLOGIC_TUMOR_STAGE.value_counts(dropna=False).to_dict(),"dx_year":[int(d.INITIAL_PATHOLOGIC_DX_YEAR.min()),int(d.INITIAL_PATHOLOGIC_DX_YEAR.max())],
 "tumor_status":d.TUMOR_STATUS.value_counts(dropna=False).to_dict()}
# ---------- CNA seg: chr3 status ----------
seg=pd.read_csv(D+"/data_cna_hg19.seg",sep="\t"); seg["len"]=seg["loc.end"]-seg["loc.start"]; seg=seg[seg.ID.isin(d.index)]
def chrmean(cid,ch):
    x=seg[(seg.ID==cid)&(seg.chrom==ch)]; return float(np.average(x["seg.mean"],weights=x["len"]))
d["chr3_mean"]=[chrmean(i,3) for i in d.index]
d["chr3_loss_frac"]=[ (lambda x: float(x.loc[x["seg.mean"]<-0.2,"len"].sum()/x["len"].sum()))(seg[(seg.ID==i)&(seg.chrom==3)]) for i in d.index]
d["M3"]=(d.chr3_mean<=-0.2).astype(int)   # monosomy 3 threshold on length-weighted mean log2 ratio
R["chr3"]={"threshold":-0.2,"n_M3":int(d.M3.sum()),"n_D3":int((1-d.M3).sum()),"mean_M3":float(d.loc[d.M3==1,"chr3_mean"].mean()),"mean_D3":float(d.loc[d.M3==0,"chr3_mean"].mean()),
 "min_M3_gap_below":float(d.loc[d.M3==1,"chr3_mean"].max()),"max_D3":float(d.loc[d.M3==0,"chr3_mean"].max()),"min_D3":float(d.loc[d.M3==0,"chr3_mean"].min())}
cyto=d.CYTOGENETIC_ABNORMALITY_TYPE.dropna(); lab=cyto.str.contains("Chromosome 3 loss"); R["chr3"]["concordance_n"]=int(len(cyto)); R["chr3"]["concordance_agree"]=int((lab.astype(int)==d.loc[cyto.index,"M3"]).sum())
R["chr3"]["gep_class"]=d.groupby("GENE_EXPRESSION_PROFILE").M3.agg(["sum","count"]).to_dict("index")
R["chr3"]["pheo_M3"]=int(d.loc["TCGA-VD-A8KB-01","M3"])
# ---------- mutations ----------
mu=pd.read_csv(D+"/data_mutations.txt",sep="\t",comment="#",low_memory=False)
silent={"Silent","Intron","3'UTR","5'UTR","3'Flank","5'Flank","IGR","RNA","Splice_Region"}; mu=mu[~mu.Variant_Classification.isin(silent)&mu.Tumor_Sample_Barcode.isin(d.index)]
GENES=["GNAQ","GNA11","CYSLTR2","PLCB4","BAP1","SF3B1","EIF1AX","SRSF2"]
for g in GENES: d["mut_"+g]=d.index.isin(mu[mu.Hugo_Symbol==g].Tumor_Sample_Barcode).astype(int)
R["mut"]={g:{"n":int(d["mut_"+g].sum()),"pct":float(d["mut_"+g].mean()*100)} for g in GENES}
d["GQ11"]=((d.mut_GNAQ+d.mut_GNA11)>0).astype(int); R["mut"]["GNAQ_or_GNA11"]=int(d.GQ11.sum()); R["mut"]["GNAQ_and_GNA11"]=int(((d.mut_GNAQ+d.mut_GNA11)==2).sum())
R["mut"]["none_GNAQ_GNA11_CYSLTR2_PLCB4"]=int((d[["mut_GNAQ","mut_GNA11","mut_CYSLTR2","mut_PLCB4"]].sum(axis=1)==0).sum())
R["mut"]["BAP1_vs_SF3B1_EIF1AX_cooccur"]={"BAP1&SF3B1":int(((d.mut_BAP1+d.mut_SF3B1)==2).sum()),"BAP1&EIF1AX":int(((d.mut_BAP1+d.mut_EIF1AX)==2).sum()),"SF3B1&EIF1AX":int(((d.mut_SF3B1+d.mut_EIF1AX)==2).sum())}
bap=mu[mu.Hugo_Symbol=="BAP1"]; R["mut"]["BAP1_classes"]=bap.drop_duplicates(["Tumor_Sample_Barcode"]).Variant_Classification.value_counts().to_dict()
gq=mu[mu.Hugo_Symbol.isin(["GNAQ","GNA11"])]; R["mut"]["hotspot_codons"]=gq.HGVSp_Short.value_counts().head(6).to_dict() if "HGVSp_Short" in gq else {}
# fisher M3 association
fi={}
for g in GENES:
    a=int(((d["mut_"+g]==1)&(d.M3==1)).sum()); b=int(((d["mut_"+g]==1)&(d.M3==0)).sum()); c=int(((d["mut_"+g]==0)&(d.M3==1)).sum()); dd=int(((d["mut_"+g]==0)&(d.M3==0)).sum())
    orr,p=stats.fisher_exact([[a,b],[c,dd]]); fi[g]={"M3_mut":a,"D3_mut":b,"pct_M3":100*a/(a+c),"pct_D3":100*b/(b+dd),"OR":float(orr),"p":float(p)}
for g,q in zip(GENES,bh([fi[g]["p"] for g in GENES])): fi[g]["q"]=float(q)
R["fisher"]=fi
# ---------- genome-wide CNA frequency ----------
chrs=list(range(1,23)); BIN=1_000_000; sizes=seg.groupby("chrom")["loc.end"].max().to_dict()
bins=[];
for ch in chrs:
    for s in range(0,int(sizes[ch]),BIN): bins.append((ch,s))
nb=len(bins); idx={(ch,s):i for i,(ch,s) in enumerate(bins)}
M=np.zeros((N,nb)); sid=list(d.index)
for r in seg.itertuples(index=False):
    ch=r.chrom; ch=int(ch) if str(ch).isdigit() else None
    if ch is None or ch>22: continue
    i=sid.index(r.ID); s0=(int(r[2])//BIN)*BIN; s1=(int(r[3])//BIN)*BIN
    for s in range(s0,s1+1,BIN):
        if (ch,s) in idx: M[i,idx[(ch,s)]]=r[5]
bch=np.array([b[0] for b in bins]); bpos=np.array([b[1] for b in bins])
cen={1:125e6,3:93.5e6,6:60e6,8:45.6e6}
def freq(mask,kind):
    sub=M[mask]; return (sub>0.2).mean(0)*100 if kind=="gain" else (sub<-0.2).mean(0)*100
m3=(d.M3.values==1); R["cna"]={}
def armfrac(ch,arm,msk,kind):
    sel=(bch==ch)&((bpos>=cen[ch]) if arm=="q" else (bpos<cen[ch])); sub=M[msk][:,sel]
    mean=sub.mean(1); return float(((mean>0.1) if kind=="gain" else (mean<-0.1)).mean()*100), int(((mean>0.1) if kind=="gain" else (mean<-0.1)).sum())
for nm,(ch,arm,kind) in {"8q_gain":(8,"q","gain"),"6p_gain":(6,"p","gain"),"1p_loss":(1,"p","loss"),"3_loss":(3,"q","loss"),"6q_loss":(6,"q","loss")}.items():
    R["cna"][nm]={"M3":armfrac(ch,arm,m3,kind),"D3":armfrac(ch,arm,~m3,kind),"all":armfrac(ch,arm,np.ones(N,bool),kind)}
# arm-level association chr8q gain by M3 (Fisher)
sel=(bch==8)&(bpos>=cen[8]); g8=(M[:,sel].mean(1)>0.1).astype(int); d["gain8q"]=g8
a=int(((g8==1)&m3).sum()); b=int(((g8==1)&~m3).sum()); c=int(((g8==0)&m3).sum()); dd=int(((g8==0)&~m3).sum()); o,p=stats.fisher_exact([[a,b],[c,dd]]); R["cna"]["8q_vs_M3"]={"a":a,"b":b,"c":c,"d":dd,"OR":float(o),"p":float(p)}
sel6=(bch==6)&(bpos<cen[6]); d["gain6p"]=(M[:,sel6].mean(1)>0.1).astype(int)
# ---------- expression ----------
ex=pd.read_csv(D+"/data_mrna_seq_v2_rsem.txt",sep="\t").dropna(subset=["Hugo_Symbol"]).drop(columns=["Entrez_Gene_Id"]).groupby("Hugo_Symbol").mean()
ex=ex[d.index]; L=np.log2(ex+1); keep=L.mean(1)>=1.0; L=L[keep]; R["expr"]={"genes_total":int(ex.shape[0]),"genes_kept":int(L.shape[0])}
top=L.var(1).sort_values(ascending=False).index[:2000]; X=L.loc[top].T; Xz=(X-X.mean())/X.std(ddof=1)
pca=PCA(n_components=10,random_state=RNG_SEED).fit(Xz); P=pca.transform(Xz); R["pca"]={"evr":[float(v) for v in pca.explained_variance_ratio_[:5]]}
sil={}; labs={}
for k in range(2,7):
    km=KMeans(n_clusters=k,n_init=50,random_state=RNG_SEED).fit(P); sil[k]=float(silhouette_score(P,km.labels_)); labs[k]=km.labels_
R["cluster"]={"silhouette":sil,"best_k":int(max(sil,key=sil.get))}
# order clusters by M3 fraction for k=4
lab4=labs[4]; fr={c_:d.M3.values[lab4==c_].mean() for c_ in range(4)}; order=sorted(fr,key=lambda c_:fr[c_]); mp={o:i+1 for i,o in enumerate(order)}; d["C4"]=[mp[v] for v in lab4]
R["cluster"]["k4"]={int(k):{"n":int((d.C4==k).sum()),"M3":int(d.loc[d.C4==k,"M3"].sum())} for k in [1,2,3,4]}
kb=R["cluster"]["best_k"]; d["Cbest"]=labs[kb]; R["cluster"]["best_tab"]={int(k):{"n":int((d.Cbest==k).sum()),"M3":int(d.loc[d.Cbest==k,"M3"].sum())} for k in range(kb)}
R["pca"]["pc1_vs_M3_p"]=float(stats.mannwhitneyu(P[m3,0],P[~m3,0]).pvalue); R["pca"]["pc1_auc"]=float(roc_auc_score(d.M3,P[:,0]))
# ---------- DE M3 vs D3 ----------
A=L.loc[:,m3]; B=L.loc[:,~m3]; t,p=stats.ttest_ind(A,B,axis=1,equal_var=False); lfc=A.mean(1)-B.mean(1); q=bh(p)
de=pd.DataFrame({"lfc":lfc,"p":p,"q":q}); R["de"]={"tested":int(len(de)),"sig":int(((de.q<0.05)&(de.lfc.abs()>1)).sum()),"up":int(((de.q<0.05)&(de.lfc>1)).sum()),"down":int(((de.q<0.05)&(de.lfc<-1)).sum()),"q05":int((de.q<0.05).sum())}
R["de"]["top_up"]=[(g,float(r.lfc),float(r.q)) for g,r in de[de.lfc>0].sort_values("p").head(8).iterrows()]; R["de"]["top_down"]=[(g,float(r.lfc),float(r.q)) for g,r in de[de.lfc<0].sort_values("p").head(8).iterrows()]
R["de"]["chr3_genes_note"]=None
# ---------- survival ----------
sv=d.dropna(subset=["OS_MONTHS"]).copy(); R["surv"]={"n_os":int(len(sv))}
def lr(df,tc,ec,grp): 
    r=multivariate_logrank_test(df[tc],df[grp],df[ec]); return float(r.test_statistic),float(r.p_value)
R["surv"]["os_m3_logrank"]=lr(sv,"OS_MONTHS","OS","M3"); R["surv"]["os_m3_events"]={"M3":int(sv.loc[sv.M3==1,"OS"].sum()),"D3":int(sv.loc[sv.M3==0,"OS"].sum()),"nM3":int((sv.M3==1).sum()),"nD3":int((sv.M3==0).sum())}
cx=sv[["OS_MONTHS","OS","M3","AGE"]].dropna(); mdl=CoxPHFitter().fit(cx,"OS_MONTHS","OS"); s=mdl.summary.loc["M3"]; R["surv"]["os_cox_M3_ageadj"]={"HR":float(np.exp(s.coef)),"lo":float(s["exp(coef) lower 95%"]),"hi":float(s["exp(coef) upper 95%"]),"p":float(s.p)}
fd=d.dropna(subset=["DFS","DFS_MONTHS"]).copy(); R["surv"]["dfs_n"]=int(len(fd)); R["surv"]["dfs_m3_logrank"]=lr(fd,"DFS_MONTHS","DFS","M3"); R["surv"]["dfs_events"]={"M3":int(fd.loc[fd.M3==1,"DFS"].sum()),"D3":int(fd.loc[fd.M3==0,"DFS"].sum()),"nM3":int((fd.M3==1).sum()),"nD3":int((fd.M3==0).sum())}
cd=fd[["DFS_MONTHS","DFS","M3","AGE"]].dropna(); m2=CoxPHFitter().fit(cd,"DFS_MONTHS","DFS"); s=m2.summary.loc["M3"]; R["surv"]["dfs_cox_M3_ageadj"]={"HR":float(np.exp(s.coef)),"lo":float(s["exp(coef) lower 95%"]),"hi":float(s["exp(coef) upper 95%"]),"p":float(s.p)}
R["surv"]["os_k4_logrank"]=lr(sv.join(d[["C4"]],rsuffix="_x") if "C4" not in sv else sv,"OS_MONTHS","OS","C4")
def rate(df,tc,ec,g,tm=36):
    k=KaplanMeierFitter().fit(df.loc[df[g]==1,tc],df.loc[df[g]==1,ec]); return float(k.predict(tm))
R["surv"]["os36"]={"M3":rate(sv,"OS_MONTHS","OS","M3"),"D3":float(KaplanMeierFitter().fit(sv.loc[sv.M3==0,"OS_MONTHS"],sv.loc[sv.M3==0,"OS"]).predict(36))}
# clinical by chr3
cl={}
for v in ["AGE","TUMOR_THICKNESS","TUMOR_BASAL_DIAMETER","TMB_NONSYNONYMOUS","MITOTIC_COUNT","ANEUPLOIDY_SCORE"]:
    x=pd.to_numeric(d[v],errors="coerce"); a=x[d.M3==1].dropna(); b=x[d.M3==0].dropna(); u=stats.mannwhitneyu(a,b)
    cl[v]={"med_M3":float(a.median()),"med_D3":float(b.median()),"n_M3":int(len(a)),"n_D3":int(len(b)),"p":float(u.pvalue)}
R["clin_by_chr3"]=cl
for nm,vv in {"epith":d.HISTOLOGICAL_DIAGNOSIS.fillna("").str.contains("Epithelioid"),"ese":d.EXTRASCLERAL_EXTENSION.eq("YES")}.items():
    a=int((vv&(d.M3==1)).sum()); b=int((vv&(d.M3==0)).sum()); c=int((~vv&(d.M3==1)).sum()); e=int((~vv&(d.M3==0)).sum()); o,p=stats.fisher_exact([[a,b],[c,e]]); R["clin_by_chr3"][nm]={"M3":a,"D3":b,"OR":float(o),"p":float(p)}
# ---------- ML: classify M3 ----------
y=d.M3.values; Lt=L.T.loc[d.index]
clinc=pd.DataFrame({"age":d.AGE,"thick":d.TUMOR_THICKNESS,"diam":pd.to_numeric(d.TUMOR_BASAL_DIAMETER,errors="coerce"),"male":(d.SEX=="Male").astype(int),"epith":d.HISTOLOGICAL_DIAGNOSIS.fillna("").str.contains("Epithelioid").astype(int)}); clinc=clinc.fillna(clinc.median())
mutc=d[["mut_"+g for g in GENES]].copy()
def run(feat,name,topk=None):
    rs=RepeatedStratifiedKFold(n_splits=5,n_repeats=5,random_state=RNG_SEED); acc=np.zeros(N); cn=np.zeros(N); aucs=[]; aps=[]
    for tr,te in rs.split(np.zeros(N),y):
        Xtr,Xte=feat.iloc[tr],feat.iloc[te]
        if topk:
            tt,pp=stats.ttest_ind(Xtr[y[tr]==1],Xtr[y[tr]==0],axis=0,equal_var=False); cols=Xtr.columns[np.argsort(pp)[:topk]]; Xtr,Xte=Xtr[cols],Xte[cols]
        sc=StandardScaler().fit(Xtr); lrm=LogisticRegression(C=0.1 if topk else 1.0,max_iter=5000,random_state=RNG_SEED).fit(sc.transform(Xtr),y[tr]); pr=lrm.predict_proba(sc.transform(Xte))[:,1]
        acc[te]+=pr; cn[te]+=1; aucs.append(roc_auc_score(y[te],pr)); aps.append(average_precision_score(y[te],pr))
    return {"name":name,"oof":acc/cn,"auc_mean":float(np.mean(aucs)),"auc_sd":float(np.std(aucs,ddof=1)),"ap_mean":float(np.mean(aps)),"ap_sd":float(np.std(aps,ddof=1))}
MODELS=[run(clinc,"Clinical (5 variables)"),run(mutc,"Driver mutations (8 genes)"),run(Lt,"Expression (top 50 genes, in-fold)",50),run(pd.concat([Lt,mutc],axis=1),"Expression + mutations (top 50, in-fold)",50)]
# the combined set: mutation columns could be dropped in top-k selection, acceptable (selected in-fold)
R["ml"]={"prevalence":float(y.mean()),"models":[{k:v for k,v in m.items() if k!="oof"} for m in MODELS]}
for m in MODELS: m["oof_auc"]=float(roc_auc_score(y,m["oof"])); m["oof_ap"]=float(average_precision_score(y,m["oof"]))
R["ml"]["oof"]={m["name"]:{"auc":m["oof_auc"],"ap":m["oof_ap"]} for m in MODELS}
# gene selection frequency (full data top 20 by t-test) for report
tt,pp=stats.ttest_ind(Lt[y==1],Lt[y==0],axis=0,equal_var=False); R["ml"]["top_genes_full"]=[(g,float(Lt[g][y==1].mean()-Lt[g][y==0].mean())) for g in Lt.columns[np.argsort(pp)[:10]]]
# ---------- candidate nanocarrier-relevant genes ----------
CAND=["TFRC","FOLR1","EGFR","CD44","ITGAV","ITGB3","ICAM1","SCARB1","LRP1","SLC2A1","MET","IGF1R","CD274","MLANA","PMEL","TYR","PRAME","HIF1A","VEGFA","BCL2"]
CAND=[g for g in CAND if g in L.index]; R["cand"]={"genes":CAND,"missing":[g for g in ["TFRC","FOLR1","EGFR","CD44","ITGAV","ITGB3","ICAM1","SCARB1","LRP1","SLC2A1","MET","IGF1R","CD274","MLANA","PMEL","TYR","PRAME","HIF1A","VEGFA","BCL2"] if g not in L.index]}
cp=[stats.mannwhitneyu(L.loc[g,m3],L.loc[g,~m3]).pvalue for g in CAND]; cq=bh(cp)
R["cand"]["tab"]={g:{"mean_D3":float(L.loc[g,~m3].mean()),"mean_M3":float(L.loc[g,m3].mean()),"lfc":float(L.loc[g,m3].mean()-L.loc[g,~m3].mean()),"p":float(p_),"q":float(q_),"mean_all":float(L.loc[g].mean())} for g,p_,q_ in zip(CAND,cp,cq)}
json.dump(R,open(S+"/out/uvm_results.json","w"),indent=1,default=lambda o: float(o) if isinstance(o,np.floating) else int(o) if isinstance(o,np.integer) else str(o))
d.to_csv(S+"/out/uvm_cohort.csv"); np.save(S+"/out/uvm_M.npy",M); np.save(S+"/out/uvm_P.npy",P); de.to_csv(S+"/out/uvm_de.csv")
import pickle; pickle.dump({"MODELS":MODELS,"y":y,"bch":bch,"bpos":bpos,"sil":sil,"labs":labs},open(S+"/out/uvm_aux.pkl","wb"))
print("OK")
