import sys, json, pickle, warnings
import os; S=os.path.dirname(os.path.dirname(os.path.abspath(__file__))); sys.path.insert(0,S+"/scripts")
import numpy as np, pandas as pd
import matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt
from matplotlib.patches import Patch, Rectangle
from matplotlib.lines import Line2D
from sklearn.metrics import roc_curve, precision_recall_curve
from lifelines import KaplanMeierFitter
from style_config import OKABE_ITO as C, DPI, apply_style
warnings.filterwarnings("ignore"); apply_style(); F=S+"/figs_uvm"
R=json.load(open(S+"/out/uvm_results.json")); d=pd.read_csv(S+"/out/uvm_cohort.csv",index_col=0); M=np.load(S+"/out/uvm_M.npy"); P=np.load(S+"/out/uvm_P.npy"); de=pd.read_csv(S+"/out/uvm_de.csv",index_col=0)
aux=pickle.load(open(S+"/out/uvm_aux.pkl","rb")); MODELS=aux["MODELS"]; y=aux["y"]; bch=aux["bch"]; bpos=aux["bpos"]; sil=aux["sil"]
cD3,cM3=C[5],C[6]   # blue, vermillion
def save(fig,n): fig.savefig(f"{F}/{n}.png",dpi=DPI,bbox_inches="tight"); plt.close(fig)
# ---- Fig 1 cohort ----
fig,axs=plt.subplots(2,3,figsize=(9.6,6.0))
ax=axs[0,0]; h=d.HISTOLOGICAL_DIAGNOSIS.fillna("NA").map(lambda s:"Spindle" if s=="Spindle Cell" else "Epithelioid" if s=="Epithelioid Cell" else "Mixed"); v=h.value_counts().reindex(["Spindle","Mixed","Epithelioid"])
ax.barh(v.index,v.values,color=[C[4],C[1],C[6]]); [ax.text(x+0.5,i,str(x),va="center",fontsize=8) for i,x in enumerate(v.values)]; ax.set_title("Histological cell type"); ax.set_xlabel("Patients"); ax.invert_yaxis()
ax=axs[0,1]; st=d.AJCC_PATHOLOGIC_TUMOR_STAGE.dropna().str.replace("Stage ","").value_counts().reindex(["IIA","IIB","IIIA","IIIB","IIIC","IV"]).fillna(0)
ax.plot(st.index,st.values,"o-",color=C[5]); [ax.text(i,x+0.8,str(int(x)),ha="center",fontsize=8) for i,x in enumerate(st.values)]; ax.set_title("AJCC pathologic stage (7th ed.)"); ax.set_ylabel("Patients"); ax.set_ylim(0,32)
ax=axs[0,2]; ax.hist(d.AGE,bins=np.arange(20,90,5),color=C[4],edgecolor="white"); ax.axvline(d.AGE.median(),color="k",ls="--",lw=1); ax.set_title("Age at diagnosis"); ax.set_xlabel("Years"); ax.set_ylabel("Patients"); ax.text(0.03,0.93,f"median {d.AGE.median():.1f}",transform=ax.transAxes,fontsize=8,va="top")
ax=axs[1,0]; dd=pd.to_numeric(d.TUMOR_BASAL_DIAMETER,errors="coerce")
for g,col,lab in [(0,cD3,"Disomy 3"),(1,cM3,"Monosomy 3")]:
    k=d.M3==g; ax.scatter(dd[k],d.TUMOR_THICKNESS[k],s=22,color=col,alpha=.85,label=lab,edgecolor="white",lw=.4)
ax.set_xlabel("Largest basal diameter (mm)"); ax.set_ylabel("Tumour thickness (mm)"); ax.set_title("Tumour dimensions"); ax.legend(frameon=False,fontsize=7.5)
ax=axs[1,1]; ax.hist(d.chr3_mean,bins=np.linspace(-1.2,0.2,29),color=C[2],edgecolor="white"); ax.axvline(-0.2,color="k",ls="--",lw=1); ax.set_xlabel("Chromosome 3 mean log2 copy ratio"); ax.set_ylabel("Patients"); ax.set_title("Chromosome 3 status"); ax.set_ylim(0,21)
ax.text(0.03,0.95,f"Monosomy 3 (n = {R['chr3']['n_M3']})\nthreshold: -0.2",transform=ax.transAxes,ha="left",fontsize=7.5,va="top"); ax.text(0.97,0.95,f"Disomy 3\n(n = {R['chr3']['n_D3']})",transform=ax.transAxes,ha="right",fontsize=7.5,va="top")
ax=axs[1,2]; sv=d.dropna(subset=["OS_MONTHS"]); k=KaplanMeierFitter().fit(sv.OS_MONTHS,sv.OS,label=f"Overall (n = {len(sv)})"); k.plot_survival_function(ax=ax,color=C[0],ci_show=True); ax.set_title("Overall survival"); ax.set_xlabel("Months"); ax.set_ylabel("Survival probability"); ax.set_ylim(0,1.02); ax.legend(frameon=False,fontsize=7.5,loc="lower left")
fig.tight_layout(); save(fig,"fig1_cohort")
# ---- Fig 2 oncoprint ----
G=["GNAQ","GNA11","CYSLTR2","PLCB4","BAP1","SF3B1","EIF1AX","SRSF2"]
mu=pd.read_csv(S+"/data_uvm/data_mutations.txt",sep="\t",comment="#",low_memory=False); silent={"Silent","Intron","3'UTR","5'UTR","3'Flank","5'Flank","IGR","RNA","Splice_Region"}; mu=mu[~mu.Variant_Classification.isin(silent)&mu.Tumor_Sample_Barcode.isin(d.index)]
cls=lambda v:"Missense" if v=="Missense_Mutation" else "Truncating" if v in("Frame_Shift_Del","Frame_Shift_Ins","Nonsense_Mutation") else "Splice site" if v=="Splice_Site" else "In-frame"
col={"Missense":C[2],"Truncating":C[6],"Splice site":C[1],"In-frame":C[4]}
key=lambda s:tuple(-int(d.loc[s,"mut_"+g]) for g in ["BAP1","SF3B1","EIF1AX","GNA11","GNAQ"])
order=sorted(d.index,key=lambda s:(d.loc[s,"M3"]*-1,)+key(s)); idx={s:i for i,s in enumerate(order)}
fig,ax=plt.subplots(figsize=(9.6,4.6)); rows=G+["Monosomy 3","Chr 8q gain","Chr 6p gain"]
for j,r in enumerate(rows):
    for s in order: ax.add_patch(Rectangle((idx[s],j),0.9,0.8,color="#EDEDED",lw=0))
for j,g in enumerate(G):
    for r in mu[mu.Hugo_Symbol==g].itertuples():
        ax.add_patch(Rectangle((idx[r.Tumor_Sample_Barcode],j),0.9,0.8,color=col[cls(r.Variant_Classification)],lw=0))
for j,(nm,c_,cc) in enumerate([("M3","M3",C[6]),("gain8q","gain8q",C[1]),("gain6p","gain6p",C[5])],start=len(G)):
    for s in order:
        if d.loc[s,c_]==1: ax.add_patch(Rectangle((idx[s],j),0.9,0.8,color=cc,lw=0))
ax.set_xlim(0,len(order)); ax.set_ylim(len(rows),0); ax.set_yticks(np.arange(len(rows))+0.4)
ax.set_yticklabels([f"{g}   {R['mut'][g]['pct']:.1f}%" for g in G]+[f"Monosomy 3   {R['chr3']['n_M3']/80*100:.1f}%",f"Chr 8q gain   {d.gain8q.mean()*100:.1f}%",f"Chr 6p gain   {d.gain6p.mean()*100:.1f}%"],fontsize=8)
ax.set_xticks([]); [s.set_visible(False) for s in ax.spines.values()]; ax.tick_params(length=0); ax.set_xlabel(f"Patients (n = {len(order)}), ordered by chromosome 3 status and driver mutation")
ax.legend(handles=[Patch(color=v,label=k) for k,v in col.items()],frameon=False,ncol=4,loc="upper center",bbox_to_anchor=(0.5,-0.06),fontsize=8)
save(fig,"fig2_oncoprint")
# ---- Fig 3 CNA landscape ----
m3=d.M3.values==1; fig,axs=plt.subplots(2,1,figsize=(9.6,5.2),sharex=True,sharey=True)
chrs=list(range(1,23)); off={}; o=0
for ch in chrs: off[ch]=o; o+=(bpos[bch==ch].max()+1e6)
x=np.array([off[c_]+p for c_,p in zip(bch,bpos)])/1e6
for ax,mask,ttl,cc in [(axs[0],~m3,f"Disomy 3 (n = {(~m3).sum()})",cD3),(axs[1],m3,f"Monosomy 3 (n = {m3.sum()})",cM3)]:
    sub=M[mask]; g=(sub>0.2).mean(0)*100; l=(sub<-0.2).mean(0)*100
    ax.fill_between(x,0,g,color=C[1],step="mid",lw=0,label="Gain"); ax.fill_between(x,0,-l,color=C[5],step="mid",lw=0,label="Loss"); ax.axhline(0,color="k",lw=.6)
    ax.set_title(ttl,loc="left",fontsize=9,color=cc,fontweight="bold"); ax.set_ylabel("Frequency (%)")
for ch in chrs:
    for ax in axs: ax.axvline(off[ch]/1e6,color="#BBBBBB",lw=.4)
mids=[(off[ch]+bpos[bch==ch].max()/2)/1e6 for ch in chrs]; axs[1].set_xticks(mids); axs[1].set_xticklabels(chrs,fontsize=7); axs[1].set_xlabel("Chromosome"); axs[1].set_xlim(0,x.max()); axs[1].set_ylim(-105,105)
axs[0].legend(frameon=False,ncol=2,loc="upper right",fontsize=8); fig.tight_layout(); save(fig,"fig3_cna")
# ---- Fig 4 driver x chr3 dumbbell ----
fi=R["fisher"]; fig,ax=plt.subplots(figsize=(7.4,4.4)); gs=["BAP1","EIF1AX","SF3B1","GNAQ","GNA11","CYSLTR2","PLCB4","SRSF2"]; yy=np.arange(len(gs))[::-1]
for yv,g in zip(yy,gs):
    a,b=fi[g]["pct_D3"],fi[g]["pct_M3"]; ax.plot([a,b],[yv,yv],color="#999",lw=1.2,zorder=1); ax.scatter(a,yv,color=cD3,s=42,zorder=3); ax.scatter(b,yv,color=cM3,s=42,zorder=3)
    ax.text(103,yv,f"q = {fi[g]['q']:.3g}",va="center",fontsize=7.5)
ax.set_yticks(yy); ax.set_yticklabels(gs); ax.set_xlim(-3,115); ax.set_xticks([0,20,40,60,80,100]); ax.set_xlabel("Tumours carrying a non-silent somatic mutation (%)"); ax.text(103,len(gs)-0.35,"Fisher, BH",fontsize=7.5,fontweight="bold")
ax.legend(handles=[Line2D([],[],marker="o",ls="",color=cD3,label=f"Disomy 3 (n = {R['chr3']['n_D3']})"),Line2D([],[],marker="o",ls="",color=cM3,label=f"Monosomy 3 (n = {R['chr3']['n_M3']})")],frameon=False,loc="upper center",bbox_to_anchor=(0.4,-0.14),ncol=2,fontsize=8)
save(fig,"fig4_drivers")
# ---- Fig 5 PCA + clusters ----
fig,axs=plt.subplots(1,3,figsize=(10.4,3.7)); ev=R["pca"]["evr"]
ax=axs[0]
for g,c_,l in [(0,cD3,"Disomy 3"),(1,cM3,"Monosomy 3")]: k=d.M3.values==g; ax.scatter(P[k,0],P[k,1],s=24,color=c_,label=l,edgecolor="white",lw=.4)
ax.set_xlabel(f"PC1 ({ev[0]*100:.1f}%)"); ax.set_ylabel(f"PC2 ({ev[1]*100:.1f}%)"); ax.set_title("Chromosome 3 status"); ax.legend(frameon=False,fontsize=7.5)
ax=axs[1]; k4=d.C4.values; pal={1:C[5],2:C[2],3:C[1],4:C[6]}
for k in [1,2,3,4]: ax.scatter(P[k4==k,0],P[k4==k,1],s=24,color=pal[k],label=f"E{k} (n = {(k4==k).sum()})",edgecolor="white",lw=.4)
ax.set_xlabel(f"PC1 ({ev[0]*100:.1f}%)"); ax.set_ylabel(f"PC2 ({ev[1]*100:.1f}%)"); ax.set_title("k-means, k = 4"); ax.legend(frameon=False,fontsize=7.5)
ax=axs[2]; ks=sorted(map(int,sil)); ax.plot(ks,[sil[str(k)] if str(k) in sil else sil[k] for k in ks],"o-",color=C[0]); ax.set_xlabel("Number of clusters (k)"); ax.set_ylabel("Mean silhouette width"); ax.set_title("Cluster-number selection"); ax.set_xticks(ks)
fig.tight_layout(); save(fig,"fig5_pca")
# ---- Fig 6 KM ----
fig,axs=plt.subplots(1,3,figsize=(10.4,3.6))
def km(ax,df,tc,ec,grp,labels,colors,ttl,xl=None):
    for g,l,c_ in zip(labels[0],labels[1],colors):
        s=df[df[grp]==g]; KaplanMeierFitter().fit(s[tc],s[ec],label=f"{l} (n = {len(s)}, events = {int(s[ec].sum())})").plot_survival_function(ax=ax,color=c_,ci_show=False,show_censors=True,censor_styles={"ms":3})
    ax.set_title(ttl,fontsize=9); ax.set_xlabel("Months"); ax.set_ylim(0,1.03); ax.legend(frameon=False,fontsize=6.8,loc="upper center",bbox_to_anchor=(0.5,-0.2))
sv=d.dropna(subset=["OS_MONTHS"]); fd=d.dropna(subset=["DFS","DFS_MONTHS"])
l1=R["surv"]["os_m3_logrank"]; km(axs[0],sv,"OS_MONTHS","OS","M3",([0,1],["Disomy 3","Monosomy 3"]),[cD3,cM3],f"Overall survival by chromosome 3\nlog-rank chi2 = {l1[0]:.2f}, p = {l1[1]:.4f}"); axs[0].set_ylabel("Survival probability")
l2=R["surv"]["dfs_m3_logrank"]; km(axs[1],fd,"DFS_MONTHS","DFS","M3",([0,1],["Disomy 3","Monosomy 3"]),[cD3,cM3],f"Disease-free survival by chromosome 3\nlog-rank chi2 = {l2[0]:.2f}, p = {l2[1]:.4f}")
l3=R["surv"]["os_k4_logrank"]; km(axs[2],sv,"OS_MONTHS","OS","C4",([1,2,3,4],["E1","E2","E3","E4"]),[pal[1],pal[2],pal[3],pal[4]],f"Overall survival by expression cluster\nlog-rank chi2 = {l3[0]:.2f}, p = {l3[1]:.1e}")
fig.tight_layout(); save(fig,"fig6_km")
# ---- Fig 7 volcano ----
fig,ax=plt.subplots(figsize=(6.6,4.9)); lq=-np.log10(de.q.clip(lower=1e-300)); sig=(de.q<0.05)&(de.lfc.abs()>1)
ax.scatter(de.lfc[~sig],lq[~sig],s=3,color="#BBBBBB",alpha=.6,lw=0,rasterized=True); up=sig&(de.lfc>0); dn=sig&(de.lfc<0)
ax.scatter(de.lfc[up],lq[up],s=5,color=cM3,lw=0,label=f"Higher in monosomy 3 (n = {R['de']['up']})",rasterized=True); ax.scatter(de.lfc[dn],lq[dn],s=5,color=cD3,lw=0,label=f"Higher in disomy 3 (n = {R['de']['down']})",rasterized=True)
ds=de[sig].assign(lq=lq[sig]); lab=pd.concat([ds[ds.lfc>0].sort_values("lq",ascending=False).head(3),ds[ds.lfc<0].sort_values("lq",ascending=False).head(3)])
for g,r in lab.iterrows(): ax.annotate(g,(r.lfc,r.lq),fontsize=7,xytext=(-4 if r.lfc<0 else 4,-9 if g in ("RAF1","SLC25A26") else 3),textcoords="offset points",ha="right" if r.lfc<0 else "left")
ax.axvline(1,color="k",ls=":",lw=.7); ax.axvline(-1,color="k",ls=":",lw=.7); ax.axhline(-np.log10(0.05),color="k",ls=":",lw=.7)
ax.set_xlabel("log2 fold change (monosomy 3 minus disomy 3)"); ax.set_ylabel("-log10 BH q-value"); ax.legend(frameon=False,fontsize=8,loc="upper center",bbox_to_anchor=(0.5,-0.14),ncol=1); save(fig,"fig7_volcano")
# ---- Fig 8 ROC/PR ----
fig,axs=plt.subplots(1,2,figsize=(8.4,3.8)); cl=[C[4],C[1],C[5],C[6]]; ls=["-","--","-.",":"]; nm={"Clinical (5 variables)":"Clinical","Driver mutations (8 genes)":"Driver mutations","Expression (top 50 genes, in-fold)":"Expression","Expression + mutations (top 50, in-fold)":"Expression + mutations"}
for i,m in enumerate(MODELS):
    f,t,_=roc_curve(y,m["oof"]); axs[0].plot(f,t,ls[i],color=cl[i],lw=1.7,label=f"{nm[m['name']]} (AUROC {m['oof_auc']:.3f})")
    p,r,_=precision_recall_curve(y,m["oof"]); axs[1].plot(r,p,ls[i],color=cl[i],lw=1.7,label=f"{nm[m['name']]} (AP {m['oof_ap']:.3f})")
axs[0].plot([0,1],[0,1],color="grey",ls=":",lw=.8); axs[0].set_xlabel("False positive rate"); axs[0].set_ylabel("True positive rate"); axs[0].set_title("ROC")
axs[1].axhline(y.mean(),color="grey",ls=":",lw=.8); axs[1].set_xlabel("Recall"); axs[1].set_ylabel("Precision"); axs[1].set_title("Precision-recall"); axs[1].set_ylim(0.4,1.03)
for a in axs: a.legend(frameon=False,fontsize=7,loc="upper center",bbox_to_anchor=(0.5,-0.2))
fig.tight_layout(); save(fig,"fig8_roc")
# ---- Fig 9 candidate targets ----
ct=R["cand"]["tab"]; gl=sorted(ct,key=lambda g:ct[g]["lfc"],reverse=True); L=len(gl)
fig,axs=plt.subplots(1,2,figsize=(8.8,5.2),gridspec_kw={"width_ratios":[1.1,1]},sharey=True)
mat=np.array([[ct[g]["mean_D3"],ct[g]["mean_M3"]] for g in gl]); im=axs[0].imshow(mat,cmap="viridis",aspect="auto"); axs[0].set_xticks([0,1]); axs[0].set_xticklabels(["Disomy 3","Monosomy 3"]); axs[0].set_yticks(range(L)); axs[0].set_yticklabels(gl)
for i in range(L):
    for j in range(2): axs[0].text(j,i,f"{mat[i,j]:.2f}",ha="center",va="center",fontsize=7,color="white" if mat[i,j]<np.median(mat) else "black")
cb=fig.colorbar(im,ax=axs[0],fraction=0.06,pad=0.03); cb.set_label("Mean log2(RSEM + 1)",fontsize=8)
yy=np.arange(L); cols=[cM3 if ct[g]["q"]<0.05 and ct[g]["lfc"]>0 else cD3 if ct[g]["q"]<0.05 else "#AAAAAA" for g in gl]
axs[1].barh(yy,[ct[g]["lfc"] for g in gl],color=cols,height=.65); axs[1].axvline(0,color="k",lw=.7); axs[1].set_xlabel("log2 fold change (M3 minus D3)")
for i,g in enumerate(gl): axs[1].text(ct[g]["lfc"]+(0.05 if ct[g]["lfc"]>=0 else -0.05),i,f"q={ct[g]['q']:.2g}",va="center",ha="left" if ct[g]["lfc"]>=0 else "right",fontsize=6.5)
axs[1].set_xlim(-1.2,2.4); axs[1].legend(handles=[Patch(color=cM3,label="q < 0.05, higher in M3"),Patch(color=cD3,label="q < 0.05, higher in D3"),Patch(color="#AAAAAA",label="q ≥ 0.05")],frameon=False,fontsize=7,loc="lower right")
fig.tight_layout(); save(fig,"fig9_targets")
print("ok")
