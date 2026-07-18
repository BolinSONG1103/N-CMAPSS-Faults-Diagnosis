"""生成 DTAE 诊断的王昆式图（检测+隔离混淆矩阵 + 潜空间），用真实残差数据。"""
import os, sys
import numpy as np, pandas as pd
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "figure_pipeline"))
import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec
import style as S
import closure_lib as C
from dtae import DTAE

S.apply()
FAM_OF = {0:1,1:2,2:2,3:3,4:3,5:4,6:4,7:5,8:5}
CLASSES = ["正常","HPT","Fan","HPC","LPT","LPC"]
DIAG_SEV = 0.05   # 预先声明：真值量程归一化严重度>0.05 视为物理可观测退化

def rich(s,W=8):
    R=s["R"];df=pd.DataFrame(R)
    mean=df.rolling(W,min_periods=1).mean().values
    slope=(df-df.shift(W)).fillna(0).values
    mx=df.rolling(W,min_periods=1).apply(lambda v:v[np.argmax(np.abs(v))],raw=True).values
    return np.hstack([R,mean,slope,mx])
def lab(s):
    y=np.zeros(len(s["cycles"]),int); y[~s["hs"]]=FAM_OF[int(s["fault_idx"][0])] if len(s["fault_idx"]) else 0
    return y
def sev_true(s): return np.max(np.maximum(0,-s["theta_true"]),axis=1)
def oh(y,k=6):
    Y=np.zeros((len(y),k));Y[np.arange(len(y)),y]=1;return Y

def train_and_eval(seed=1):
    data=C.load_sequences(lam=0.01); dev=data["calRaw"]+data["idRaw"]
    rng=np.random.RandomState(seed); Xtr,ytr,Xte,yte,ste=[],[],[],[],[]
    for s in dev:
        X=rich(s);y=lab(s);sv=sev_true(s);n=len(y);idx=rng.permutation(n);cut=int(0.7*n)
        Xtr.append(X[idx[:cut]]);ytr.append(y[idx[:cut]])
        Xte.append(X[idx[cut:]]);yte.append(y[idx[cut:]]);ste.append(sv[idx[cut:]])
    Xtr=np.vstack(Xtr);ytr=np.concatenate(ytr);Xte=np.vstack(Xte);yte=np.concatenate(yte);ste=np.concatenate(ste)
    mu=Xtr.mean(0);sd=Xtr.std(0)+1e-8;Xtr=(Xtr-mu)/sd;Xte=(Xte-mu)/sd
    net=DTAE(Xtr.shape[1],16,6,lambda_cls=3.0,noise=0.3,mask=0.1,lr=3e-3,epochs=500,batch=128,seed=seed).fit(Xtr,oh(ytr))
    yp=np.argmax(net.predict_proba(Xte),1); Z=net.encode(Xte)
    return yte,yp,ste,Z

def draw_conf(ax,M,labels,tag,title):
    row=M.sum(1,keepdims=True); N=M/np.where(row==0,1,row)
    ax.imshow(N,cmap=S.CONF_CMAP,vmin=0,vmax=1,aspect="equal")
    ax.set_xticks(range(len(labels)));ax.set_yticks(range(len(labels)))
    ax.set_xticklabels(labels,rotation=30,ha="right");ax.set_yticklabels(labels)
    ax.set_xlabel("预测");ax.set_ylabel("真值");ax.set_title(title,pad=8)
    ax.set_xticks(np.arange(-.5,len(labels),1),minor=True);ax.set_yticks(np.arange(-.5,len(labels),1),minor=True)
    ax.grid(which="minor",color="white",linewidth=1.4);ax.tick_params(which="minor",length=0)
    for sp in ax.spines.values(): sp.set_visible(False)
    for i in range(len(labels)):
        for j in range(len(labels)):
            if N[i,j]>=0.005:
                ax.text(j,i,f"{N[i,j]*100:.0f}\n({M[i,j]})",ha="center",va="center",
                        fontsize=9,color="white" if N[i,j]>0.55 else "#1a1a1a")
    S.panel_tag(ax,tag)

def make():
    yte,yp,ste,Z=train_and_eval()
    fig=plt.figure(figsize=(13.6,4.6))
    gs=GridSpec(1,3,width_ratios=[1,1.12,1.15],wspace=0.5,figure=fig)
    # (a) 检测混淆
    ax=fig.add_subplot(gs[0])
    det_t=(yte>0).astype(int);det_p=(yp>0).astype(int)
    Md=np.zeros((2,2),int)
    for a,b in zip(det_t,det_p): Md[a,b]+=1
    draw_conf(ax,Md,["正常","故障"],"(a)","故障检测")
    # (b) 隔离混淆（可观测退化样本）
    ax=fig.add_subplot(gs[1])
    diag=(yte>0)&(ste>DIAG_SEV)
    Mi=np.zeros((5,5),int)
    for a,b in zip(yte[diag],yp[diag]):
        bb=b if b>0 else a  # 极少数漏检归对角外? 这里只统计预测为故障
        if b>0: Mi[a-1,b-1]+=1
    draw_conf(ax,Mi,CLASSES[1:],"(b)","部件族隔离")
    # (c) 潜空间 PCA
    ax=fig.add_subplot(gs[2])
    Zc=Z-Z.mean(0); U,s,Vt=np.linalg.svd(Zc,full_matrices=False); P=Zc@Vt[:2].T
    cols={0:"#9AA0A6",1:"#D62728",2:"#2CA02C",3:"#1F77B4",4:"#FF7F0E",5:"#9467BD"}
    for c in range(6):
        mk=yte==c
        if mk.any(): ax.scatter(P[mk,0],P[mk,1],s=10,c=cols[c],label=CLASSES[c],alpha=0.6,edgecolor="none")
    ax.set_xlabel("潜维度 1");ax.set_ylabel("潜维度 2");ax.set_title("DTAE 潜空间",pad=8)
    ax.legend(loc="upper right",fontsize=8,ncol=2,markerscale=1.5);S.panel_tag(ax,"(c)")
    fig.suptitle("基于 DTAE 的分布内气路故障诊断",y=1.02,fontsize=13.5)
    out=os.path.join(os.path.dirname(__file__),"dtae_diagnosis_preview")
    fig.savefig(out+".png",dpi=200,bbox_inches="tight");plt.close(fig)
    print("saved",out+".png")

if __name__=="__main__":
    make()
