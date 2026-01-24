#!/usr/bin/env python3
import argparse, sys, numpy as np, h5py
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

def robust_mode(v, bins=128):
    if v is None or v.size==0: return np.nan
    h,e = np.histogram(v, bins=bins); k = np.argmax(h); return 0.5*(e[k]+e[k+1])

def infer_hi_mass(m, sample=200_000):
    if m is None or m.size==0: return np.nan
    idx = np.random.choice(m.size, size=min(sample, m.size), replace=False)
    mode = robust_mode(m[idx]); 
    if not np.isfinite(mode): return np.nan
    u = np.unique(np.round(m[idx],10)); return u[np.argmin(np.abs(u-mode))]

def load_species(f, ptype):
    gname = f'PartType{ptype}'
    if gname not in f: return None,None,None
    g=f[gname]; pos=g['Coordinates'][:]
    mass=g['Masses'][:] if 'Masses' in g else None
    soft=g['Softening'][:] if 'Softening' in g else (g['SofteningComoving'][:] if 'SofteningComoving' in g else None)
    return pos,mass,soft

def wrap_delta(d, L):
    if L is None or not np.isfinite(L) or L<=0: return d
    return ((d+0.5*L)%L)-0.5*L

def slice_mask(pos, axis, ctr, W, T, L=None):
    if pos is None or pos.size==0: return np.array([],dtype=int)
    p = pos - ctr[None,:]
    if L is not None: p = wrap_delta(p, L)
    a=[0,1,2]; a.remove(axis)
    m = (np.abs(p[:,axis])<=T) & (np.abs(p[:,a[0]])<=W/2) & (np.abs(p[:,a[1]])<=W/2)
    return np.where(m)[0]

def classify(m, hi, tol):
    if m is None or m.size==0 or not np.isfinite(hi): return None
    lo=(1-tol)*hi; hi2=(1+tol)*hi; return (m>=lo)&(m<=hi2)

def radial_mask(pos, ctr, rmin, L=None):
    if pos is None or pos.size==0: return np.array([],dtype=int)
    d = pos - ctr[None,:]
    if L is not None: d = wrap_delta(d,L)
    r = np.sqrt((d*d).sum(axis=1)); return np.where(r>=rmin)[0]

def choose_center(mode, allpos, L):
    if mode=='box' and L is not None: return np.array([0.5*L]*3),'box'
    if mode=='median': return np.median(allpos,axis=0),'median'
    if mode=='densest':
        try:
            from scipy.spatial import cKDTree
            samp = allpos[np.random.choice(allpos.shape[0], size=min(300000,allpos.shape[0]), replace=False)]
            tree=cKDTree(samp); d,_=tree.query(samp,k=32)
            rho=1/np.maximum(d[:,-1],1e-30); return samp[np.argmax(rho)],'densest'
        except Exception: return np.median(allpos,axis=0),'median(fallback)'
    if L is not None: return np.array([0.5*L]*3),'box(auto)'
    return 0.5*(allpos.min(0)+allpos.max(0)),'midspan(auto)'

def main():
    ap=argparse.ArgumentParser(description='IC resolution slice viewer')
    ap.add_argument('ic')
    ap.add_argument('--axis',default='z',choices=['x','y','z'])
    ap.add_argument('--center',nargs=3,type=float)
    ap.add_argument('--auto-center',default='auto',choices=['auto','box','median','densest'])
    ap.add_argument('--periodic',action='store_true')
    ap.add_argument('--width',type=float)
    ap.add_argument('--thickness',type=float)
    ap.add_argument('--buffer-radius',type=float)
    ap.add_argument('--dm-hi-mass',type=float,default=np.nan)
    ap.add_argument('--gas-hi-mass',type=float,default=np.nan)
    ap.add_argument('--hi-mass-tol',type=float,default=0.25)
    ap.add_argument('--auto-mass',type=int,default=200000)
    ap.add_argument('--downsample',type=int,default=1)
    ap.add_argument('--savefig',default='ic_slices.png')
    args=ap.parse_args()
    axis={'x':0,'y':1,'z':2}[args.axis]
    with h5py.File(args.ic,'r') as f:
        hdr=f.get('Header')
        L=float(hdr.attrs['BoxSize']) if (hdr is not None and 'BoxSize' in hdr.attrs) else None
        periodic = True if (L is not None) else bool(args.periodic)
        pos_dm,m_dm,_ = load_species(f,1)
        pos_g ,m_g ,_ = load_species(f,0)
        if pos_dm is None: print('ERROR: no PartType1',file=sys.stderr); sys.exit(2)
        allpos = pos_dm if pos_g is None else np.vstack([pos_dm,pos_g])
        span=(allpos.max(0)-allpos.min(0)); 
        if args.center is not None: ctr=np.asarray(args.center); cm='user'
        else: ctr,cm=choose_center(args.auto_center,allpos,L)
        W=args.width if args.width is not None else 0.9*float(span.max())
        T=args.thickness if args.thickness is not None else W/40.0
        print(f'[info] center={ctr}, center_mode={cm}, width={W:.3g}, thickness={T:.3g}, periodic={periodic}, boxsize={L}')
        dm_hi = args.dm_hi_mass if np.isfinite(args.dm_hi_mass) else infer_hi_mass(m_dm,args.auto_mass)
        gas_hi= args.gas_hi_mass if np.isfinite(args.gas_hi_mass) else (infer_hi_mass(m_g,args.auto_mass) if m_g is not None else np.nan)
        print(f'[info] inferred/provided high-res masses: DM={dm_hi:.6g}  Gas={(gas_hi if np.isfinite(gas_hi) else np.nan)}')
        idx_dm = slice_mask(pos_dm,axis,ctr,W,T,L if periodic else None)
        idx_dm = idx_dm[::max(1,int(args.downsample))]
        print(f'[info] DM in slice: {idx_dm.size} / {pos_dm.shape[0]}')
        if m_dm is not None: print(f'[info] unique DM masses (rounded 12dp): {np.unique(np.round(m_dm,12)).size}')
        if pos_g is not None:
            idx_g  = slice_mask(pos_g ,axis,ctr,W,T,L if periodic else None)
            idx_g  = idx_g[::max(1,int(args.downsample))]
            print(f'[info] Gas in slice: {idx_g.size} / {pos_g.shape[0]}')
            if m_g is not None: print(f'[info] unique Gas masses (rounded 12dp): {np.unique(np.round(m_g,12)).size}')
        else:
            idx_g=np.array([],dtype=int)
        if idx_dm.size + idx_g.size == 0:
            T2=T*10.0
            idx_dm = slice_mask(pos_dm,axis,ctr,W,T2,L if periodic else None)
            if pos_g is not None: idx_g = slice_mask(pos_g,axis,ctr,W,T2,L if periodic else None)
            print(f'[warn] initial slice empty; retried with thickness={T2} -> DM={idx_dm.size}, Gas={idx_g.size if pos_g is not None else 0}')
        pos_dm_sl=pos_dm[idx_dm]; m_dm_sl = (m_dm[idx_dm] if m_dm is not None else None)
        if pos_g is not None:
            pos_g_sl = pos_g[idx_g]; m_g_sl = (m_g[idx_g] if m_g is not None else None)
        else:
            pos_g_sl = np.zeros((0,3)); m_g_sl=None
        # classify AFTER building slices (fixes UnboundLocalError)
        dm_is_hi = classify(m_dm_sl, dm_hi, args.hi_mass_tol) if (m_dm_sl is not None and np.isfinite(dm_hi)) else None
        g_is_hi  = classify(m_g_sl , gas_hi, args.hi_mass_tol) if (m_g_sl  is not None and np.isfinite(gas_hi)) else None
        # diagnostics
        gas_out=None
        if pos_g is not None and args.buffer_radius is not None:
            ridx = radial_mask(pos_g, ctr, args.buffer_radius, L if periodic else None)
            gas_out=len(ridx); print(f'[diagnostic] Gas particles with r >= {args.buffer_radius}: {gas_out}')
        # plane coords
        a=[0,1,2]; a.remove(axis)
        def pc(p): return p[:,a[0]]-ctr[a[0]], p[:,a[1]]-ctr[a[1]]
        xdm,ydm=pc(pos_dm_sl); xg,yg=pc(pos_g_sl)
        # figure
        fig=plt.figure(figsize=(11,9)); gs=fig.add_gridspec(2,2,height_ratios=[1.0,0.6],hspace=0.28,wspace=0.22)
        ax1=fig.add_subplot(gs[0,0])
        if m_dm_sl is not None and m_dm_sl.size>0 and dm_is_hi is not None:
            ax1.scatter(xdm[ dm_is_hi], ydm[ dm_is_hi], s=0.15, alpha=0.7, rasterized=True, label='DM hi-res')
            ax1.scatter(xdm[~dm_is_hi], ydm[~dm_is_hi], s=0.15, alpha=0.7, rasterized=True, label='DM low-res')
            ax1.legend(loc='upper right', markerscale=10)
        else:
            ax1.scatter(xdm, ydm, s=0.2, alpha=0.7, rasterized=True)
        ax1.set_title('DM slice (hi vs low by mass)')
        ax1.set_xlabel(f"{'xyz'[a[0]]} - center"); ax1.set_ylabel(f"{'xyz'[a[1]]} - center")
        ax1.set_xlim(-W/2,W/2); ax1.set_ylim(-W/2,W/2)
        if args.buffer_radius: 
            import matplotlib.patches as mp; ax1.add_patch(mp.Circle((0,0),args.buffer_radius,fill=False,linestyle='--'))
        ax2=fig.add_subplot(gs[0,1])
        if pos_g_sl.shape[0]>0 and m_g_sl is not None and g_is_hi is not None:
            ax2.scatter(xg[ g_is_hi], yg[ g_is_hi], s=0.15, alpha=0.7, rasterized=True, label='Gas hi-res')
            ax2.scatter(xg[~g_is_hi], yg[~g_is_hi], s=0.15, alpha=0.7, rasterized=True, label='Gas low-res')
            ax2.legend(loc='upper right', markerscale=10)
        else:
            ax2.text(0.5,0.5,'No gas in file or slice',ha='center',va='center',transform=ax2.transAxes)
        ax2.set_title('Gas slice (hi vs low by mass)')
        ax2.set_xlabel(f"{'xyz'[a[0]]} - center"); ax2.set_ylabel(f"{'xyz'[a[1]]} - center")
        ax2.set_xlim(-W/2,W/2); ax2.set_ylim(-W/2,W/2)
        if args.buffer_radius: 
            import matplotlib.patches as mp; ax2.add_patch(mp.Circle((0,0),args.buffer_radius,fill=False,linestyle='--'))
        ax3=fig.add_subplot(gs[1,0])
        if m_dm is not None and m_dm.size>0:
            ax3.hist(np.log10(m_dm),bins=64,histtype='step'); ax3.set_xlabel('log10(M_DM)'); ax3.set_ylabel('count'); ax3.set_title('DM mass histogram (global)')
            if np.isfinite(dm_hi): ax3.axvline(np.log10(dm_hi),linestyle='--',label='hi-res target'); ax3.legend()
        else: ax3.text(0.5,0.5,'No DM masses',ha='center',va='center',transform=ax3.transAxes)
        ax4=fig.add_subplot(gs[1,1])
        if m_g is not None and m_g.size>0:
            ax4.hist(np.log10(m_g),bins=64,histtype='step'); ax4.set_xlabel('log10(M_gas)'); ax4.set_ylabel('count'); ax4.set_title('Gas mass histogram (global)')
            if np.isfinite(gas_hi): ax4.axvline(np.log10(gas_hi),linestyle='--',label='hi-res target'); ax4.legend()
        else: ax4.text(0.5,0.5,'No gas masses',ha='center',va='center',transform=ax4.transAxes)
        # summaries
        if m_dm is not None and m_dm.size>0 and np.isfinite(dm_hi):
            msk=classify(m_dm,dm_hi,args.hi_mass_tol); 
            if msk is not None: print(f"[summary] DM hi-res fraction (global, within tol): {msk.mean():.3f}")
        if m_g is not None and m_g.size>0 and np.isfinite(gas_hi):
            msk=classify(m_g,gas_hi,args.hi_mass_tol); 
            if msk is not None: print(f"[summary] Gas hi-res fraction (global, within tol): {msk.mean():.3f}")
        # warnings
        if m_dm is not None and m_dm.size>0:
            if np.unique(np.round(m_dm,12)).size==1: print('[warn] DM appears single-mass globally (no low-res shells).')
        if (m_g is not None and m_g.size>0 and args.buffer_radius is not None and gas_out is not None and gas_out>0):
            print(f"[warn] Gas exists outside buffer radius {args.buffer_radius}. Low-res region may not be DM-only.")
        # title + save
        notes = [f"axis={args.axis}", f"width={W:g}", f"2*thickness={2*T:g}", f"center=({ctr[0]:.3g},{ctr[1]:.3g},{ctr[2]:.3g})", f"DM_hi={dm_hi:.6g}", f"Gas_hi={(gas_hi if np.isfinite(gas_hi) else np.nan)}"]
        fig.suptitle('IC Resolution Check\n' + ' | '.join(notes), y=0.98)
        plt.savefig(args.savefig,bbox_inches='tight',dpi=200)
        print(f'[done] Saved figure to {args.savefig}')

if __name__=='__main__':
    main()