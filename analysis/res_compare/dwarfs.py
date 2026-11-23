"""Stack halos with 1e10 <= M200 < 1e11 Msun (z=2.5/3): stars, gas, SF gas density, and the Dalla Vecchia & Schaye (2012)
critical density above which a heating event radiates its energy before it can expand:
n_H,tc ~ 100 cm^-3 (dT/10^7.5 K)^1.5 (m_gas/1e6 Msun)^-0.5."""
import glob, sys, numpy as np, h5py
X, MP = 0.76, 1.6726e-24
for arg in sys.argv[1:]:
    lab, sd, gd, dT = arg.split(":"); dT = float(dT)
    D = {0: {}, 4: {}}
    for f in sorted(glob.glob(sd + "/*.hdf5")):
        h = h5py.File(f, "r")
        for t, ks in ((0, ("Coordinates", "Masses", "Density", "StarFormationRate")), (4, ("Coordinates", "Masses", "StellarFormationTime"))):
            for k in ks: D[t].setdefault(k, []).append(h[f"PartType{t}"][k][:])
        H = dict(h["Header"].attrs); P = dict(h["Parameters"].attrs)
    D = {t: {k: np.concatenate(v) for k, v in d.items()} for t, d in D.items()}
    a, hh, box = H["Time"], P["HubbleParam"], H["BoxSize"]
    G = {}
    for f in sorted(glob.glob(gd + "/*.hdf5")):
        g = h5py.File(f, "r").get("Group")
        if g is None or "GroupPos" not in g: continue
        for k in ("GroupPos", "Group_M_Crit200", "Group_R_Crit200"): G.setdefault(k, []).append(g[k][:])
    G = {k: np.concatenate(v) for k, v in G.items()}
    M200 = G["Group_M_Crit200"] * 1e10 / hh
    sel = np.where((M200 >= 1e10) & (M200 < 1e11))[0]
    from scipy.spatial import cKDTree
    tg, ts = cKDTree(D[0]["Coordinates"], boxsize=box), cKDTree(D[4]["Coordinates"], boxsize=box)
    ig = np.unique(np.concatenate([tg.query_ball_point(G["GroupPos"][i], G["Group_R_Crit200"][i]) for i in sel]).astype(int))
    is_ = np.unique(np.concatenate([ts.query_ball_point(G["GroupPos"][i], G["Group_R_Crit200"][i]) for i in sel]).astype(int))
    mg = D[0]["Masses"][ig] * 1e10 / hh; sfr = D[0]["StarFormationRate"][ig]
    nH = D[0]["Density"][ig].astype(np.float64) * 1e10 / hh / (a / hh) ** 3 * 1.989e33 / 3.0857e21**3 * X / MP
    sf = sfr > 0
    mgas = np.median(D[0]["Masses"]) * 1e10 / hh
    ncrit = 100 * (dT / 10**7.5) ** 1.5 * (mgas / 1e6) ** -0.5
    ms = D[4]["Masses"][is_] * 1e10 / hh
    o = np.argsort(nH[sf]); w = mg[sf][o]; c = np.cumsum(w) / w.sum()
    p = lambda q: nH[sf][o][np.searchsorted(c, q)]
    print(f"{lab:9s} z={H['Redshift']:.2f} N_halo={len(sel):3d}  M*={ms.sum()/1e9:6.2f}e9  Mgas={mg.sum()/1e9:6.2f}e9  M_SF={mg[sf].sum()/1e9:5.2f}e9  "
          f"SFR={sfr.sum():5.1f}  t_dep={mg[sf].sum()/max(sfr.sum(),1e-9)/1e9:4.2f} Gyr  SF nH p50/p90={p(.5):5.1f}/{p(.9):5.1f}  "
          f"n_crit={ncrit:5.0f}  SF mass above n_crit={(w[nH[sf][o]>ncrit]).sum()/w.sum()*100:4.1f}%")
