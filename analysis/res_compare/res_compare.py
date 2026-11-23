"""Main halo of halo441 at z=3 / 2.5 across resolutions and SN heating temperatures.
Usage: python3 res_compare.py <z-label> label:snapdir:groupdir ...
Physical units: Msun, pkpc, Msun/yr, cm^-3.
"""
import glob, sys
import numpy as np, h5py
from scipy.integrate import quad

X_H, MP, KB = 0.76, 1.6726e-24, 1.3807e-16
REF = np.array([28350., 24848., 25717.])   # main halo at z=3 (comoving kpc/h), same in all runs


def read(snapdir, types=(0, 1, 2, 4, 6)):
    out = {t: {} for t in types}
    for f in sorted(glob.glob(snapdir + "/*.hdf5")):
        h = h5py.File(f, "r")
        for t in types:
            g = h.get(f"PartType{t}")
            if g is None:
                continue
            for k in g.keys():
                out[t].setdefault(k, []).append(g[k][:])
    H = dict(h["Header"].attrs); P = dict(h["Parameters"].attrs)
    S = {t: {k: np.concatenate(v) for k, v in d.items()} for t, d in out.items()}
    for t in S:
        if "Coordinates" in S[t] and "Masses" not in S[t]:
            S[t]["Masses"] = np.full(len(S[t]["Coordinates"]), H["MassTable"][t])
    return S, H, P


def groups(gdir):
    G = {}
    for f in sorted(glob.glob(gdir + "/*.hdf5")):
        g = h5py.File(f, "r").get("Group")
        if g is None or "GroupPos" not in g:
            continue
        for k in ("GroupPos", "Group_M_Crit200", "Group_R_Crit200"):
            G.setdefault(k, []).append(g[k][:])
    return {k: np.concatenate(v) for k, v in G.items()}


def age_Gyr(a, h, Om):
    return quad(lambda x: 1 / (x * np.sqrt(Om / x**3 + 1 - Om)), 0, a)[0] * 977.8 / (100 * h)


def analyse(label, snapdir, gdir):
    S, H, P = read(snapdir)
    a, h, Om, Ob = H["Time"], P["HubbleParam"], P["Omega0"], P["OmegaBaryon"]
    box = H["BoxSize"]
    G = groups(gdir)
    # main halo: most massive group within 1 cMpc/h of the reference position
    d = np.linalg.norm(G["GroupPos"] - REF, axis=1)
    cand = np.where(d < 1000)[0]
    gi = cand[np.argmax(G["Group_M_Crit200"][cand])]
    c = G["GroupPos"][gi].copy()
    R200 = G["Group_R_Crit200"][gi] * a / h               # pkpc
    M200 = G["Group_M_Crit200"][gi] * 1e10 / h
    L = a / h                                               # ckpc/h -> pkpc
    M = 1e10 / h                                            # code mass -> Msun

    def rel(t):
        x = S[t]["Coordinates"] - c
        x -= box * np.round(x / box)
        return x * L
    # shrinking sphere on stars for the galaxy centre
    xs = rel(4); ms = S[4]["Masses"]
    cen = np.zeros(3); r = 0.3 * R200
    for _ in range(30):
        s = np.linalg.norm(xs - cen, axis=1) < r
        if s.sum() < 50:
            break
        cen = np.average(xs[s], axis=0, weights=ms[s]); r *= 0.85
    out = {"M200 [1e11]": M200 / 1e11, "R200 [pkpc]": R200}
    Hz = 100 * h * np.sqrt(Om / a**3 + 1 - Om) / 1000.     # km/s/kpc
    vel = {}
    pos = {}
    for t in S:
        if "Coordinates" not in S[t]:
            continue
        pos[t] = rel(t) - cen
        v = S[t]["Velocities"] * np.sqrt(a)
        vel[t] = v
    # bulk velocity of the galaxy from stars within 10 pkpc
    sel = np.linalg.norm(pos[4], axis=1) < 10
    vb = np.average(vel[4][sel], axis=0, weights=ms[sel])
    rr = {t: np.linalg.norm(pos[t], axis=1) for t in pos}
    m = {t: S[t]["Masses"] * M for t in pos}

    g = S[0]
    rho = g["Density"].astype(np.float64) * M / (L**3) * 1.989e33 / (3.0857e21**3)   # g/cm^3 physical
    nH = rho * X_H / MP
    mu = 4 / (1 + 3 * X_H + 4 * X_H * g["ElectronAbundance"])
    T = g["InternalEnergy"] * 1e10 * (2 / 3) * mu * MP / KB
    sfr = g["StarFormationRate"]
    sfg = sfr > 0
    gal, halo = 10.0, R200
    inR = {t: rr[t] < halo for t in rr}
    ing = rr[0] < gal

    tot = sum(m[t][inR[t]].sum() for t in m)
    bary = m[0][inR[0]].sum() + m[4][inR[4]].sum() + (m[6][inR[6]].sum() if 6 in m else 0)
    out["f_bary(<R200) / cosmic"] = bary / tot / (Ob / Om)
    out["M_gas(<R200) [1e10]"] = m[0][inR[0]].sum() / 1e10
    for lo, hi, name in [(0, 3e4, "T<3e4"), (3e4, 3e5, "3e4-3e5"), (3e5, 3e6, "3e5-3e6"), (3e6, 3e7, "3e6-3e7"), (3e7, 1e12, ">3e7")]:
        s = inR[0] & (T >= lo) & (T < hi) & ~sfg
        out[f"  gas(<R200) non-SF {name} [1e10]"] = m[0][s].sum() / 1e10
    out["  gas(<R200) SF [1e10]"] = m[0][inR[0] & sfg].sum() / 1e10
    out["M*(<10 pkpc) [1e10]"] = m[4][rr[4] < gal].sum() / 1e10
    out["M*(<R200) [1e10]"] = m[4][inR[4]].sum() / 1e10
    xs4 = rr[4][rr[4] < gal]; ws = m[4][rr[4] < gal]; o = np.argsort(xs4)
    out["r_half,* (<10 pkpc) [pkpc]"] = xs4[o][np.searchsorted(np.cumsum(ws[o]), ws.sum() / 2)]
    out["M_SFgas(<10 pkpc) [1e10]"] = m[0][ing & sfg].sum() / 1e10
    out["M_gas(<10 pkpc) [1e10]"] = m[0][ing].sum() / 1e10
    out["SFR(<10 pkpc) [Msun/yr]"] = sfr[ing].sum()
    out["SFR(<R200) [Msun/yr]"] = sfr[inR[0]].sum()
    out["t_dep = M_SF/SFR (<10) [Gyr]"] = m[0][ing & sfg].sum() / max(sfr[ing].sum(), 1e-9) / 1e9
    w = m[0][ing & sfg]
    for p in (10, 50, 90):
        x = np.sort(nH[ing & sfg]); ww = w[np.argsort(nH[ing & sfg])]
        out[f"SF gas nH p{p} (mass-wt) [cm^-3]"] = x[np.searchsorted(np.cumsum(ww), p / 100 * ww.sum())] if len(x) else np.nan
    out["M_dust(<10 pkpc) [1e7]"] = m[6][rr[6] < gal].sum() / 1e7 if 6 in m else np.nan
    out["Z_SFgas (<10) mass-wt"] = np.average(g["Metallicity"][ing & sfg], weights=w) if w.sum() > 0 else np.nan
    out["gas particle mass [1e5]"] = np.median(m[0]) / 1e5
    out["N SF gas (<10 pkpc)"] = int((ing & sfg).sum())
    # radial gas flows through the 0.2-0.3 R200 shell and at 10-20 pkpc
    for (r0, r1, nm) in [(10, 20, "10-20 pkpc"), (0.2 * R200, 0.3 * R200, "0.2-0.3 R200"), (0.9 * R200, 1.0 * R200, "0.9-1.0 R200")]:
        s = (rr[0] > r0) & (rr[0] < r1)
        vr = np.einsum("ij,ij->i", pos[0][s], vel[0][s] - vb) / rr[0][s] + Hz * rr[0][s]   # km/s
        f = m[0][s] * vr / (r1 - r0) * 1.0227e-9   # Msun/yr  (km/s / kpc = 1.0227e-9 /yr)
        out[f"gas outflow {nm} [Msun/yr]"] = f[vr > 0].sum()
        out[f"gas inflow  {nm} [Msun/yr]"] = -f[vr < 0].sum()
        hot = T[s] > 1e6
        out[f"  hot(>1e6 K) outflow {nm}"] = f[(vr > 0) & hot].sum()
    # star formation history of stars now within 10 pkpc (birth mass)
    st = S[4]
    born = st["StellarFormationTime"]; bm = st.get("StellarBirthMass", st["Masses"]) * M
    s = (rr[4] < gal) & (born > 0)
    edges_z = [12, 8, 6, 5, 4, 3.5, 3.0, 2.5]
    tnow = age_Gyr(a, h, Om)
    for z0, z1 in zip(edges_z[:-1], edges_z[1:]):
        a0, a1 = 1 / (1 + z0), 1 / (1 + z1)
        if a0 >= a:
            continue
        a1 = min(a1, a)
        sb = s & (born >= a0) & (born < a1)
        dt = (age_Gyr(a1, h, Om) - age_Gyr(a0, h, Om)) * 1e9
        out[f"SFH(<10 pkpc) z={z0}-{z1} [Msun/yr]"] = bm[sb].sum() / dt
    return label, out


zlab = sys.argv[1]
res = [analyse(*arg.split(":")) for arg in sys.argv[2:]]
keys = list(dict.fromkeys(k for _, o in res for k in o))
print(f"\n=== main halo at {zlab} ===")
print(f"{'':42s}" + "".join(f"{l:>12s}" for l, _ in res))
for k in keys:
    row = f"{k:42s}"
    for _, o in res:
        v = o.get(k, np.nan)
        row += f"{v:12.3g}" if isinstance(v, (float, np.floating)) else f"{v:>12}"
    print(row)
