#!/usr/bin/env python3
"""Score small-grain bump and galaxy properties for DustShatteringMach test runs.

Usage: python3 scripts/mach_scan_score.py RUN_DIR [RUN_DIR ...] [--snap 29]
Centre: shrinking sphere on stars, started from the halo 441 position. Dust in a 5-20 kpc spherical shell.
"""
import argparse, glob
import numpy as np
import h5py

START = np.array([29455., 23169., 27855.])  # ckpc/h, halo 441 main galaxy at z=0


def load(run, snap):
    fs = sorted(glob.glob(f"{run}/snapdir_{snap:03d}/*.hdf5"))
    S, SM, G, GM, GSF = [], [], [], [], []
    D = {k: [] for k in ["Coordinates", "Masses", "GrainRadius", "CarbonMassFraction", "DustFormationTime"]}
    for fn in fs:
        with h5py.File(fn) as f:
            p = dict(f["Parameters"].attrs)
            if "PartType4" in f:
                S.append(f["PartType4/Coordinates"][:]); SM.append(f["PartType4/Masses"][:])
            G.append(f["PartType0/Coordinates"][:]); GM.append(f["PartType0/Masses"][:])
            GSF.append(f["PartType0/StarFormationRate"][:])
            if "PartType6" in f:
                for k in D:
                    D[k].append(f["PartType6/" + k][:])
    cat = np.concatenate
    return p, cat(S), cat(SM), cat(G), cat(GM), cat(GSF), {k: cat(v) for k, v in D.items()}


def half_mass_radius(r, m):
    o = np.argsort(r); c = np.cumsum(m[o])
    return r[o][np.searchsorted(c, 0.5 * c[-1])]


def score(run, snap):
    p, S, SM, G, GM, GSF, D = load(run, snap)
    h = p["HubbleParam"]; mu = 1e10 / h
    c = START.copy()
    for R in [200, 100, 50, 30, 20, 10, 5]:
        s = np.linalg.norm(S - c, axis=1) < R
        if s.sum() < 50:
            break
        c = np.average(S[s], axis=0, weights=SM[s])
    rs = np.linalg.norm(S - c, axis=1) / h
    rg = np.linalg.norm(G - c, axis=1) / h
    rd = np.linalg.norm(D["Coordinates"] - c, axis=1) / h
    a, m, cf = D["GrainRadius"], D["Masses"], D["CarbonMassFraction"]
    out = {"run": run.split("halo441_output_")[-1]}
    out["Mstar"] = SM[rs < 30].sum() * mu
    out["SFR"] = GSF[rg < 30].sum()
    out["Mdust"] = m[rd < 30].sum() * mu
    out["rh_dust"] = half_mass_radius(rd[rd < 30], m[rd < 30])
    sh = (rd >= 5) & (rd < 20); M = m[sh].sum()
    f = lambda sel: m[sh & sel].sum() / M
    out["f<5"] = f(a < 5); out["f<10"] = f(a < 10); out["f5-15"] = f((a >= 5) & (a < 15)); out["f<20"] = f(a < 20)
    out["a_med"] = np.interp(0.5, np.cumsum(m[sh][np.argsort(a[sh])]) / M, np.sort(a[sh]))
    out["CF"] = np.average(cf[rd < 30], weights=m[rd < 30])
    out["CF_5-20nm"] = np.average(cf[sh & (a >= 5) & (a < 20)], weights=m[sh & (a >= 5) & (a < 20)]) if (sh & (a >= 5) & (a < 20)).any() else np.nan
    e = np.logspace(0, 3, 31); hist = np.histogram(a[sh], e, weights=m[sh])[0]; cen = np.sqrt(e[1:] * e[:-1])
    out["peaks<40nm"] = [round(float(cen[i]), 1) for i in range(1, len(hist) - 1) if hist[i] > hist[i - 1] and hist[i] > hist[i + 1] and cen[i] < 40 and hist[i] / M > 0.01]
    return out


if __name__ == "__main__":
    ap = argparse.ArgumentParser(); ap.add_argument("runs", nargs="+"); ap.add_argument("--snap", type=int, default=29)
    a = ap.parse_args()
    print(f"{'run':28s} {'M*':>8s} {'SFR':>5s} {'Mdust':>8s} {'rh_d':>5s} {'f<5':>6s} {'f<10':>6s} {'f5-15':>6s} {'f<20':>6s} {'a_med':>6s} {'CF':>5s} {'CFsm':>5s}  peaks<40nm")
    for r in a.runs:
        try:
            o = score(r, a.snap)
        except Exception as ex:
            print(f"{r}: {ex}"); continue
        print(f"{o['run']:28s} {o['Mstar']:8.2e} {o['SFR']:5.2f} {o['Mdust']:8.2e} {o['rh_dust']:5.1f} {o['f<5']:6.3f} {o['f<10']:6.3f} {o['f5-15']:6.3f} {o['f<20']:6.3f} {o['a_med']:6.1f} {o['CF']:5.2f} {o['CF_5-20nm']:5.2f}  {o['peaks<40nm']}")
