"""Where the zoom SFR is: per FOF group (top by M200) vs outside groups."""
import glob, sys, numpy as np, h5py
for arg in sys.argv[1:]:
    lab, sd, gd = arg.split(":")
    sfr, pos, mstar_pos = [], [], []
    for f in sorted(glob.glob(sd + "/*.hdf5")):
        h = h5py.File(f, "r"); g = h["PartType0"]
        sfr.append(g["StarFormationRate"][:]); pos.append(g["Coordinates"][:])
        H = dict(h["Header"].attrs); P = dict(h["Parameters"].attrs)
    sfr = np.concatenate(sfr); pos = np.concatenate(pos); box = H["BoxSize"]
    G = {}
    for f in sorted(glob.glob(gd + "/*.hdf5")):
        g = h5py.File(f, "r").get("Group")
        if g is None or "GroupPos" not in g: continue
        for k in ("GroupPos", "Group_M_Crit200", "Group_R_Crit200"): G.setdefault(k, []).append(g[k][:])
    G = {k: np.concatenate(v) for k, v in G.items()}
    o = np.argsort(-G["Group_M_Crit200"])
    s = sfr > 0; ps = pos[s]; fs = sfr[s]; assigned = np.zeros(s.sum(), bool)
    rows = []
    for rank, i in enumerate(o[:300]):
        if G["Group_R_Crit200"][i] <= 0: continue
        d = ps - G["GroupPos"][i]; d -= box * np.round(d / box)
        inn = (np.linalg.norm(d, axis=1) < G["Group_R_Crit200"][i]) & ~assigned
        assigned |= inn
        rows.append((G["Group_M_Crit200"][i] * 1e10 / P["HubbleParam"], fs[inn].sum()))
    rows = np.array(rows)
    def band(lo, hi): k = (rows[:, 0] >= lo) & (rows[:, 0] < hi); return rows[k, 1].sum(), k.sum()
    tot = fs.sum()
    print(f"{lab:9s} z={H['Redshift']:.2f} total SFR {tot:6.1f} | main {rows[0,1]:5.1f} | "
          f"1e11-1e12 halos {band(1e11,1e12)[0]:5.1f} (n={band(1e11,1e12)[1]}) | 1e10-1e11 {band(1e10,1e11)[0]:5.1f} (n={band(1e10,1e11)[1]}) | "
          f"<1e10 {band(0,1e10)[0]:5.1f} | outside R200 of top-300 {fs[~assigned].sum():5.1f}")
