"""For young stars (3-40 Myr, SNII window) at z=3: feedback kernel the code would build.
r0 = 2 * Hsml(nearest gas); count gas within r0 (no cap on r0) and nearest-gas temperature."""
import glob, sys, numpy as np, h5py
from scipy.spatial import cKDTree
from scipy.integrate import quad

def load(d):
    gas = {k: [] for k in ("pos", "hsml", "u", "rho", "ne")}; st = {"pos": [], "a": [], "res": []}
    for f in sorted(glob.glob(d + "/*.hdf5")):
        h = h5py.File(f, "r")
        H = dict(h["Header"].attrs)
        g = h["PartType0"]
        gas["pos"].append(g["Coordinates"][:]); gas["hsml"].append(g["SmoothingLength"][:])
        gas["u"].append(g["InternalEnergy"][:]); gas["rho"].append(g["Density"][:]); gas["ne"].append(g["ElectronAbundance"][:])
        if "PartType4" in h:
            s = h["PartType4"]; st["pos"].append(s["Coordinates"][:]); st["a"].append(s["StellarFormationTime"][:]); st["res"].append(s["EnergyReservoir"][:])
    P = dict(h["Parameters"].attrs) if "Parameters" in h else {}
    return {k: np.concatenate(v) for k, v in gas.items()}, {k: np.concatenate(v) for k, v in st.items()}, H, P

def t_Gyr(a, H0=67.74, Om=0.3089):
    OL = 1 - Om
    f = lambda x: 1 / (x * np.sqrt(Om / x**3 + OL))
    return quad(f, 0, a)[0] * 977.8 / H0

for name in sys.argv[1:]:
    gas, st, H, P = load(name)
    a = H["Time"]; h = H.get("HubbleParam", P.get("HubbleParam", 0.6774)); Om = H.get("Omega0", P.get("Omega0", 0.3089))
    tnow = t_Gyr(a, 100 * h, Om)
    sel = st["a"] > 0
    ages = np.array([ (tnow - t_Gyr(x, 100 * h, Om)) * 1e3 for x in st["a"][sel] ])
    young = np.where(sel)[0][(ages >= 3) & (ages <= 40)]
    T = gas["u"] * (1e5)**2 * (4 / (1 + 3 * 0.76 + 4 * 0.76 * gas["ne"])) * 1.6726e-24 * (5/3 - 1) / 1.3807e-16
    tree = cKDTree(gas["pos"])
    d, j = tree.query(st["pos"][young], k=1)
    r0 = 2 * gas["hsml"][j]
    n_in = np.array(tree.query_ball_point(st["pos"][young], r0, return_length=True))
    kpc = a / h   # comoving kpc/h -> physical kpc
    print(f"== {name}: z={1/a-1:.2f}, gas {len(gas['hsml'])}, stars {sel.sum()}, young(3-40 Myr) {len(young)}, stars with reservoir {(st['res']>0).sum()}")
    q = lambda x: " ".join(f"{np.percentile(x,p):9.3g}" for p in (50, 90, 99, 99.9)) + f"  max {x.max():9.3g}"
    print("   percentiles            p50       p90       p99     p99.9")
    print("   r0 [pkpc]          ", q(r0 * kpc))
    print("   gas in r0          ", q(n_in.astype(float)))
    print("   T nearest gas [K]  ", q(T[j]))
    print(f"   sum gas-in-r0 over young stars: {n_in.sum():.3e}   (events with >1e4: {(n_in>1e4).sum()}, >1e5: {(n_in>1e5).sum()})")
    hot = T[j] > 1e7
    print(f"   nearest gas hotter than 1e7 K: {hot.sum()} stars; their median gas-in-r0 {np.median(n_in[hot]) if hot.any() else 0:.0f}, sum {n_in[hot].sum():.3e}")
