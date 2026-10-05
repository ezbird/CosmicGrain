#!/usr/bin/env python3
"""
Quantify stellar morphology for all stars and for configurable stellar-age bins.

The circularity proxy is epsilon = j_z/[R V_circ(R)], with
V_circ(R) = sqrt(G M(<R)/R) from all particle types.  A single orientation,
bulk velocity, and circular-speed curve derived from the complete galaxy are
used for every age bin so that the populations can be compared directly.
"""
import argparse
import glob

import h5py
import matplotlib.pyplot as plt
import numpy as np

G = 4.30091e-6  # kpc (km/s)^2 / Msun
HUBBLE_TIME_GYR = 9.778  # 1/H0 in Gyr when H0 = 100 km/s/Mpc
FORMATION_FIELDS = (
    "StellarFormationTime",
    "GFM_StellarFormationTime",
    "FormationTime",
)


def scalar(x):
    return float(np.asarray(x).reshape(-1)[0])


def get_attr(header, params, names, default=None):
    for name in names:
        if params is not None and name in params.attrs:
            return scalar(params.attrs[name])
        if name in header.attrs:
            return scalar(header.attrs[name])
    if default is not None:
        return default
    raise RuntimeError("Missing attribute; tried " + ", ".join(names))


def info(fn):
    with h5py.File(fn, "r") as f:
        header = f["Header"]
        params = f.get("Parameters")
        a = scalar(header.attrs["Time"])
        box = scalar(header.attrs["BoxSize"])
        h = get_attr(header, params, ("HubbleParam",))
        omega_m = get_attr(header, params, ("Omega0", "OmegaM", "OmegaMatter"))
        omega_l = get_attr(
            header, params, ("OmegaLambda", "OmegaL", "OmegaVacuum"),
            default=1.0 - omega_m,
        )
    return a, 1.0 / a - 1.0, box, h, omega_m, omega_l


def find_formation_field(files):
    for fn in files:
        with h5py.File(fn, "r") as f:
            if "PartType4" not in f:
                continue
            g = f["PartType4"]
            for name in FORMATION_FIELDS:
                if name in g:
                    return name
    raise RuntimeError(
        "No stellar formation-time field found in PartType4; tried: "
        + ", ".join(FORMATION_FIELDS)
    )


def load(files, particle_type, need_vel=False, extra_field=None):
    coords, masses, velocities, extra = [], [], [], []
    for fn in files:
        with h5py.File(fn, "r") as f:
            name = f"PartType{particle_type}"
            if name not in f:
                continue
            g = f[name]
            n = len(g["Coordinates"])
            coords.append(g["Coordinates"][:])
            if "Masses" in g:
                masses.append(g["Masses"][:])
            else:
                mass = f["Header"].attrs["MassTable"][particle_type]
                if mass <= 0:
                    raise RuntimeError(f"No masses for type {particle_type}")
                masses.append(np.full(n, mass))
            if need_vel:
                velocities.append(g["Velocities"][:])
            if extra_field is not None:
                if extra_field not in g:
                    raise RuntimeError(f"{name}/{extra_field} missing from {fn}")
                extra.append(g[extra_field][:])
    if not coords:
        return None
    out = {"Coordinates": np.concatenate(coords), "Masses": np.concatenate(masses)}
    if need_vel:
        out["Velocities"] = np.concatenate(velocities)
    if extra_field is not None:
        out[extra_field] = np.concatenate(extra)
    return out


def cosmic_age_gyr(scale_factor, h, omega_m, omega_l):
    """Cosmic age for a flat matter+Lambda cosmology (radiation neglected)."""
    scale_factor = np.asarray(scale_factor, dtype=float)
    if omega_m <= 0 or omega_l < 0:
        raise RuntimeError("Age conversion requires Omega0 > 0 and OmegaLambda >= 0")
    if omega_l == 0:
        return (HUBBLE_TIME_GYR / h) * 2.0 * scale_factor**1.5 / (3.0 * np.sqrt(omega_m))
    prefactor = (HUBBLE_TIME_GYR / h) * 2.0 / (3.0 * np.sqrt(omega_l))
    return prefactor * np.arcsinh(np.sqrt(omega_l / omega_m) * scale_factor**1.5)


def dperiodic(x, center, box):
    delta = x.astype(float) - center
    delta -= box * np.rint(delta / box)
    return delta


def rotation(j):
    zhat = j / np.linalg.norm(j)
    ref = np.array([0.0, 0.0, 1.0])
    if abs(ref @ zhat) > 0.9:
        ref = np.array([0.0, 1.0, 0.0])
    xhat = np.cross(ref, zhat)
    xhat /= np.linalg.norm(xhat)
    yhat = np.cross(zhat, xhat)
    return np.vstack((xhat, yhat, zhat))


def binned_profile(x, weights, edges, volume=None):
    mass = np.histogram(x, bins=edges, weights=weights)[0]
    if volume is not None:
        mass = mass / volume
    centers = 0.5 * (edges[:-1] + edges[1:])
    return centers, mass


def expfit(x, y, xmin, xmax):
    selected = (x >= xmin) & (x <= xmax) & np.isfinite(y) & (y > 0)
    if selected.sum() < 3:
        return np.nan, np.nan, selected
    slope, intercept = np.polyfit(x[selected], np.log(y[selected]), 1)
    scale = -1.0 / slope if slope < 0 else np.nan
    return scale, intercept, selected


def safe_fraction(weights, mask):
    total = np.sum(weights)
    return np.sum(weights[mask]) / total if total > 0 else np.nan


def age_label(lo, hi):
    if np.isinf(hi):
        return rf"$>{lo:g}$ Gyr"
    if lo == 0:
        return rf"$<{hi:g}$ Gyr"
    return rf"${lo:g}$--${hi:g}$ Gyr"


parser = argparse.ArgumentParser()
parser.add_argument("snapshot")
parser.add_argument("--center", nargs=3, type=float, required=True)
parser.add_argument("--orientation-radius", type=float, default=20.0)
parser.add_argument("--analysis-radius", type=float, default=30.0)
parser.add_argument("--disk-zmax", type=float, default=5.0)
parser.add_argument("--radial-bin", type=float, default=0.5)
parser.add_argument("--vertical-bin", type=float, default=0.25)
parser.add_argument("--rd-fit", nargs=2, type=float, default=[3.0, 12.0], metavar=("RMIN", "RMAX"))
parser.add_argument("--hz-fit", nargs=2, type=float, default=[0.5, 3.0], metavar=("ZMIN", "ZMAX"))
parser.add_argument(
    "--age-bins", nargs="+", type=float, default=[0.0, 2.0, 6.0, np.inf],
    metavar="GYR", help="Age-bin edges in Gyr; default: 0 2 6 inf",
)
parser.add_argument(
    "--min-age-bin-stars", type=int, default=100,
    help="Minimum number of stars inside the analysis aperture for age-bin fits",
)
parser.add_argument("--prefix", default="stellar_morphology_quant")
args = parser.parse_args()

files = sorted(glob.glob(args.snapshot))
if not files:
    raise FileNotFoundError(args.snapshot)
if len(args.age_bins) < 2 or np.any(np.diff(args.age_bins) <= 0) or args.age_bins[0] < 0:
    raise ValueError("--age-bins must be at least two strictly increasing, non-negative edges")

aa, redshift, box, h, omega_m, omega_l = info(files[0])
center = np.array(args.center, float)
formation_field = find_formation_field(files)

stars = load(files, 4, need_vel=True, extra_field=formation_field)
parts = []
for particle_type in range(7):
    part = load(files, particle_type)
    if part is not None:
        parts.append((particle_type, part))

rs = dperiodic(stars["Coordinates"], center, box) * aa / h
ms = stars["Masses"].astype(float) * 1e10 / h
vs = stars["Velocities"].astype(float)
formation_a = stars[formation_field].astype(float)
valid_formation = (formation_a > 0) & (formation_a <= aa * (1.0 + 1e-6))
stellar_age = np.full(len(formation_a), np.nan)
stellar_age[valid_formation] = cosmic_age_gyr(aa, h, omega_m, omega_l) - cosmic_age_gyr(
    formation_a[valid_formation], h, omega_m, omega_l
)
stellar_age[(stellar_age < 0) & (stellar_age > -1e-6)] = 0.0

r3 = np.linalg.norm(rs, axis=1)
orientation_selection = (r3 < args.orientation_radius) & valid_formation
if orientation_selection.sum() < 3:
    raise RuntimeError("Fewer than three valid stars inside the orientation radius")
bulk_velocity = np.average(vs[orientation_selection], axis=0, weights=ms[orientation_selection])
angular_momentum = np.sum(
    np.cross(rs[orientation_selection], vs[orientation_selection] - bulk_velocity)
    * ms[orientation_selection, None], axis=0,
)
if not np.all(np.isfinite(angular_momentum)) or np.linalg.norm(angular_momentum) == 0:
    raise RuntimeError("Stellar angular-momentum vector is zero or non-finite")
jhat = angular_momentum / np.linalg.norm(angular_momentum)
rot = rotation(jhat)
r = rs @ rot.T
v = (vs - bulk_velocity) @ rot.T
radius = np.hypot(r[:, 0], r[:, 1])
jz = r[:, 0] * v[:, 1] - r[:, 1] * v[:, 0]

# Enclosed total mass curve from all particle types.
rall, mall = [], []
for particle_type, part in parts:
    rp = np.linalg.norm(dperiodic(part["Coordinates"], center, box) * aa / h, axis=1)
    mp = part["Masses"].astype(float) * 1e10 / h
    keep = rp <= max(args.analysis_radius * 1.5, 50.0)
    rall.append(rp[keep])
    mall.append(mp[keep])
rall = np.concatenate(rall)
mall = np.concatenate(mall)
order = np.argsort(rall)
rall = rall[order]
cum = np.cumsum(mall[order])
menc = np.interp(np.maximum(radius, 1e-6), rall, cum, left=cum[0], right=cum[-1])
vcirc = np.sqrt(G * menc / np.maximum(radius, 1e-6))
jcirc = radius * vcirc
epsilon = np.full_like(jz, np.nan, dtype=float)
valid_epsilon = (jcirc > 0) & np.isfinite(jcirc)
epsilon[valid_epsilon] = jz[valid_epsilon] / jcirc[valid_epsilon]

analysis = (r3 < args.analysis_radius) & valid_formation
redges = np.arange(0, args.analysis_radius + args.radial_bin, args.radial_bin)
areas = np.pi * (redges[1:] ** 2 - redges[:-1] ** 2)
zedges = np.arange(0, 10.0 + args.vertical_bin, args.vertical_bin)
absz = np.abs(r[:, 2])
annulus_area = np.pi * (args.rd_fit[1] ** 2 - args.rd_fit[0] ** 2)
volumes = np.full(len(zedges) - 1, 2.0 * annulus_area * args.vertical_bin)


def measure(selection, allow_fits=True):
    finite = selection & np.isfinite(epsilon)
    weights = ms[finite]
    result = {
        "nstar": int(selection.sum()),
        "mass": float(np.sum(ms[selection])),
        "disk70": safe_fraction(weights, epsilon[finite] > 0.7),
        "disk50": safe_fraction(weights, epsilon[finite] > 0.5),
        "counter": safe_fraction(weights, epsilon[finite] < 0),
    }
    radial_selection = selection & (absz < args.disk_zmax)
    rc, sigma = binned_profile(radius[radial_selection], ms[radial_selection], redges, areas)
    vertical_selection = selection & (radius >= args.rd_fit[0]) & (radius <= args.rd_fit[1])
    zc, rho = binned_profile(absz[vertical_selection], ms[vertical_selection], zedges, volumes)
    if allow_fits:
        rd, rd_intercept, _ = expfit(rc, sigma, *args.rd_fit)
        hz, hz_intercept, _ = expfit(zc, rho, *args.hz_fit)
    else:
        rd = rd_intercept = hz = hz_intercept = np.nan
    result.update(rc=rc, sigma=sigma, rd=rd, rd_intercept=rd_intercept,
                  zc=zc, rho=rho, hz=hz, hz_intercept=hz_intercept)
    return result


all_result = measure(analysis)
age_results = []
for index, (lo, hi) in enumerate(zip(args.age_bins[:-1], args.age_bins[1:])):
    if index == len(args.age_bins) - 2:
        age_selection = analysis & (stellar_age >= lo) & (stellar_age <= hi)
    else:
        age_selection = analysis & (stellar_age >= lo) & (stellar_age < hi)
    age_results.append(measure(age_selection, age_selection.sum() >= args.min_age_bin_stars))

print("=" * 78)
print("STELLAR MORPHOLOGY QUANTIFICATION BY AGE")
print("=" * 78)
print(f"a={aa:.8f} z={redshift:.8f} h={h:.8f} Omega_m={omega_m:.6f} Omega_L={omega_l:.6f}")
print(f"formation-time field: PartType4/{formation_field}")
print(f"valid stellar ages: {valid_formation.sum():,}/{len(stars['Masses']):,}")
print(f"Nstar(<{args.analysis_radius:g} pkpc)={analysis.sum():,}")
print("j_hat =", jhat)
print("circularity proxy: epsilon = j_z/[R V_circ(R)]")
print(f"all: f(eps>0.7)={all_result['disk70']:.4f}  f(eps>0.5)={all_result['disk50']:.4f}  "
      f"f(eps<0)={all_result['counter']:.4f}  Rd={all_result['rd']:.3f} kpc  hz={all_result['hz']:.3f} kpc")
print("\nAGE-RESOLVED RESULTS")
print(f"{'age (Gyr)':>14} {'Nstar':>9} {'Mstar (Msun)':>14} {'f>0.7':>9} {'f>0.5':>9} {'f<0':>9} {'Rd':>9} {'hz':>9}")
for (lo, hi), result in zip(zip(args.age_bins[:-1], args.age_bins[1:]), age_results):
    age_text = f"{lo:g}-{'inf' if np.isinf(hi) else f'{hi:g}'}"
    print(f"{age_text:>14} {result['nstar']:9,d} {result['mass']:14.6e} "
          f"{result['disk70']:9.4f} {result['disk50']:9.4f} {result['counter']:9.4f} "
          f"{result['rd']:9.3f} {result['hz']:9.3f}")
    if result["nstar"] < args.min_age_bin_stars:
        print(f"  WARNING: fits suppressed: Nstar < {args.min_age_bin_stars}")

# Preserve the original all-star plots.
epsilon_edges = np.linspace(-1.5, 1.5, 61)
epsilon_centers = 0.5 * (epsilon_edges[:-1] + epsilon_edges[1:])
finite_all = analysis & np.isfinite(epsilon)
all_hist, _ = np.histogram(epsilon[finite_all], bins=epsilon_edges, weights=ms[finite_all])
all_hist = all_hist / all_hist.sum()

fig, ax = plt.subplots(figsize=(6.5, 4.8))
ax.step(epsilon_centers, all_hist, where="mid")
ax.axvline(0.7, ls="--", lw=1)
ax.set(xlabel=r"Circularity proxy $\epsilon=j_z/[R\,V_{\rm circ}(R)]$",
       ylabel="Stellar mass fraction per bin", xlim=(-1.5, 1.5))
fig.tight_layout()
fig.savefig(args.prefix + "_circularity.png", dpi=240, bbox_inches="tight")
fig.savefig(args.prefix + "_circularity.pdf", bbox_inches="tight")
plt.close(fig)

fig, ax = plt.subplots(figsize=(6.5, 4.8))
positive = all_result["sigma"] > 0
ax.semilogy(all_result["rc"][positive], all_result["sigma"][positive], "o-", ms=3)
if np.isfinite(all_result["rd"]):
    xx = np.linspace(*args.rd_fit, 200)
    ax.plot(xx, np.exp(all_result["rd_intercept"] - xx / all_result["rd"]), ls="--",
            label=rf"$R_d={all_result['rd']:.2f}$ kpc")
    ax.legend()
ax.set(xlabel="Cylindrical radius (kpc)", ylabel=r"$\Sigma_\star$ ($M_\odot$ kpc$^{-2}$)")
fig.tight_layout()
fig.savefig(args.prefix + "_radial_profile.png", dpi=240, bbox_inches="tight")
fig.savefig(args.prefix + "_radial_profile.pdf", bbox_inches="tight")
plt.close(fig)

fig, ax = plt.subplots(figsize=(6.5, 4.8))
positive = all_result["rho"] > 0
ax.semilogy(all_result["zc"][positive], all_result["rho"][positive], "o-", ms=3)
if np.isfinite(all_result["hz"]):
    xx = np.linspace(*args.hz_fit, 200)
    ax.plot(xx, np.exp(all_result["hz_intercept"] - xx / all_result["hz"]), ls="--",
            label=rf"$h_z={all_result['hz']:.2f}$ kpc")
    ax.legend()
ax.set(xlabel=r"$|z'|$ (kpc)", ylabel=r"$\rho_\star$ ($M_\odot$ kpc$^{-3}$)")
fig.tight_layout()
fig.savefig(args.prefix + "_vertical_profile.png", dpi=240, bbox_inches="tight")
fig.savefig(args.prefix + "_vertical_profile.pdf", bbox_inches="tight")
plt.close(fig)

# Age-comparison plots.
colors = plt.cm.viridis(np.linspace(0.15, 0.85, len(age_results)))
age_hists = []
fig, ax = plt.subplots(figsize=(7.0, 5.0))
for (lo, hi), result, color in zip(zip(args.age_bins[:-1], args.age_bins[1:]), age_results, colors):
    selected = analysis & (stellar_age >= lo) & (stellar_age <= hi if np.isinf(hi) else stellar_age < hi) & np.isfinite(epsilon)
    hist, _ = np.histogram(epsilon[selected], bins=epsilon_edges, weights=ms[selected])
    hist = hist / hist.sum() if hist.sum() > 0 else np.full_like(hist, np.nan, dtype=float)
    age_hists.append(hist)
    ax.step(epsilon_centers, hist, where="mid", color=color, label=age_label(lo, hi))
ax.axvline(0.7, color="0.4", ls="--", lw=1)
ax.set(xlabel=r"Circularity proxy $\epsilon=j_z/[R\,V_{\rm circ}(R)]$",
       ylabel="Stellar mass fraction per bin", xlim=(-1.5, 1.5))
ax.legend(title="Stellar age")
fig.tight_layout()
fig.savefig(args.prefix + "_circularity_by_age.png", dpi=240, bbox_inches="tight")
fig.savefig(args.prefix + "_circularity_by_age.pdf", bbox_inches="tight")
plt.close(fig)

fig, ax = plt.subplots(figsize=(7.0, 5.0))
for (lo, hi), result, color in zip(zip(args.age_bins[:-1], args.age_bins[1:]), age_results, colors):
    positive = result["sigma"] > 0
    ax.semilogy(result["rc"][positive], result["sigma"][positive], "o-", ms=3,
                color=color, label=age_label(lo, hi) + rf", $R_d={result['rd']:.2f}$ kpc")
ax.set(xlabel="Cylindrical radius (kpc)", ylabel=r"$\Sigma_\star$ ($M_\odot$ kpc$^{-2}$)")
ax.legend(fontsize=9)
fig.tight_layout()
fig.savefig(args.prefix + "_radial_profile_by_age.png", dpi=240, bbox_inches="tight")
fig.savefig(args.prefix + "_radial_profile_by_age.pdf", bbox_inches="tight")
plt.close(fig)

fig, ax = plt.subplots(figsize=(7.0, 5.0))
for (lo, hi), result, color in zip(zip(args.age_bins[:-1], args.age_bins[1:]), age_results, colors):
    positive = result["rho"] > 0
    ax.semilogy(result["zc"][positive], result["rho"][positive], "o-", ms=3,
                color=color, label=age_label(lo, hi) + rf", $h_z={result['hz']:.2f}$ kpc")
ax.set(xlabel=r"$|z'|$ (kpc)", ylabel=r"$\rho_\star$ ($M_\odot$ kpc$^{-3}$)")
ax.legend(fontsize=9)
fig.tight_layout()
fig.savefig(args.prefix + "_vertical_profile_by_age.png", dpi=240, bbox_inches="tight")
fig.savefig(args.prefix + "_vertical_profile_by_age.pdf", bbox_inches="tight")
plt.close(fig)

np.savez_compressed(
    args.prefix + "_data.npz",
    epsilon=epsilon[finite_all], epsilon_mass=ms[finite_all], stellar_age_gyr=stellar_age[analysis],
    stellar_mass=ms[analysis], age_bin_edges_gyr=np.asarray(args.age_bins),
    age_bin_labels=np.asarray([age_label(lo, hi) for lo, hi in zip(args.age_bins[:-1], args.age_bins[1:])]),
    epsilon_bin_centers=epsilon_centers, epsilon_hist_all=all_hist, epsilon_hist_by_age=np.asarray(age_hists),
    R_profile=all_result["rc"], Sigma_star=all_result["sigma"],
    Sigma_star_by_age=np.asarray([x["sigma"] for x in age_results]),
    z_profile=all_result["zc"], rho_star=all_result["rho"],
    rho_star_by_age=np.asarray([x["rho"] for x in age_results]),
    Rd_kpc=all_result["rd"], hz_kpc=all_result["hz"],
    Rd_by_age_kpc=np.asarray([x["rd"] for x in age_results]),
    hz_by_age_kpc=np.asarray([x["hz"] for x in age_results]),
    disk_fraction_eps07=all_result["disk70"], disk_fraction_eps05=all_result["disk50"],
    counterrotating_fraction=all_result["counter"],
    disk_fraction_eps07_by_age=np.asarray([x["disk70"] for x in age_results]),
    disk_fraction_eps05_by_age=np.asarray([x["disk50"] for x in age_results]),
    counterrotating_fraction_by_age=np.asarray([x["counter"] for x in age_results]),
    nstar_by_age=np.asarray([x["nstar"] for x in age_results]),
    mstar_by_age_msun=np.asarray([x["mass"] for x in age_results]),
    j_hat=jhat, rotation_matrix=rot, formation_time_field=formation_field,
)

with open(args.prefix + "_summary.txt", "w", encoding="utf-8") as stream:
    stream.write(f"formation_time_field {formation_field}\n")
    stream.write(f"R_d_kpc {all_result['rd']}\n")
    stream.write(f"h_z_kpc {all_result['hz']}\n")
    stream.write(f"disk_fraction_eps_gt_0.7 {all_result['disk70']}\n")
    stream.write(f"disk_fraction_eps_gt_0.5 {all_result['disk50']}\n")
    stream.write(f"counterrotating_fraction_eps_lt_0 {all_result['counter']}\n")
    stream.write("circularity_definition j_z/[R V_circ(R)]\n")
    stream.write("# age_lo_gyr age_hi_gyr nstar mstar_msun f_eps_gt_0.7 f_eps_gt_0.5 f_eps_lt_0 Rd_kpc hz_kpc\n")
    for (lo, hi), result in zip(zip(args.age_bins[:-1], args.age_bins[1:]), age_results):
        stream.write(f"{lo} {hi} {result['nstar']} {result['mass']} {result['disk70']} "
                     f"{result['disk50']} {result['counter']} {result['rd']} {result['hz']}\n")

print("\nSaved all-star and age-resolved circularity/radial/vertical PNG/PDF, data NPZ, and summary TXT.")
