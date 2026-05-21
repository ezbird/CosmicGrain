#!/usr/bin/env python3
"""
plot_smoothing_lengths.py
--------------------------
Visualize gas smoothing length (Hsml) distributions across CosmicGrain
snapshots, broken down by gas phase (density / temperature). Restricts
loading to the high-resolution zoom region via the FOF/SubFind group
catalog — this is essential at 2048³ where uniform random subsampling
overwhelmingly draws background low-resolution particles.

The smoothing length is Gadget-4's adaptive SPH kernel radius — it is
the most direct measure of local resolution. Comparing Hsml against the
dust hash cell size and the physical shock radius tells you whether your
clumping factor and shock destruction are operating in the resolved or
subgrid regime.

Usage:
    # All snapshots, 1024^3, one run — zoom region only
    python plot_smoothing_lengths.py --run S10 --res 1024

    # Just a specific snapshot
    python plot_smoothing_lengths.py --run S10 --res 1024 --snap 150

    # Larger aperture (default 3×R200)
    python plot_smoothing_lengths.py --run S10 --res 2048 --aperture 5.0

    # Compare two runs side-by-side at a single snapshot
    python plot_smoothing_lengths.py --run S10 S5 --res 1024 --snap 150

    # Evolution only (skip the per-snapshot detail panel)
    python plot_smoothing_lengths.py --run S10 --res 1024 --evolution-only
"""

import os
import re
import glob
import argparse
import numpy as np
import h5py
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec

plt.style.use('scripts/sleek.mplstyle')

# ─────────────────────────────────────────────────────────────────────────────
# Gas phase definitions (temperature × density bins)
# ─────────────────────────────────────────────────────────────────────────────
PHASES = [
    dict(label="Diffuse CGM",       nH_lo=None,  nH_hi=0.01,  T_lo=1e4,  T_hi=None,  color="#5b9bd5", ls="-"),
    dict(label="Warm ISM",          nH_lo=0.01,  nH_hi=1.0,   T_lo=1e3,  T_hi=1e5,   color="#70ad47", ls="-"),
    dict(label="Cool ISM",          nH_lo=1.0,   nH_hi=100,   T_lo=None, T_hi=1e4,   color="#ed7d31", ls="-"),
    dict(label="Dense/SF",          nH_lo=100,   nH_hi=None,  T_lo=None, T_hi=None,  color="#c00000", ls="-"),
    dict(label="Hot gas (T>10⁶ K)", nH_lo=None,  nH_hi=None,  T_lo=1e6,  T_hi=None,  color="#7030a0", ls="--"),
]

FIGDIR = "dust_figures/smoothing_lengths"
os.makedirs(FIGDIR, exist_ok=True)

PROTONMASS        = 1.6726e-24
HYDROGEN_MASSFRAC = 0.76

# ─────────────────────────────────────────────────────────────────────────────
# Snapshot / catalog helpers — same pattern as compare_grid_dust / mckinnon
# ─────────────────────────────────────────────────────────────────────────────

def find_snapshots(run, res):
    output_dir = f"{run}_output_{res}"
    if not os.path.isdir(output_dir):
        return []
    seen, bases = set(), []
    for snapdir in sorted(glob.glob(os.path.join(output_dir, "snapdir_*"))):
        for f in sorted(glob.glob(os.path.join(snapdir, "snapshot_*.hdf5"))):
            # Normalise away any chunk suffix so that snapshot_030.0.hdf5,
            # snapshot_030.3.hdf5, and a chunk-only file like snapshot_001.1.hdf5
            # all resolve to the same base (e.g. .../snapshot_030).
            import re as _re
            base = _re.sub(r"\.\d+\.hdf5$", "", f)   # strip .N.hdf5
            base = _re.sub(r"\.hdf5$",         "", base) # strip plain .hdf5
            if base not in seen:
                seen.add(base); bases.append(base)
    return sorted(bases)


def subfiles(snap_base):
    files = sorted(glob.glob(snap_base + ".*.hdf5"))
    if not files:
        s = snap_base + ".hdf5"
        files = [s] if os.path.exists(s) else []
    return files


def read_header(snap_base):
    defaults = dict(h=0.6774, a=1.0, z=0.0, um=1.989e43, ul=3.085678e21, uv=1e5)
    for suffix in [".0.hdf5", ".hdf5"]:
        f = snap_base + suffix
        if os.path.exists(f):
            try:
                with h5py.File(f, "r") as hf:
                    attrs = hf["Header"].attrs
                    a = float(attrs.get("Time", defaults["a"]))
                    return dict(
                        h  = float(attrs.get("HubbleParam",              defaults["h"])),
                        a  = a,
                        z  = float(attrs.get("Redshift", 1.0/a - 1.0)),
                        um = float(attrs.get("UnitMass_in_g",            defaults["um"])),
                        ul = float(attrs.get("UnitLength_in_cm",         defaults["ul"])),
                        uv = float(attrs.get("UnitVelocity_in_cm_per_s", defaults["uv"])),
                    )
            except Exception:
                pass
    return defaults


def find_catalog(run, snap_base, res):
    """
    Locate the FOF/SubFind group catalog for this snapshot.
    Returns path to the first catalog file (.0.hdf5) or None.
    Mirrors the logic used in compare_grid_dust and mckinnon_comparison.
    """
    # Strip any trailing file-index suffix (e.g. snapshot_030.3 → snapshot_030)
    clean_base = re.sub(r"\.\d+$", "", snap_base)
    m = re.search(r"snapshot_(\d+)$", clean_base)
    if not m:
        return None
    snap_num   = m.group(1)
    groups_dir = os.path.join(f"{run}_output_{res}", f"groups_{snap_num}")
    cats = sorted(glob.glob(
        os.path.join(groups_dir, f"fof_subhalo_tab_{snap_num}.*.hdf5")))
    if not cats:
        cats = sorted(glob.glob(
            os.path.join(groups_dir, f"fof_subhalo_tab_{snap_num}.hdf5")))
    return cats[0] if cats else None


def get_halo_center_r200(run, snap_base, res):
    """
    Return (center_ckpch, r200_ckpch) in comoving kpc/h by reading
    GroupPos / Group_R_Mean200 directly from the SubFind catalog.

    This guarantees unit consistency with snapshot Coordinates, which
    are also in comoving kpc/h in Gadget-4. Returns (None, None) if no
    catalog is available (e.g. early snapshots before the first FOF run).
    """
    catalog = find_catalog(run, snap_base, res)
    if catalog is None:
        return None, None
    try:
        with h5py.File(catalog, "r") as hf:
            if "Group" not in hf:
                return None, None
            grp = hf["Group"]
            if "GroupPos" not in grp or grp["GroupPos"].shape[0] == 0:
                return None, None
            ctr  = grp["GroupPos"][0].astype(float)           # ckpc/h
            r200 = (float(grp["Group_R_Mean200"][0])
                    if "Group_R_Mean200" in grp else None)     # ckpc/h

            # Sanity check: in a 50 Mpc/h box, GroupPos should be O(1e4) ckpc/h.
            # If values look like Mpc/h (< 500), rescale.
            if ctr[0] < 500.0 and r200 is not None and r200 < 1.0:
                print("    [WARNING] GroupPos looks like Mpc/h — converting ×1000")
                ctr  *= 1000.0
                r200 *= 1000.0

            return ctr, r200
    except Exception as e:
        print(f"    [catalog error] {snap_base}: {e}")
        return None, None


# ─────────────────────────────────────────────────────────────────────────────
# Gas phase mask
# ─────────────────────────────────────────────────────────────────────────────

def phase_mask(data, phase):
    nH, T = data["nH"], data["T"]
    mask  = np.ones(len(nH), bool)
    if phase["nH_lo"] is not None: mask &= nH >= phase["nH_lo"]
    if phase["nH_hi"] is not None: mask &= nH <  phase["nH_hi"]
    if phase["T_lo"]  is not None: mask &= T  >= phase["T_lo"]
    if phase["T_hi"]  is not None: mask &= T  <  phase["T_hi"]
    return mask


# ─────────────────────────────────────────────────────────────────────────────
# Snapshot loader — zoom-region filtered
# ─────────────────────────────────────────────────────────────────────────────

def read_snap(snap_base, run, res, aperture_r200=3.0, max_particles=None):
    """
    Load PartType0 gas restricted to aperture_r200 × R200 around the main
    halo center. All coordinate arithmetic is done in comoving kpc/h (the
    native Gadget-4 frame) to avoid unit mismatches.

    If no catalog is found (early snapshots), loads all gas as a fallback.

    Parameters
    ----------
    snap_base      : snapshot base path (no extension)
    run            : run label (e.g. "S10") — used to locate the catalog
    res            : resolution integer — used to build output_dir path
    aperture_r200  : load gas within this multiple of R200 (default 3.0)
    max_particles  : subsample to at most this many after the spatial cut

    Returns None if no gas data is found.
    """
    hdr = read_header(snap_base)
    h, a = hdr["h"], hdr["a"]
    ul   = hdr["ul"]        # cm per code length
    uv   = hdr["uv"]        # cm/s per code velocity
    ud   = hdr["um"] / ul**3  # g/cm³ per code density

    # ── Halo center and aperture in comoving kpc/h ────────────────────────────
    ctr_ckpch, r200_ckpch = get_halo_center_r200(run, snap_base, res)

    if ctr_ckpch is None:
        print("    [no catalog] loading all gas without zoom filter")
        rmax_ckpch = None
        r200_pkpc  = None
    else:
        rmax_ckpch = r200_ckpch * aperture_r200
        r200_pkpc  = r200_ckpch * a / h
        print(f"    center=({ctr_ckpch[0]:.0f}, {ctr_ckpch[1]:.0f}, "
              f"{ctr_ckpch[2]:.0f}) ckpc/h  "
              f"R200={r200_pkpc:.0f} pkpc  "
              f"aperture={aperture_r200:.1f}×R200="
              f"{rmax_ckpch * a / h:.0f} pkpc")

    hsml_list, nH_list, T_list, m_list = [], [], [], []

    for fname in subfiles(snap_base):
        try:
            with h5py.File(fname, "r") as hf:
                if "PartType0" not in hf:
                    continue
                pt0 = hf["PartType0"]

                pos  = pt0["Coordinates"][:]             # comoving kpc/h
                hsml = pt0["SmoothingLength"][:] * (a / h)  # → physical kpc
                dens = pt0["Density"][:]
                m    = pt0["Masses"][:]

                # ── Spatial filter in comoving kpc/h ─────────────────────────
                if rmax_ckpch is not None:
                    box = float(hf["Header"].attrs.get("BoxSize", 50000.0))
                    dx  = pos - ctr_ckpch
                    # Periodic nearest-image
                    dx  = dx - box * np.round(dx / box)
                    r   = np.sqrt((dx**2).sum(axis=1))
                    keep = r < rmax_ckpch
                    hsml = hsml[keep]
                    dens = dens[keep]
                    m    = m[keep]
                    # Internal energy needs same mask — load separately below
                    u_raw_all = None
                else:
                    keep      = np.ones(len(hsml), dtype=bool)
                    u_raw_all = None

                if len(hsml) == 0:
                    continue

                # ── Physical density → nH [cm⁻³] ─────────────────────────────
                # Gadget-4 stores comoving density: ρ_phys = ρ_code × a⁻³ × ud
                rho_cgs = dens * (a**-3) * ud
                nH      = rho_cgs * HYDROGEN_MASSFRAC / PROTONMASS

                # ── Temperature from internal energy ──────────────────────────
                if "InternalEnergy" in pt0:
                    u_raw_all = pt0["InternalEnergy"][:]
                elif "InternalEnergyOld" in pt0:
                    u_raw_all = pt0["InternalEnergyOld"][:]
                else:
                    u_raw_all = np.zeros(len(dens))

                u_raw = u_raw_all[keep] if rmax_ckpch is not None else u_raw_all

                u_cgs = u_raw * uv**2
                # T = (γ-1) u μ m_p / k_B  with μ = 0.6 (ionised cosmic mix)
                T = (2.0 / 3.0) * u_cgs * 0.6 * PROTONMASS / 1.38065e-16

                hsml_list.append(hsml)
                nH_list.append(nH)
                T_list.append(T)
                m_list.append(m)

        except Exception as e:
            print(f"    [WARNING] {fname}: {e}")

    if not hsml_list:
        return None

    hsml = np.concatenate(hsml_list)
    nH   = np.concatenate(nH_list)
    T    = np.concatenate(T_list)
    m    = np.concatenate(m_list)
    n_zoom = len(hsml)

    # ── Optional subsample after spatial cut ──────────────────────────────────
    if max_particles and n_zoom > max_particles:
        idx  = np.random.choice(n_zoom, max_particles, replace=False)
        hsml = hsml[idx]; nH = nH[idx]; T = T[idx]; m = m[idx]

    print(f"    {n_zoom:,} gas particles in zoom region"
          + (f" → subsampled to {len(hsml):,}"
             if max_particles and n_zoom > max_particles else ""))

    return dict(hsml=hsml, nH=nH, T=T, m=m, hdr=hdr,
                n_zoom=n_zoom, r200_pkpc=r200_pkpc)


# ─────────────────────────────────────────────────────────────────────────────
# Summary statistics
# ─────────────────────────────────────────────────────────────────────────────

def summarize(data):
    rows = []
    hsml_all = data["hsml"]
    rows.append(dict(
        phase="ALL GAS", n=len(hsml_all), frac=1.0,
        p5=np.percentile(hsml_all,  5),  p25=np.percentile(hsml_all, 25),
        median=np.median(hsml_all),
        p75=np.percentile(hsml_all, 75), p95=np.percentile(hsml_all, 95),
        color="black",
    ))
    for ph in PHASES:
        mask = phase_mask(data, ph)
        n    = mask.sum()
        if n < 5: continue
        hh   = data["hsml"][mask]
        rows.append(dict(
            phase=ph["label"], n=n, frac=n / len(hsml_all),
            p5=np.percentile(hh,  5),  p25=np.percentile(hh, 25),
            median=np.median(hh),
            p75=np.percentile(hh, 75), p95=np.percentile(hh, 95),
            color=ph["color"],
        ))
    return rows


# ─────────────────────────────────────────────────────────────────────────────
# Single-snapshot detail figure
# ─────────────────────────────────────────────────────────────────────────────

def plot_snapshot_detail(snap_base, run, res, aperture_r200, outpath):
    print(f"  Loading {snap_base} ...")
    data = read_snap(snap_base, run, res,
                     aperture_r200=aperture_r200, max_particles=500_000)
    if data is None:
        print("  No gas data found — skipping.")
        return

    hdr      = data["hdr"]
    z, a     = hdr["z"], hdr["a"]
    r200_str = (f"  R200={data['r200_pkpc']:.0f} pkpc"
                if data["r200_pkpc"] else "")
    rows     = summarize(data)

    fig = plt.figure(figsize=(15, 10))
    fig.patch.set_facecolor("white")
    gs  = gridspec.GridSpec(2, 3, figure=fig,
                            hspace=0.38, wspace=0.32,
                            left=0.07, right=0.97, top=0.91, bottom=0.08)

    # ── Panel 1: PDF by gas phase ──────────────────────────────────────────
    ax1 = fig.add_subplot(gs[0, :2])
    bins = np.logspace(np.log10(1e-3), np.log10(500), 80)

    ax1.hist(data["hsml"], bins=bins, density=True,
             color="lightgray", alpha=0.6, label="All gas", zorder=1)
    for ph in PHASES:
        mask = phase_mask(data, ph)
        if mask.sum() < 5: continue
        ax1.hist(data["hsml"][mask], bins=bins, density=True,
                 histtype="step", color=ph["color"], lw=1.6,
                 ls=ph["ls"], label=ph["label"], zorder=3)

    ax1.set_xscale("log")
    ax1.set_xlabel(r"Smoothing length $H_{\rm sml}$ [physical kpc]", fontsize=11)
    ax1.set_ylabel("Probability density", fontsize=11)
    ax1.set_title(f"Smoothing length PDF by gas phase  |  {run} {res}³  |  z={z:.2f}",
                  fontsize=10)
    ax1.legend(fontsize=8, loc="upper left", framealpha=0.9)
    ax1.grid(True, which="both", ls=":", alpha=0.3)
    _ylim = ax1.get_ylim()
    for xref, label, col in [(0.05,"50 pc","firebrick"),(0.5,"0.5 kpc","navy")]:
        ax1.axvline(xref, color=col, lw=1.0, ls="--", alpha=0.7)
        ax1.text(xref*1.1, _ylim[1]*0.85, label, color=col, fontsize=7, va="top")

    # ── Panel 2: Box-and-whisker per phase ────────────────────────────────
    ax2 = fig.add_subplot(gs[0, 2])
    for ii, row in enumerate(rows):
        y = len(rows) - ii
        ax2.barh(y, row["p75"]-row["p25"], left=row["p25"],
                 height=0.5, color=row["color"], alpha=0.55, zorder=3)
        ax2.plot([row["median"]]*2, [y-0.25, y+0.25],
                 color=row["color"], lw=2.0, zorder=4)
        for seg in [(row["p5"], row["p75"]), (row["p25"], row["p95"])]:
            ax2.plot(list(seg), [y, y], color=row["color"], lw=0.8, alpha=0.7)
        for x in [row["p5"], row["p95"]]:
            ax2.plot([x, x], [y-0.12, y+0.12], color=row["color"], lw=0.8, alpha=0.7)
        ax2.text(row["median"]*1.05, y, f'{row["median"]*1000:.0f} pc',
                 va="center", ha="left", fontsize=7, color=row["color"])

    ax2.set_xscale("log")
    ax2.set_yticks(list(range(1, len(rows)+1))[::-1])
    ax2.set_yticklabels([r["phase"] for r in rows], fontsize=8)
    ax2.set_xlabel(r"$H_{\rm sml}$ [physical kpc]", fontsize=10)
    ax2.set_title("Median ± IQR (whiskers: 5–95%)", fontsize=9)
    ax2.grid(True, which="both", ls=":", alpha=0.3, axis="x")
    for xref, col in [(0.05,"firebrick"),(0.5,"navy")]:
        ax2.axvline(xref, color=col, lw=0.8, ls="--", alpha=0.6)

    # ── Panel 3: 2D Hsml vs nH coloured by T ─────────────────────────────
    ax3 = fig.add_subplot(gs[1, :2])
    idx = np.random.choice(len(data["nH"]), min(60000, len(data["nH"])), replace=False)
    sc  = ax3.scatter(data["nH"][idx], data["hsml"][idx]*1000,
                      c=np.log10(np.clip(data["T"][idx], 10, 1e8)),
                      cmap="plasma", s=0.8, alpha=0.5, rasterized=True,
                      vmin=2, vmax=7)
    cb  = plt.colorbar(sc, ax=ax3, pad=0.01)
    cb.set_label(r"$\log_{10}$ T [K]", fontsize=9)
    cb.set_ticks([2, 3, 4, 5, 6, 7])

    ax3.set_xscale("log"); ax3.set_yscale("log")
    ax3.set_xlabel(r"Hydrogen number density $n_H$ [cm$^{-3}$]", fontsize=11)
    ax3.set_ylabel(r"$H_{\rm sml}$ [physical pc]", fontsize=11)
    ax3.set_title("Smoothing length vs density (coloured by temperature)", fontsize=10)
    ax3.grid(True, which="both", ls=":", alpha=0.25)
    for nref in [0.01, 1.0, 100.0]:
        ax3.axvline(nref, color="gray", lw=0.7, ls=":", alpha=0.5)
    for yref, label, col in [(50,"50 pc","firebrick"),(500,"500 pc","navy")]:
        ax3.axhline(yref, color=col, lw=1.0, ls="--", alpha=0.7)
        ax3.text(ax3.get_xlim()[0]*2, yref*1.05, label, color=col, fontsize=7)

    # ── Panel 4: summary stats table ──────────────────────────────────────
    ax4 = fig.add_subplot(gs[1, 2])
    ax4.axis("off")
    col_labels = ["Phase", "N", "f%", "p5\n[pc]", "med\n[pc]", "p95\n[pc]"]
    table_data = [
        [r["phase"], f"{r['n']:,}", f"{r['frac']*100:.1f}",
         f"{r['p5']*1000:.0f}", f"{r['median']*1000:.0f}",
         f"{r['p95']*1000:.0f}"]
        for r in rows
    ]
    tbl = ax4.table(cellText=table_data, colLabels=col_labels,
                    cellLoc="center", loc="center",
                    bbox=[0.0, 0.0, 1.0, 1.0])
    tbl.auto_set_font_size(False); tbl.set_fontsize(7.5)
    for ii, row in enumerate(rows):
        tbl[ii+1, 0].set_facecolor(row["color"])
        tbl[ii+1, 0].set_alpha(0.25)
    for j in range(len(col_labels)):
        tbl[0, j].set_facecolor("#dddddd")
        tbl[0, j].set_text_props(weight="bold")
    ax4.set_title("Summary statistics", fontsize=9, pad=4)

    fig.suptitle(
        f"Gas Smoothing Lengths  |  {run}  {res}³  |  "
        f"z = {z:.3f}  (a = {a:.3f}){r200_str}  |  "
        f"N_zoom = {data['n_zoom']:,}  (aperture {aperture_r200:.1f}×R200)",
        fontsize=11, fontweight="bold")

    fig.savefig(outpath, dpi=150, bbox_inches="tight")
    print(f"  Saved: {outpath}")
    plt.close(fig)


# ─────────────────────────────────────────────────────────────────────────────
# Evolution plot
# ─────────────────────────────────────────────────────────────────────────────

def plot_evolution(snap_bases, run, res, aperture_r200, outpath, stride=1):
    print(f"\n  Building evolution track "
          f"({len(snap_bases)} snaps, stride={stride}) ...")

    z_vals = []
    stats  = {ph["label"]: dict(p25=[], median=[], p75=[]) for ph in PHASES}
    stats["ALL GAS"] = dict(p25=[], median=[], p75=[])

    for sb in snap_bases[::stride]:
        print(f"    {os.path.basename(sb)}", end=" ", flush=True)
        data = read_snap(sb, run, res,
                         aperture_r200=aperture_r200, max_particles=80_000)
        if data is None:
            print("(skip)"); continue

        z_vals.append(data["hdr"]["z"])
        hsml_all = data["hsml"]
        stats["ALL GAS"]["p25"].append(np.percentile(hsml_all, 25))
        stats["ALL GAS"]["median"].append(np.median(hsml_all))
        stats["ALL GAS"]["p75"].append(np.percentile(hsml_all, 75))

        for ph in PHASES:
            lbl  = ph["label"]
            mask = phase_mask(data, ph)
            if mask.sum() < 10:
                for k in ("p25","median","p75"):
                    stats[lbl][k].append(np.nan)
            else:
                hh = data["hsml"][mask]
                stats[lbl]["p25"].append(np.percentile(hh, 25))
                stats[lbl]["median"].append(np.median(hh))
                stats[lbl]["p75"].append(np.percentile(hh, 75))
        print("✓")

    if not z_vals:
        print("  No valid snapshots — skipping evolution plot.")
        return

    z_arr = np.array(z_vals)
    order = np.argsort(z_arr)[::-1]   # high-z first (left → right on x-axis)
    z_arr = z_arr[order]

    fig, axes = plt.subplots(2, 1, figsize=(12, 9), sharex=True)
    fig.patch.set_facecolor("white")

    # Top: all-gas with IQR band
    ax = axes[0]
    med = np.array(stats["ALL GAS"]["median"])[order] * 1000  # → pc
    p25 = np.array(stats["ALL GAS"]["p25"])[order]    * 1000
    p75 = np.array(stats["ALL GAS"]["p75"])[order]    * 1000
    ax.fill_between(z_arr, p25, p75, color="gray", alpha=0.25, label="IQR (all gas)")
    ax.plot(z_arr, med, color="black", lw=2.0, label="Median (all gas)")
    ax.set_yscale("log")
    ax.set_ylabel(r"$H_{\rm sml}$ [physical pc]", fontsize=11)
    ax.set_title(f"Smoothing length evolution  |  {run}  {res}³  "
                 f"|  aperture {aperture_r200:.1f}×R200", fontsize=11)
    for yref, label, col in [(50,"50 pc","firebrick"),(500,"500 pc","navy")]:
        ax.axhline(yref, color=col, lw=1.0, ls="--", alpha=0.7)
        ax.text(z_arr.max()*0.98, yref*1.05, label,
                color=col, fontsize=8, ha="right")
    ax.legend(fontsize=9, loc="upper right")
    ax.grid(True, which="both", ls=":", alpha=0.3)
    ax.invert_xaxis()

    # Bottom: per-phase medians
    ax2 = axes[1]
    for ph in PHASES:
        lbl = ph["label"]
        med = np.array(stats[lbl]["median"])[order] * 1000
        p25 = np.array(stats[lbl]["p25"])[order]    * 1000
        p75 = np.array(stats[lbl]["p75"])[order]    * 1000
        if np.all(np.isnan(med)): continue
        ax2.fill_between(z_arr, p25, p75, color=ph["color"], alpha=0.15)
        ax2.plot(z_arr, med, color=ph["color"], lw=1.8,
                 ls=ph["ls"], label=lbl)
    ax2.set_yscale("log")
    ax2.set_xlabel("Redshift $z$", fontsize=11)
    ax2.set_ylabel(r"Median $H_{\rm sml}$ [physical pc]", fontsize=11)
    ax2.set_title("Per gas phase (zoom region)", fontsize=10)
    for yref, col in [(50,"firebrick"),(500,"navy")]:
        ax2.axhline(yref, color=col, lw=1.0, ls="--", alpha=0.7)
    ax2.legend(fontsize=8, loc="upper right", ncol=2)
    ax2.grid(True, which="both", ls=":", alpha=0.3)

    # Secondary x-axis: scale factor
    ax_top = ax.twiny()
    zticks = np.array([z for z in ax.get_xticks() if 0 <= z <= z_arr.max()])
    ax_top.set_xlim(ax.get_xlim())
    ax_top.set_xticks(zticks)
    ax_top.set_xticklabels([f"{1/(1+z):.2f}" for z in zticks], fontsize=8)
    ax_top.set_xlabel("Scale factor $a$", fontsize=9)

    fig.tight_layout()
    fig.savefig(outpath, dpi=150, bbox_inches="tight")
    print(f"  Saved: {outpath}")
    plt.close(fig)


# ─────────────────────────────────────────────────────────────────────────────
# Multi-run comparison
# ─────────────────────────────────────────────────────────────────────────────

def plot_run_comparison(snap_bases_by_run, target_z, res, aperture_r200, outpath):
    RUN_COLORS = ["#1f77b4","#d62728","#2ca02c","#ff7f0e",
                  "#9467bd","#8c564b","#e377c2"]

    fig, axes = plt.subplots(1, 2, figsize=(13, 6))
    fig.patch.set_facecolor("white")
    bins = np.logspace(np.log10(1e-3), np.log10(500), 80)

    for ii, (run, snaps) in enumerate(snap_bases_by_run.items()):
        best, best_dz = None, 1e30
        for sb in snaps:
            dz = abs(read_header(sb)["z"] - target_z)
            if dz < best_dz:
                best_dz = dz; best = sb
        if best is None or best_dz > 0.5:
            print(f"  [{run}] No snap near z={target_z:.1f} — skipping"); continue
        print(f"  [{run}] {os.path.basename(best)} "
              f"(z={read_header(best)['z']:.3f})")
        data = read_snap(best, run, res,
                         aperture_r200=aperture_r200, max_particles=300_000)
        if data is None: continue
        color = RUN_COLORS[ii % len(RUN_COLORS)]

        axes[0].hist(data["hsml"]*1000, bins=bins*1000, density=True,
                     histtype="step", color=color, lw=1.8, label=run)
        mask = phase_mask(data, PHASES[3])  # Dense/SF
        if mask.sum() > 5:
            axes[1].hist(data["hsml"][mask]*1000, bins=bins*1000, density=True,
                         histtype="step", color=color, lw=1.8, label=run)

    for ax, title in zip(axes, ["All gas (zoom region)",
                                 "Dense/SF gas only (nH > 100 cm⁻³)"]):
        ax.set_xscale("log")
        ax.set_xlabel(r"$H_{\rm sml}$ [physical pc]", fontsize=11)
        ax.set_ylabel("Probability density", fontsize=11)
        ax.set_title(title, fontsize=10)
        ax.legend(fontsize=9)
        ax.grid(True, which="both", ls=":", alpha=0.3)
        for xref, col in [(50,"firebrick"),(500,"navy")]:
            ax.axvline(xref, color=col, lw=1.0, ls="--", alpha=0.7)

    fig.suptitle(
        f"Smoothing length comparison  |  {res}³  |  z ≈ {target_z:.1f}"
        f"  |  aperture {aperture_r200:.1f}×R200",
        fontsize=12, fontweight="bold")
    fig.tight_layout()
    fig.savefig(outpath, dpi=150, bbox_inches="tight")
    print(f"  Saved: {outpath}")
    plt.close(fig)


# ─────────────────────────────────────────────────────────────────────────────
# CLI
# ─────────────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--run", nargs="+", required=True,
                        help="Run label(s), e.g. S10  or  S5 S10")
    parser.add_argument("--res", type=int, default=1024,
                        help="Resolution (default: 1024)")
    parser.add_argument("--snap", type=int, default=None,
                        help="Specific snapshot number for detail panel "
                             "(default: last available)")
    parser.add_argument("--aperture", type=float, default=3.0,
                        help="Load gas within this multiple of R200 "
                             "(default: 3.0 — captures ISM + inner CGM)")
    parser.add_argument("--target-z", type=float, default=0.0,
                        help="Target redshift for multi-run comparison "
                             "(default: 0.0)")
    parser.add_argument("--evolution-stride", type=int, default=5,
                        help="Load every Nth snapshot for evolution plot "
                             "(default: 5; use 1 for full track)")
    parser.add_argument("--evolution-only", action="store_true",
                        help="Skip the per-snapshot detail panel")
    parser.add_argument("--no-evolution", action="store_true",
                        help="Skip the evolution plot")
    args = parser.parse_args()

    print(f"\nRuns:       {args.run}")
    print(f"Resolution: {args.res}³")
    print(f"Aperture:   {args.aperture:.1f} × R200")

    snap_map = {}
    for run in args.run:
        snaps = find_snapshots(run, args.res)
        if not snaps:
            print(f"[{run}] No snapshots found in {run}_output_{args.res}/")
            continue
        snap_map[run] = snaps
        print(f"[{run}] Found {len(snaps)} snapshots")

    if not snap_map:
        print("No valid runs found. Exiting.")
        return

    # ── Per-snapshot detail ───────────────────────────────────────────────
    if not args.evolution_only:
        for run, snaps in snap_map.items():
            if args.snap is not None:
                matches = (
                    [s for s in snaps if re.search(rf"snapshot_{args.snap:03d}$", s)]
                 or [s for s in snaps if re.search(rf"snapshot_{args.snap:04d}$", s)])
                snap_base = matches[0] if matches else snaps[-1]
                if not matches:
                    print(f"[{run}] Snapshot {args.snap} not found — using last")
            else:
                snap_base = snaps[-1]

            snap_num = re.search(r"snapshot_(\d+)$", snap_base)
            snap_num = snap_num.group(1) if snap_num else "last"
            outpath  = os.path.join(FIGDIR,
                f"hsml_detail_{run}_{args.res}_snap{snap_num}.png")
            print(f"\n[{run}] Detail plot → {snap_base}")
            plot_snapshot_detail(snap_base, run, args.res, args.aperture, outpath)

    # ── Multi-run comparison ──────────────────────────────────────────────
    if len(snap_map) > 1:
        outpath = os.path.join(FIGDIR,
            f"hsml_compare_{'_'.join(snap_map.keys())}"
            f"_{args.res}_z{args.target_z:.1f}.png")
        print(f"\nComparison plot at z≈{args.target_z}")
        plot_run_comparison(snap_map, args.target_z, args.res,
                            args.aperture, outpath)

    # ── Evolution ────────────────────────────────────────────────────────
    if not args.no_evolution:
        for run, snaps in snap_map.items():
            outpath = os.path.join(FIGDIR,
                f"hsml_evolution_{run}_{args.res}.png")
            print(f"\n[{run}] Evolution plot")
            plot_evolution(snaps, run, args.res, args.aperture,
                           outpath, stride=args.evolution_stride)

    print("\nDone.")


if __name__ == "__main__":
    main()
