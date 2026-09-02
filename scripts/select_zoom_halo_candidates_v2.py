#!/usr/bin/env python3
"""
select_zoom_halo_candidates.py

Rank isolated z~0 parent-box halos for future MUSIC2 zoom simulations.

Designed for GADGET-4 FOF/SUBFIND catalogs containing:
    Group/GroupPos
    Group/Group_M_Crit200
    Group/Group_R_Crit200
    Group/GroupLen
    Group/GroupNsubs

Main selection ideas
--------------------
1. Use M200c as the halo mass.
2. Require a minimum distance from the box edge (for convenient zoom regions).
3. Measure isolation using PERIODIC distances.
4. For each candidate, find the nearest halo with mass >= f_neighbor * M200c.
5. Express isolation both in Mpc/h and in units of candidate R200c.
6. Select a configurable number of halos in logarithmic mass bins.

GADGET conventions assumed:
    Group_M_Crit200 : 1e10 Msun/h
    Group_R_Crit200 : comoving kpc/h (or same length units as GroupPos)
    GroupPos        : same length units as BoxSize

The script reads Header attributes where possible and prints the inferred units.
"""

import argparse
import csv
import math
import sys

import h5py
import numpy as np

try:
    from scipy.spatial import cKDTree
except ImportError:
    cKDTree = None


def get_header_attr(f, key, default=None):
    if "Header" in f and key in f["Header"].attrs:
        return f["Header"].attrs[key]
    if key in f.attrs:
        return f.attrs[key]
    return default


def infer_length_scale_to_mpc_h(boxsize):
    """
    Infer whether coordinates are stored in kpc/h or Mpc/h.

    For this project a 50 Mpc/h box is normally BoxSize~50000 if units are
    kpc/h, or BoxSize~50 if units are Mpc/h.
    """
    if boxsize > 1000.0:
        return 1.0e-3, "kpc/h"
    return 1.0, "Mpc/h"


def periodic_delta(a, b, box):
    d = np.abs(a - b)
    return np.minimum(d, box - d)


def periodic_distance(a, b, box):
    d = periodic_delta(a, b, box)
    return np.sqrt(np.sum(d * d, axis=-1))


def nearest_massive_neighbor_bruteforce(pos, mass, box, frac, candidates):
    """
    Fallback implementation if scipy is unavailable.

    This can be slower than the cKDTree path but remains practical after
    applying the requested mass cut.
    """
    n = len(mass)
    nearest_dist = np.full(n, np.inf)
    nearest_idx = np.full(n, -1, dtype=np.int64)
    nearest_mass = np.full(n, np.nan)

    order = np.argsort(mass)[::-1]

    for ii, i in enumerate(candidates):
        threshold = frac * mass[i]

        valid = order[mass[order] >= threshold]
        valid = valid[valid != i]

        if len(valid) == 0:
            continue

        d = periodic_distance(pos[valid], pos[i], box)
        jloc = np.argmin(d)
        j = valid[jloc]

        nearest_dist[i] = d[jloc]
        nearest_idx[i] = j
        nearest_mass[i] = mass[j]

    return nearest_dist, nearest_idx, nearest_mass


def nearest_massive_neighbor_tree(pos, mass, box, frac, candidates):
    """
    Use one periodic cKDTree and progressively query neighbors until one
    satisfies M_neighbor >= frac * M_candidate.

    This avoids building a separate tree for every halo.
    """
    tree = cKDTree(pos, boxsize=box)

    nearest_dist = np.full(len(mass), np.inf)
    nearest_idx = np.full(len(mass), -1, dtype=np.int64)
    nearest_mass = np.full(len(mass), np.nan)

    # Start with modest k, expand only when necessary.
    for count, i in enumerate(candidates, start=1):
        threshold = frac * mass[i]
        k = 8

        while True:
            k_use = min(k, len(mass))
            dists, inds = tree.query(pos[i], k=k_use)

            dists = np.atleast_1d(dists)
            inds = np.atleast_1d(inds)

            found = False
            for d, j in zip(dists, inds):
                if j == i:
                    continue
                if mass[j] >= threshold:
                    nearest_dist[i] = d
                    nearest_idx[i] = j
                    nearest_mass[i] = mass[j]
                    found = True
                    break

            if found or k_use == len(mass):
                break

            k *= 2

        if count % 500 == 0:
            print(f"  isolation: processed {count}/{len(candidates)} candidates",
                  file=sys.stderr)

    return nearest_dist, nearest_idx, nearest_mass


def normalize_score(x, floor, ceiling):
    if ceiling <= floor:
        return np.ones_like(x)
    return np.clip((x - floor) / (ceiling - floor), 0.0, 1.0)


def main():
    p = argparse.ArgumentParser(
        description="Rank isolated FOF halos for future zoom simulations."
    )
    p.add_argument("catalog", help="fof_subhalo_tab_XXX.hdf5")
    p.add_argument("--output-prefix", default="zoom_halo_candidates",
                   help="Output prefix [default: zoom_halo_candidates]")

    p.add_argument("--min-logm", type=float, default=10.5,
                   help="Minimum log10(M200c/Msun) [default: 10.5]")
    p.add_argument("--max-logm", type=float, default=12.5,
                   help="Maximum log10(M200c/Msun) [default: 12.5]")
    p.add_argument("--bin-width", type=float, default=0.5,
                   help="Mass-bin width in dex [default: 0.5]")
    p.add_argument("--per-bin", type=int, default=3,
                   help="Number of preferred candidates per mass bin [default: 3]")

    p.add_argument("--edge-min", type=float, default=5.0,
                   help="Minimum distance from box edge in Mpc/h [default: 5]")
    p.add_argument("--neighbor-mass-frac", type=float, default=0.5,
                   help="Isolation neighbor must have >= this fraction of candidate mass [default: 0.5]")
    p.add_argument("--min-isolation-r200", type=float, default=5.0,
                   help="Preferred minimum neighbor distance / R200c [default: 5]")
    p.add_argument("--min-selected-separation", type=float, default=5.0,
                   help="Minimum periodic separation between selected targets in Mpc/h [default: 5]")
    p.add_argument("--min-particles", type=int, default=500,
                   help="Minimum FOF particle count [default: 500]")
    p.add_argument("--h", type=float, default=None,
                   help="Override Hubble parameter h if absent from header")
    args = p.parse_args()

    with h5py.File(args.catalog, "r") as f:
        required = [
            "Group/GroupPos",
            "Group/Group_M_Crit200",
            "Group/Group_R_Crit200",
            "Group/GroupLen",
            "Group/GroupNsubs",
        ]
        missing = [x for x in required if x not in f]
        if missing:
            raise RuntimeError("Missing required datasets: " + ", ".join(missing))

        pos_raw = f["Group/GroupPos"][:].astype(np.float64)
        m200_raw = f["Group/Group_M_Crit200"][:].astype(np.float64)
        r200_raw = f["Group/Group_R_Crit200"][:].astype(np.float64)
        group_len = f["Group/GroupLen"][:].astype(np.int64)
        nsubs = f["Group/GroupNsubs"][:].astype(np.int64)

        box_raw = get_header_attr(f, "BoxSize", None)
        time = get_header_attr(f, "Time", np.nan)
        redshift = get_header_attr(f, "Redshift", np.nan)
        hubble = get_header_attr(f, "HubbleParam", args.h)

    if box_raw is None:
        # Use project box size as a last-resort fallback.
        print("WARNING: BoxSize not found in header; assuming 50 Mpc/h.",
              file=sys.stderr)
        box_mpc_h = 50.0
        length_scale = 1.0 if np.nanmax(pos_raw) < 1000 else 1.0e-3
        raw_length_unit = "Mpc/h" if length_scale == 1.0 else "kpc/h"
    else:
        box_raw = float(np.asarray(box_raw).ravel()[0])
        length_scale, raw_length_unit = infer_length_scale_to_mpc_h(box_raw)
        box_mpc_h = box_raw * length_scale

    pos = pos_raw * length_scale
    r200_mpc_h = r200_raw * length_scale

    # Standard GADGET mass unit for group catalogs.
    # Convert from 1e10 Msun/h to physical Msun if h is known.
    if hubble is None:
        hubble = 0.6732
        print("WARNING: HubbleParam not found; using h=0.6732.", file=sys.stderr)
    hubble = float(np.asarray(hubble).ravel()[0])

    m200_msun = m200_raw * 1.0e10 / hubble

    # Exclude zero/invalid SO masses before taking logs.
    valid_mass = np.isfinite(m200_msun) & (m200_msun > 0.0)
    logm = np.full(len(m200_msun), np.nan)
    logm[valid_mass] = np.log10(m200_msun[valid_mass])

    # Non-periodic distance to nearest geometric box face.
    edge_dist = np.min(np.minimum(pos, box_mpc_h - pos), axis=1)

    base = (
        valid_mass
        & (logm >= args.min_logm)
        & (logm < args.max_logm)
        & (edge_dist >= args.edge_min)
        & (group_len >= args.min_particles)
        & (r200_mpc_h > 0.0)
    )
    candidate_idx = np.where(base)[0]

    print("")
    print("=== Parent halo catalog ===")
    print(f"Catalog                 : {args.catalog}")
    print(f"Groups                  : {len(m200_msun):,}")
    print(f"Scale factor             : {float(np.asarray(time).ravel()[0]) if np.size(time) else np.nan:.6f}")
    print(f"Redshift                 : {float(np.asarray(redshift).ravel()[0]) if np.size(redshift) else np.nan:.6f}")
    print(f"h                        : {hubble:.6f}")
    print(f"Raw coordinate unit      : inferred {raw_length_unit}")
    print(f"Box size                 : {box_mpc_h:.3f} Mpc/h")
    print(f"Mass range               : 10^{args.min_logm:.2f} - 10^{args.max_logm:.2f} Msun")
    print(f"Edge cut                 : >= {args.edge_min:.2f} Mpc/h")
    print(f"Selected-target spacing  : >= {args.min_selected_separation:.2f} Mpc/h")
    print(f"Minimum FOF particles    : {args.min_particles}")
    print(f"Candidates before isolation: {len(candidate_idx):,}")
    print("")

    if len(candidate_idx) == 0:
        raise RuntimeError("No halos satisfy the initial cuts.")

    print("Computing periodic isolation...")
    if cKDTree is not None:
        nearest_dist, nearest_idx, nearest_mass = nearest_massive_neighbor_tree(
            pos, m200_msun, box_mpc_h,
            args.neighbor_mass_frac, candidate_idx
        )
    else:
        print("WARNING: scipy not available; using slower brute-force isolation.",
              file=sys.stderr)
        nearest_dist, nearest_idx, nearest_mass = nearest_massive_neighbor_bruteforce(
            pos, m200_msun, box_mpc_h,
            args.neighbor_mass_frac, candidate_idx
        )

    isolation_r200 = np.full(len(m200_msun), np.inf)
    isolation_r200[candidate_idx] = (
        nearest_dist[candidate_idx] / r200_mpc_h[candidate_idx]
    )

    # Ranking score.
    # Once a halo passes the hard edge cut, additional edge clearance is not rewarded.
    # The score is therefore based purely on isolation quality.
    iso_score = normalize_score(
        isolation_r200[candidate_idx],
        args.min_isolation_r200,
        max(args.min_isolation_r200 * 3.0, args.min_isolation_r200 + 1.0)
    )

    score = np.full(len(m200_msun), -np.inf)
    score[candidate_idx] = iso_score

    # Prefer halos actually satisfying the requested isolation threshold.
    preferred = base & (isolation_r200 >= args.min_isolation_r200)

    # Select the top N in each mass bin while enforcing spatial separation
    # between already-selected targets.
    edges = np.arange(
        args.min_logm,
        args.max_logm + 0.5 * args.bin_width,
        args.bin_width
    )

    selected = []
    bin_labels = {}

    def far_enough_from_selected(i):
        if not selected:
            return True
        d = periodic_distance(pos[np.asarray(selected, dtype=np.int64)], pos[i], box_mpc_h)
        return np.all(d >= args.min_selected_separation)

    for lo, hi in zip(edges[:-1], edges[1:]):
        in_bin_preferred = np.where(
            preferred & (logm >= lo) & (logm < hi)
        )[0]

        in_bin_all = np.where(
            base & (logm >= lo) & (logm < hi)
        )[0]

        rank_pref = in_bin_preferred[np.argsort(score[in_bin_preferred])[::-1]]
        rank_all = in_bin_all[np.argsort(score[in_bin_all])[::-1]]

        chosen = []

        # First pass: preferred halos that also satisfy selected-target separation.
        for i in rank_pref:
            i = int(i)
            if far_enough_from_selected(i):
                chosen.append(i)
                selected.append(i)
                bin_labels[i] = f"{lo:.1f}-{hi:.1f}"
                if len(chosen) == args.per_bin:
                    break

        # Second pass: relax the isolation preference if needed, but keep target separation.
        if len(chosen) < args.per_bin:
            for i in rank_all:
                i = int(i)
                if i in selected:
                    continue
                if far_enough_from_selected(i):
                    chosen.append(i)
                    selected.append(i)
                    bin_labels[i] = f"{lo:.1f}-{hi:.1f}"
                    if len(chosen) == args.per_bin:
                        break

        # Third pass: if the requested separation is too strict to fill the bin,
        # fill with the best remaining halos and warn the user in the terminal.
        if len(chosen) < args.per_bin:
            print(
                f"WARNING: mass bin {lo:.1f}-{hi:.1f} could only supply "
                f"{len(chosen)}/{args.per_bin} halos with "
                f"{args.min_selected_separation:.2f} Mpc/h target separation.",
                file=sys.stderr
            )
            for i in rank_all:
                i = int(i)
                if i in selected:
                    continue
                chosen.append(i)
                selected.append(i)
                bin_labels[i] = f"{lo:.1f}-{hi:.1f}"
                if len(chosen) == args.per_bin:
                    break

    # Rank all base candidates globally for the full CSV.
    ranked_all = candidate_idx[np.argsort(score[candidate_idx])[::-1]]

    def row_for(i, selected_flag=False):
        j = nearest_idx[i]
        nmass = nearest_mass[i]
        return {
            "group_index": int(i),
            "selected": int(selected_flag),
            "mass_bin_log10": bin_labels.get(int(i), ""),
            "M200c_Msun": m200_msun[i],
            "log10_M200c_Msun": logm[i],
            "R200c_kpc_h": r200_mpc_h[i] * 1000.0,
            "GroupLen": int(group_len[i]),
            "GroupNsubs": int(nsubs[i]),
            "x_Mpc_h": pos[i, 0],
            "y_Mpc_h": pos[i, 1],
            "z_Mpc_h": pos[i, 2],
            "edge_distance_Mpc_h": edge_dist[i],
            "neighbor_group_index": int(j),
            "neighbor_M200c_Msun": nmass,
            "neighbor_mass_ratio": nmass / m200_msun[i] if np.isfinite(nmass) else np.nan,
            "neighbor_distance_Mpc_h": nearest_dist[i],
            "neighbor_distance_over_R200c": isolation_r200[i],
            "score": score[i],
        }

    fields = [
        "group_index", "selected", "mass_bin_log10",
        "M200c_Msun", "log10_M200c_Msun", "R200c_kpc_h",
        "GroupLen", "GroupNsubs",
        "x_Mpc_h", "y_Mpc_h", "z_Mpc_h",
        "edge_distance_Mpc_h",
        "neighbor_group_index", "neighbor_M200c_Msun",
        "neighbor_mass_ratio", "neighbor_distance_Mpc_h",
        "neighbor_distance_over_R200c", "score",
    ]

    csv_all = args.output_prefix + "_all_ranked.csv"
    with open(csv_all, "w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=fields)
        w.writeheader()
        selected_set = set(selected)
        for i in ranked_all:
            w.writerow(row_for(int(i), int(i) in selected_set))

    csv_sel = args.output_prefix + "_selected.csv"
    with open(csv_sel, "w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=fields)
        w.writeheader()
        for i in selected:
            w.writerow(row_for(i, True))

    txt_sel = args.output_prefix + "_selected.txt"
    with open(txt_sel, "w") as fh:
        fh.write("# Recommended parent-box halos for MUSIC2 zooms\n")
        fh.write("# Coordinates are in Mpc/h.\n")
        fh.write("# Masses are M200c in physical Msun.\n")
        fh.write("# Isolation neighbor has M >= "
                 f"{args.neighbor_mass_frac:.3f} * candidate M200c.\n")
        fh.write("#\n")
        fh.write(
            "# idx  bin          logM200   M200[Msun]      "
            "x[Mpc/h]   y[Mpc/h]   z[Mpc/h]   edge   "
            "d_neigh   d/R200   Npart   Nsubs\n"
        )
        for i in selected:
            fh.write(
                f"{i:6d} "
                f"{bin_labels.get(i,''):>9s} "
                f"{logm[i]:9.4f} "
                f"{m200_msun[i]:12.5e} "
                f"{pos[i,0]:10.5f} "
                f"{pos[i,1]:10.5f} "
                f"{pos[i,2]:10.5f} "
                f"{edge_dist[i]:7.3f} "
                f"{nearest_dist[i]:8.3f} "
                f"{isolation_r200[i]:8.2f} "
                f"{group_len[i]:8d} "
                f"{nsubs[i]:6d}\n"
            )

    print("")
    print("=== Recommended candidates ===")
    print(
        " idx    mass bin     logM200       M200 [Msun]       "
        "x       y       z      edge   d_neigh  d/R200  Npart"
    )
    print("-" * 115)

    for i in selected:
        print(
            f"{i:6d}  "
            f"{bin_labels.get(i,''):>9s}  "
            f"{logm[i]:8.3f}  "
            f"{m200_msun[i]:12.4e}  "
            f"{pos[i,0]:7.2f} "
            f"{pos[i,1]:7.2f} "
            f"{pos[i,2]:7.2f} "
            f"{edge_dist[i]:7.2f} "
            f"{nearest_dist[i]:8.2f} "
            f"{isolation_r200[i]:7.1f} "
            f"{group_len[i]:7d}"
        )

    print("")
    print(f"Selected halos           : {len(selected)}")
    print(f"Wrote                    : {csv_sel}")
    print(f"Wrote                    : {txt_sel}")
    print(f"Wrote full ranked sample : {csv_all}")
    print("")
    print("Interpretation:")
    print("  d_neigh = periodic distance to nearest halo with")
    print(f"            M200 >= {args.neighbor_mass_frac:.2f} * candidate M200")
    print("  d/R200  = d_neigh divided by candidate R200c")
    print("  edge    = ordinary distance to nearest box face (not periodic)")
    print(f"  selected targets are kept >= {args.min_selected_separation:.2f} Mpc/h apart when possible")
    print("")


if __name__ == "__main__":
    main()
