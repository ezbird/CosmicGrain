#!/usr/bin/env python3
"""
cosmicgrain_trace_highrv_agb_particles.py

Trace the AGB-origin dust particles identified by
cosmicgrain_highrv_agb_history_250pc.py backward through all available
snapshots of the CURRENT CG10 run.

The target particle IDs come from:
    <prefix>_particles.txt

For each target ParticleID and snapshot, record:
  scale factor, redshift, cosmic time, time since DustFormationTime,
  GrainRadius, CarbonMassFraction, DustTemperature, dust mass, DustSource,
  position relative to the z=0 CG10 center in the z=0 face-on frame.

This is deliberately ParticleID-based: no nearest-neighbour matching.

The script also reports:
  * first snapshot in which each target is found
  * whether DustSource remains AGB
  * evolution of radius/carbon fraction with time since formation
  * ensemble medians in age bins
  * an approximate "injection-like" lifetime, defined by configurable
    thresholds in radius and carbon fraction.

HubbleParam is read from Parameters, matching this CosmicGrain dataset.
"""

import argparse
import glob
import re
from pathlib import Path

import h5py
import numpy as np
import matplotlib.pyplot as plt

MPC_KM = 3.0856775814913673e19
SEC_PER_GYR = 365.25 * 86400.0 * 1.0e9


def natural_key(s):
    return [int(t) if t.isdigit() else t.lower()
            for t in re.split(r"(\d+)", s)]


def cosmic_age_gyr(a, h, om, ol):
    a = np.asarray(a, dtype=float)
    H0 = 100.0 * h / MPC_KM
    pref = 2.0 / (3.0 * H0 * np.sqrt(ol))
    return pref * np.arcsinh(np.sqrt(ol / om) * a**1.5) / SEC_PER_GYR


def periodic_delta(coords, center, box):
    d = coords - center[None, :]
    d -= box * np.rint(d / box)
    return d


def snapshot_number(fn):
    m = re.search(r"snapshot_(\d+)", fn)
    return int(m.group(1)) if m else -1


def read_targets(fn):
    # First column written by the previous diagnostic is ParticleID.
    arr = np.loadtxt(fn)
    if arr.ndim == 1:
        arr = arr[None, :]
    # IDs were written numerically; round safely back to uint64.
    ids = np.rint(arr[:, 0]).astype(np.uint64)
    return np.unique(ids)


def discover_snapshots(root):
    pats = [
        str(Path(root) / "snapdir_*" / "snapshot_*.0.hdf5"),
        str(Path(root) / "snapshot_*.0.hdf5"),
    ]
    leaders = []
    for pat in pats:
        leaders.extend(glob.glob(pat))
    leaders = sorted(set(leaders), key=natural_key)
    if not leaders:
        raise FileNotFoundError(f"No snapshot leaders found under {root}")

    groups = []
    for leader in leaders:
        # Convert ...snapshot_029.0.hdf5 -> ...snapshot_029.*.hdf5
        stem = re.sub(r"\.0\.hdf5$", ".*.hdf5", leader)
        files = sorted(glob.glob(stem), key=natural_key)
        if files:
            groups.append((snapshot_number(leader), files))
    return groups


def load_master_geometry(npz):
    m = np.load(npz)
    return (np.asarray(m["center"], float),
            np.asarray(m["rotation"], float))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("output_root",
                    help="e.g. ../halo441_output_1024")
    ap.add_argument("target_particles",
                    help="..._particles.txt from high-RV AGB history diagnostic")
    ap.add_argument("master_npz",
                    help="z=0 master map NPZ for center/rotation")
    ap.add_argument("--omega-m", type=float, default=0.3158)
    ap.add_argument("--omega-l", type=float, default=0.6842)
    ap.add_argument("--injection-radius-min", type=float, default=90.0,
                    help="nm; injection-like if radius >= this")
    ap.add_argument("--injection-carbon-min", type=float, default=0.55,
                    help="injection-like if carbon fraction >= this")
    ap.add_argument("--prefix",
                    default="halo441_current_highrv_agb_particle_tracks")
    A = ap.parse_args()

    targets = read_targets(A.target_particles)
    target_set = set(int(x) for x in targets)
    center, rotation = load_master_geometry(A.master_npz)
    snaps = discover_snapshots(A.output_root)

    print("INPUT")
    print("=" * 78)
    print(f"target ParticleIDs = {len(targets)}")
    print(f"snapshots found    = {len(snaps)}")
    print(f"first/last snap    = {snaps[0][0]} / {snaps[-1][0]}")
    print(f"injection-like     = radius >= {A.injection_radius_min:g} nm "
          f"AND carbon >= {A.injection_carbon_min:g}")

    rows = []
    found_by_snap = []

    for snapnum, files in snaps:
        snap_rows = []
        a = box = h = None

        for fn in files:
            with h5py.File(fn, "r") as f:
                if a is None:
                    a = float(f["Header"].attrs["Time"])
                    box = float(f["Header"].attrs["BoxSize"])
                    h = float(f["Parameters"].attrs["HubbleParam"])

                if "PartType6" not in f:
                    continue

                g = f["PartType6"]
                ids = np.asarray(g["ParticleIDs"][:], dtype=np.uint64)

                # Efficient membership test for only 23 target IDs.
                q = np.isin(ids, targets)
                if not np.any(q):
                    continue

                ii = np.where(q)[0]
                coords = np.asarray(g["Coordinates"][ii], float)
                mass = np.asarray(g["Masses"][ii], float)
                radius = np.asarray(g["GrainRadius"][ii], float)
                carbon = np.asarray(g["CarbonMassFraction"][ii], float)
                source = np.asarray(g["DustSource"][ii], np.int64)
                form = np.asarray(g["DustFormationTime"][ii], float)
                temp = np.asarray(g["DustTemperature"][ii], float)

                # Physical displacement from the z=0 center, represented in
                # the fixed z=0 face-on orientation. This is for visualization;
                # target matching itself depends only on ParticleID.
                xyz = periodic_delta(coords, center, box) * a / h
                xyz = xyz @ rotation.T

                t = float(cosmic_age_gyr(a, h, A.omega_m, A.omega_l))
                age = np.full(len(ii), np.nan)
                goodf = np.isfinite(form) & (form > 0) & (form <= a)
                age[goodf] = t - cosmic_age_gyr(
                    form[goodf], h, A.omega_m, A.omega_l
                )

                for j, loc in enumerate(ii):
                    pid = int(ids[loc])
                    snap_rows.append([
                        pid, snapnum, a, 1.0/a - 1.0, t,
                        form[j], age[j], mass[j], radius[j], carbon[j],
                        temp[j], int(source[j]),
                        xyz[j,0], xyz[j,1], xyz[j,2],
                        np.hypot(xyz[j,0], xyz[j,1]),
                        abs(xyz[j,2])
                    ])

        rows.extend(snap_rows)
        found_by_snap.append((snapnum, a, len(snap_rows)))
        print(f"snap {snapnum:03d}  a={a:.6f}  z={1/a-1:.4f}  "
              f"targets found={len(snap_rows):2d}/{len(targets)}")

    if not rows:
        raise RuntimeError("None of the target ParticleIDs were found.")

    data = np.asarray(rows, dtype=float)
    # columns
    PID,SNAP,AA,ZZ,TIME,FORM,AGE,MASS,RADIUS,CARBON,TEMP,SOURCE,X,Y,Z,RGAL,ABSZ = range(17)

    # Sort by ParticleID, then cosmic time.
    order = np.lexsort((data[:, TIME], data[:, PID]))
    data = data[order]

    header = (
        "ParticleID snapshot scale_factor redshift cosmic_time_Gyr "
        "formation_scale_factor age_since_formation_Gyr mass_code "
        "grain_radius_nm carbon_mass_fraction dust_temperature_K DustSource "
        "x_pkpc_z0frame y_pkpc_z0frame z_pkpc_z0frame "
        "Rgal_pkpc_z0frame absz_pkpc_z0frame"
    )
    np.savetxt(A.prefix + "_tracks.txt", data, header=header)

    print("\nPARTICLE TRACK SUMMARY")
    print("=" * 78)
    print("ParticleID       Nsnap first last   age_last[Gyr]  a_last[nm]  C_last  source_ok")
    per_particle = []

    for pid in targets:
        q = data[:, PID].astype(np.uint64) == pid
        d = data[q]
        if not len(d):
            print(f"{int(pid):<16d}  0    --    --")
            continue
        d = d[np.argsort(d[:, TIME])]
        source_ok = np.all(d[:, SOURCE] == 1)
        per_particle.append((pid, len(d), int(d[0,SNAP]), int(d[-1,SNAP]),
                             d[-1,AGE], d[-1,RADIUS], d[-1,CARBON], source_ok))
        print(f"{int(pid):<16d} {len(d):5d} {int(d[0,SNAP]):5d} {int(d[-1,SNAP]):4d}"
              f" {d[-1,AGE]:14.5f} {d[-1,RADIUS]:11.3f}"
              f" {d[-1,CARBON]:7.4f} {str(source_ok):>10s}")

    # Ensemble age bins.
    finite_age = np.isfinite(data[:, AGE]) & (data[:, AGE] >= 0)
    print("\nENSEMBLE EVOLUTION BY TIME SINCE FORMATION")
    print("=" * 78)
    bins = [(0,.1),(.1,.25),(.25,.5),(.5,1),(1,2),(2,4),(4,8)]
    print("age range [Gyr]       N    radius p16/med/p84 [nm]       C p16/med/p84")
    for lo,hi in bins:
        q=finite_age&(data[:,AGE]>=lo)&(data[:,AGE]<hi)
        if not q.any():
            continue
        rr=np.percentile(data[q,RADIUS],[16,50,84])
        cc=np.percentile(data[q,CARBON],[16,50,84])
        print(f"{lo:4.2f}-{hi:<4.2f}       {q.sum():4d}    "
              f"{rr[0]:7.2f}/{rr[1]:7.2f}/{rr[2]:7.2f}       "
              f"{cc[0]:.4f}/{cc[1]:.4f}/{cc[2]:.4f}")

    # Approximate lifetime for which each particle remains injection-like.
    # Because snapshots are discrete, report the maximum sampled age satisfying
    # the criterion, not an interpolated transition time.
    inj = finite_age & (data[:,RADIUS] >= A.injection_radius_min) & \
          (data[:,CARBON] >= A.injection_carbon_min)

    lifetimes = []
    print("\nSAMPLED INJECTION-LIKE LIFETIME")
    print("=" * 78)
    print("ParticleID       max sampled age satisfying criterion [Gyr]")
    for pid in targets:
        qpid = data[:,PID].astype(np.uint64)==pid
        q=qpid&inj
        life=np.max(data[q,AGE]) if q.any() else np.nan
        lifetimes.append(life)
        print(f"{int(pid):<16d} {life:12.5f}")

    lf=np.asarray(lifetimes,float)
    if np.isfinite(lf).any():
        p=np.nanpercentile(lf,[16,50,84])
        print(f"\nAcross targets: p16/median/p84 = "
              f"{p[0]:.4f} / {p[1]:.4f} / {p[2]:.4f} Gyr")
    print("NOTE: this is a snapshot-sampled lower-resolution diagnostic, "
          "not an exact transition time.")

    # Plot individual tracks plus ensemble.
    fig,ax=plt.subplots(figsize=(8,5.7))
    for pid in targets:
        q=(data[:,PID].astype(np.uint64)==pid)&finite_age
        if q.any():
            d=data[q]; o=np.argsort(d[:,AGE])
            ax.plot(d[o,AGE],d[o,RADIUS],alpha=.35,lw=1)
            ax.scatter(d[o,AGE],d[o,RADIUS],s=12,alpha=.5)
    ax.axhline(A.injection_radius_min,ls="--",lw=1)
    ax.set_xlabel("Time since dust formation (Gyr)")
    ax.set_ylabel("Grain radius (nm)")
    ax.set_title("CG10 high-R(V) AGB particle histories")
    fig.tight_layout()
    fig.savefig(A.prefix+"_radius_vs_age.png",dpi=220)
    plt.close(fig)

    fig,ax=plt.subplots(figsize=(8,5.7))
    for pid in targets:
        q=(data[:,PID].astype(np.uint64)==pid)&finite_age
        if q.any():
            d=data[q]; o=np.argsort(d[:,AGE])
            ax.plot(d[o,AGE],d[o,CARBON],alpha=.35,lw=1)
            ax.scatter(d[o,AGE],d[o,CARBON],s=12,alpha=.5)
    ax.axhline(A.injection_carbon_min,ls="--",lw=1)
    ax.set_xlabel("Time since dust formation (Gyr)")
    ax.set_ylabel("Carbon mass fraction")
    ax.set_title("CG10 high-R(V) AGB composition histories")
    fig.tight_layout()
    fig.savefig(A.prefix+"_carbon_vs_age.png",dpi=220)
    plt.close(fig)

    # Radius vs carbon, colored by age.
    q=finite_age
    fig,ax=plt.subplots(figsize=(7.2,5.8))
    sc=ax.scatter(data[q,RADIUS],data[q,CARBON],c=data[q,AGE],
                  s=24,alpha=.75)
    ax.axvline(A.injection_radius_min,ls="--",lw=1)
    ax.axhline(A.injection_carbon_min,ls="--",lw=1)
    ax.set_xlabel("Grain radius (nm)")
    ax.set_ylabel("Carbon mass fraction")
    ax.set_title("Evolution of z=0 high-R(V) AGB grains")
    cb=fig.colorbar(sc,ax=ax)
    cb.set_label("Time since formation (Gyr)")
    fig.tight_layout()
    fig.savefig(A.prefix+"_radius_carbon_age.png",dpi=220)
    plt.close(fig)

    print("\nOUTPUTS")
    print("=" * 78)
    print(A.prefix+"_tracks.txt")
    print(A.prefix+"_radius_vs_age.png")
    print(A.prefix+"_carbon_vs_age.png")
    print(A.prefix+"_radius_carbon_age.png")


if __name__ == "__main__":
    main()
