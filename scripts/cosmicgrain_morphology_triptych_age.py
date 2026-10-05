#!/usr/bin/env python3
"""Make categorical and continuous-age stellar morphology triptychs."""
import argparse
import glob

import h5py
import matplotlib.pyplot as plt
from matplotlib.colors import Normalize
from matplotlib.patches import Patch
import numpy as np

HUBBLE_TIME_GYR = 9.778
FORMATION_FIELDS = ("StellarFormationTime", "GFM_StellarFormationTime", "FormationTime")


def scalar(value):
    return float(np.asarray(value).reshape(-1)[0])


def attribute(header, parameters, names, default=None):
    for name in names:
        if parameters is not None and name in parameters.attrs:
            return scalar(parameters.attrs[name])
        if name in header.attrs:
            return scalar(header.attrs[name])
    if default is not None:
        return default
    raise RuntimeError("Missing snapshot attribute: " + ", ".join(names))


def snapshot_info(filename):
    with h5py.File(filename, "r") as handle:
        header = handle["Header"]
        parameters = handle.get("Parameters")
        scale_factor = scalar(header.attrs["Time"])
        box = scalar(header.attrs["BoxSize"])
        hubble = attribute(header, parameters, ("HubbleParam",))
        omega_m = attribute(header, parameters, ("Omega0", "OmegaM", "OmegaMatter"))
        omega_l = attribute(
            header, parameters, ("OmegaLambda", "OmegaL", "OmegaVacuum"),
            default=1.0 - omega_m,
        )
    return scale_factor, box, hubble, omega_m, omega_l


def formation_field(files):
    for filename in files:
        with h5py.File(filename, "r") as handle:
            if "PartType4" not in handle:
                continue
            for field in FORMATION_FIELDS:
                if field in handle["PartType4"]:
                    return field
    raise RuntimeError("No stellar formation-time field found in PartType4")


def load(files, particle_type, velocity=False, extra=None):
    coordinates, masses, velocities, extra_values = [], [], [], []
    for filename in files:
        with h5py.File(filename, "r") as handle:
            group_name = f"PartType{particle_type}"
            if group_name not in handle:
                continue
            group = handle[group_name]
            count = len(group["Coordinates"])
            coordinates.append(group["Coordinates"][:])
            if "Masses" in group:
                masses.append(group["Masses"][:])
            else:
                mass = handle["Header"].attrs["MassTable"][particle_type]
                if mass <= 0:
                    raise RuntimeError(f"No mass data for PartType{particle_type}")
                masses.append(np.full(count, mass))
            if velocity:
                velocities.append(group["Velocities"][:])
            if extra:
                if extra not in group:
                    raise RuntimeError(f"{group_name}/{extra} missing in {filename}")
                extra_values.append(group[extra][:])
    if not coordinates:
        return None
    result = {"coordinates": np.concatenate(coordinates), "masses": np.concatenate(masses)}
    if velocity:
        result["velocities"] = np.concatenate(velocities)
    if extra:
        result[extra] = np.concatenate(extra_values)
    return result


def periodic_delta(coordinates, center, box):
    delta = coordinates.astype(float) - center
    delta -= box * np.rint(delta / box)
    return delta


def rotation_matrix(angular_momentum):
    zhat = angular_momentum / np.linalg.norm(angular_momentum)
    reference = np.array([0.0, 0.0, 1.0])
    if abs(reference @ zhat) > 0.9:
        reference = np.array([0.0, 1.0, 0.0])
    xhat = np.cross(reference, zhat)
    xhat /= np.linalg.norm(xhat)
    yhat = np.cross(zhat, xhat)
    return np.vstack((xhat, yhat, zhat))


def cosmic_age(scale_factor, hubble, omega_m, omega_l):
    scale_factor = np.asarray(scale_factor, dtype=float)
    if omega_l == 0:
        return HUBBLE_TIME_GYR / hubble * 2 * scale_factor**1.5 / (3 * np.sqrt(omega_m))
    prefactor = HUBBLE_TIME_GYR / hubble * 2 / (3 * np.sqrt(omega_l))
    return prefactor * np.arcsinh(np.sqrt(omega_l / omega_m) * scale_factor**1.5)


def histogram2d(x, y, weights, xlim, ylim, nx, ny):
    values, _, _ = np.histogram2d(
        x, y, bins=(nx, ny), range=(xlim, ylim), weights=weights,
    )
    return values.T


def stretch(values, percentile=99.5, softness=0.03):
    positive = values[values > 0]
    if positive.size == 0:
        return np.zeros_like(values, dtype=float)
    high = np.percentile(positive, percentile)
    if high <= 0:
        return np.zeros_like(values, dtype=float)
    return np.clip(np.arcsinh(values / (softness * high)) / np.arcsinh(1 / softness), 0, 1)


def categorical_rgb(x, y, mass, ages, xlim, ylim, nx, ny, cuts):
    colors = np.array([
        [0.05, 0.75, 1.00],  # young: cyan-blue
        [1.00, 0.58, 0.08],  # intermediate: orange
        [0.90, 0.08, 0.22],  # old: crimson
    ])
    selections = (ages < cuts[0], (ages >= cuts[0]) & (ages < cuts[1]), ages >= cuts[1])
    layers = []
    for selected in selections:
        density = histogram2d(x[selected], y[selected], mass[selected], xlim, ylim, nx, ny)
        layers.append(stretch(density))
    layers = np.asarray(layers)
    rgb = np.einsum("ihw,ic->hwc", layers, colors)
    maximum = np.max(rgb, axis=2, keepdims=True)
    rgb /= np.maximum(1.0, maximum)
    return np.clip(rgb, 0, 1), colors


def continuous_age_rgba(x, y, mass, ages, xlim, ylim, nx, ny, age_max):
    total = histogram2d(x, y, mass, xlim, ylim, nx, ny)
    weighted_age = histogram2d(x, y, mass * ages, xlim, ylim, nx, ny)
    mean_age = np.divide(weighted_age, total, out=np.full_like(total, np.nan), where=total > 0)
    # Plasma has no pale midpoint, so intermediate ages retain a clear hue.
    color_values = plt.get_cmap("plasma")(np.clip(mean_age / age_max, 0, 1))[:, :, :3]
    brightness = stretch(total)
    # Black background; surface density controls brightness and age controls hue.
    rgba = np.zeros((*total.shape, 4))
    rgba[:, :, :3] = color_values * brightness[:, :, None]
    rgba[:, :, 3] = 1.0
    return rgba, mean_age


def gas_image(x, y, mass, xlim, ylim, nx, ny):
    density = histogram2d(x, y, mass, xlim, ylim, nx, ny)
    return stretch(density)


def style_axis(axis, title, xlim, ylim, xlabel, ylabel):
    axis.set_title(title)
    axis.set_xlim(xlim)
    axis.set_ylim(ylim)
    axis.set_xlabel(xlabel)
    axis.set_ylabel(ylabel)
    axis.set_aspect("equal")


parser = argparse.ArgumentParser()
parser.add_argument("snapshot")
parser.add_argument("--center", nargs=3, type=float, required=True)
parser.add_argument("--orientation-radius", type=float, default=20.0)
parser.add_argument("--xy-half-size", type=float, default=25.0)
parser.add_argument("--z-half-size", type=float, default=10.0)
parser.add_argument("--stellar-zmax", type=float, default=25.0,
                    help="Face-on stellar projection half-depth in kpc")
parser.add_argument("--gas-zmax", type=float, default=20.0,
                    help="Face-on gas projection half-depth in kpc")
parser.add_argument("--pixels", type=int, default=220)
parser.add_argument("--age-cuts", nargs=2, type=float, default=[2.0, 6.0],
                    metavar=("YOUNG_MAX", "OLD_MIN"))
parser.add_argument("--continuous-age-max", type=float, default=12.0)
parser.add_argument("--prefix", default="stellar_morphology_age")
args = parser.parse_args()

if args.age_cuts[0] <= 0 or args.age_cuts[1] <= args.age_cuts[0]:
    raise ValueError("--age-cuts must be positive and strictly increasing")

files = sorted(glob.glob(args.snapshot))
if not files:
    raise FileNotFoundError(args.snapshot)
scale_factor, box, hubble, omega_m, omega_l = snapshot_info(files[0])
center = np.asarray(args.center, dtype=float)
age_field = formation_field(files)
stars = load(files, 4, velocity=True, extra=age_field)
gas = load(files, 0)
if stars is None or gas is None:
    raise RuntimeError("Both PartType4 stars and PartType0 gas are required")

star_position = periodic_delta(stars["coordinates"], center, box) * scale_factor / hubble
gas_position = periodic_delta(gas["coordinates"], center, box) * scale_factor / hubble
star_mass = stars["masses"].astype(float) * 1e10 / hubble
gas_mass = gas["masses"].astype(float) * 1e10 / hubble
velocity = stars["velocities"].astype(float)
formation_scale_factor = stars[age_field].astype(float)
valid = (formation_scale_factor > 0) & (formation_scale_factor <= scale_factor * (1 + 1e-6))
ages = np.full(len(star_mass), np.nan)
ages[valid] = cosmic_age(scale_factor, hubble, omega_m, omega_l) - cosmic_age(
    formation_scale_factor[valid], hubble, omega_m, omega_l,
)
ages[(ages < 0) & (ages > -1e-6)] = 0

radius3d = np.linalg.norm(star_position, axis=1)
orient = valid & (radius3d < args.orientation_radius)
if orient.sum() < 3:
    raise RuntimeError("Too few valid stars inside the orientation radius")
bulk_velocity = np.average(velocity[orient], axis=0, weights=star_mass[orient])
angular_momentum = np.sum(
    np.cross(star_position[orient], velocity[orient] - bulk_velocity)
    * star_mass[orient, None], axis=0,
)
rotation = rotation_matrix(angular_momentum)
star_rotated = star_position @ rotation.T
gas_rotated = gas_position @ rotation.T

star_keep = valid & (np.abs(star_rotated[:, 2]) < args.stellar_zmax)
gas_keep = np.abs(gas_rotated[:, 2]) < args.gas_zmax
edge_keep = valid & (np.abs(star_rotated[:, 1]) < args.stellar_zmax)
xlim = (-args.xy_half_size, args.xy_half_size)
ylim = xlim
zlim = (-args.z_half_size, args.z_half_size)
nx = args.pixels
ny_edge = max(60, int(round(args.pixels * args.z_half_size / args.xy_half_size)))

face_rgb, category_colors = categorical_rgb(
    star_rotated[star_keep, 0], star_rotated[star_keep, 1], star_mass[star_keep], ages[star_keep],
    xlim, ylim, nx, nx, args.age_cuts,
)
edge_rgb, _ = categorical_rgb(
    star_rotated[edge_keep, 0], star_rotated[edge_keep, 2], star_mass[edge_keep], ages[edge_keep],
    xlim, zlim, nx, ny_edge, args.age_cuts,
)
gas_face = gas_image(
    gas_rotated[gas_keep, 0], gas_rotated[gas_keep, 1], gas_mass[gas_keep],
    xlim, ylim, nx, nx,
)

# Unblended 2x3 stellar decomposition.  Every population receives its own
# density stretch because the young component contains much less mass.
category_selections = (
    valid & (ages < args.age_cuts[0]),
    valid & (ages >= args.age_cuts[0]) & (ages < args.age_cuts[1]),
    valid & (ages >= args.age_cuts[1]),
)
category_titles = (
    rf"Young: $< {args.age_cuts[0]:g}$ Gyr",
    rf"Intermediate: ${args.age_cuts[0]:g}$–${args.age_cuts[1]:g}$ Gyr",
    rf"Old: $> {args.age_cuts[1]:g}$ Gyr",
)
face_density_by_age = []
edge_density_by_age = []
for selected in category_selections:
    face_selected = selected & (np.abs(star_rotated[:, 2]) < args.stellar_zmax)
    edge_selected = selected & (np.abs(star_rotated[:, 1]) < args.stellar_zmax)
    face_density_by_age.append(histogram2d(
        star_rotated[face_selected, 0], star_rotated[face_selected, 1],
        star_mass[face_selected], xlim, ylim, nx, nx,
    ))
    edge_density_by_age.append(histogram2d(
        star_rotated[edge_selected, 0], star_rotated[edge_selected, 2],
        star_mass[edge_selected], xlim, zlim, nx, ny_edge,
    ))

fig, axes = plt.subplots(2, 3, figsize=(13.8, 8.0))
for column, (title, face_density, edge_density) in enumerate(zip(
        category_titles, face_density_by_age, edge_density_by_age)):
    axes[0, column].imshow(stretch(face_density), origin="lower", extent=(*xlim, *ylim),
                           cmap="magma", vmin=0, vmax=1, interpolation="nearest")
    axes[1, column].imshow(stretch(edge_density), origin="lower", extent=(*xlim, *zlim),
                           cmap="magma", vmin=0, vmax=1, interpolation="nearest")
    axes[0, column].set_title(title)
    style_axis(axes[0, column], title, xlim, ylim, "$x'$ (kpc)", "$y'$ (kpc)")
    style_axis(axes[1, column], "", xlim, zlim, "$x'$ (kpc)", "$z'$ (kpc)")
axes[0, 0].text(-0.18, 0.5, "Face-on", transform=axes[0, 0].transAxes,
                rotation=90, ha="center", va="center", fontsize=13, fontweight="bold")
axes[1, 0].text(-0.18, 0.5, "Edge-on", transform=axes[1, 0].transAxes,
                rotation=90, ha="center", va="center", fontsize=13, fontweight="bold")
fig.suptitle("Stellar morphology by age", fontsize=15)
fig.text(0.5, 0.015, "Each panel is independently surface-density stretched",
         ha="center", color="0.35")
fig.tight_layout(rect=(0.03, 0.04, 1, 0.96))
fig.savefig(args.prefix + "_age_decomposition.png", dpi=240, bbox_inches="tight")
fig.savefig(args.prefix + "_age_decomposition.pdf", bbox_inches="tight")
plt.close(fig)

# The gas is kept as a separate reference image rather than interrupting the
# stellar age sequence.
fig, axis = plt.subplots(figsize=(6.2, 5.2))
image = axis.imshow(gas_face, origin="lower", extent=(*xlim, *ylim), cmap="viridis",
                    vmin=0, vmax=1, interpolation="nearest")
style_axis(axis, "Gas — Face-on", xlim, ylim, "$x'$ (kpc)", "$y'$ (kpc)")
fig.colorbar(image, ax=axis, label="Stretched gas surface density")
fig.tight_layout()
fig.savefig(args.prefix + "_gas_faceon.png", dpi=240, bbox_inches="tight")
fig.savefig(args.prefix + "_gas_faceon.pdf", bbox_inches="tight")
plt.close(fig)

fig, axes = plt.subplots(1, 3, figsize=(15.0, 5.1), gridspec_kw={"width_ratios": [1, 1, 1]})
axes[0].imshow(face_rgb, origin="lower", extent=(*xlim, *ylim), interpolation="nearest")
axes[1].imshow(gas_face, origin="lower", extent=(*xlim, *ylim), cmap="viridis", vmin=0, vmax=1,
               interpolation="nearest")
axes[2].imshow(edge_rgb, origin="lower", extent=(*xlim, *zlim), interpolation="nearest")
style_axis(axes[0], "Stars — Face-on, colored by age", xlim, ylim, "$x'$ (kpc)", "$y'$ (kpc)")
style_axis(axes[1], "Gas — Face-on", xlim, ylim, "$x'$ (kpc)", "$y'$ (kpc)")
style_axis(axes[2], "Stars — Edge-on, colored by age", xlim, zlim, "$x'$ (kpc)", "$z'$ (kpc)")
cuts = args.age_cuts
legend = [
    Patch(color=category_colors[0], label=rf"Young: $< {cuts[0]:g}$ Gyr"),
    Patch(color=category_colors[1], label=rf"Intermediate: ${cuts[0]:g}$–${cuts[1]:g}$ Gyr"),
    Patch(color=category_colors[2], label=rf"Old: $> {cuts[1]:g}$ Gyr"),
]
fig.legend(handles=legend, loc="lower center", ncol=3, frameon=False, bbox_to_anchor=(0.5, -0.01))
fig.tight_layout(rect=(0, 0.07, 1, 1))
fig.savefig(args.prefix + "_categorical.png", dpi=240, bbox_inches="tight")
fig.savefig(args.prefix + "_categorical.pdf", bbox_inches="tight")
plt.close(fig)

face_rgba, face_mean_age = continuous_age_rgba(
    star_rotated[star_keep, 0], star_rotated[star_keep, 1], star_mass[star_keep], ages[star_keep],
    xlim, ylim, nx, nx, args.continuous_age_max,
)
edge_rgba, edge_mean_age = continuous_age_rgba(
    star_rotated[edge_keep, 0], star_rotated[edge_keep, 2], star_mass[edge_keep], ages[edge_keep],
    xlim, zlim, nx, ny_edge, args.continuous_age_max,
)
fig, axes = plt.subplots(1, 3, figsize=(15.0, 5.1), gridspec_kw={"width_ratios": [1, 1, 1]})
axes[0].imshow(face_rgba, origin="lower", extent=(*xlim, *ylim), interpolation="nearest")
axes[1].imshow(gas_face, origin="lower", extent=(*xlim, *ylim), cmap="viridis", vmin=0, vmax=1,
               interpolation="nearest")
axes[2].imshow(edge_rgba, origin="lower", extent=(*xlim, *zlim), interpolation="nearest")
style_axis(axes[0], "Stars — Face-on, mean age", xlim, ylim, "$x'$ (kpc)", "$y'$ (kpc)")
style_axis(axes[1], "Gas — Face-on", xlim, ylim, "$x'$ (kpc)", "$y'$ (kpc)")
style_axis(axes[2], "Stars — Edge-on, mean age", xlim, zlim, "$x'$ (kpc)", "$z'$ (kpc)")
mapper = plt.cm.ScalarMappable(norm=Normalize(0, args.continuous_age_max), cmap="plasma")
mapper.set_array([])
colorbar = fig.colorbar(mapper, ax=[axes[0], axes[2]], orientation="horizontal",
                        fraction=0.055, pad=0.14, aspect=35)
colorbar.set_label("Mass-weighted mean stellar age (Gyr)")
fig.subplots_adjust(bottom=0.19, wspace=0.28)
fig.savefig(args.prefix + "_continuous.png", dpi=240, bbox_inches="tight")
fig.savefig(args.prefix + "_continuous.pdf", bbox_inches="tight")
plt.close(fig)

np.savez_compressed(
    args.prefix + "_maps.npz", faceon_mean_age_gyr=face_mean_age,
    edgeon_mean_age_gyr=edge_mean_age, categorical_faceon_rgb=face_rgb,
    categorical_edgeon_rgb=edge_rgb, gas_faceon_stretch=gas_face,
    faceon_density_by_age=np.asarray(face_density_by_age),
    edgeon_density_by_age=np.asarray(edge_density_by_age),
    age_cuts_gyr=np.asarray(args.age_cuts), j_hat=angular_momentum / np.linalg.norm(angular_momentum),
    rotation_matrix=rotation, formation_time_field=age_field,
)

inside30 = valid & (np.linalg.norm(star_position, axis=1) < 30)
print("=" * 78)
print("AGE-COLORED MORPHOLOGY TRIPTYCH")
print("=" * 78)
print(f"a={scale_factor:.8f} z={1/scale_factor-1:.8f} h={hubble:.8f}")
print(f"formation-time field: PartType4/{age_field}")
print("j_hat =", angular_momentum / np.linalg.norm(angular_momentum))
print(f"valid stellar ages: {valid.sum():,}/{len(valid):,}")
print(f"Nstar(<30 pkpc)={inside30.sum():,}")
for label, selected in (
    (f"age < {cuts[0]:g} Gyr", valid & (ages < cuts[0])),
    (f"{cuts[0]:g} <= age < {cuts[1]:g} Gyr", valid & (ages >= cuts[0]) & (ages < cuts[1])),
    (f"age >= {cuts[1]:g} Gyr", valid & (ages >= cuts[1])),
):
    local = selected & inside30
    print(f"{label}: N={local.sum():,}, Mstar={star_mass[local].sum():.6e} Msun")
print("\nSaved 2x3 age decomposition, separate gas map, categorical and continuous-age PNG/PDF figures, plus map NPZ.")
