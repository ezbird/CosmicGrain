#!/usr/bin/env python3
"""
Plot stellar evolution from Gadget-4 simulation
Uses official Behroozi et al. 2013 stellar-to-halo mass relation via halotools
"""
import numpy as np
import matplotlib.pyplot as plt
import sys
import os
from halotools.empirical_models import Behroozi10SmHm

# Argument handling
if len(sys.argv) < 2:
    print("\n" + "="*70)
    print("🌌 Oops! You forgot to tell me where your stellar evolution data is! 🌌")
    print("="*70)
    print("\nUsage:")
    print("  python plot_stellar_evolution.py <output_directory>")
    print("\nExample:")
    print("  python plot_stellar_evolution.py ../5_output_zoom_512_halo569_50Mpc_dust/")
    print("\n" + "="*70 + "\n")
    sys.exit(1)

output_dir = sys.argv[1].rstrip('/')
data_file = os.path.join(output_dir, 'stellar_evolution.txt')

if not os.path.exists(data_file):
    print(f"\n❌ Error: Could not find {data_file}")
    print(f"   Make sure the stellar_evolution.txt file exists in {output_dir}/\n")
    sys.exit(1)

print(f"\n📊 Loading data from: {data_file}")

# =============================================================================
# BEHROOZI ET AL. 2013 - OFFICIAL IMPLEMENTATION
# =============================================================================

def cosmic_age_approx(z):
    """Approximate cosmic age in Gyr (simple fitting formula)"""
    a = 1.0 / (1.0 + z)
    t = 13.8 * (0.88 * a**(3/2) + 0.12)
    return t

# Initialize Behroozi+2013 model
behroozi_model = Behroozi10SmHm()

def get_behroozi_shmr(M_halo, z):
    """
    Get stellar-to-halo mass relation using official Behroozi et al. 2013
    
    Parameters:
    -----------
    M_halo : array-like
        Halo mass in M_sun (h=1 units)
    z : array-like
        Redshift
    
    Returns:
    --------
    M_star : array
        Stellar mass in M_sun
    ratio : array
        M_star / M_halo
    """
    M_halo = np.atleast_1d(M_halo)
    z = np.atleast_1d(z)
    
    # Ensure arrays are same shape
    if len(M_halo) == 1 and len(z) > 1:
        M_halo = np.full_like(z, M_halo[0])
    elif len(z) == 1 and len(M_halo) > 1:
        z = np.full_like(M_halo, z[0])
    
    # Get stellar mass from Behroozi model
    M_star = np.array([behroozi_model.mean_stellar_mass(prim_haloprop=mh, redshift=zz) 
                       for mh, zz in zip(M_halo, z)])
    
    ratio = M_star / M_halo
    return M_star, ratio

# =============================================================================
# LOAD DATA
# =============================================================================

data = np.loadtxt(data_file)
time = data[:,0]
m_star = data[:,1]
m_halo = data[:,2]
ratio = data[:,3]

# Check if gas data is available (6 columns instead of 4)
has_gas = (data.shape[1] >= 5)
if has_gas:
    m_gas = data[:,4]
    if data.shape[1] >= 6:
        f_baryon = data[:,5]
    else:
        f_baryon = (m_star + m_gas) / m_halo
    print("✓ Gas mass data found!")
else:
    print("⚠ No gas mass data - run with updated C++ code for gas tracking")
    m_gas = None
    f_baryon = None

z = 1/time - 1

print(f"✓ Loaded {len(time)} snapshots")
print(f"  Redshift range: z = {z.max():.2f} → {z.min():.2f}")
print(f"  Final M_star:    {m_star[-1]:.2e} M_☉")
if has_gas:
    print(f"  Final M_gas:     {m_gas[-1]:.2e} M_☉")
    print(f"  Final f_baryon:  {f_baryon[-1]:.4f} ({f_baryon[-1]*100:.2f}%)")
print(f"  Final M_halo:    {m_halo[-1]:.2e} M_☉")
print(f"  Final M*/M_halo: {ratio[-1]:.4f} ({ratio[-1]*100:.2f}%)")

# =============================================================================
# CALCULATE BEHROOZI PREDICTIONS
# =============================================================================

print("\n🔍 Computing Behroozi+2013 predictions...")

# Behroozi relation valid only for z <= 8
valid_mask = z <= 8.0
z_valid = z[valid_mask]
m_halo_valid = m_halo[valid_mask]

# Prediction for this halo's actual mass evolution
if len(z_valid) > 0:
    _, behroozi_ratio_valid = get_behroozi_shmr(m_halo_valid, z_valid)
    print(f"  ✓ Computed predictions for your halo's mass evolution")
else:
    behroozi_ratio_valid = np.array([])
    print(f"  ⚠ All redshifts > 8, skipping Behroozi predictions")

# Reference predictions for fixed MW and LMC masses
z_range = np.linspace(0, min(z.max(), 8.0), 100)
M_halo_mw = np.full_like(z_range, 1.2e12)
M_halo_lmc = np.full_like(z_range, 1.5e11)

_, mw_ratio = get_behroozi_shmr(M_halo_mw, z_range)
_, lmc_ratio = get_behroozi_shmr(M_halo_lmc, z_range)
print(f"  ✓ Computed reference curves for MW and LMC masses")

# Observational data at z=0
obs_data = {
    'Milky Way': {'M_star': 6e10, 'M_halo': 1.2e12, 'ratio': 0.05, 'color': 'red', 'marker': '*', 'size': 300},
    'LMC': {'M_star': 2.5e9, 'M_halo': 1.5e11, 'ratio': 0.017, 'color': 'blue', 'marker': 's', 'size': 150},
    'SMC': {'M_star': 5e8, 'M_halo': 4e10, 'ratio': 0.012, 'color': 'green', 'marker': 'D', 'size': 120}
}

# =============================================================================
# CREATE FIGURE - 2 PANELS ONLY
# =============================================================================

fig, axes = plt.subplots(2, 1, figsize=(10, 8))
ax1, ax2 = axes

# =============================================================================
# PANEL 1: MASSES VS REDSHIFT
# =============================================================================

ax1.invert_xaxis()

# Plot simulation data (will be in main legend)
line_star = ax1.plot(z, m_star, label='M$_*$ (simulation)', color='C0', linewidth=2.5, zorder=3)[0]
line_halo = ax1.plot(z, m_halo, label='M$_{halo}$ (simulation)', color='C1', linewidth=2.5, zorder=3)[0]

sim_lines = [line_star, line_halo]
if has_gas:
    line_gas = ax1.plot(z, m_gas, label='M$_{gas}$ (simulation)', color='C2', linewidth=2.5, zorder=3, linestyle='--')[0]
    line_baryon = ax1.plot(z, m_star + m_gas, label='M$_{baryon}$ (stars+gas)', color='purple', linewidth=2, zorder=3, alpha=0.7)[0]
    sim_lines.extend([line_gas, line_baryon])

# Plot observational data (will be in separate legend)
obs_handles = []
obs_labels = []
for name, obs in obs_data.items():
    h1 = ax1.scatter(0, obs['M_star'], marker=obs['marker'], s=obs['size'], 
                color=obs['color'], edgecolors='black', linewidth=1.5,
                zorder=5)
    h2 = ax1.scatter(0, obs['M_halo'], marker=obs['marker'], s=obs['size'], 
                color=obs['color'], edgecolors='black', linewidth=1.5,
                facecolors='none', zorder=5)
    obs_handles.extend([h1, h2])
    obs_labels.extend([f'{name} M$_*$', f'{name} M$_{{halo}}$'])

ax1.set_yscale('log')
ax1.set_xlabel('Redshift', fontsize=12)
ax1.set_ylabel('Mass (M$_\\odot$)', fontsize=12)

# Main legend (simulation data) - automatically find best location
leg1 = ax1.legend(sim_lines, [l.get_label() for l in sim_lines], 
                  fontsize=9, loc='best', framealpha=0.9)

# Observational data legend - lower left
leg2 = ax1.legend(obs_handles, obs_labels, fontsize=7.5, loc='lower left', 
                  ncol=1, framealpha=0.9, title='Observations (z=0)')
ax1.add_artist(leg1)  # Add first legend back since second one replaces it

ax1.grid(True, alpha=0.3, linestyle='--')
ax1.set_title(f'Stellar Evolution: {os.path.basename(output_dir)}', fontsize=13, fontweight='bold', pad=10)

# Info box - top left
textstr = f'Final (z={z[-1]:.2f}):\n'
textstr += f'M$_*$ = {m_star[-1]:.2e}\n'
if has_gas:
    textstr += f'M$_{{gas}}$ = {m_gas[-1]:.2e}\n'
textstr += f'M$_{{halo}}$ = {m_halo[-1]:.2e}'
props = dict(boxstyle='round', facecolor='wheat', alpha=0.8)
ax1.text(0.02, 0.98, textstr, transform=ax1.transAxes, fontsize=8,
         verticalalignment='top', bbox=props)

# =============================================================================
# PANEL 2: M*/M_HALO RATIO
# =============================================================================

ax2.invert_xaxis()
ax2.plot(z, ratio, color='darkblue', linewidth=2.5, label='Simulation', zorder=4)

# Behroozi prediction for YOUR halo's actual mass evolution
if len(z_valid) > 0:
    ax2.plot(z_valid, behroozi_ratio_valid, 'k--', linewidth=2, alpha=0.8, 
             label='Behroozi+2013\n(same M$_{halo}$(z))', zorder=3)

# Reference predictions for fixed MW and LMC masses
ax2.plot(z_range, mw_ratio, color='red', linewidth=1.5, alpha=0.6, linestyle=':', 
         label='Behroozi+2013\n(MW-mass: 1.2×10$^{12}$ M$_\\odot$)', zorder=2)
ax2.plot(z_range, lmc_ratio, color='blue', linewidth=1.5, alpha=0.6, linestyle=':', 
         label='Behroozi+2013\n(LMC-mass: 1.5×10$^{11}$ M$_\\odot$)', zorder=2)

# Plot observational points but DON'T add to legend (already in panel 1)
for name, obs in obs_data.items():
    ax2.scatter(0, obs['ratio'], marker=obs['marker'], s=obs['size'],
                color=obs['color'], edgecolors='black', linewidth=1.5, zorder=5)

# Mark peak on MW curve
peak_idx = np.argmax(mw_ratio)
ax2.scatter(z_range[peak_idx], mw_ratio[peak_idx], marker='o', s=100, 
           color='red', edgecolors='white', linewidth=2, zorder=6, alpha=0.7)

ax2.set_xlabel('Redshift', fontsize=12)
ax2.set_ylabel('M$_*$/M$_{halo}$', fontsize=12)
ax2.legend(fontsize=8.5, loc='best', framealpha=0.9)
ax2.grid(True, alpha=0.3, linestyle='--')
ax2.set_ylim(0, 0.06)

# Add cosmic age axis on top
ax2_top = ax2.twiny()
ax2_top.set_xlim(ax2.get_xlim())
z_ticks = np.array([0, 1, 2, 3, 4, 6, 8])
z_ticks = z_ticks[z_ticks <= z.max()]
ages = cosmic_age_approx(z_ticks)
ax2_top.set_xticks(z_ticks)
ax2_top.set_xticklabels([f'{a:.1f}' for a in ages], fontsize=9)
ax2_top.set_xlabel('Age of Universe (Gyr)', fontsize=11)

# =============================================================================
# SAVE AND PRINT DIAGNOSTICS
# =============================================================================

plt.tight_layout()

output_file = os.path.join(output_dir, 'stellar_evolution.png')
plt.savefig(output_file, dpi=150, bbox_inches='tight')

# Print diagnostics
print(f"\n📈 Comparison to Behroozi+2013:")
print(f"   Your final ratio:     {ratio[-1]:.4f} ({ratio[-1]*100:.2f}%)")

if len(z_valid) > 0 and len(behroozi_ratio_valid) > 0:
    print(f"   Expected ratio:       {behroozi_ratio_valid[-1]:.4f} ({behroozi_ratio_valid[-1]*100:.2f}%)")
    diff_percent = (ratio[-1]/behroozi_ratio_valid[-1] - 1)*100
    print(f"   Difference:           {diff_percent:+.1f}%")
    
    if abs(diff_percent) < 20:
        print(f"   ✓ Good agreement with observations!")
    elif abs(diff_percent) < 50:
        print(f"   ⚠ Moderate deviation from observations")
    else:
        print(f"   ⚠⚠ Large deviation from observations")

# Behroozi predictions at z=0 for reference masses
M_star_mw_z0, ratio_mw_z0 = get_behroozi_shmr(np.array([1.2e12]), np.array([0]))
M_star_lmc_z0, ratio_lmc_z0 = get_behroozi_shmr(np.array([1.5e11]), np.array([0]))

print(f"\n🌌 Behroozi+2013 predictions at z=0:")
print(f"   MW-mass (1.2×10¹² M☉):  M* = {M_star_mw_z0[0]:.2e} M☉, ratio = {ratio_mw_z0[0]:.4f}")
print(f"   LMC-mass (1.5×10¹¹ M☉): M* = {M_star_lmc_z0[0]:.2e} M☉, ratio = {ratio_lmc_z0[0]:.4f}")

if has_gas:
    print(f"\n🌌 Baryon Budget:")
    print(f"   Universal f_baryon:   0.16 (16%)")
    print(f"   Your f_baryon:        {f_baryon[-1]:.4f} ({f_baryon[-1]*100:.2f}%)")
    print(f"   In stars:             {ratio[-1]:.4f} ({ratio[-1]*100:.2f}%)")
    print(f"   In gas:               {m_gas[-1]/m_halo[-1]:.4f} ({m_gas[-1]/m_halo[-1]*100:.2f}%)")
    print(f"   Ejected/unaccreted:   {0.16 - f_baryon[-1]:.4f} ({(0.16-f_baryon[-1])*100:.2f}%)")
    
    if f_baryon[-1] < 0.10:
        print(f"   ⚠⚠ Severe baryon depletion - strong feedback!")
    elif f_baryon[-1] < 0.13:
        print(f"   ⚠ Moderate baryon depletion")
    else:
        print(f"   ✓ Reasonable baryon retention")

print(f"\n✓ Plot saved to: {output_file}\n")
