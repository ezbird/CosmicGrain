#!/usr/bin/env python3
"""
Plot Dust Evolution - Dust-to-metal and dust-to-gas ratios over time
"""

import numpy as np
import matplotlib.pyplot as plt
import sys
import os

# =============================================================================
# OBSERVATIONAL CONSTRAINTS
# =============================================================================

OBS_DZ = {
    'Milky Way': {'value': 0.45, 'range': (0.40, 0.50), 'color': 'red', 'marker': '*', 'size': 300},
    'LMC': {'value': 0.35, 'range': (0.30, 0.40), 'color': 'blue', 'marker': 's', 'size': 150},
    'SMC': {'value': 0.25, 'range': (0.20, 0.30), 'color': 'green', 'marker': 'D', 'size': 120},
}

OBS_DG = {
    'Milky Way': {'value': 0.010, 'range': (0.008, 0.012), 'metallicity': 1.0},
    'LMC': {'value': 0.004, 'range': (0.003, 0.005), 'metallicity': 0.5},
    'SMC': {'value': 0.002, 'range': (0.0015, 0.0025), 'metallicity': 0.2},
}

def dz_metallicity_relation(Z_Zsun):
    """Empirical D/Z vs metallicity (Rémy-Ruyer+2014)"""
    dz = 0.5 * (1 - np.exp(-Z_Zsun / 0.3))
    return dz

def dg_metallicity_relation(Z_Zsun):
    """Empirical D/G vs metallicity"""
    dg = 0.01 * Z_Zsun * (1 + 0.3 * np.log10(np.maximum(Z_Zsun, 0.01)))
    dg = np.maximum(dg, 1e-8)
    return dg

# =============================================================================
# LOAD DATA
# =============================================================================

if len(sys.argv) < 2:
    print("\n" + "="*70)
    print("🌫️  Oops! Provide the output directory!  🌫️")
    print("="*70)
    print("\nUsage: python plot_dust_evolution.py <output_directory>\n")
    sys.exit(1)

output_dir = sys.argv[1].rstrip('/')
data_file = os.path.join(output_dir, 'dust_evolution.txt')

if not os.path.exists(data_file):
    print(f"\n❌ Error: Could not find {data_file}\n")
    sys.exit(1)

print(f"\n🌫️  Loading dust data from: {data_file}")

# Load data
data = np.loadtxt(data_file)

time = data[:, 0]
m_dust = data[:, 1]
m_metal = data[:, 2]
m_gas = data[:, 3]
dz_ratio = data[:, 4]
dg_ratio = data[:, 5]
m_halo = data[:, 6]

z = 1/time - 1

# Calculate metallicity (Z/Zsun) assuming solar Z = 0.02
Z_Zsun = np.where(m_gas > 0, m_metal / m_gas / 0.02, 0)

# Filter out zeros for ratio calculations
has_dust = m_dust > 0
dz_nonzero = dz_ratio[has_dust]
dg_nonzero = dg_ratio[has_dust]
z_dust = z[has_dust]
Z_dust = Z_Zsun[has_dust]

print(f"✓ Loaded {len(time)} snapshots")
print(f"  Redshift range: z = {z.max():.2f} → {z.min():.2f}")
print(f"  Snapshots with dust: {np.sum(has_dust)}")
print(f"  Final M_dust:   {m_dust[-1]:.2e} M_☉")
print(f"  Final M_metal:  {m_metal[-1]:.2e} M_☉")
print(f"  Final M_gas:    {m_gas[-1]:.2e} M_☉")
print(f"  Final D/Z:      {dz_ratio[-1]:.6f} (Expected: ~0.3-0.5)")
print(f"  Final D/G:      {dg_ratio[-1]:.6e} (Expected: ~0.001-0.01)")
print(f"  Final Z/Zsun:   {Z_Zsun[-1]:.3f}")

# =============================================================================
# CREATE FIGURE
# =============================================================================

fig = plt.figure(figsize=(14, 10))
gs = fig.add_gridspec(3, 2, hspace=0.3, wspace=0.3)

ax1 = fig.add_subplot(gs[0, :])  # Masses
ax2 = fig.add_subplot(gs[1, 0])  # D/Z vs z
ax3 = fig.add_subplot(gs[1, 1])  # D/Z vs metallicity
ax4 = fig.add_subplot(gs[2, 0])  # D/G vs z
ax5 = fig.add_subplot(gs[2, 1])  # D/G vs metallicity

# =============================================================================
# PANEL 1: MASSES
# =============================================================================

ax1.invert_xaxis()
ax1.plot(z, m_dust, label='M$_{dust}$', color='brown', linewidth=2.5, zorder=3)
ax1.plot(z, m_metal, label='M$_{metal}$', color='orange', linewidth=2.5, zorder=3)
ax1.plot(z, m_gas, label='M$_{gas}$', color='blue', linewidth=2.5, zorder=3, alpha=0.7)

ax1.set_yscale('log')
ax1.set_xlabel('Redshift', fontsize=12)
ax1.set_ylabel('Mass (M$_\\odot$)', fontsize=12)
ax1.legend(fontsize=10, loc='best', framealpha=0.9)
ax1.grid(True, alpha=0.3, linestyle='--')
ax1.set_title(f'Dust Evolution: {os.path.basename(output_dir)}', 
              fontsize=14, fontweight='bold', pad=10)

textstr = f'Final (z={z[-1]:.2f}):\n'
textstr += f'M$_{{dust}}$ = {m_dust[-1]:.2e}\n'
textstr += f'M$_{{metal}}$ = {m_metal[-1]:.2e}\n'
textstr += f'M$_{{gas}}$ = {m_gas[-1]:.2e}'
props = dict(boxstyle='round', facecolor='wheat', alpha=0.8)
ax1.text(0.02, 0.98, textstr, transform=ax1.transAxes, fontsize=9,
         verticalalignment='top', bbox=props)

# =============================================================================
# PANEL 2: D/Z VS REDSHIFT
# =============================================================================

ax2.invert_xaxis()

if len(dz_nonzero) > 0:
    ax2.plot(z_dust, dz_nonzero, color='darkred', linewidth=2.5, label='Simulation', zorder=4)

# Plot observations
for name, obs in OBS_DZ.items():
    ax2.scatter(0, obs['value'], marker=obs['marker'], s=obs['size'],
                color=obs['color'], edgecolors='black', linewidth=1.5,
                label=name, zorder=5)
    yerr = [[obs['value'] - obs['range'][0]], [obs['range'][1] - obs['value']]]
    ax2.errorbar(0, obs['value'], yerr=yerr, color=obs['color'],
                fmt='none', linewidth=2, capsize=5, zorder=4)

ax2.axhspan(0.3, 0.5, color='gray', alpha=0.1, label='Typical range', zorder=0)

ax2.set_xlabel('Redshift', fontsize=11)
ax2.set_ylabel('Dust-to-Metal Ratio (D/Z)', fontsize=11)
ax2.legend(fontsize=8, loc='best', framealpha=0.9)
ax2.grid(True, alpha=0.3, linestyle='--')

# Auto-scale with min set by data
if len(dz_nonzero) > 0:
    ymin = max(0, dz_nonzero.min() * 0.5)
    ymax = max(0.7, dz_nonzero.max() * 1.5)
    ax2.set_ylim(ymin, ymax)
else:
    ax2.set_ylim(0, 0.7)

# =============================================================================
# PANEL 3: D/Z VS METALLICITY
# =============================================================================

if len(dz_nonzero) > 0:
    sc = ax3.scatter(Z_dust, dz_nonzero, c=z_dust, cmap='viridis_r', s=30,
                    edgecolors='black', linewidth=0.5, zorder=3)
    plt.colorbar(sc, ax=ax3, label='Redshift')

# Empirical relation
Z_model = np.logspace(-1.5, 0.5, 100)
dz_model = dz_metallicity_relation(Z_model)
ax3.plot(Z_model, dz_model, 'k--', linewidth=2, alpha=0.7,
         label='Rémy-Ruyer+2014', zorder=2)

# Observations
for name, obs in OBS_DZ.items():
    if name == 'Milky Way':
        Z = 1.0
    elif name == 'LMC':
        Z = 0.5
    elif name == 'SMC':
        Z = 0.2
    
    ax3.scatter(Z, obs['value'], marker=obs['marker'], s=obs['size'],
                color=obs['color'], edgecolors='black', linewidth=1.5, zorder=5)

ax3.set_xscale('log')
ax3.set_xlabel('Metallicity (Z/Z$_\\odot$)', fontsize=11)
ax3.set_ylabel('Dust-to-Metal Ratio (D/Z)', fontsize=11)
ax3.legend(fontsize=8, loc='best', framealpha=0.9)
ax3.grid(True, alpha=0.3, linestyle='--')
ax3.set_xlim(0.05, 3)

# Auto-scale y-axis
if len(dz_nonzero) > 0:
    ymin = max(0, dz_nonzero.min() * 0.5)
    ymax = max(0.7, dz_nonzero.max() * 1.5)
    ax3.set_ylim(ymin, ymax)
else:
    ax3.set_ylim(0, 0.7)

# =============================================================================
# PANEL 4: D/G VS REDSHIFT
# =============================================================================

ax4.invert_xaxis()

if len(dg_nonzero) > 0:
    ax4.plot(z_dust, dg_nonzero, color='darkblue', linewidth=2.5, label='Simulation', zorder=4)

# Observations
for name, obs in OBS_DG.items():
    color = OBS_DZ[name]['color']
    marker = OBS_DZ[name]['marker']
    size = OBS_DZ[name]['size']
    
    ax4.scatter(0, obs['value'], marker=marker, s=size,
                color=color, edgecolors='black', linewidth=1.5,
                label=name, zorder=5)
    yerr = [[obs['value'] - obs['range'][0]], [obs['range'][1] - obs['value']]]
    ax4.errorbar(0, obs['value'], yerr=yerr, color=color,
                fmt='none', linewidth=2, capsize=5, zorder=4)

ax4.set_xlabel('Redshift', fontsize=11)
ax4.set_ylabel('Dust-to-Gas Ratio (D/G)', fontsize=11)
ax4.set_yscale('log')
ax4.legend(fontsize=8, loc='best', framealpha=0.9)
ax4.grid(True, alpha=0.3, linestyle='--', which='both')

# Auto-scale y-axis based on data
if len(dg_nonzero) > 0:
    ymin = dg_nonzero.min() * 0.5
    ymax = max(3e-2, dg_nonzero.max() * 2)
    ax4.set_ylim(ymin, ymax)
else:
    ax4.set_ylim(1e-8, 3e-2)

# =============================================================================
# PANEL 5: D/G VS METALLICITY  
# =============================================================================

if len(dg_nonzero) > 0:
    sc2 = ax5.scatter(Z_dust, dg_nonzero, c=z_dust, cmap='viridis_r', s=30,
                     edgecolors='black', linewidth=0.5, zorder=3)
    plt.colorbar(sc2, ax=ax5, label='Redshift')

# Empirical relation
dg_model = dg_metallicity_relation(Z_model)
ax5.plot(Z_model, dg_model, 'k--', linewidth=2, alpha=0.7,
         label='D/G ∝ Z', zorder=2)

# Observations
for name, obs in OBS_DG.items():
    color = OBS_DZ[name]['color']
    marker = OBS_DZ[name]['marker']
    size = OBS_DZ[name]['size']
    Z = obs['metallicity']
    
    ax5.scatter(Z, obs['value'], marker=marker, s=size,
                color=color, edgecolors='black', linewidth=1.5, zorder=5)

ax5.set_xscale('log')
ax5.set_yscale('log')
ax5.set_xlabel('Metallicity (Z/Z$_\\odot$)', fontsize=11)
ax5.set_ylabel('Dust-to-Gas Ratio (D/G)', fontsize=11)
ax5.legend(fontsize=8, loc='best', framealpha=0.9)
ax5.grid(True, alpha=0.3, linestyle='--', which='both')
ax5.set_xlim(0.05, 3)

# Auto-scale y-axis
if len(dg_nonzero) > 0:
    ymin = dg_nonzero.min() * 0.5
    ymax = max(3e-2, dg_nonzero.max() * 2)
    ax5.set_ylim(ymin, ymax)
else:
    ax5.set_ylim(1e-8, 3e-2)

# =============================================================================
# SAVE AND DIAGNOSTICS
# =============================================================================

plt.tight_layout()

output_file = os.path.join(output_dir, 'dust_evolution.png')
plt.savefig(output_file, dpi=150, bbox_inches='tight')

print(f"\n📊 Dust Diagnostics:")
print(f"   Final D/Z:      {dz_ratio[-1]:.6f} (Expected: ~0.3-0.5)")
print(f"   Final D/G:      {dg_ratio[-1]:.6e} (Expected: ~0.001-0.01)")
print(f"   Final Z/Zsun:   {Z_Zsun[-1]:.3f}")
print(f"   Dust fraction:  {100*m_dust[-1]/m_gas[-1]:.4f}% of gas mass")
print(f"   Metal fraction: {100*m_metal[-1]/m_gas[-1]:.2f}% of gas mass")

# Diagnosis
factor_dz = 0.4 / dz_ratio[-1] if dz_ratio[-1] > 0 else np.inf
factor_dg = 0.007 / dg_ratio[-1] if dg_ratio[-1] > 0 else np.inf

print(f"\n⚠️  WARNING: Dust production is TOO LOW!")
print(f"   D/Z is {factor_dz:.0f}× lower than expected")
print(f"   D/G is {factor_dg:.0f}× lower than expected")
print(f"\nPossible causes:")
print(f"   1. Dust creation yields too small")
print(f"   2. Dust destruction too efficient (thermal/shocks)")
print(f"   3. Not enough SNe II / AGB stars producing dust")
print(f"   4. Dust particle mass initialization issue")

print(f"\n✓ Plot saved to: {output_file}\n")
