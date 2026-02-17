import numpy as np
import matplotlib.pyplot as plt
import re

# Parse dust creation events
creation_data = []
with open('dust.txt', 'r') as f:
    for line in f:
        if '[DUST_CREATE]' in line:
            # Extract: vel_dust=245.3 km/s vel_star=12.1 km/s rho=5.234e+00 cm^-3 grain_r=10.00 nm feedback_type=1
            vel_dust = float(re.search(r'vel_dust=([\d.]+)', line).group(1))
            vel_star = float(re.search(r'vel_star=([\d.]+)', line).group(1))
            rho = float(re.search(r'rho=([\d.e+-]+)', line).group(1))
            feedback = int(re.search(r'feedback_type=(\d)', line).group(1))
            creation_data.append([vel_dust, vel_star, rho, feedback])

creation = np.array(creation_data)

# Parse drag events  
drag_data = []
with open('dust.txt', 'r') as f:
    for line in f:
        if '[DUST_DRAG]' in line:
            # Extract: r=234.5 kpc vel=178.2 km/s rho=1.234e-03 cm^-3 t_drag=8.5 Myr grain_r=10.50 nm
            r = float(re.search(r'r=([\d.]+)', line).group(1))
            vel = float(re.search(r'vel=([\d.]+)', line).group(1))
            rho = float(re.search(r'rho=([\d.e+-]+)', line).group(1))
            t_drag = float(re.search(r't_drag=([\d.]+)', line).group(1))
            drag_data.append([r, vel, rho, t_drag])

drag = np.array(drag_data)

print(f"\n{'='*60}")
print(f"DUST DIAGNOSTICS SUMMARY")
print(f"{'='*60}")

if len(creation) > 0:
    print(f"\n--- DUST CREATION ({len(creation)} events) ---")
    
    sn_mask = creation[:,3] == 1
    agb_mask = creation[:,3] == 2
    
    print(f"\nSupernovae ({np.sum(sn_mask)} events):")
    if np.sum(sn_mask) > 0:
        print(f"  Dust velocity: median={np.median(creation[sn_mask,0]):.1f} km/s, "
              f"90th%={np.percentile(creation[sn_mask,0], 90):.1f} km/s")
        print(f"  Star velocity: median={np.median(creation[sn_mask,1]):.1f} km/s")
        print(f"  Density: median={np.median(creation[sn_mask,2]):.2e} cm^-3")
    
    print(f"\nAGB winds ({np.sum(agb_mask)} events):")
    if np.sum(agb_mask) > 0:
        print(f"  Dust velocity: median={np.median(creation[agb_mask,0]):.1f} km/s, "
              f"90th%={np.percentile(creation[agb_mask,0], 90):.1f} km/s")
        print(f"  Star velocity: median={np.median(creation[agb_mask,1]):.1f} km/s")
        print(f"  Density: median={np.median(creation[agb_mask,2]):.2e} cm^-3")
    
    # CHECK FOR PROBLEM: Velocities >200 km/s?
    high_vel = creation[creation[:,0] > 200]
    if len(high_vel) > 0:
        print(f"\n⚠️  WARNING: {len(high_vel)} particles created with v>200 km/s!")
        print(f"    This could explain dust escaping to 500+ kpc")

if len(drag) > 0:
    print(f"\n--- DRAG APPLICATION ({len(drag)} events) ---")
    
    # Binned by radius
    inner = drag[drag[:,0] < 100]
    mid = drag[(drag[:,0] >= 100) & (drag[:,0] < 200)]
    outer = drag[(drag[:,0] >= 200) & (drag[:,0] < 500)]
    very_outer = drag[drag[:,0] >= 500]
    
    print(f"\nInner (<100 kpc): {len(inner)} samples")
    if len(inner) > 0:
        print(f"  Velocity: median={np.median(inner[:,1]):.1f} km/s")
        print(f"  Density: median={np.median(inner[:,2]):.2e} cm^-3")
        print(f"  Drag time: median={np.median(inner[:,3]):.1f} Myr")
    
    print(f"\nMid (100-200 kpc): {len(mid)} samples")
    if len(mid) > 0:
        print(f"  Velocity: median={np.median(mid[:,1]):.1f} km/s")
        print(f"  Density: median={np.median(mid[:,2]):.2e} cm^-3")
        print(f"  Drag time: median={np.median(mid[:,3]):.1f} Myr")
    
    print(f"\nOuter (200-500 kpc): {len(outer)} samples")
    if len(outer) > 0:
        print(f"  Velocity: median={np.median(outer[:,1]):.1f} km/s")
        print(f"  Density: median={np.median(outer[:,2]):.2e} cm^-3")
        print(f"  Drag time: median={np.median(outer[:,3]):.1f} Myr")
    
    print(f"\nVery outer (>500 kpc): {len(very_outer)} samples")
    if len(very_outer) > 0:
        print(f"  Velocity: median={np.median(very_outer[:,1]):.1f} km/s")
        print(f"  Density: median={np.median(very_outer[:,2]):.2e} cm^-3")
        print(f"  Drag time: median={np.median(very_outer[:,3]):.1f} Myr")
        
        # CHECK FOR PROBLEM: High velocities at large radii?
        if np.median(very_outer[:,1]) > 100:
            print(f"\n⚠️  WARNING: Dust still moving fast (>100 km/s) at 500+ kpc!")
            print(f"    Drag may be too weak in low-density regions")
        
        # CHECK FOR PROBLEM: Very long drag times?
        if np.median(very_outer[:,3]) > 100:
            print(f"\n⚠️  WARNING: Drag timescales >100 Myr at large radii!")
            print(f"    Dust travels too far before drag can slow it down")

print(f"\n{'='*60}\n")

# Make plots
if len(creation) > 0 and len(drag) > 0:
    fig, axes = plt.subplots(2, 2, figsize=(12, 10))
    
    # Plot 1: Creation velocities
    axes[0,0].hist(creation[:,0], bins=30, alpha=0.7, color='blue', label='Dust')
    axes[0,0].axvline(200, color='red', linestyle='--', linewidth=2, label='200 km/s threshold')
    axes[0,0].set_xlabel('Velocity (km/s)')
    axes[0,0].set_ylabel('Count')
    axes[0,0].set_title('Initial dust velocities at creation')
    axes[0,0].legend()
    axes[0,0].set_yscale('log')
    
    # Plot 2: Drag timescale vs radius
    sc = axes[0,1].scatter(drag[:,0], drag[:,3], c=np.log10(drag[:,2]), 
                           alpha=0.5, s=20, cmap='viridis')
    axes[0,1].set_xlabel('Radius (kpc)')
    axes[0,1].set_ylabel('Drag timescale (Myr)')
    axes[0,1].set_title('Drag timescale vs radius')
    axes[0,1].set_yscale('log')
    axes[0,1].axhline(50, color='r', linestyle='--', label='50 Myr')
    axes[0,1].axhline(100, color='orange', linestyle='--', label='100 Myr')
    axes[0,1].legend()
    cbar = plt.colorbar(sc, ax=axes[0,1])
    cbar.set_label('log₁₀(Density [cm⁻³])')
    
    # Plot 3: Velocity vs radius
    sc2 = axes[1,0].scatter(drag[:,0], drag[:,1], c=np.log10(drag[:,2]),
                            alpha=0.5, s=20, cmap='plasma')
    axes[1,0].set_xlabel('Radius (kpc)')
    axes[1,0].set_ylabel('Velocity (km/s)')
    axes[1,0].set_title('Dust velocity vs radius')
    axes[1,0].set_yscale('log')
    axes[1,0].axhline(100, color='r', linestyle='--', label='100 km/s')
    axes[1,0].legend()
    cbar2 = plt.colorbar(sc2, ax=axes[1,0])
    cbar2.set_label('log₁₀(Density [cm⁻³])')
    
    # Plot 4: Density vs radius
    axes[1,1].scatter(drag[:,0], drag[:,2], alpha=0.5, s=20, color='green')
    axes[1,1].set_xlabel('Radius (kpc)')
    axes[1,1].set_ylabel('Density (cm⁻³)')
    axes[1,1].set_title('Gas density vs radius')
    axes[1,1].set_yscale('log')
    axes[1,1].axhline(0.001, color='r', linestyle='--', label='0.001 cm⁻³ (IGM)')
    axes[1,1].axhline(0.1, color='orange', linestyle='--', label='0.1 cm⁻³ (CGM)')
    axes[1,1].legend()
    
    plt.tight_layout()
    plt.savefig('dust_drag_diagnostics.png', dpi=150, bbox_inches='tight')
    print(f"Saved plot: dust_drag_diagnostics.png\n")
