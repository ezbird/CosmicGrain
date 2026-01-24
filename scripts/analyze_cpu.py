#!/usr/bin/env python3
"""
Analyze Gadget-4 cpu.txt output to diagnose performance bottlenecks.

Usage:
    python analyze_cpu.py cpu.txt
    python analyze_cpu.py /path/to/output/cpu.txt --plot
"""

import argparse
import re
import numpy as np
import sys

def parse_cpu_file(filename):
    """Parse Gadget-4 cpu.txt file."""
    
    steps = []
    current_step = None
    
    with open(filename, 'r') as f:
        for line in f:
            line = line.strip()
            
            # Parse step header
            # Step 616258, Time: 0.0496837, CPUs: 24, HighestActiveTimeBin: 7
            step_match = re.match(r'Step\s+(\d+),\s+Time:\s+([\d.e+-]+),\s+CPUs:\s+(\d+),\s+HighestActiveTimeBin:\s+(\d+)', line)
            if step_match:
                if current_step is not None:
                    steps.append(current_step)
                current_step = {
                    'step': int(step_match.group(1)),
                    'time': float(step_match.group(2)),
                    'cpus': int(step_match.group(3)),
                    'highest_bin': int(step_match.group(4)),
                    'timings': {}
                }
                continue
            
            # Parse timing lines
            # total                     0.10  100.0%  144884.73  100.0%
            timing_match = re.match(r'\s*(\w+)\s+([\d.]+)\s+[\d.]+%\s+([\d.]+)\s+([\d.]+)%', line)
            if timing_match and current_step is not None:
                name = timing_match.group(1)
                diff = float(timing_match.group(2))
                cumul = float(timing_match.group(3))
                cumul_pct = float(timing_match.group(4))
                current_step['timings'][name] = {
                    'diff': diff,
                    'cumulative': cumul,
                    'cumulative_pct': cumul_pct
                }
    
    # Don't forget last step
    if current_step is not None:
        steps.append(current_step)
    
    return steps


def analyze_performance(steps):
    """Analyze performance from parsed cpu data."""
    
    if not steps:
        print("ERROR: No steps found in file!")
        return
    
    print("=" * 70)
    print("GADGET-4 PERFORMANCE ANALYSIS")
    print("=" * 70)
    
    # Basic stats
    first_step = steps[0]
    last_step = steps[-1]
    
    n_steps = last_step['step'] - first_step['step']
    time_start = first_step['time']
    time_end = last_step['time']
    z_start = 1.0/time_start - 1 if time_start > 0 else float('inf')
    z_end = 1.0/time_end - 1 if time_end > 0 else float('inf')
    
    total_walltime = last_step['timings'].get('total', {}).get('cumulative', 0)
    
    print(f"\n📊 SIMULATION PROGRESS")
    print(f"   Steps in file: {len(steps)} entries")
    print(f"   Total steps taken: {last_step['step']:,}")
    print(f"   Scale factor: {time_start:.6f} → {time_end:.6f}")
    print(f"   Redshift: z={z_start:.2f} → z={z_end:.2f}")
    print(f"   Total walltime: {total_walltime:.1f} sec = {total_walltime/3600:.2f} hours")
    print(f"   CPUs: {last_step['cpus']}")
    
    # Timestep analysis
    print(f"\n⏱️  TIMESTEP ANALYSIS")
    avg_walltime_per_step = total_walltime / max(1, last_step['step'])
    print(f"   Average walltime per step: {avg_walltime_per_step:.4f} sec")
    print(f"   Steps per hour: {3600/avg_walltime_per_step:.0f}")
    
    # This is the key diagnostic!
    delta_a = time_end - time_start
    steps_per_delta_a = n_steps / max(delta_a, 1e-10)
    print(f"   Steps per Δa=0.01: {steps_per_delta_a * 0.01:.0f}")
    
    # Highest active timebin distribution
    bins = [s['highest_bin'] for s in steps]
    print(f"\n   HighestActiveTimeBin statistics:")
    print(f"   Min: {min(bins)}, Max: {max(bins)}, Mean: {np.mean(bins):.1f}")
    
    if max(bins) < 10:
        print(f"   ⚠️  WARNING: Low timebin values suggest particles on VERY small timesteps!")
        print(f"   ⚠️  This is likely causing your slowdown!")
    
    # Timing breakdown
    print(f"\n📈 TIMING BREAKDOWN (cumulative)")
    print(f"   {'Category':<25} {'Time (hrs)':<12} {'Percentage':<10} {'Assessment'}")
    print(f"   {'-'*25} {'-'*12} {'-'*10} {'-'*20}")
    
    timings = last_step['timings']
    total_time = timings.get('total', {}).get('cumulative', 1)
    
    categories = [
        ('treegrav', 'Gravity (tree)', 40, 60),
        ('pm_grav', 'Gravity (PM)', 5, 15),
        ('sph', 'SPH/Hydro', 10, 25),
        ('domain', 'Domain decomp', 1, 5),
        ('drift/kicks', 'Drift/Kicks', 2, 8),
        ('sfrcool', 'SF+Cooling', 1, 5),
        ('misc', 'Miscellaneous', 0, 5),
    ]
    
    for key, label, low, high in categories:
        if key in timings:
            t = timings[key]['cumulative']
            pct = timings[key]['cumulative_pct']
            hrs = t / 3600
            
            if pct < low:
                assessment = "✓ OK"
            elif pct <= high:
                assessment = "✓ Normal"
            else:
                assessment = f"⚠️ HIGH (expect {low}-{high}%)"
            
            print(f"   {label:<25} {hrs:<12.2f} {pct:<10.1f}% {assessment}")
    
    # Sub-category analysis
    print(f"\n📊 DETAILED BREAKDOWN")
    
    # Tree gravity details
    if 'treebuild' in timings and 'treeforce' in timings:
        build_pct = timings['treebuild']['cumulative_pct']
        force_pct = timings.get('treeforce', {}).get('cumulative_pct', 0)
        imbal_pct = timings.get('treeimbalance', {}).get('cumulative_pct', 0)
        
        print(f"\n   Tree Gravity:")
        print(f"     Tree building:  {build_pct:.1f}%", end="")
        if build_pct > 15:
            print(f" ⚠️ HIGH - tree rebuilt too often!")
        else:
            print(" ✓")
        
        print(f"     Tree force:     {force_pct:.1f}%")
        print(f"     Load imbalance: {imbal_pct:.1f}%", end="")
        if imbal_pct > 5:
            print(f" ⚠️ HIGH - poor domain decomposition!")
        else:
            print(" ✓")
    
    # SPH details
    if 'density' in timings and 'hydro' in timings:
        dens_pct = timings['density']['cumulative_pct']
        hydro_pct = timings['hydro']['cumulative_pct']
        hydro_imbal = timings.get('hydroimbalance', {}).get('cumulative_pct', 0)
        
        print(f"\n   SPH:")
        print(f"     Density:        {dens_pct:.1f}%")
        print(f"     Hydro forces:   {hydro_pct:.1f}%")
        print(f"     Load imbalance: {hydro_imbal:.1f}%", end="")
        if hydro_imbal > 3:
            print(f" ⚠️ HIGH")
        else:
            print(" ✓")
    
    # Misc analysis
    misc_pct = timings.get('misc', {}).get('cumulative_pct', 0)
    if misc_pct > 5:
        print(f"\n   ⚠️ 'misc' category is {misc_pct:.1f}% - this includes MPI waits and sync overhead!")
    
    # Performance estimates
    print(f"\n🔮 PERFORMANCE PROJECTIONS")
    
    # Current rate
    delta_z = z_start - z_end
    if delta_z > 0 and total_walltime > 0:
        hours_per_delta_z_1 = (total_walltime / 3600) / delta_z
        print(f"   Current rate: {hours_per_delta_z_1:.1f} hours per Δz=1")
        
        # Projections (assuming constant rate, which is optimistic)
        remaining_z = z_end - 0
        est_hours_remaining = remaining_z * hours_per_delta_z_1
        print(f"   Estimated time to z=0: {est_hours_remaining:.0f} hours ({est_hours_remaining/24:.1f} days)")
        print(f"   ⚠️  Note: Rate slows significantly at low-z due to structure formation!")
    
    # Recommendations
    print(f"\n💡 RECOMMENDATIONS")
    print("=" * 70)
    
    recommendations = []
    
    # Based on timestep bins
    if max(bins) < 10:
        recommendations.append(
            "CRITICAL: Particles on very small timesteps (HighestActiveBin < 10)\n"
            "   → Check softening lengths - they may be too small!\n"
            "   → Check MinSizeTimestep - may be forcing tiny steps\n"
            "   → Look for particles with extreme accelerations"
        )
    
    # Based on tree building
    if timings.get('treebuild', {}).get('cumulative_pct', 0) > 15:
        recommendations.append(
            "Tree building is expensive (>15%)\n"
            "   → Increase TreeDomainUpdateFrequency\n"
            "   → Check ActivePartFracForNewDomainDecomp"
        )
    
    # Based on imbalance
    total_imbalance = (
        timings.get('treeimbalance', {}).get('cumulative_pct', 0) +
        timings.get('hydroimbalance', {}).get('cumulative_pct', 0) +
        timings.get('densimbalance', {}).get('cumulative_pct', 0)
    )
    if total_imbalance > 10:
        recommendations.append(
            f"Load imbalance is {total_imbalance:.1f}%\n"
            "   → Try different number of MPI tasks\n"
            "   → For zooms: use PLACEHIGHRESREGION\n"
            "   → Adjust TopNodeFactor (try 2.0-4.0)"
        )
    
    # Based on misc
    if misc_pct > 10:
        recommendations.append(
            f"'misc' overhead is high ({misc_pct:.1f}%)\n"
            "   → This often indicates MPI synchronization overhead\n"
            "   → Try fewer MPI tasks (24 may be better than 124 for zooms)\n"
            "   → Check network bandwidth between nodes"
        )
    
    # Based on steps
    if last_step['step'] > 500000 and z_end > 10:
        recommendations.append(
            f"Excessive timesteps ({last_step['step']:,} to reach z={z_end:.0f})\n"
            "   → MOST LIKELY: Softening lengths are too small for your units!\n"
            "   → Check: UnitLength_in_cm and softening values\n"
            "   → If using kpc units, softening of 0.001 = 1 pc (very small!)\n"
            "   → Try softening 0.5-2.0 for high-res particles (= 0.5-2 kpc)"
        )
    
    if recommendations:
        for i, rec in enumerate(recommendations, 1):
            print(f"\n{i}. {rec}")
    else:
        print("   No major issues detected!")
    
    print("\n" + "=" * 70)
    
    return {
        'total_steps': last_step['step'],
        'z_end': z_end,
        'walltime_hours': total_walltime / 3600,
        'bins': bins,
        'timings': timings
    }


def plot_performance(steps, output_file='cpu_analysis.png'):
    """Create performance visualization."""
    try:
        import matplotlib.pyplot as plt
    except ImportError:
        print("matplotlib not available for plotting")
        return
    
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    
    # Extract data
    step_nums = [s['step'] for s in steps]
    times = [s['time'] for s in steps]
    redshifts = [1/t - 1 if t > 0 else 100 for t in times]
    bins = [s['highest_bin'] for s in steps]
    
    # 1. Redshift vs Step
    ax = axes[0, 0]
    ax.plot(step_nums, redshifts, 'b-', linewidth=0.5)
    ax.set_xlabel('Step Number')
    ax.set_ylabel('Redshift')
    ax.set_title('Simulation Progress')
    ax.set_yscale('log')
    ax.grid(True, alpha=0.3)
    ax.invert_yaxis()
    
    # 2. Highest Active Bin vs Step
    ax = axes[0, 1]
    ax.scatter(step_nums, bins, c=redshifts, cmap='viridis_r', s=1, alpha=0.5)
    ax.set_xlabel('Step Number')
    ax.set_ylabel('Highest Active TimeBin')
    ax.set_title('Timestep Bin Evolution (color = redshift)')
    ax.grid(True, alpha=0.3)
    cbar = plt.colorbar(ax.collections[0], ax=ax)
    cbar.set_label('Redshift')
    
    # 3. Cumulative timing breakdown
    ax = axes[1, 0]
    last = steps[-1]
    categories = ['treegrav', 'sph', 'pm_grav', 'domain', 'sfrcool', 'misc']
    labels = ['Tree Gravity', 'SPH/Hydro', 'PM Gravity', 'Domain', 'SF+Cool', 'Misc']
    values = [last['timings'].get(c, {}).get('cumulative_pct', 0) for c in categories]
    colors = plt.cm.Set2(np.linspace(0, 1, len(categories)))
    
    bars = ax.barh(labels, values, color=colors)
    ax.set_xlabel('Percentage of Total Time')
    ax.set_title('Cumulative Time Distribution')
    ax.set_xlim(0, 100)
    
    for bar, val in zip(bars, values):
        if val > 2:
            ax.text(val + 1, bar.get_y() + bar.get_height()/2, 
                   f'{val:.1f}%', va='center', fontsize=9)
    
    # 4. Steps per delta_a
    ax = axes[1, 1]
    if len(steps) > 10:
        window = max(1, len(steps) // 50)
        step_arr = np.array(step_nums)
        time_arr = np.array(times)
        
        # Calculate steps per delta_a in windows
        steps_rate = []
        z_centers = []
        for i in range(window, len(steps), window):
            da = time_arr[i] - time_arr[i-window]
            ds = step_arr[i] - step_arr[i-window]
            if da > 0:
                rate = ds / da * 0.01  # steps per delta_a = 0.01
                steps_rate.append(rate)
                z_centers.append(1/time_arr[i] - 1)
        
        if steps_rate:
            ax.semilogy(z_centers, steps_rate, 'g-', linewidth=1)
            ax.set_xlabel('Redshift')
            ax.set_ylabel('Steps per Δa=0.01')
            ax.set_title('Timestep Resolution vs Redshift')
            ax.invert_xaxis()
            ax.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(output_file, dpi=150, bbox_inches='tight')
    print(f"\nSaved plot to: {output_file}")
    plt.close()


def main():
    parser = argparse.ArgumentParser(description='Analyze Gadget-4 cpu.txt output')
    parser.add_argument('cpu_file', help='Path to cpu.txt file')
    parser.add_argument('--plot', action='store_true', help='Generate performance plot')
    parser.add_argument('--output', default='cpu_analysis.png', help='Plot output file')
    
    args = parser.parse_args()
    
    print(f"Reading: {args.cpu_file}")
    steps = parse_cpu_file(args.cpu_file)
    
    if not steps:
        print("ERROR: Could not parse any steps from file!")
        sys.exit(1)
    
    print(f"Parsed {len(steps)} step entries\n")
    
    results = analyze_performance(steps)
    
    if args.plot:
        plot_performance(steps, args.output)


if __name__ == '__main__':
    main()
