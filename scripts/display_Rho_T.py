#!/usr/bin/env python3
"""
Display 16 evenly spaced frames from your Rho-T plots in a 4x4 grid
"""

import matplotlib.pyplot as plt
import matplotlib.image as mpimg
import glob
import numpy as np
import os

def create_4x4_evenly_spaced():
    """
    Create a 4x4 grid showing 16 evenly spaced frames from all available plots
    """
    # Find all PNG files and sort them
    png_files = sorted(glob.glob("/home/cygnus/gadget4/rho_T_frames/rho_T_snap_*.png"))
    
    if not png_files:
        print("No PNG files found in rho_T_frames/!")
        return
    
    n_total = len(png_files)
    print(f"Found {n_total} total frames")
    
    # Select 16 evenly spaced indices
    if n_total < 16:
        print(f"Warning: Only {n_total} frames available, will show all with some empty spaces")
        selected_indices = list(range(n_total))
        # Pad with last frame if needed
        while len(selected_indices) < 16:
            selected_indices.append(n_total - 1)
    else:
        # Evenly space 16 frames across the total range
        selected_indices = np.linspace(0, n_total - 1, 16, dtype=int)
    
    # Create the 4x4 figure
    fig, axes = plt.subplots(4, 4, figsize=(16, 16))
    fig.suptitle(f'Rho-T Evolution', 
                 fontsize=20, y=0.98)
    
    # Plot each selected frame
    for i, idx in enumerate(selected_indices):
        row = i // 4
        col = i % 4
        
        if idx < n_total:
            png_file = png_files[idx]
            
            try:
                # Load and display image
                img = mpimg.imread(png_file)
                axes[row, col].imshow(img)
                axes[row, col].axis('off')
                
                # Extract info from filename for title
                basename = os.path.basename(png_file)
                parts = basename.replace('.png', '').split('_')
                
                if len(parts) >= 4:
                    snap_num = parts[2]  # snap_XXX
                    z_part = parts[3]    # zX.XX
                    
                    # Clean up the redshift display
                    z_value = z_part.replace('z', '')
                    title = f"Snap {snap_num}\nz = {z_value}"
                else:
                    title = f"Frame {idx+1}"
                
                # Add frame number info
                title += f"\n({idx+1}/{n_total})"
                
                #axes[row, col].set_title(title, fontsize=12, pad=10)
                
            except Exception as e:
                print(f"Error loading {png_file}: {e}")
                axes[row, col].text(0.5, 0.5, f'Error\nFrame {idx+1}', 
                                   ha='center', va='center', transform=axes[row, col].transAxes)
                axes[row, col].axis('off')
        else:
            # Empty frame
            axes[row, col].axis('off')
    
    plt.tight_layout()
    plt.subplots_adjust(top=0.94)  # Make room for suptitle
    
    # Save the result
    output_file = "/home/cygnus/gadget4/rho-T_grid.png"
    plt.savefig(output_file, dpi=200, bbox_inches='tight')
    print(f"\nSaved 4x4 overview: {output_file}")
    
    # Also create a summary of what frames were selected
    print(f"\nSelected frames:")
    for i, idx in enumerate(selected_indices[:min(16, n_total)]):
        basename = os.path.basename(png_files[idx])
        print(f"  Position {i+1:2d}: Frame {idx+1:2d}/{n_total} - {basename}")
    
    # Show the plot
    print(f"\nDisplaying 4x4 grid...")
    plt.show()

def create_time_spacing_info():
    """
    Create a small plot showing the time spacing of selected frames
    """
    png_files = sorted(glob.glob("rho_T_frames/rho_T_snap_*.png"))
    n_total = len(png_files)
    
    if n_total < 16:
        selected_indices = list(range(n_total))
    else:
        selected_indices = np.linspace(0, n_total - 1, 16, dtype=int)
    
    # Extract redshifts
    redshifts = []
    snap_nums = []
    
    for idx in selected_indices:
        if idx < n_total:
            basename = os.path.basename(png_files[idx])
            parts = basename.replace('.png', '').split('_')
            if len(parts) >= 4:
                try:
                    snap_num = int(parts[2])
                    z_str = parts[3].replace('z', '')
                    z = float(z_str)
                    redshifts.append(z)
                    snap_nums.append(snap_num)
                except:
                    continue
    
    if len(redshifts) > 0:
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 4))
        
        # Plot redshift vs frame
        ax1.plot(range(1, len(redshifts)+1), redshifts, 'bo-', linewidth=2, markersize=8)
        ax1.set_xlabel('4x4 Grid Position')
        ax1.set_ylabel('Redshift')
        ax1.set_title('Redshift of Selected Frames')
        ax1.grid(True, alpha=0.3)
        ax1.invert_yaxis()
        
        # Plot snapshot number vs frame
        ax2.plot(range(1, len(snap_nums)+1), snap_nums, 'ro-', linewidth=2, markersize=8)
        ax2.set_xlabel('4x4 Grid Position')
        ax2.set_ylabel('Snapshot Number')
        ax2.set_title('Snapshot Numbers of Selected Frames')
        ax2.grid(True, alpha=0.3)
        
        plt.tight_layout()
        spacing_file = "rho_T_frames/frame_spacing_info.png"
        plt.savefig(spacing_file, dpi=150, bbox_inches='tight')
        print(f"Saved spacing info: {spacing_file}")
        plt.close()

if __name__ == "__main__":
    print("Creating 4x4 evenly spaced Rho-T overview...")
    create_4x4_evenly_spaced()
    print("\nCreating spacing information...")
    create_time_spacing_info()
    print("\nDone!")
