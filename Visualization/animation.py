#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Fixed GIF animation generation script
Specifically addresses animation "freezing" issues
"""

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as animation
import os
import glob
from pathlib import Path

def create_fixed_animation():
    """Create fixed version of animation"""
    print("Starting creation of fixed GIF animation...")
    
    # Set font parameters
    plt.rcParams['font.sans-serif'] = ['DejaVu Sans', 'Arial']
    plt.rcParams['axes.unicode_minus'] = False
    
    # Get data files
    data_folder = Path("animation_data")
    output_folder = Path("animations")
    
    data_files = sorted(glob.glob(str(data_folder / "velocity_magnitude_*.dat")))
    print(f"Found {len(data_files)} data files")
    
    # Read first file to get grid information
    with open(data_files[0], 'r') as f:
        header = f.readline().strip()
        nx = int(header.split('nx=')[1].split()[0])
        ny = int(header.split('ny=')[1].split()[0])
    
    print(f"Grid size: {nx} x {ny}")
    
    # Preload all data to ensure correct range calculation
    print("Preloading data...")
    all_data = []
    timesteps = []
    
    for i, filename in enumerate(data_files):
        with open(filename, 'r') as f:
            f.readline()  # Skip header
            data = [float(line.strip()) for line in f]
        
        # Convert to 2D array and flip y-axis
        velocity_field = np.array(data).reshape(ny, nx)
        velocity_field = np.flipud(velocity_field)
        all_data.append(velocity_field)
        
        # Extract timestep
        timestep = int(filename.split('_')[-1].split('.')[0])
        timesteps.append(timestep)
        
        if i % 50 == 0:
            print(f"Loading progress: {i+1}/{len(data_files)}")
    
    # Calculate global color range
    print("Calculating color range...")
    all_values = []
    for data in all_data:
        non_zero_values = data[data > 0]  # Only consider non-zero values
        if len(non_zero_values) > 0:
            all_values.extend(non_zero_values)
    
    vmin = np.min(all_values)
    vmax = np.max(all_values)
    print(f"Color range: [{vmin:.6f}, {vmax:.6f}]")
    
    # Create figure
    fig, ax = plt.subplots(figsize=(10, 8))
    ax.set_aspect('equal')
    
    # Initialize display
    initial_data = all_data[0].copy()
    initial_data[initial_data == 0] = vmin * 0.5  # Obstacles shown in darker color
    
    im = ax.imshow(initial_data, cmap='viridis', vmin=vmin, vmax=vmax, 
                   origin='lower', extent=[0, nx, 0, ny])
    
    title = ax.set_title(f'LBM Fluid Simulation - Velocity Magnitude\nTimestep: {timesteps[0]}', 
                        fontsize=14, pad=20)
    ax.set_xlabel('X Coordinate', fontsize=12)
    ax.set_ylabel('Y Coordinate', fontsize=12)
    ax.grid(True, alpha=0.3)
    
    # Add colorbar
    cbar = plt.colorbar(im, ax=ax, shrink=0.8)
    cbar.set_label('Velocity Magnitude', fontsize=12)
    
    plt.tight_layout()
    
    def animate(frame):
        """Animation update function - fixed version"""
        if frame < len(all_data):
            # Get current frame data
            current_data = all_data[frame].copy()
            current_timestep = timesteps[frame]
            
            # Handle obstacle display
            current_data[current_data == 0] = vmin * 0.5
            
            # Update image data
            im.set_array(current_data)
            
            # Update title
            title.set_text(f'LBM Fluid Simulation - Velocity Magnitude\nTimestep: {current_timestep}')
            
            # Force refresh
            im.set_clim(vmin, vmax)
            
            # Show progress
            if frame % 20 == 0:
                print(f"Animation progress: {frame+1}/{len(all_data)} (Timestep: {current_timestep})")
        
        return [im, title]
    
    # Create animation - use slower frame rate to ensure visibility
    print("Creating animation object...")
    anim = animation.FuncAnimation(
        fig, animate, 
        frames=len(all_data),
        interval=150,  # Increase interval time
        blit=False,
        repeat=True
    )
    
    # Save GIF
    gif_path = output_folder / "velocity_magnitude_animation_fixed.gif"
    print(f"Saving GIF to: {gif_path}")
    
    # Use pillow writer to save with slower fps
    anim.save(gif_path, writer='pillow', fps=6, dpi=80)
    
    print("Fixed GIF animation creation completed!")
    
    # Also create a quick preview version (key frames only)
    print("Creating quick preview version...")
    create_preview_animation(all_data, timesteps, vmin, vmax, nx, ny, output_folder)
    
    plt.close(fig)
    return True

def create_preview_animation(all_data, timesteps, vmin, vmax, nx, ny, output_folder):
    """Create quick preview animation (key frames only)"""
    
    # Select key frames (every 20th frame)
    step = 20
    preview_data = all_data[::step]
    preview_timesteps = timesteps[::step]
    
    print(f"Preview version contains {len(preview_data)} frames")
    
    # Create preview figure
    fig, ax = plt.subplots(figsize=(8, 6))
    ax.set_aspect('equal')
    
    # Initialize
    initial_data = preview_data[0].copy()
    initial_data[initial_data == 0] = vmin * 0.5
    
    im = ax.imshow(initial_data, cmap='viridis', vmin=vmin, vmax=vmax, 
                   origin='lower', extent=[0, nx, 0, ny])
    
    title = ax.set_title(f'LBM Fluid Simulation Preview\nTimestep: {preview_timesteps[0]}', fontsize=12)
    ax.set_xlabel('X Coordinate')
    ax.set_ylabel('Y Coordinate')
    
    cbar = plt.colorbar(im, ax=ax, shrink=0.8)
    cbar.set_label('Velocity Magnitude')
    
    plt.tight_layout()
    
    def animate_preview(frame):
        if frame < len(preview_data):
            current_data = preview_data[frame].copy()
            current_data[current_data == 0] = vmin * 0.5
            im.set_array(current_data)
            title.set_text(f'LBM Fluid Simulation Preview\nTimestep: {preview_timesteps[frame]}')
        return [im, title]
    
    # Create preview animation
    anim_preview = animation.FuncAnimation(
        fig, animate_preview,
        frames=len(preview_data),
        interval=300,  # Slower frame rate
        blit=False,
        repeat=True
    )
    
    # Save preview version
    preview_path = output_folder / "velocity_magnitude_preview.gif"
    anim_preview.save(preview_path, writer='pillow', fps=3, dpi=60)
    
    print(f"Preview GIF saved to: {preview_path}")
    plt.close(fig)

def main():
    """Main function"""
    try:
        success = create_fixed_animation()
        if success:
            print("\n" + "="*50)
            print("✅ Fixed animation creation successful!")
            print("Generated files:")
            print("- velocity_magnitude_animation_fixed.gif (Full version)")
            print("- velocity_magnitude_preview.gif (Preview version)")
            print("="*50)
        return 0
    except Exception as e:
        print(f"❌ Error: {e}")
        import traceback
        traceback.print_exc()
        return 1

if __name__ == "__main__":
    exit(main())
