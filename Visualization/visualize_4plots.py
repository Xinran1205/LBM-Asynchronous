
# This code is for generating visualizations of lid-driven cavity flow simulation results.

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from matplotlib.colors import LinearSegmentedColormap
import matplotlib.colors as mcolors


plt.rcParams['font.sans-serif'] = ['DejaVu Sans', 'Arial']
plt.rcParams['axes.unicode_minus'] = False

def load_final_state(filename):
    """
    Load final state data file
    
    Parameters:
        filename: data file path
        
    Returns:
        x_coords: x coordinate array
        y_coords: y coordinate array  
        u_x: x-direction velocity component
        u_y: y-direction velocity component
        u_mag: velocity magnitude
        pressure: pressure
        obstacles: obstacle flags
    """
    print(f"Loading data file: {filename}")
    
    data = []
    with open(filename, 'r') as f:
        for line in f:
            parts = line.strip().split()
            if len(parts) == 7:
                data.append([float(x) for x in parts])
    
    data = np.array(data)
    
    # Extract data
    x_coords = data[:, 0].astype(int)
    y_coords = data[:, 1].astype(int)
    u_x = data[:, 2]
    u_y = data[:, 3]
    u_mag = data[:, 4]
    pressure = data[:, 5]
    obstacles = data[:, 6].astype(int)
    
    print(f"Data loading completed, {len(data)} grid points")
    print(f"Grid size: {x_coords.max()+1} x {y_coords.max()+1}")
    
    return x_coords, y_coords, u_x, u_y, u_mag, pressure, obstacles

def reshape_data(x_coords, y_coords, u_x, u_y, u_mag, pressure, obstacles):
    """
    Reshape 1D data to 2D grid
    
    Parameters:
        x_coords, y_coords: coordinate arrays
        u_x, u_y, u_mag: velocity related arrays
        pressure: pressure array
        obstacles: obstacle array
        
    Returns:
        reshaped 2D arrays
    """
    nx = int(x_coords.max()) + 1
    ny = int(y_coords.max()) + 1
    
    print(f"Reshaping data to {nx} x {ny} grid")
    
    # Reshape to 2D arrays (note: y-direction needs to be flipped because matplotlib's imshow default origin is at top-left)
    u_x_2d = u_x.reshape(ny, nx)
    u_y_2d = u_y.reshape(ny, nx)
    u_mag_2d = u_mag.reshape(ny, nx)
    pressure_2d = pressure.reshape(ny, nx)
    obstacles_2d = obstacles.reshape(ny, nx)
    
    # Flip y-direction to match physical coordinate system
    u_x_2d = np.flipud(u_x_2d)
    u_y_2d = np.flipud(u_y_2d)
    u_mag_2d = np.flipud(u_mag_2d)
    pressure_2d = np.flipud(pressure_2d)
    obstacles_2d = np.flipud(obstacles_2d)
    
    return u_x_2d, u_y_2d, u_mag_2d, pressure_2d, obstacles_2d

def create_velocity_magnitude_plot(u_mag_2d, obstacles_2d, nx, ny):
    """
    Create velocity magnitude distribution plot
    """
    fig, ax = plt.subplots(figsize=(10, 8))
    
    # Create grid
    x = np.linspace(0, nx-1, nx)
    y = np.linspace(0, ny-1, ny)
    X, Y = np.meshgrid(x, y)
    
    # Handle obstacle regions (set to NaN for not displaying)
    u_mag_plot = u_mag_2d.copy()
    u_mag_plot[obstacles_2d == 1] = np.nan
    
    # Velocity magnitude contour plot
    levels = np.linspace(0, u_mag_plot[~np.isnan(u_mag_plot)].max(), 20)
    contour = ax.contourf(X, Y, u_mag_plot, levels=levels, cmap='viridis', extend='max')
    ax.set_title('Velocity Magnitude Distribution', fontsize=14, fontweight='bold')
    ax.set_xlabel('X Coordinate')
    ax.set_ylabel('Y Coordinate')
    ax.set_aspect('equal')
    
    # Add colorbar
    cbar = plt.colorbar(contour, ax=ax)
    cbar.set_label('Velocity Magnitude', rotation=270, labelpad=15)
    
    # Add obstacle boundaries
    obstacle_boundary = np.zeros_like(obstacles_2d)
    for i in range(1, ny-1):
        for j in range(1, nx-1):
            if obstacles_2d[i, j] == 1:
                # Check if surrounded by non-obstacles
                if (obstacles_2d[i-1, j] == 0 or obstacles_2d[i+1, j] == 0 or 
                    obstacles_2d[i, j-1] == 0 or obstacles_2d[i, j+1] == 0):
                    obstacle_boundary[i, j] = 1
    
    # Draw obstacle boundaries
    ax.contour(X, Y, obstacle_boundary, levels=[0.5], colors='black', linewidths=2)
    
    plt.tight_layout()
    return fig

def create_pressure_plot(pressure_2d, obstacles_2d, nx, ny):
    """
    Create pressure distribution plot
    """
    fig, ax = plt.subplots(figsize=(10, 8))
    
    x = np.linspace(0, nx-1, nx)
    y = np.linspace(0, ny-1, ny)
    X, Y = np.meshgrid(x, y)
    
    # Handle obstacle regions
    pressure_plot = pressure_2d.copy()
    pressure_plot[obstacles_2d == 1] = np.nan
    
    # Pressure contour plot
    levels = np.linspace(pressure_plot[~np.isnan(pressure_plot)].min(), 
                        pressure_plot[~np.isnan(pressure_plot)].max(), 25)
    contour = ax.contourf(X, Y, pressure_plot, levels=levels, cmap='plasma', extend='both')
    
    ax.set_title('Pressure Distribution', fontsize=16, fontweight='bold')
    ax.set_xlabel('X Coordinate')
    ax.set_ylabel('Y Coordinate')
    ax.set_aspect('equal')
    
    # Add colorbar
    cbar = plt.colorbar(contour, ax=ax)
    cbar.set_label('Pressure', rotation=270, labelpad=15)
    
    # Add obstacle boundaries
    obstacle_boundary = np.zeros_like(obstacles_2d)
    for i in range(1, ny-1):
        for j in range(1, nx-1):
            if obstacles_2d[i, j] == 1:
                if (obstacles_2d[i-1, j] == 0 or obstacles_2d[i+1, j] == 0 or 
                    obstacles_2d[i, j-1] == 0 or obstacles_2d[i, j+1] == 0):
                    obstacle_boundary[i, j] = 1
    
    ax.contour(X, Y, obstacle_boundary, levels=[0.5], colors='black', linewidths=2)
    
    plt.tight_layout()
    return fig

def create_x_velocity_plot(u_x_2d, obstacles_2d, nx, ny):
    """
    Create X-direction velocity component plot
    """
    fig, ax = plt.subplots(figsize=(10, 8))
    
    x = np.linspace(0, nx-1, nx)
    y = np.linspace(0, ny-1, ny)
    X, Y = np.meshgrid(x, y)
    
    # Handle obstacle regions
    u_x_plot = u_x_2d.copy()
    u_x_plot[obstacles_2d == 1] = np.nan
    
    # X-velocity contour plot
    contour = ax.contourf(X, Y, u_x_plot, levels=20, cmap='RdBu_r')
    ax.set_title('X-Direction Velocity Component', fontsize=12, fontweight='bold')
    ax.set_xlabel('X Coordinate')
    ax.set_ylabel('Y Coordinate')
    ax.set_aspect('equal')
    plt.colorbar(contour, ax=ax, shrink=0.8, label='X-Velocity Component')
    
    # Add obstacle boundaries
    obstacle_boundary = np.zeros_like(obstacles_2d)
    for i in range(1, ny-1):
        for j in range(1, nx-1):
            if obstacles_2d[i, j] == 1:
                if (obstacles_2d[i-1, j] == 0 or obstacles_2d[i+1, j] == 0 or 
                    obstacles_2d[i, j-1] == 0 or obstacles_2d[i, j+1] == 0):
                    obstacle_boundary[i, j] = 1
    
    ax.contour(X, Y, obstacle_boundary, levels=[0.5], colors='black', linewidths=1.5)
    
    plt.tight_layout()
    return fig

def create_y_velocity_plot(u_y_2d, obstacles_2d, nx, ny):
    """
    Create Y-direction velocity component plot
    """
    fig, ax = plt.subplots(figsize=(10, 8))
    
    x = np.linspace(0, nx-1, nx)
    y = np.linspace(0, ny-1, ny)
    X, Y = np.meshgrid(x, y)
    
    # Handle obstacle regions
    u_y_plot = u_y_2d.copy()
    u_y_plot[obstacles_2d == 1] = np.nan
    
    # Y-velocity contour plot
    contour = ax.contourf(X, Y, u_y_plot, levels=20, cmap='RdBu_r')
    ax.set_title('Y-Direction Velocity Component', fontsize=12, fontweight='bold')
    ax.set_xlabel('X Coordinate')
    ax.set_ylabel('Y Coordinate')
    ax.set_aspect('equal')
    plt.colorbar(contour, ax=ax, shrink=0.8, label='Y-Velocity Component')
    
    # Add obstacle boundaries
    obstacle_boundary = np.zeros_like(obstacles_2d)
    for i in range(1, ny-1):
        for j in range(1, nx-1):
            if obstacles_2d[i, j] == 1:
                if (obstacles_2d[i-1, j] == 0 or obstacles_2d[i+1, j] == 0 or 
                    obstacles_2d[i, j-1] == 0 or obstacles_2d[i, j+1] == 0):
                    obstacle_boundary[i, j] = 1
    
    ax.contour(X, Y, obstacle_boundary, levels=[0.5], colors='black', linewidths=1.5)
    
    plt.tight_layout()
    return fig

def main():
    """
    Main function
    """
    print("=" * 60)
    print("Lid-Driven Cavity Visualization Program")
    print("=" * 60)
    
    # Data file path
    data_file = "../MPI/final_state.dat"
    
    try:
        # Load data
        x_coords, y_coords, u_x, u_y, u_mag, pressure, obstacles = load_final_state(data_file)
        
        # Reshape data
        u_x_2d, u_y_2d, u_mag_2d, pressure_2d, obstacles_2d = reshape_data(
            x_coords, y_coords, u_x, u_y, u_mag, pressure, obstacles)
        
        nx = int(x_coords.max()) + 1
        ny = int(y_coords.max()) + 1
        
        print(f"\nData Statistics:")
        print(f"Velocity range: {u_mag.min():.2e} - {u_mag.max():.2e}")
        print(f"Pressure range: {pressure.min():.2e} - {pressure.max():.2e}")
        print(f"Number of obstacles: {np.sum(obstacles)}")
        print(f"Number of fluid grid points: {np.sum(obstacles == 0)}")
        
        # Create visualization plots
        print("\nGenerating visualization plots...")
        
        # 1. Velocity magnitude plot
        print("Generating velocity magnitude plot...")
        vel_mag_fig = create_velocity_magnitude_plot(u_mag_2d, obstacles_2d, nx, ny)
        vel_mag_fig.savefig('error_velocity_magnitude_distribution.png', dpi=300, bbox_inches='tight')
        print("Velocity magnitude plot saved as error_velocity_magnitude_distribution.png")
        
        # 2. Pressure distribution plot
        print("Generating pressure distribution plot...")
        pressure_fig = create_pressure_plot(pressure_2d, obstacles_2d, nx, ny)
        pressure_fig.savefig('pressure_distribution.png', dpi=300, bbox_inches='tight')
        print("Pressure distribution plot saved as pressure_distribution.png")
        
        # 3. X-velocity component plot
        print("Generating X-direction velocity component plot...")
        x_vel_fig = create_x_velocity_plot(u_x_2d, obstacles_2d, nx, ny)
        x_vel_fig.savefig('x_velocity_component.png', dpi=300, bbox_inches='tight')
        print("X-velocity component plot saved as x_velocity_component.png")
        
        # 4. Y-velocity component plot
        print("Generating Y-direction velocity component plot...")
        y_vel_fig = create_y_velocity_plot(u_y_2d, obstacles_2d, nx, ny)
        y_vel_fig.savefig('y_velocity_component.png', dpi=300, bbox_inches='tight')
        print("Y-velocity component plot saved as y_velocity_component.png")
        
        print("\nAll visualization plots generated successfully!")
        print("\nOutput files:")
        print("- error_velocity_magnitude_distribution.png: Velocity magnitude distribution")
        print("- pressure_distribution.png: Pressure distribution")
        print("- x_velocity_component.png: X-direction velocity component")
        print("- y_velocity_component.png: Y-direction velocity component")
        
        # Show plots
        plt.show()
        
    except FileNotFoundError:
        print(f"Error: Data file {data_file} not found")
        print("Please ensure final_state.dat file is in the current directory")
    except Exception as e:
        print(f"Error: {e}")
        print("Please check the data file format")

if __name__ == "__main__":
    main()
