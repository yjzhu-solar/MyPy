#!/usr/bin/env python3
"""
Example script demonstrating the standalone plot_slit_position function.

This example shows how to:
1. Create slit data programmatically 
2. Plot slit positions on custom matplotlib axes
3. Create publication-quality figures with multiple slits
4. Customize appearance for different use cases
"""

import numpy as np
import matplotlib.pyplot as plt
from slit_interactive import generate_straight_slit_data, plot_slit_position

# Create synthetic data for demonstration
np.random.seed(42)
nx, ny, nt = 200, 150, 100

# Create synthetic time-series data with some structure
data_cube = np.random.random((ny, nx, nt)) * 100
# Add some temporal variations
for t in range(nt):
    data_cube[:, :, t] += 50 * np.sin(t * 0.1) * np.exp(-((np.arange(nx)[None, :] - 100)**2 + (np.arange(ny)[:, None] - 75)**2) / 1000)

# Compute standard deviation for visualization
std_image = np.std(data_cube, axis=2)

print("Creating example plots with standalone slit position plotting...")

# Example 1: Single slit with default styling
print("1. Creating single slit example...")
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))

# Display images
ax1.imshow(data_cube[:, :, 0], cmap='viridis', origin='lower')
ax2.imshow(std_image, cmap='magma', origin='lower')

# Generate slit data
result = generate_straight_slit_data(100, 75, 80, 30, data_cube, 'NDArray', line_width=5)

# Plot slit position with automatic styling
plot_elements1 = plot_slit_position(ax1, result, show_legend=True)
plot_elements2 = plot_slit_position(ax2, result, boundary_color='white', 
                                   curve_color='cyan', show_legend=False)

ax1.set_title('Frame 0 with Slit')
ax2.set_title('Standard Deviation with Slit')
ax1.set_xlabel('X [pixels]')
ax1.set_ylabel('Y [pixels]')
ax2.set_xlabel('X [pixels]')
ax2.set_ylabel('Y [pixels]')
plt.tight_layout()
plt.savefig('single_slit_example.png', dpi=150, bbox_inches='tight')
plt.close()

# Example 2: Multiple slits with different orientations and colors
print("2. Creating multiple slits example...")
slits = [
    generate_straight_slit_data(60, 40, 60, 0, data_cube, 'NDArray'),    # Horizontal
    generate_straight_slit_data(100, 75, 60, 90, data_cube, 'NDArray'),   # Vertical
    generate_straight_slit_data(140, 110, 60, 45, data_cube, 'NDArray'),   # Diagonal
    generate_straight_slit_data(80, 100, 40, -30, data_cube, 'NDArray'),   # Other diagonal
]

colors = ['red', 'blue', 'green', 'orange']
labels = ['Horizontal', 'Vertical', '45° Diagonal', '-30° Diagonal']

fig, ax = plt.subplots(figsize=(10, 8))
ax.imshow(std_image, cmap='gray', origin='lower', alpha=0.8)

# Plot each slit with custom colors
for i, (slit_result, color, label) in enumerate(zip(slits, colors, labels)):
    # Only show legend for the first slit to avoid clutter
    show_legend = (i == 0)
    plot_elements = plot_slit_position(
        ax, slit_result, 
        curve_color=color, 
        point_color=color,
        boundary_color=color,
        boundary_alpha=0.3,
        curve_width=2.5,
        point_size=8,
        show_legend=show_legend
    )

ax.set_title('Multiple Slits with Different Orientations', fontsize=14)
ax.set_xlabel('X [pixels]')
ax.set_ylabel('Y [pixels]')

# Create custom legend
from matplotlib.lines import Line2D
custom_lines = [Line2D([0], [0], color=color, lw=2) for color in colors]
ax.legend(custom_lines, labels, loc='upper right')

plt.tight_layout()
plt.savefig('multiple_slits_example.png', dpi=150, bbox_inches='tight')
plt.close()

# Example 3: Publication-style figure with minimal overlay
print("3. Creating publication-style example...")
fig, ax = plt.subplots(figsize=(8, 6))

# High-contrast display for publication
ax.imshow(data_cube[:, :, 25], cmap='plasma', origin='lower')

# Single prominent slit
result = generate_straight_slit_data(100, 75, 100, 60, data_cube, 'NDArray', line_width=7)

# Minimal styling for clarity
plot_elements = plot_slit_position(
    ax, result,
    show_boundary=True,
    show_curve=True, 
    show_control_points=True,
    boundary_color='white',
    curve_color='white',
    point_color='yellow',
    boundary_alpha=0.8,
    curve_alpha=1.0,
    point_alpha=1.0,
    curve_width=3,
    point_size=10,
    show_legend=False
)

ax.set_title('Solar Data Analysis: Slit Definition', fontsize=16, pad=20)
ax.set_xlabel('Solar X [pixels]', fontsize=12)
ax.set_ylabel('Solar Y [pixels]', fontsize=12)

# Add scale bar or other annotations as needed
ax.text(0.02, 0.98, 'Slit Length: 100 pixels\nAngle: 60°', 
        transform=ax.transAxes, fontsize=10, verticalalignment='top',
        bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))

plt.tight_layout()
plt.savefig('publication_slit_example.png', dpi=300, bbox_inches='tight')
plt.close()

# Example 4: Batch processing visualization
print("4. Creating batch processing example...")
fig, axes = plt.subplots(2, 3, figsize=(15, 10))
axes = axes.flatten()

# Generate slits at different positions
centers = [(50, 40), (100, 50), (150, 60), (80, 100), (120, 110), (160, 120)]
angles = [0, 30, 60, 90, 120, 150]

for i, (ax, center, angle) in enumerate(zip(axes, centers, angles)):
    # Show different frames for variety
    frame_idx = i * 8
    ax.imshow(data_cube[:, :, frame_idx], cmap='viridis', origin='lower')
    
    # Generate slit
    result = generate_straight_slit_data(center[0], center[1], 50, angle, 
                                       data_cube, 'NDArray', line_width=3)
    
    # Plot with consistent styling
    plot_elements = plot_slit_position(
        ax, result,
        boundary_color='white',
        curve_color='red', 
        point_color='yellow',
        boundary_alpha=0.6,
        show_legend=False
    )
    
    ax.set_title(f'Frame {frame_idx}, Angle {angle}°', fontsize=10)
    ax.set_xlabel('X [pix]', fontsize=8)
    ax.set_ylabel('Y [pix]', fontsize=8)

plt.suptitle('Batch Processing: Multiple Slits Across Different Frames', fontsize=14)
plt.tight_layout()
plt.savefig('batch_processing_example.png', dpi=150, bbox_inches='tight')
plt.close()

print("\nExample plots saved:")
print("- single_slit_example.png: Basic usage with two panels")
print("- multiple_slits_example.png: Multiple slits with custom colors")
print("- publication_slit_example.png: Publication-quality styling")
print("- batch_processing_example.png: Batch processing visualization")

print("\nStandalone slit plotting examples completed successfully!")
print("The plot_slit_position function provides flexible, non-GUI slit visualization.")