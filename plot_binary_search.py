#!/usr/bin/env python3
"""
Binary Search Convergence Visualizer

Creates visualizations specifically designed for binary search:
1. Convergence trace - how bounds narrow over iterations
2. Sorted values with cutoff point
3. Distribution around cutoff
4. CDF with cutoff marked

Usage:
    python3 plot_binary_search.py convergence.csv [--importance values.csv]
"""

import numpy as np
import matplotlib.pyplot as plt
import sys
import argparse

def plot_convergence_trace(convergence_data, output_file=None):
    """Plot how binary search bounds narrow over iterations"""
    
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 10))
    
    iterations = convergence_data['iteration']
    min_bounds = convergence_data['min_bound']
    max_bounds = convergence_data['max_bound']
    test_points = convergence_data['test_point']
    counts = convergence_data['count_above']
    target = convergence_data['target'][0]
    
    # Plot 1: Bounds narrowing
    ax1.fill_between(iterations, min_bounds, max_bounds, alpha=0.3, label='Search Range')
    ax1.plot(iterations, min_bounds, 'b-', linewidth=2, label='Lower Bound')
    ax1.plot(iterations, max_bounds, 'r-', linewidth=2, label='Upper Bound')
    ax1.plot(iterations, test_points, 'go-', linewidth=2, markersize=8, label='Test Point')
    
    # Mark final cutoff
    final_cutoff = test_points.iloc[-1]
    ax1.axhline(y=final_cutoff, color='purple', linestyle='--', linewidth=2, 
                label=f'Final Cutoff: {final_cutoff:.2f}')
    
    ax1.set_xlabel('Iteration', fontsize=12)
    ax1.set_ylabel('Importance Value', fontsize=12)
    ax1.set_title('Binary Search Convergence: Bounds Narrowing', fontsize=14, fontweight='bold')
    ax1.legend(loc='best')
    ax1.grid(True, alpha=0.3)
    
    # Plot 2: Count above vs target
    ax2.plot(iterations, counts, 'bo-', linewidth=2, markersize=8, label='Count Above Test Point')
    ax2.axhline(y=target, color='red', linestyle='--', linewidth=2, label=f'Target: {target}')
    
    # Shade the region where we're converging
    converged_mask = np.abs(counts - target) <= 3
    if converged_mask.any():
        ax2.fill_between(iterations, 0, counts.max() * 1.1, 
                        where=converged_mask, alpha=0.2, color='green', 
                        label='Converged Region')
    
    ax2.set_xlabel('Iteration', fontsize=12)
    ax2.set_ylabel('Count Above Test Point', fontsize=12)
    ax2.set_title('Convergence: Count Above vs Target', fontsize=14, fontweight='bold')
    ax2.legend(loc='best')
    ax2.grid(True, alpha=0.3)
    
    plt.tight_layout()
    
    if output_file:
        plt.savefig(output_file, dpi=300, bbox_inches='tight')
        print(f"Convergence plot saved: {output_file}")
    else:
        plt.show()
    
    plt.close()


def plot_cutoff_in_sorted_data(importance_values, cutoff, output_file=None):
    """Show where cutoff falls in sorted values"""
    
    sorted_vals = np.sort(importance_values)
    
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 10))
    
    # Plot 1: Full sorted data with cutoff
    indices = np.arange(len(sorted_vals))
    colors = ['red' if val > cutoff else 'blue' for val in sorted_vals]
    
    ax1.scatter(indices, sorted_vals, c=colors, s=10, alpha=0.6)
    ax1.axhline(y=cutoff, color='green', linestyle='--', linewidth=3, 
                label=f'Cutoff: {cutoff:.2f}')
    
    # Find cutoff index
    cutoff_idx = np.searchsorted(sorted_vals, cutoff)
    ax1.axvline(x=cutoff_idx, color='green', linestyle=':', linewidth=2, alpha=0.7,
                label=f'Cutoff Index: {cutoff_idx}')
    
    ax1.set_xlabel('Element Index (sorted)', fontsize=12)
    ax1.set_ylabel('Importance Value', fontsize=12)
    ax1.set_title('Cutoff Position in Sorted Data', fontsize=14, fontweight='bold')
    ax1.legend(loc='best')
    ax1.grid(True, alpha=0.3)
    
    # Add text annotations
    selected = np.sum(sorted_vals > cutoff)
    rejected = len(sorted_vals) - selected
    ax1.text(0.02, 0.98, f'Selected (above cutoff): {selected}\nRejected (below cutoff): {rejected}',
             transform=ax1.transAxes, fontsize=10, verticalalignment='top',
             bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))
    
    # Plot 2: Zoom around cutoff
    window = 200
    start_idx = max(0, cutoff_idx - window)
    end_idx = min(len(sorted_vals), cutoff_idx + window)
    
    zoom_indices = np.arange(start_idx, end_idx)
    zoom_vals = sorted_vals[start_idx:end_idx]
    zoom_colors = ['red' if val > cutoff else 'blue' for val in zoom_vals]
    
    ax2.scatter(zoom_indices, zoom_vals, c=zoom_colors, s=20, alpha=0.6)
    ax2.axhline(y=cutoff, color='green', linestyle='--', linewidth=3, 
                label=f'Cutoff: {cutoff:.2f}')
    ax2.axvline(x=cutoff_idx, color='green', linestyle=':', linewidth=2, alpha=0.7)
    
    ax2.set_xlabel('Element Index (sorted)', fontsize=12)
    ax2.set_ylabel('Importance Value', fontsize=12)
    ax2.set_title(f'Zoom: ±{window} elements around cutoff', fontsize=14, fontweight='bold')
    ax2.legend(loc='best')
    ax2.grid(True, alpha=0.3)
    
    plt.tight_layout()
    
    if output_file:
        plt.savefig(output_file, dpi=300, bbox_inches='tight')
        print(f"Cutoff position plot saved: {output_file}")
    else:
        plt.show()
    
    plt.close()


def plot_distribution_around_cutoff(importance_values, cutoff, output_file=None):
    """Show distribution around the cutoff point"""
    
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 6))
    
    # Plot 1: Histogram around cutoff
    range_size = cutoff * 0.2  # ±20% of cutoff
    vals_in_range = importance_values[(importance_values >= cutoff - range_size) & 
                                      (importance_values <= cutoff + range_size)]
    
    ax1.hist(vals_in_range, bins=50, alpha=0.7, color='skyblue', edgecolor='black')
    ax1.axvline(x=cutoff, color='red', linestyle='--', linewidth=3, label=f'Cutoff: {cutoff:.2f}')
    
    ax1.set_xlabel('Importance Value', fontsize=12)
    ax1.set_ylabel('Count', fontsize=12)
    ax1.set_title(f'Distribution ±20% around Cutoff', fontsize=14, fontweight='bold')
    ax1.legend(loc='best')
    ax1.grid(True, alpha=0.3, axis='y')
    
    # Plot 2: Log scale histogram (full range)
    ax2.hist(importance_values, bins=100, alpha=0.7, color='lightcoral', edgecolor='black')
    ax2.axvline(x=cutoff, color='red', linestyle='--', linewidth=3, label=f'Cutoff: {cutoff:.2f}')
    ax2.set_yscale('log')
    
    ax2.set_xlabel('Importance Value', fontsize=12)
    ax2.set_ylabel('Count (log scale)', fontsize=12)
    ax2.set_title('Full Distribution (Log Scale)', fontsize=14, fontweight='bold')
    ax2.legend(loc='best')
    ax2.grid(True, alpha=0.3, which='both')
    
    plt.tight_layout()
    
    if output_file:
        plt.savefig(output_file, dpi=300, bbox_inches='tight')
        print(f"Distribution plot saved: {output_file}")
    else:
        plt.show()
    
    plt.close()


def plot_cdf(importance_values, cutoff, output_file=None):
    """Plot cumulative distribution function with cutoff"""
    
    sorted_vals = np.sort(importance_values)
    cdf = np.arange(1, len(sorted_vals) + 1) / len(sorted_vals) * 100
    
    fig, ax = plt.subplots(figsize=(12, 6))
    
    # Plot CDF
    ax.plot(sorted_vals, cdf, 'b-', linewidth=2, label='CDF')
    
    # Mark cutoff
    cutoff_idx = np.searchsorted(sorted_vals, cutoff)
    cutoff_percentile = (cutoff_idx / len(sorted_vals)) * 100
    
    ax.axvline(x=cutoff, color='red', linestyle='--', linewidth=3, 
              label=f'Cutoff: {cutoff:.2f} ({cutoff_percentile:.1f}th percentile)')
    ax.axhline(y=cutoff_percentile, color='red', linestyle=':', linewidth=2, alpha=0.5)
    
    # Shade selected/rejected regions
    ax.fill_between(sorted_vals, 0, cdf, where=(sorted_vals <= cutoff), 
                    alpha=0.2, color='blue', label='Rejected')
    ax.fill_between(sorted_vals, 0, cdf, where=(sorted_vals > cutoff), 
                    alpha=0.2, color='green', label='Selected')
    
    ax.set_xlabel('Importance Value', fontsize=12)
    ax.set_ylabel('Cumulative Percentage', fontsize=12)
    ax.set_title('Cumulative Distribution Function (CDF)', fontsize=14, fontweight='bold')
    ax.legend(loc='best')
    ax.grid(True, alpha=0.3)
    
    # Add percentile markers
    percentiles = [10, 25, 50, 75, 90, 95, 99]
    for p in percentiles:
        idx = int(len(sorted_vals) * p / 100)
        if idx < len(sorted_vals):
            val = sorted_vals[idx]
            ax.plot(val, p, 'ko', markersize=5)
            if p in [50, 75, 95]:  # Label some key percentiles
                ax.annotate(f'P{p}', xy=(val, p), xytext=(10, 10), 
                           textcoords='offset points', fontsize=8)
    
    plt.tight_layout()
    
    if output_file:
        plt.savefig(output_file, dpi=300, bbox_inches='tight')
        print(f"CDF plot saved: {output_file}")
    else:
        plt.show()
    
    plt.close()


def create_all_plots(convergence_file, importance_file=None, save_plots=False):
    """Create all binary search visualizations"""
    
    import pandas as pd
    
    # Load convergence data
    conv_data = pd.read_csv(convergence_file)
    
    print(f"Loaded convergence data: {len(conv_data)} iterations")
    print(f"Final cutoff: {conv_data['test_point'].iloc[-1]:.2f}")
    print(f"Target: {conv_data['target'].iloc[0]}")
    print(f"Final count: {conv_data['count_above'].iloc[-1]}")
    
    # Plot convergence
    output = f"{convergence_file.replace('.csv', '_convergence.png')}" if save_plots else None
    plot_convergence_trace(conv_data, output)
    
    # If importance values provided, create additional plots
    if importance_file:
        importance_data = pd.read_csv(importance_file)
        importance_values = importance_data['importance'].values
        cutoff = conv_data['test_point'].iloc[-1]
        
        print(f"\nLoaded importance values: {len(importance_values)} values")
        print(f"Range: [{importance_values.min():.2f}, {importance_values.max():.2f}]")
        
        # Plot cutoff in sorted data
        output = f"{convergence_file.replace('.csv', '_sorted.png')}" if save_plots else None
        plot_cutoff_in_sorted_data(importance_values, cutoff, output)
        
        # Plot distribution around cutoff
        output = f"{convergence_file.replace('.csv', '_distribution.png')}" if save_plots else None
        plot_distribution_around_cutoff(importance_values, cutoff, output)
        
        # Plot CDF
        output = f"{convergence_file.replace('.csv', '_cdf.png')}" if save_plots else None
        plot_cdf(importance_values, cutoff, output)
        
        print("\nAll plots created successfully!")


def main():
    parser = argparse.ArgumentParser(
        description='Visualize binary search convergence',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
    # View convergence trace only
    python3 plot_binary_search.py convergence.csv
    
    # View all plots with importance values
    python3 plot_binary_search.py convergence.csv --importance values.csv
    
    # Save plots to files
    python3 plot_binary_search.py convergence.csv --importance values.csv --save
        """
    )
    
    parser.add_argument('convergence_file', 
                       help='CSV file with convergence data (iteration, bounds, counts)')
    parser.add_argument('--importance', '-i', 
                       help='CSV file with importance values (for additional plots)')
    parser.add_argument('--save', '-s', action='store_true',
                       help='Save plots as PNG files instead of displaying')
    
    args = parser.parse_args()
    
    create_all_plots(args.convergence_file, args.importance, args.save)


if __name__ == '__main__':
    main()