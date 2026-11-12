#!/usr/bin/env python3
"""
plot_cutoff.py - Visualization tool for cutoff binary search test results

Usage:
    python plot_cutoff.py cutoff_test_layer0_q_proj.csv
    python plot_cutoff.py cutoff_test_layer0_q_proj.csv --bins 100 --save
"""

import numpy as np
import matplotlib.pyplot as plt
import argparse
import sys

def parse_csv(filename):
    """Parse CSV file and extract metadata + importance values"""
    metadata = {}
    values = []
    
    with open(filename, 'r') as f:
        for line in f:
            line = line.strip()
            
            if line.startswith('#'):
                # Parse metadata
                if ':' in line:
                    key, val = line[1:].split(':', 1)
                    key = key.strip()
                    val = val.strip()
                    
                    # Try to convert to number
                    try:
                        val = float(val)
                    except:
                        pass
                    
                    metadata[key] = val
            elif line and line != 'importance_value':
                # Parse data
                try:
                    values.append(float(line))
                except:
                    pass
    
    return metadata, np.array(values)

def plot_distribution(metadata, values, bins=64, save=False):
    """Create comprehensive visualization of importance distribution"""
    
    fig = plt.figure(figsize=(16, 10))
    
    # Get cutoff value
    cutoff = float(metadata.get('Cutoff', 0))
    effort = float(metadata.get('Effort', 0.5))
    above = int(metadata.get('Above', 0))
    below = int(metadata.get('Below', 0))
    
    # ========================================
    # 1. Histogram with cutoff line
    # ========================================
    ax1 = plt.subplot(2, 3, 1)
    
    counts, bin_edges, patches = ax1.hist(values, bins=bins, 
                                          color='steelblue', 
                                          alpha=0.7, 
                                          edgecolor='black')
    
    # Color bins based on cutoff
    for i, patch in enumerate(patches):
        if bin_edges[i] > cutoff:
            patch.set_facecolor('coral')
        else:
            patch.set_facecolor('lightsteelblue')
    
    ax1.axvline(cutoff, color='red', linestyle='--', linewidth=2, 
               label=f'Cutoff = {cutoff:.0f}')
    ax1.set_xlabel('Importance Value')
    ax1.set_ylabel('Frequency')
    ax1.set_title('Importance Distribution with Cutoff')
    ax1.legend()
    ax1.grid(True, alpha=0.3)
    
    # ========================================
    # 2. Cumulative distribution
    # ========================================
    ax2 = plt.subplot(2, 3, 2)
    
    sorted_vals = np.sort(values)
    cumulative = np.arange(1, len(sorted_vals) + 1)
    
    ax2.plot(sorted_vals, cumulative, linewidth=2, color='navy')
    ax2.axvline(cutoff, color='red', linestyle='--', linewidth=2, 
               label=f'Cutoff = {cutoff:.0f}')
    ax2.axhline(above, color='green', linestyle=':', linewidth=2,
               label=f'Selected: {above}')
    ax2.set_xlabel('Importance Value')
    ax2.set_ylabel('Cumulative Count')
    ax2.set_title('Cumulative Distribution Function')
    ax2.legend()
    ax2.grid(True, alpha=0.3)
    
    # ========================================
    # 3. Log scale histogram
    # ========================================
    ax3 = plt.subplot(2, 3, 3)
    
    ax3.hist(values, bins=bins, color='steelblue', alpha=0.7, 
            edgecolor='black', log=True)
    ax3.axvline(cutoff, color='red', linestyle='--', linewidth=2,
               label=f'Cutoff = {cutoff:.0f}')
    ax3.set_xlabel('Importance Value')
    ax3.set_ylabel('Frequency (log scale)')
    ax3.set_title('Log-Scale Distribution')
    ax3.legend()
    ax3.grid(True, alpha=0.3)
    
    # ========================================
    # 4. Box plot and statistics
    # ========================================
    ax4 = plt.subplot(2, 3, 4)
    
    # Split values by cutoff
    above_cutoff = values[values > cutoff]
    below_cutoff = values[values <= cutoff]
    
    bp = ax4.boxplot([below_cutoff, above_cutoff, values],
                     labels=['Below Cutoff', 'Above Cutoff', 'All'],
                     patch_artist=True,
                     showmeans=True)
    
    # Color boxes
    bp['boxes'][0].set_facecolor('lightsteelblue')
    bp['boxes'][1].set_facecolor('coral')
    bp['boxes'][2].set_facecolor('lightgray')
    
    ax4.axhline(cutoff, color='red', linestyle='--', linewidth=2,
               alpha=0.5, label='Cutoff')
    ax4.set_ylabel('Importance Value')
    ax4.set_title('Distribution Statistics')
    ax4.legend()
    ax4.grid(True, alpha=0.3, axis='y')
    
    # ========================================
    # 5. Percentile plot
    # ========================================
    ax5 = plt.subplot(2, 3, 5)
    
    percentiles = np.linspace(0, 100, 1000)
    percentile_values = np.percentile(values, percentiles)
    
    ax5.plot(percentiles, percentile_values, linewidth=2, color='purple')
    
    # Mark effort level
    effort_pct = (1.0 - effort) * 100
    ax5.axvline(effort_pct, color='green', linestyle=':', linewidth=2,
               label=f'Effort = {effort:.2f} ({effort_pct:.1f}th pct)')
    ax5.axhline(cutoff, color='red', linestyle='--', linewidth=2,
               label=f'Cutoff = {cutoff:.0f}')
    
    ax5.set_xlabel('Percentile')
    ax5.set_ylabel('Importance Value')
    ax5.set_title('Percentile Plot')
    ax5.legend()
    ax5.grid(True, alpha=0.3)
    
    # ========================================
    # 6. Summary statistics text
    # ========================================
    ax6 = plt.subplot(2, 3, 6)
    ax6.axis('off')
    
    stats_text = f"""
    CUTOFF ANALYSIS SUMMARY
    {'='*40}
    
    Configuration:
      Layer:        {metadata.get('Layer', 'N/A')}
      Weight:       {metadata.get('Weight', 'N/A')}
      Effort:       {effort:.3f}
    
    Cutoff Value:   {cutoff:.2f}
    
    Selection:
      Above cutoff: {above:,} ({above/len(values)*100:.1f}%)
      Below cutoff: {below:,} ({below/len(values)*100:.1f}%)
      Total:        {len(values):,}
    
    Statistics (All Values):
      Min:          {np.min(values):.2f}
      Max:          {np.max(values):.2f}
      Mean:         {np.mean(values):.2f}
      Median:       {np.median(values):.2f}
      Std Dev:      {np.std(values):.2f}
    
    Statistics (Above Cutoff):
      Count:        {len(above_cutoff):,}
      Mean:         {np.mean(above_cutoff) if len(above_cutoff) > 0 else 0:.2f}
      Min:          {np.min(above_cutoff) if len(above_cutoff) > 0 else 0:.2f}
      Max:          {np.max(above_cutoff) if len(above_cutoff) > 0 else 0:.2f}
    
    Statistics (Below Cutoff):
      Count:        {len(below_cutoff):,}
      Mean:         {np.mean(below_cutoff) if len(below_cutoff) > 0 else 0:.2f}
      Min:          {np.min(below_cutoff) if len(below_cutoff) > 0 else 0:.2f}
      Max:          {np.max(below_cutoff) if len(below_cutoff) > 0 else 0:.2f}
    """
    
    ax6.text(0.05, 0.95, stats_text, transform=ax6.transAxes,
            fontsize=9, verticalalignment='top', fontfamily='monospace',
            bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.3))
    
    # ========================================
    # Overall title
    # ========================================
    fig.suptitle(f'Cutoff Binary Search Analysis - Layer {metadata.get("Layer", "?")} '
                f'{metadata.get("Weight", "?")} (Effort={effort:.2f})',
                fontsize=14, fontweight='bold')
    
    plt.tight_layout()
    
    if save:
        output_file = filename.replace('.csv', '.png')
        plt.savefig(output_file, dpi=150, bbox_inches='tight')
        print(f"Saved plot to {output_file}")
    
    plt.show()

def plot_heatmap_2d(metadata, values, save=False):
    """Create 2D heatmap showing distribution density"""
    
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))
    
    cutoff = float(metadata.get('Cutoff', 0))
    
    # 2D histogram
    bins = 50
    h, xedges, yedges = np.histogram2d(
        np.arange(len(values)),
        values,
        bins=[100, bins]
    )
    
    # Plot heatmap
    im = ax1.imshow(h.T, aspect='auto', origin='lower',
                   extent=[0, len(values), np.min(values), np.max(values)],
                   cmap='hot', interpolation='bilinear')
    
    ax1.axhline(cutoff, color='cyan', linestyle='--', linewidth=2,
               label=f'Cutoff = {cutoff:.0f}')
    ax1.set_xlabel('Value Index')
    ax1.set_ylabel('Importance Value')
    ax1.set_title('2D Density Heatmap')
    ax1.legend()
    plt.colorbar(im, ax=ax1, label='Density')
    
    # Sorted heatmap
    sorted_vals = np.sort(values)
    h2, xedges2, yedges2 = np.histogram2d(
        np.arange(len(sorted_vals)),
        sorted_vals,
        bins=[100, bins]
    )
    
    im2 = ax2.imshow(h2.T, aspect='auto', origin='lower',
                    extent=[0, len(sorted_vals), np.min(sorted_vals), np.max(sorted_vals)],
                    cmap='hot', interpolation='bilinear')
    
    ax2.axhline(cutoff, color='cyan', linestyle='--', linewidth=2,
               label=f'Cutoff = {cutoff:.0f}')
    ax2.set_xlabel('Sorted Index')
    ax2.set_ylabel('Importance Value')
    ax2.set_title('2D Density Heatmap (Sorted)')
    ax2.legend()
    plt.colorbar(im2, ax=ax2, label='Density')
    
    plt.tight_layout()
    
    if save:
        output_file = filename.replace('.csv', '_heatmap.png')
        plt.savefig(output_file, dpi=150, bbox_inches='tight')
        print(f"Saved heatmap to {output_file}")
    
    plt.show()

if __name__ == '__main__':
    parser = argparse.ArgumentParser(
        description='Visualize cutoff binary search test results'
    )
    parser.add_argument('filename', help='CSV file with importance values')
    parser.add_argument('--bins', type=int, default=64, 
                       help='Number of histogram bins (default: 64)')
    parser.add_argument('--save', action='store_true',
                       help='Save plots to file')
    parser.add_argument('--heatmap', action='store_true',
                       help='Also generate 2D heatmap visualization')
    
    args = parser.parse_args()
    
    try:
        # Parse data
        print(f"Loading data from {args.filename}...")
        metadata, values = parse_csv(args.filename)
        
        print(f"Loaded {len(values)} importance values")
        print(f"Metadata: {metadata}")
        print()
        
        # Main visualization
        plot_distribution(metadata, values, bins=args.bins, save=args.save)
        
        # Optional heatmap
        if args.heatmap:
            plot_heatmap_2d(metadata, values, save=args.save)
        
    except Exception as e:
        print(f"Error: {e}", file=sys.stderr)
        sys.exit(1)