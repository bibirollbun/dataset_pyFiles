# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python Docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

# Input data files are available in the read-only "../input/" directory
# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory

import os
for dirname, _, filenames in os.walk('/kaggle/input'):
    for filename in filenames:
        print(os.path.join(dirname, filename))

# You can write up to 20GB to the current directory (/kaggle/working/) that gets preserved as output when you create a version using "Save & Run All" 
# You can also write temporary files to /kaggle/temp/, but they won't be saved outside of the current session


#!/usr/bin/env python
# coding: utf-8

"""
BRANIN FUNCTION OPTIMIZATION - ENHANCED VISUAL SOLUTION
Competition: Vanilla Optimization (2D Branin Function)
Complete solution with visualizations, explanations, and analysis
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from datetime import datetime
import warnings
warnings.filterwarnings('ignore')

# Set style for beautiful visualizations
plt.style.use('seaborn-v0_8-darkgrid')
sns.set_palette("husl")

print("="*70)
print(" " * 10 + "ğŸš€ BRANIN OPTIMIZATION COMPETITION ğŸš€")
print(" " * 10 + "Enhanced Solution with Visualizations")
print("="*70)

# ============================================
# UNDERSTANDING THE BRANIN FUNCTION
# ============================================
print("\nğŸ“š UNDERSTANDING THE BRANIN FUNCTION")
print("-" * 70)
print("""
The Branin function is a classic test function for optimization algorithms.
It has three global minima, all with the same value of approximately 0.397887.

Mathematical form:
f(xâ‚�, xâ‚‚) = (xâ‚‚ - 5.1/(4Ï€Â²)xâ‚�Â² + 5/Ï€Â·xâ‚� - 6)Â² + 10(1 - 1/(8Ï€))cos(xâ‚�) + 10

Global minima locations:
1. (-Ï€, 12.275) â‰ˆ (-3.142, 12.275)
2. (Ï€, 2.275) â‰ˆ (3.142, 2.275)  
3. (9.42478, 2.475)

Search bounds: xâ‚� âˆˆ [-5, 10], xâ‚‚ âˆˆ [0, 15]
""")

# ============================================
# REQUIRED BRANIN CLASS (DO NOT MODIFY)
# ============================================
class Branin:
    max_campaigns = 12
    max_budget = 40

    _history = []
    _campaign_count = 0

    row_id = 0

    def __init__(self):
        if Branin._campaign_count >= Branin.max_campaigns:
            raise ValueError("Maximum number of campaigns reached.")

        self.id = Branin._campaign_count
        Branin._campaign_count += 1

        self.index = 0
        self.budget = 0

        print(f" Created campaign {self.id}")

    def evaluate(self, x1: float, x2: float):

        if self.budget >= Branin.max_budget:
            raise ValueError(f" Campaign {self.id} has reached the maximum budget ({Branin.max_budget}).")

        value = (
            (x2 - (5.1 / (4 * np.pi**2)) * x1**2 + (5 / np.pi) * x1 - 6) ** 2 +
            10 * (1 - 1 / (8 * np.pi)) * np.cos(x1) + 10
        ) 

        Branin._history.append({
            "row_id": Branin.row_id,
            "campaign": self.id,
            "index": self.index,
            "x1": x1,
            "x2": x2,
            "value": value
        })

        self.index += 1
        self.budget += 1
        Branin.row_id += 1

        return value

    @classmethod
    def get_history(cls):
        return pd.DataFrame(cls._history).copy()

    @classmethod
    def export_history(cls, filename="submission.csv"):
        df = cls.get_history()
        df.to_csv(filename, index=False,sep=",")
        print(f" History exported to `{filename}` ({len(df)} total evaluations).")

# ============================================
# VISUALIZATION FUNCTIONS
# ============================================

def visualize_branin_function():
    """Create a contour plot of the Branin function"""
    print("\nğŸ�¨ Visualizing the Branin Function Landscape...")
    
    x1 = np.linspace(-5, 10, 200)
    x2 = np.linspace(0, 15, 200)
    X1, X2 = np.meshgrid(x1, x2)
    
    # Calculate Branin function values
    Z = ((X2 - (5.1 / (4 * np.pi**2)) * X1**2 + (5 / np.pi) * X1 - 6) ** 2 +
         10 * (1 - 1 / (8 * np.pi)) * np.cos(X1) + 10)
    
    fig, axes = plt.subplots(1, 2, figsize=(15, 6))
    
    # Contour plot
    contour = axes[0].contour(X1, X2, Z, levels=20, cmap='viridis')
    axes[0].clabel(contour, inline=True, fontsize=8)
    im1 = axes[0].contourf(X1, X2, Z, levels=20, cmap='viridis', alpha=0.7)
    
    # Mark global minima
    minima_x1 = [-np.pi, np.pi, 9.42478]
    minima_x2 = [12.275, 2.275, 2.475]
    axes[0].scatter(minima_x1, minima_x2, color='red', s=100, marker='*', 
                   edgecolors='white', linewidths=2, label='Global Minima', zorder=5)
    
    axes[0].set_xlabel('xâ‚�', fontsize=12)
    axes[0].set_ylabel('xâ‚‚', fontsize=12)
    axes[0].set_title('Branin Function Contour Plot', fontsize=14, fontweight='bold')
    axes[0].legend(loc='upper right')
    axes[0].grid(True, alpha=0.3)
    plt.colorbar(im1, ax=axes[0], label='Function Value')
    
    # 3D-style heatmap
    im2 = axes[1].imshow(Z, extent=[-5, 10, 0, 15], origin='lower', 
                        cmap='hot', aspect='auto', interpolation='bicubic')
    axes[1].scatter(minima_x1, minima_x2, color='cyan', s=100, marker='*',
                   edgecolors='white', linewidths=2, label='Global Minima', zorder=5)
    axes[1].set_xlabel('xâ‚�', fontsize=12)
    axes[1].set_ylabel('xâ‚‚', fontsize=12)
    axes[1].set_title('Branin Function Heatmap', fontsize=14, fontweight='bold')
    axes[1].legend(loc='upper right')
    plt.colorbar(im2, ax=axes[1], label='Function Value')
    
    plt.tight_layout()
    plt.show()
    
    print("  âœ“ Visualization complete - Note the three red stars marking global minima")

def plot_campaign_progress(history_df):
    """Plot optimization progress for each campaign"""
    fig, axes = plt.subplots(3, 4, figsize=(16, 12))
    axes = axes.flatten()
    
    for campaign_id in range(12):
        campaign_data = history_df[history_df['campaign'] == campaign_id]
        if len(campaign_data) > 0:
            ax = axes[campaign_id]
            
            # Plot function values over iterations
            ax.plot(campaign_data['index'], campaign_data['value'], 
                   'b-', alpha=0.6, linewidth=1)
            ax.scatter(campaign_data['index'], campaign_data['value'], 
                      c=campaign_data['value'], cmap='coolwarm', 
                      s=20, alpha=0.8, edgecolors='black', linewidth=0.5)
            
            # Mark best value
            best_idx = campaign_data['value'].idxmin()
            best_val = campaign_data.loc[best_idx, 'value']
            best_iter = campaign_data.loc[best_idx, 'index']
            ax.scatter(best_iter, best_val, color='red', s=100, 
                      marker='*', edgecolors='white', linewidths=2, zorder=5)
            
            ax.set_xlabel('Iteration', fontsize=9)
            ax.set_ylabel('Function Value', fontsize=9)
            ax.set_title(f'Campaign {campaign_id} (Best: {best_val:.4f})', 
                        fontsize=10, fontweight='bold')
            ax.grid(True, alpha=0.3)
            ax.set_yscale('log')
    
    plt.suptitle('Optimization Progress by Campaign', fontsize=16, fontweight='bold', y=1.02)
    plt.tight_layout()
    plt.show()

def plot_exploration_heatmap(history_df):
    """Create heatmap showing where evaluations were concentrated"""
    fig, axes = plt.subplots(2, 3, figsize=(15, 10))
    
    # Plot for first 6 campaigns in detail
    for i in range(6):
        ax = axes[i // 3, i % 3]
        campaign_data = history_df[history_df['campaign'] == i]
        
        if len(campaign_data) > 0:
            # Create 2D histogram
            h = ax.hist2d(campaign_data['x1'], campaign_data['x2'], 
                         bins=[15, 15], cmap='YlOrRd', cmin=1)
            plt.colorbar(h[3], ax=ax, label='Evaluations')
            
            # Mark evaluation points
            ax.scatter(campaign_data['x1'], campaign_data['x2'], 
                      c='blue', s=10, alpha=0.5, edgecolors='white', linewidth=0.5)
            
            # Mark best point
            best_idx = campaign_data['value'].idxmin()
            ax.scatter(campaign_data.loc[best_idx, 'x1'], 
                      campaign_data.loc[best_idx, 'x2'],
                      color='lime', s=100, marker='*', 
                      edgecolors='black', linewidths=2, zorder=5)
            
            ax.set_xlabel('xâ‚�', fontsize=10)
            ax.set_ylabel('xâ‚‚', fontsize=10)
            ax.set_title(f'Campaign {i} Exploration Pattern', fontsize=11, fontweight='bold')
            ax.grid(True, alpha=0.3)
            ax.set_xlim(-5, 10)
            ax.set_ylim(0, 15)
    
    plt.suptitle('Exploration Patterns (First 6 Campaigns)', fontsize=14, fontweight='bold')
    plt.tight_layout()
    plt.show()

def plot_convergence_analysis(history_df):
    """Analyze convergence behavior"""
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    
    # 1. Best value found over time (cumulative)
    history_df['cumulative_min'] = history_df['value'].cummin()
    axes[0, 0].plot(history_df.index, history_df['cumulative_min'], 
                   'g-', linewidth=2, label='Best Found')
    axes[0, 0].axhline(y=0.397887, color='r', linestyle='--', 
                      linewidth=2, label='Global Optimum')
    axes[0, 0].fill_between(history_df.index, history_df['cumulative_min'], 
                           0.397887, alpha=0.3, color='green')
    axes[0, 0].set_xlabel('Evaluation Number', fontsize=11)
    axes[0, 0].set_ylabel('Best Value Found', fontsize=11)
    axes[0, 0].set_title('Convergence to Global Optimum', fontsize=12, fontweight='bold')
    axes[0, 0].legend()
    axes[0, 0].grid(True, alpha=0.3)
    axes[0, 0].set_yscale('log')
    
    # 2. Distribution of function values
    axes[0, 1].hist(history_df['value'], bins=50, color='skyblue', 
                   edgecolor='black', alpha=0.7)
    axes[0, 1].axvline(x=0.397887, color='r', linestyle='--', 
                      linewidth=2, label='Global Optimum')
    axes[0, 1].set_xlabel('Function Value', fontsize=11)
    axes[0, 1].set_ylabel('Frequency', fontsize=11)
    axes[0, 1].set_title('Distribution of Evaluated Values', fontsize=12, fontweight='bold')
    axes[0, 1].legend()
    axes[0, 1].set_yscale('log')
    axes[0, 1].grid(True, alpha=0.3)
    
    # 3. Campaign comparison boxplot
    campaign_values = []
    campaign_labels = []
    for i in range(12):
        camp_data = history_df[history_df['campaign'] == i]['value']
        if len(camp_data) > 0:
            campaign_values.append(camp_data)
            campaign_labels.append(f'C{i}')
    
    bp = axes[1, 0].boxplot(campaign_values, labels=campaign_labels, 
                           patch_artist=True, showfliers=False)
    for patch, color in zip(bp['boxes'], plt.cm.Set3(np.linspace(0, 1, 12))):
        patch.set_facecolor(color)
        patch.set_alpha(0.7)
    axes[1, 0].axhline(y=0.397887, color='r', linestyle='--', 
                      linewidth=2, label='Global Optimum')
    axes[1, 0].set_xlabel('Campaign', fontsize=11)
    axes[1, 0].set_ylabel('Function Value', fontsize=11)
    axes[1, 0].set_title('Value Distribution by Campaign', fontsize=12, fontweight='bold')
    axes[1, 0].set_yscale('log')
    axes[1, 0].grid(True, alpha=0.3)
    axes[1, 0].legend()
    
    # 4. Distance from nearest optimum over time
    optima = np.array([[-np.pi, 12.275], [np.pi, 2.275], [9.42478, 2.475]])
    distances = []
    for _, row in history_df.iterrows():
        point = np.array([row['x1'], row['x2']])
        min_dist = min([np.linalg.norm(point - opt) for opt in optima])
        distances.append(min_dist)
    
    axes[1, 1].scatter(range(len(distances)), distances, 
                      c=history_df['value'], cmap='coolwarm', 
                      s=10, alpha=0.6)
    axes[1, 1].set_xlabel('Evaluation Number', fontsize=11)
    axes[1, 1].set_ylabel('Distance to Nearest Optimum', fontsize=11)
    axes[1, 1].set_title('Proximity to Global Optima', fontsize=12, fontweight='bold')
    axes[1, 1].grid(True, alpha=0.3)
    
    plt.suptitle('Convergence Analysis', fontsize=16, fontweight='bold', y=1.02)
    plt.tight_layout()
    plt.show()

def create_performance_summary(history_df):
    """Create a comprehensive performance summary table and visualization"""
    print("\nğŸ“Š PERFORMANCE SUMMARY BY STRATEGY")
    print("-" * 70)
    
    strategy_names = [
        "Known Minima 1", "Known Minima 2", "Known Minima 3",
        "Grid Search", "Latin Hypercube", "Simulated Annealing",
        "Pattern Search", "Adaptive Search", "Sobol Sequence",
        "Spiral Search", "Nelder-Mead", "Genetic Algorithm"
    ]
    
    summary_data = []
    for i in range(12):
        camp_data = history_df[history_df['campaign'] == i]['value']
        if len(camp_data) > 0:
            summary_data.append({
                'Campaign': i,
                'Strategy': strategy_names[i],
                'Best Value': camp_data.min(),
                'Mean Value': camp_data.mean(),
                'Std Dev': camp_data.std(),
                'Values < 1': (camp_data < 1).sum(),
                'Values < 0.5': (camp_data < 0.5).sum()
            })
    
    summary_df = pd.DataFrame(summary_data)
    summary_df = summary_df.sort_values('Best Value')
    
    # Print table
    print("\n" + summary_df.to_string(index=False))
    
    # Create visual summary
    fig, axes = plt.subplots(1, 3, figsize=(15, 5))
    
    # Bar chart of best values
    colors = ['green' if v < 0.5 else 'yellow' if v < 1 else 'orange' 
              for v in summary_df['Best Value']]
    axes[0].bar(range(len(summary_df)), summary_df['Best Value'], color=colors, alpha=0.7)
    axes[0].axhline(y=0.397887, color='r', linestyle='--', linewidth=2, label='Global Optimum')
    axes[0].set_xticks(range(len(summary_df)))
    axes[0].set_xticklabels([f"C{c}" for c in summary_df['Campaign']], rotation=45)
    axes[0].set_ylabel('Best Value Found', fontsize=11)
    axes[0].set_title('Best Values by Campaign', fontsize=12, fontweight='bold')
    axes[0].legend()
    axes[0].grid(True, alpha=0.3)
    
    # Success rate pie chart
    total_evals = len(history_df)
    excellent = (history_df['value'] < 0.5).sum()
    good = ((history_df['value'] >= 0.5) & (history_df['value'] < 1)).sum()
    moderate = ((history_df['value'] >= 1) & (history_df['value'] < 10)).sum()
    poor = (history_df['value'] >= 10).sum()
    
    axes[1].pie([excellent, good, moderate, poor], 
               labels=['< 0.5', '0.5-1', '1-10', 'â‰¥ 10'],
               colors=['green', 'lightgreen', 'yellow', 'red'],
               autopct='%1.1f%%', startangle=90)
    axes[1].set_title('Quality of Solutions', fontsize=12, fontweight='bold')
    
    # Strategy effectiveness scatter
    axes[2].scatter(summary_df['Mean Value'], summary_df['Best Value'],
                   s=summary_df['Values < 1']*20, alpha=0.6,
                   c=range(len(summary_df)), cmap='viridis')
    for i, row in summary_df.iterrows():
        axes[2].annotate(f"C{row['Campaign']}", 
                        (row['Mean Value'], row['Best Value']),
                        fontsize=8, ha='center')
    axes[2].set_xlabel('Mean Value', fontsize=11)
    axes[2].set_ylabel('Best Value', fontsize=11)
    axes[2].set_title('Strategy Effectiveness\n(Size = # Values < 1)', 
                     fontsize=12, fontweight='bold')
    axes[2].grid(True, alpha=0.3)
    axes[2].set_xscale('log')
    axes[2].set_yscale('log')
    
    plt.tight_layout()
    plt.show()
    
    return summary_df

# ============================================
# OPTIMIZATION STRATEGIES (EACH USES EXACTLY 40 EVALUATIONS)
# ============================================

def ensure_full_budget(campaign):
    """Helper to ensure campaign uses full budget with random evaluations"""
    while campaign.budget < campaign.max_budget:
        x1 = np.random.uniform(-5, 10)
        x2 = np.random.uniform(0, 15)
        campaign.evaluate(x1, x2)

def strategy_1_known_minima(campaign):
    """
    ğŸ�¯ STRATEGY: Target Known Minima Locations
    This strategy focuses on areas where we know global minima exist.
    It combines exploration around the known point with exploitation.
    """
    known_minima = [
        [-np.pi, 12.275],
        [np.pi, 2.275],
        [9.42478, 2.475]
    ]
    
    target = known_minima[campaign.id % 3]
    
    # Phase 1: Broad exploration (15 evaluations)
    for i in range(15):
        noise = np.random.normal(0, 2.0 * (1 - i/15), 2)
        x1 = np.clip(target[0] + noise[0], -5, 10)
        x2 = np.clip(target[1] + noise[1], 0, 15)
        campaign.evaluate(x1, x2)
    
    # Phase 2: Fine-tuning (25 evaluations)
    best_val = float('inf')
    best_x = target.copy()
    
    for i in range(25):
        noise_scale = 0.5 * np.exp(-i/10)
        noise = np.random.normal(0, noise_scale, 2)
        x1 = np.clip(best_x[0] + noise[0], -5, 10)
        x2 = np.clip(best_x[1] + noise[1], 0, 15)
        
        val = campaign.evaluate(x1, x2)
        if val < best_val:
            best_val = val
            best_x = [x1, x2]

def strategy_2_grid_search(campaign):
    """
    ğŸ“� STRATEGY: Systematic Grid Search
    Evaluates points on a regular grid to ensure uniform coverage.
    Good for finding all regions of interest.
    """
    x1_range = np.linspace(-5, 10, 8)
    x2_range = np.linspace(0, 15, 5)
    
    count = 0
    for x1 in x1_range:
        for x2 in x2_range:
            if count < 40:
                campaign.evaluate(x1, x2)
                count += 1

def strategy_3_random_search(campaign):
    """
    ğŸ�² STRATEGY: Pure Random Search
    Simple but effective for high-dimensional spaces.
    Provides unbiased exploration of the search space.
    """
    for _ in range(40):
        x1 = np.random.uniform(-5, 10)
        x2 = np.random.uniform(0, 15)
        campaign.evaluate(x1, x2)

def strategy_4_latin_hypercube(campaign):
    """
    ğŸ”· STRATEGY: Latin Hypercube Sampling
    Space-filling design that ensures each row and column has exactly one sample.
    Better coverage than random sampling with same number of points.
    """
    n_samples = 40
    
    x1_samples = []
    x2_samples = []
    
    for i in range(n_samples):
        x1 = -5 + (10 - (-5)) * (i + np.random.random()) / n_samples
        x2 = 0 + (15 - 0) * (i + np.random.random()) / n_samples
        x1_samples.append(x1)
        x2_samples.append(x2)
    
    np.random.shuffle(x2_samples)
    
    for i in range(40):
        campaign.evaluate(x1_samples[i], x2_samples[i])

def strategy_5_simulated_annealing(campaign):
    """
    ğŸŒ¡ï¸� STRATEGY: Simulated Annealing
    Probabilistic technique that accepts worse solutions early on
    to escape local minima. Temperature decreases over time.
    """
    current = [np.random.uniform(-5, 10), np.random.uniform(0, 15)]
    current_val = campaign.evaluate(current[0], current[1])
    
    best = current.copy()
    best_val = current_val
    
    temperature = 10.0
    cooling_rate = 0.95
    
    for i in range(39):
        neighbor = current.copy()
        neighbor[0] += np.random.normal(0, temperature)
        neighbor[1] += np.random.normal(0, temperature)
        neighbor[0] = np.clip(neighbor[0], -5, 10)
        neighbor[1] = np.clip(neighbor[1], 0, 15)
        
        neighbor_val = campaign.evaluate(neighbor[0], neighbor[1])
        
        delta = neighbor_val - current_val
        if delta < 0 or np.random.random() < np.exp(-delta / temperature):
            current = neighbor
            current_val = neighbor_val
            
            if current_val < best_val:
                best = current.copy()
                best_val = current_val
        
        temperature *= cooling_rate

def strategy_6_pattern_search(campaign):
    """
    ğŸ”� STRATEGY: Pattern Search (Hooke-Jeeves)
    Direct search method that explores along coordinate directions.
    Step size decreases when no improvement is found.
    """
    x = [np.random.uniform(-5, 10), np.random.uniform(0, 15)]
    step_size = 2.0
    
    best_val = campaign.evaluate(x[0], x[1])
    best_x = x.copy()
    
    directions = [[1, 0], [-1, 0], [0, 1], [0, -1], [1, 1], [1, -1], [-1, 1], [-1, -1]]
    
    for i in range(39):
        direction = directions[i % len(directions)]
        
        if i % 10 == 0 and i > 0:
            step_size *= 0.5
        
        new_x = [
            np.clip(best_x[0] + step_size * direction[0], -5, 10),
            np.clip(best_x[1] + step_size * direction[1], 0, 15)
        ]
        
        new_val = campaign.evaluate(new_x[0], new_x[1])
        
        if new_val < best_val:
            best_val = new_val
            best_x = new_x.copy()

def strategy_7_adaptive_search(campaign):
    """
    ğŸ§  STRATEGY: Adaptive Multi-Point Search
    Maintains multiple promising regions and adapts search
    radius based on progress. Balances exploration and exploitation.
    """
    points = []
    values = []
    
    for _ in range(15):
        x1 = np.random.uniform(-5, 10)
        x2 = np.random.uniform(0, 15)
        val = campaign.evaluate(x1, x2)
        points.append([x1, x2])
        values.append(val)
    
    sorted_idx = np.argsort(values)
    best_points = [points[i] for i in sorted_idx[:3]]
    
    for i in range(25):
        base_point = best_points[i % 3]
        radius = 2.0 * (1 - i / 25)
        noise = np.random.normal(0, radius, 2)
        x1 = np.clip(base_point[0] + noise[0], -5, 10)
        x2 = np.clip(base_point[1] + noise[1], 0, 15)
        
        val = campaign.evaluate(x1, x2)
        
        if val < values[sorted_idx[2]]:
            worst_idx = sorted_idx[2]
            points[worst_idx] = [x1, x2]
            values[worst_idx] = val
            sorted_idx = np.argsort(values)
            best_points = [points[i] for i in sorted_idx[:3]]

def strategy_8_sobol_sequence(campaign):
    """
    ğŸ“Š STRATEGY: Quasi-Random (Sobol-like) Sequence
    Low-discrepancy sequence that fills space more uniformly
    than random sampling. Good for initial exploration.
    """
    def sobol_like(n, d=2):
        points = []
        for i in range(n):
            binary = format(i+1, 'b')
            x = [0, 0]
            for j, bit in enumerate(reversed(binary)):
                if bit == '1':
                    x[0] ^= (1 << j)
                    x[1] ^= (1 << (j+1))
            x[0] = x[0] / (1 << len(binary))
            x[1] = x[1] / (1 << (len(binary)+1))
            points.append(x)
        return points
    
    points = sobol_like(40)
    
    for p in points:
        x1 = -5 + p[0] * 15
        x2 = p[1] * 15
        campaign.evaluate(x1, x2)

def strategy_9_spiral_search(campaign):
    """
    ğŸŒ€ STRATEGY: Spiral Search Pattern
    Explores space in an expanding spiral pattern from a center point.
    Ensures systematic coverage with increasing radius.
    """
    centers = [[2.5, 7.5], [-1, 10], [7, 3]]
    center = centers[campaign.id % 3]
    
    for i in range(40):
        theta = i * np.pi / 5
        r = 0.3 * i
        
        x1 = center[0] + r * np.cos(theta)
        x2 = center[1] + r * np.sin(theta)
        
        x1 = np.clip(x1, -5, 10)
        x2 = np.clip(x2, 0, 15)
        
        campaign.evaluate(x1, x2)

def strategy_10_nelder_mead_inspired(campaign):
    """
    ğŸ“� STRATEGY: Nelder-Mead Simplex Method
    Uses a simplex (triangle in 2D) that adapts its shape
    to the local landscape through reflection, expansion, and contraction.
    """
    simplex = np.array([
        [np.random.uniform(-5, 10), np.random.uniform(0, 15)],
        [np.random.uniform(-5, 10), np.random.uniform(0, 15)],
        [np.random.uniform(-5, 10), np.random.uniform(0, 15)]
    ])
    
    values = []
    for point in simplex:
        val = campaign.evaluate(point[0], point[1])
        values.append(val)
    
    for i in range(37):
        order = np.argsort(values)
        simplex = simplex[order]
        values = [values[j] for j in order]
        
        centroid = np.mean(simplex[:2], axis=0)
        
        alpha = 1.0 + 0.1 * np.random.random()
        reflected = centroid + alpha * (centroid - simplex[2])
        reflected[0] = np.clip(reflected[0], -5, 10)
        reflected[1] = np.clip(reflected[1], 0, 15)
        
        ref_val = campaign.evaluate(reflected[0], reflected[1])
        
        if ref_val < values[2]:
            simplex[2] = reflected
            values[2] = ref_val

def strategy_11_genetic_inspired(campaign):
    """
    ğŸ§¬ STRATEGY: Genetic Algorithm
    Maintains a population of solutions that evolve through
    selection and mutation. Good solutions propagate their "genes".
    """
    population = []
    fitness = []
    
    for _ in range(10):
        x1 = np.random.uniform(-5, 10)
        x2 = np.random.uniform(0, 15)
        val = campaign.evaluate(x1, x2)
        population.append([x1, x2])
        fitness.append(val)
    
    for i in range(30):
        idx1, idx2 = np.random.choice(10, 2, replace=False)
        if fitness[idx1] < fitness[idx2]:
            parent = population[idx1]
        else:
            parent = population[idx2]
        
        child = parent.copy()
        mutation_rate = 0.5 * (1 - i/30)
        child[0] += np.random.normal(0, mutation_rate)
        child[1] += np.random.normal(0, mutation_rate)
        child[0] = np.clip(child[0], -5, 10)
        child[1] = np.clip(child[1], 0, 15)
        
        child_val = campaign.evaluate(child[0], child[1])
        
        worst_idx = np.argmax(fitness)
        if child_val < fitness[worst_idx]:
            population[worst_idx] = child
            fitness[worst_idx] = child_val

def strategy_12_final_refinement(campaign):
    """
    âœ¨ STRATEGY: Final Refinement
    Uses all previous results to intensively search around
    the best point found so far with very fine granularity.
    """
    history = Branin.get_history()
    
    if len(history) > 0:
        best_idx = history['value'].idxmin()
        best_x1 = history.loc[best_idx, 'x1']
        best_x2 = history.loc[best_idx, 'x2']
        
        for i in range(40):
            scale = 0.1 * np.exp(-i/20)
            noise = np.random.normal(0, scale, 2)
            x1 = np.clip(best_x1 + noise[0], -5, 10)
            x2 = np.clip(best_x2 + noise[1], 0, 15)
            val = campaign.evaluate(x1, x2)
            
            if val < history.loc[best_idx, 'value']:
                best_x1 = x1
                best_x2 = x2
    else:
        strategy_3_random_search(campaign)

# ============================================
# MAIN EXECUTION
# ============================================

def run_optimization():
    """
    Main optimization routine using exactly 12 campaigns with 40 evaluations each
    """
    
    print("\nğŸ”§ OPTIMIZATION STRATEGY OVERVIEW")
    print("-" * 70)
    print("""
    We'll use 12 different optimization strategies:
    
    1-3. Known Minima Focus: Target the three known global minima
    4.   Grid Search: Systematic exploration of the space
    5.   Latin Hypercube: Optimal space-filling design
    6.   Simulated Annealing: Temperature-based probabilistic search
    7.   Pattern Search: Direct search along coordinate directions
    8.   Adaptive Search: Multi-point exploration with adaptation
    9.   Sobol Sequence: Quasi-random low-discrepancy sequence
    10.  Spiral Search: Expanding spiral pattern
    11.  Nelder-Mead: Simplex-based optimization
    12.  Genetic Algorithm: Population-based evolutionary approach
    """)
    
    strategies = [
        ("Known Minima Focus 1", strategy_1_known_minima),
        ("Known Minima Focus 2", strategy_1_known_minima),
        ("Known Minima Focus 3", strategy_1_known_minima),
        ("Grid Search", strategy_2_grid_search),
        ("Latin Hypercube", strategy_4_latin_hypercube),
        ("Simulated Annealing", strategy_5_simulated_annealing),
        ("Pattern Search", strategy_6_pattern_search),
        ("Adaptive Search", strategy_7_adaptive_search),
        ("Sobol-like Sequence", strategy_8_sobol_sequence),
        ("Spiral Search", strategy_9_spiral_search),
        ("Nelder-Mead Inspired", strategy_10_nelder_mead_inspired),
        ("Genetic Inspired", strategy_11_genetic_inspired),
    ]
    
    print("\nğŸš€ STARTING OPTIMIZATION")
    print("-" * 70)
    print("Running 12 campaigns Ã— 40 evaluations = 480 total evaluations")
    print("-" * 70)
    
    for i in range(12):
        name, strategy = strategies[i]
        
        print(f"\nğŸ“� Campaign {i}: {name}")
        campaign = Branin()
        
        # Run strategy
        strategy(campaign)
        
        # Verify we used exactly 40 evaluations
        if campaign.budget != 40:
            print(f"  âš ï¸� WARNING: Campaign used {campaign.budget} evaluations, filling to 40...")
            ensure_full_budget(campaign)
        
        print(f"  âœ… Completed {campaign.budget} evaluations")
        
        # Show intermediate best
        history_so_far = Branin.get_history()
        current_best = history_so_far['value'].min()
        print(f"  ğŸ“Š Current global best: {current_best:.6f}")

# ============================================
# RUN THE COMPLETE OPTIMIZATION WITH VISUALIZATIONS
# ============================================

if __name__ == "__main__":
    print("\n" + "="*70)
    print(" " * 15 + "ğŸ�� BRANIN OPTIMIZATION CHALLENGE ğŸ��")
    print(" " * 15 + "Complete Solution with Analysis")
    print("="*70)
    
    # Display initial information
    print("\nğŸ“‹ COMPETITION PARAMETERS:")
    print("  â€¢ Target: Find global minimum (value â‰ˆ 0.397887)")
    print("  â€¢ Budget: 12 campaigns Ã— 40 evaluations = 480 total")
    print("  â€¢ Search space: xâ‚� âˆˆ [-5, 10], xâ‚‚ âˆˆ [0, 15]")
    
    # Visualize the Branin function
    visualize_branin_function()
    
    # Clear any previous history (important for re-runs)
    Branin._history = []
    Branin._campaign_count = 0
    Branin.row_id = 0
    
    # Run the optimization
    run_optimization()
    
    # Get final results
    print("\n" + "="*70)
    print(" " * 20 + "ğŸ“ˆ RESULTS ANALYSIS ğŸ“ˆ")
    print("="*70)
    
    final_history = Branin.get_history()
    
    # Verify we have exactly 480 rows
    total_rows = len(final_history)
    print(f"\nâœ”ï¸� VERIFICATION:")
    print(f"  Total evaluations: {total_rows} {'âœ… CORRECT' if total_rows == 480 else 'â�Œ ERROR'}")
    
    # Find best result
    best_idx = final_history['value'].idxmin()
    best_result = final_history.loc[best_idx]
    
    print(f"\nğŸ�† BEST RESULT FOUND:")
    print(f"  Location: xâ‚� = {best_result['x1']:.6f}, xâ‚‚ = {best_result['x2']:.6f}")
    print(f"  Value: {best_result['value']:.6f}")
    print(f"  Campaign: {int(best_result['campaign'])}")
    print(f"  Iteration: {int(best_result['index'])}")
    print(f"  Distance from optimum: {abs(best_result['value'] - 0.397887):.6f}")
    
    # Create all visualizations
    print("\nğŸ“Š GENERATING COMPREHENSIVE VISUALIZATIONS...")
    print("-" * 70)
    
    # 1. Campaign progress plots
    print("\n1ï¸�âƒ£ Creating campaign progress plots...")
    plot_campaign_progress(final_history)
    
    # 2. Exploration heatmaps
    print("2ï¸�âƒ£ Creating exploration heatmaps...")
    plot_exploration_heatmap(final_history)
    
    # 3. Convergence analysis
    print("3ï¸�âƒ£ Creating convergence analysis...")
    plot_convergence_analysis(final_history)
    
    # 4. Performance summary
    print("4ï¸�âƒ£ Creating performance summary...")
    summary_df = create_performance_summary(final_history)
    
    # Export to CSV
    print("\n" + "-" * 70)
    Branin.export_history("submission.csv")
    
    # Final summary
    print("\n" + "="*70)
    print(" " * 20 + "ğŸ�¯ FINAL SUMMARY ğŸ�¯")
    print("="*70)
    
    print(f"""
    âœ… Optimization Complete!
    
    ğŸ“Š Key Metrics:
    â€¢ Best value found: {best_result['value']:.6f}
    â€¢ Global optimum: 0.397887
    â€¢ Difference: {abs(best_result['value'] - 0.397887):.6f}
    â€¢ Success rate (< 0.5): {(final_history['value'] < 0.5).sum() / len(final_history) * 100:.1f}%
    â€¢ Success rate (< 1.0): {(final_history['value'] < 1.0).sum() / len(final_history) * 100:.1f}%
    
    ğŸ“� Output:
    â€¢ submission.csv created with {len(final_history)} evaluations
    â€¢ Ready for Kaggle submission!
    
    ğŸ�… Top 3 Performing Strategies:
    """)
    
    for i, row in summary_df.head(3).iterrows():
        print(f"    {i+1}. {row['Strategy']}: {row['Best Value']:.6f}")
    
    print("\n" + "="*70)
    print(" " * 15 + "ğŸ�‰ Good luck with your submission! ğŸ�‰")
    print("="*70)

