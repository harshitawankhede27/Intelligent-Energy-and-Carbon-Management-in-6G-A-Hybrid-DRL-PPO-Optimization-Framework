import matplotlib.pyplot as plt
import numpy as np

# Set professional style (No specific font dependency to avoid errors)
plt.rcParams['font.family'] = 'sans-serif'
plt.rcParams['axes.linewidth'] = 2
plt.rcParams['xtick.major.width'] = 2
plt.rcParams['ytick.major.width'] = 2
plt.rcParams['font.size'] = 16
plt.rcParams['axes.spines.top'] = False
plt.rcParams['axes.spines.right'] = False

# Colors
COLOR_LEGACY = '#4A4A4A'  # Dark Grey
COLOR_PROPOSED = '#00A896' # Teal/Cyan
COLOR_ACCENT = '#D62728'   # Red for arrows/bad trends

# ==========================================
# CHART 1: Carbon Reduction (Clean Bar Chart)
# ==========================================
def plot_carbon_reduction():
    fig, ax = plt.subplots(figsize=(6, 5))
    
    categories = ['Legacy System', 'Proposed Framework']
    values = [100, 15.5]  # Normalized values (100% vs 15.5%)
    colors = [COLOR_LEGACY, COLOR_PROPOSED]
    
    bars = ax.bar(categories, values, color=colors, width=0.5)
    
    # Add simple percentage labels
    ax.text(0, 100, 'High Carbon', ha='center', va='bottom', fontsize=14, fontweight='bold', color=COLOR_LEGACY)
    ax.text(1, 15.5, '-84.5% Reduction', ha='center', va='bottom', fontsize=14, fontweight='bold', color=COLOR_PROPOSED)

    # Arrow
    ax.annotate('', 
                xy=(1, 16), xytext=(0, 95),
                arrowprops=dict(facecolor='black', arrowstyle='->', lw=2, linestyle='--'))
    
    ax.set_ylabel('Carbon Emissions (Normalized)', fontweight='bold')
    ax.set_title('Carbon Footprint', fontweight='bold', pad=20)
    ax.set_yticks([]) # Remove y-axis numbers for cleaner look
    
    plt.tight_layout()
    plt.savefig('Chart1_Carbon_Clean.png', dpi=300)
    print("Generated Chart 1 (Clean)")

# ==========================================
# CHART 2: QoS Reliability (Clean Line Chart)
# ==========================================
def plot_qos_reliability():
    fig, ax = plt.subplots(figsize=(6, 5))
    
    x = np.linspace(0, 100, 100) # Traffic Load 0-100%
    
    # Legacy: Fails after 50% load
    y_legacy = np.zeros_like(x)
    y_legacy[x > 50] = (x[x > 50] - 50) * 0.8
    
    # Proposed: Always stable
    y_proposed = np.zeros_like(x)
    
    ax.plot(x, y_legacy, color=COLOR_ACCENT, linestyle='--', linewidth=3, label='Legacy (Static)')
    ax.plot(x, y_proposed, color=COLOR_PROPOSED, linewidth=4, label='Proposed (AI)')
    
    ax.set_xlabel('Network Traffic Load (%)', fontweight='bold')
    ax.set_ylabel('Packet Loss', fontweight='bold')
    ax.set_title('Reliability / QoS', fontweight='bold', pad=20)
    
    # Annotate "Zero Violations"
    ax.text(50, 2, 'Zero Violations', color=COLOR_PROPOSED, fontweight='bold', fontsize=12)
    
    ax.legend(frameon=False)
    ax.set_ylim(-2, 45)
    ax.set_yticks([]) # Remove y-axis numbers
    
    plt.tight_layout()
    plt.savefig('Chart2_QoS_Clean.png', dpi=300)
    print("Generated Chart 2 (Clean)")

# ==========================================
# CHART 3: Training Convergence (Smooth Curve)
# ==========================================
def plot_convergence():
    fig, ax = plt.subplots(figsize=(6, 5))
    
    episodes = np.linspace(0, 100, 100)
    # Logarithmic learning curve
    reward = -100 * np.exp(-0.08 * episodes) 
    
    ax.plot(episodes, reward, color=COLOR_PROPOSED, linewidth=4)
    
    ax.set_xlabel('Training Episodes', fontweight='bold')
    ax.set_ylabel('Agent Reward', fontweight='bold')
    ax.set_title('AI Learning Curve', fontweight='bold', pad=20)
    
    # Arrows to show "Better"
    ax.text(80, -5, 'Optimized', ha='center', color=COLOR_PROPOSED, fontweight='bold')
    
    ax.grid(True, linestyle=':', alpha=0.3)
    ax.set_yticks([]) # Remove numbers
    
    plt.tight_layout()
    plt.savefig('Chart3_Convergence_Clean.png', dpi=300)
    print("Generated Chart 3 (Clean)")

if __name__ == "__main__":
    plot_carbon_reduction()
    plot_qos_reliability()
    plot_convergence()
    