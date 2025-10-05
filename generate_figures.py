import numpy as np
import matplotlib.pyplot as plt
from matplotlib import cm
import graphviz
from datetime import datetime
import shutil

# Set random seed for reproducibility
np.random.seed(42)

# ================================================================
# Figure 1: Subsidy Gradient (subsidy_gradient.png)
# ================================================================

def plot_subsidy_gradient():
    """
    Generate Figure 1: Subsidy Gradient σ(t) from 1600 to 2025.
    Corresponds to Section 2.5 — transition from patronage to enclosure.
    """
    years = np.linspace(1600, 2025, 100)

    # Subsidy gradient: σ(t) = (P - C)/(P + C)
    P_subsidy = np.exp(-0.002 * (years - 1600)) * 100   # Institutional subsidy declines
    C_subsidy = np.where(years < 2010, 50,
                         50 + 0.1 * (years - 2010) ** 2)  # User costs rise post-2010
    sigma = (P_subsidy - C_subsidy) / (P_subsidy + C_subsidy + 1e-6)

    plt.figure(figsize=(8, 4))
    plt.plot(years, sigma, label=r'$\sigma(t)$', color='blue')
    plt.axhline(0, color='black', linestyle='--', alpha=0.5)
    plt.axvline(2010, color='red', linestyle='--', label='Inversion (~2010)')
    plt.xlabel('Year')
    plt.ylabel(r'Subsidy Gradient $\sigma(t)$')
    plt.title('Figure 1: Subsidy Gradient (1600–2025)')
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.savefig('subsidy_gradient.png', dpi=300, bbox_inches='tight')
    plt.close()
    print("✓ subsidy_gradient.png created.")


# ================================================================
# Figure 2: RSVP Phase Portraits (rsvp_free.png, rsvp_attractor.png)
# ================================================================
def simulate_rsvp(Lx=1.0, Ly=1.0, Nx=60, Ny=60, T_max=0.5, dt=5e-5, h=0.02):
    """
    Simulate RSVP dynamics for free (distributed) and attractor regimes.
    Modified for numerical stability, bounded fields, and visible patterns.
    """
    x = np.linspace(0, Lx, Nx)
    y = np.linspace(0, Ly, Ny)
    X, Y = np.meshgrid(x, y)

    # Parameters (tuned for stability)
    lambda_PhiS = 0.02  # Reduced coupling
    eta_vS = 0.01       # Reduced velocity coupling
    alpha = 0.2         # Reduced diffusion
    beta = 0.1          # Reduced nonlinear term
    gamma = 0.01        # Reduced damping
    kappa = 0.2         # Reduced attractor strength
    mu = 0.02
    nu = 0.01

    # Initial fields with structured perturbation
    Phi = 0.1 * np.random.randn(Nx, Ny) + 0.5 * np.sin(2 * np.pi * X) * np.cos(2 * np.pi * Y)
    vx = 0.05 * np.random.randn(Nx, Ny) + 0.1 * np.sin(2 * np.pi * X)
    vy = 0.05 * np.random.randn(Nx, Ny) + 0.1 * np.cos(2 * np.pi * Y)
    S = 1.0 + 0.1 * np.random.randn(Nx, Ny) + 0.2 * np.sin(2 * np.pi * Y)

    # Centered Gaussian attractor
    xc, yc = Lx / 2, Ly / 2
    A, w = 1.0, 0.1  # Reduced amplitude, wider spread
    sink = A * np.exp(-((X - xc)**2 + (Y - yc)**2) / (2 * w**2))

    def div_v(vx, vy):
        return (np.roll(vx, -1, 0) - np.roll(vx, 1, 0) +
                np.roll(vy, -1, 1) - np.roll(vy, 1, 1)) / (2 * h)

    def laplacian(F):
        return (np.roll(F, -1, 0) + np.roll(F, 1, 0) +
                np.roll(F, -1, 1) + np.roll(F, 1, 1) - 4 * F) / h**2

    def grad(F):
        gx = (np.roll(F, -1, 0) - np.roll(F, 1, 0)) / (2 * h)
        gy = (np.roll(F, -1, 1) - np.roll(F, 1, 1)) / (2 * h)
        return gx, gy

    # Value clipping to prevent overflow
    def clip_fields(Phi, vx, vy, S, max_val=1e3):
        return (np.clip(Phi, -max_val, max_val),
                np.clip(vx, -max_val, max_val),
                np.clip(vy, -max_val, max_val),
                np.clip(S, -max_val, max_val))

    # ---------------- Free Regime ----------------
    for i in np.arange(0, T_max, dt):
        div_v_val = div_v(vx, vy)
        grad_Phi_x, grad_Phi_y = grad(Phi)
        grad_S_x, grad_S_y = grad(S)
        lap_S = laplacian(S)
        grad_vx_x, grad_vx_y = grad(vx)
        grad_vy_x, grad_vy_y = grad(vy)

        forcing = 0.1 * np.sin(6 * np.pi * X) * np.cos(6 * np.pi * Y)

        dPhi_dt = -Phi * div_v_val - vx * grad_Phi_x - vy * grad_Phi_y - lambda_PhiS * S + forcing
        dvx_dt = -(vx * grad_vx_x + vy * grad_vx_y) - grad_Phi_x + eta_vS * grad_S_x
        dvy_dt = -(vx * grad_vy_x + vy * grad_vy_y) - grad_Phi_y + eta_vS * grad_S_y
        dS_dt = alpha * lap_S + beta * div_v_val**2 - gamma * Phi + mu * forcing

        Phi += dt * dPhi_dt
        vx += dt * dvx_dt
        vy += dt * dvy_dt
        S += dt * dS_dt

        # Clip fields to prevent overflow
        Phi, vx, vy, S = clip_fields(Phi, vx, vy, S)

        # Diagnostic check
        if np.any(np.isnan(Phi)) or np.any(np.isinf(Phi)):
            print(f"NaN/Inf detected in Phi at step {i:.6f} (free regime)")
            break

        # Periodic diagnostic output
        if i % 0.1 < dt:
            print(f"Free regime step {i:.3f} - Phi min: {Phi.min():.4f}, max: {Phi.max():.4f}")

    # Print final ranges
    print(f"Free regime - Phi min: {Phi.min():.4f}, max: {Phi.max():.4f}")
    print(f"Free regime - vx min: {vx.min():.4f}, max: {vx.max():.4f}")
    print(f"Free regime - vy min: {vy.min():.4f}, max: {vy.max():.4f}")

    # Normalize for plotting
    Phi_plot = (Phi - Phi.min()) / (Phi.max() - Phi.min() + 1e-9)
    if Phi.max() - Phi.min() < 1e-6:
        print("Warning: Phi range too small for normalization, using raw Phi")
        Phi_plot = Phi

    plt.figure(figsize=(6, 5))
    plt.contourf(X, Y, Phi_plot, levels=30, cmap=cm.viridis)
    plt.quiver(X[::2, ::2], Y[::2, ::2], vx[::2, ::2], vy[::2, ::2],
               color='white', scale=10)  # Adjusted scale
    plt.colorbar(label=r'Normalized $\Phi$')
    plt.title('Figure 2a: Free-Market Baseline')
    plt.xlabel('x')
    plt.ylabel('y')
    plt.savefig('rsvp_free.png', dpi=300, bbox_inches='tight')
    plt.close()

    # ---------------- Attractor Regime ----------------
    for i in np.arange(0, T_max, dt):
        div_v_val = div_v(vx, vy)
        grad_Phi_x, grad_Phi_y = grad(Phi)
        grad_S_x, grad_S_y = grad(S)
        lap_S = laplacian(S)
        grad_vx_x, grad_vx_y = grad(vx)
        grad_vy_x, grad_vy_y = grad(vy)

        forcing = 0.1 * np.sin(6 * np.pi * X) * np.cos(6 * np.pi * Y)

        dPhi_dt = -Phi * div_v_val - vx * grad_Phi_x - vy * grad_Phi_y - lambda_PhiS * S + forcing
        dvx_dt = -(vx * grad_vx_x + vy * grad_vx_y) - grad_Phi_x + eta_vS * grad_S_x
        dvy_dt = -(vx * grad_vy_x + vy * grad_vy_y) - grad_Phi_y + eta_vS * grad_S_y
        dS_dt = alpha * lap_S + beta * div_v_val**2 - gamma * Phi + mu * forcing

        Phi += dt * dPhi_dt
        vx += dt * dvx_dt - kappa * sink * (X - xc) / (w**2)
        vy += dt * dvy_dt - kappa * sink * (Y - yc) / (w**2)
        S += dt * dS_dt

        # Clip fields
        Phi, vx, vy, S = clip_fields(Phi, vx, vy, S)

        # Diagnostic check
        if np.any(np.isnan(Phi)) or np.any(np.isinf(Phi)):
            print(f"NaN/Inf detected in Phi at step {i:.6f} (attractor regime)")
            break

        # Periodic diagnostic output
        if i % 0.1 < dt:
            print(f"Attractor regime step {i:.3f} - Phi min: {Phi.min():.4f}, max: {Phi.max():.4f}")

    # Print final ranges
    print(f"Attractor regime - Phi min: {Phi.min():.4f}, max: {Phi.max():.4f}")
    print(f"Attractor regime - vx min: {vx.min():.4f}, max: {vx.max():.4f}")
    print(f"Attractor regime - vy min: {vy.min():.4f}, max: {vy.max():.4f}")

    # Normalize for plotting
    Phi_plot = (Phi - Phi.min()) / (Phi.max() - Phi.min() + 1e-9)
    if Phi.max() - Phi.min() < 1e-6:
        print("Warning: Phi range too small for normalization, using raw Phi")
        Phi_plot = Phi

    plt.figure(figsize=(6, 5))
    plt.contourf(X, Y, Phi_plot, levels=30, cmap=cm.viridis)
    plt.quiver(X[::2, ::2], Y[::2, ::2], vx[::2, ::2], vy[::2, ::2],
               color='white', scale=10)
    plt.colorbar(label=r'Normalized $\Phi$')
    plt.title('Figure 2b: Platform Attractor')
    plt.xlabel('x')
    plt.ylabel('y')
    plt.savefig('rsvp_attractor.png', dpi=300, bbox_inches='tight')
    plt.close()

    print("✓ rsvp_free.png created.")
    print("✓ rsvp_attractor.png created.")

# ================================================================
# Figure 3: Compression Theft Flowchart (compression_theft.png)
# ================================================================

def plot_compression_theft():
    """
    Generate Figure 3: Flowchart of compression theft.
    Guarantees PNG output even if Graphviz cleanup removes .gv.
    """
    dot = graphviz.Digraph('compression_theft', format='png')
    dot.attr(rankdir='TB', size='6,6')

    # Nodes
    dot.node('A', 'User Creates\nNovel Ontology\n(ΔK > 0)', shape='box',
             style='filled', fillcolor='lightblue')
    dot.node('B', 'Platform Detects\nLow Engagement', shape='box')
    dot.node('C', 'Algorithmic\nSuppression', shape='box',
             style='filled', fillcolor='lightcoral')
    dot.node('D', 'Platform Adopts\nOntology (χ > 0)', shape='box')
    dot.node('E', 'User Receives\nNo Reward', shape='box',
             style='filled', fillcolor='lightyellow')
    dot.node('F', 'Compression Commons\n(Micropayments for ΔK)', shape='box',
             style='filled', fillcolor='lightgreen')

    # Edges
    dot.edge('A', 'B', label='Low Tokenized Reward')
    dot.edge('B', 'C', label='Visibility Reduced')
    dot.edge('C', 'D', label='Platform Capture')
    dot.edge('D', 'E', label='No Compensation')
    dot.edge('E', 'F', label='Proposed Solution', style='dashed')

    out_path = dot.render('compression_theft', cleanup=True)
    if not out_path.endswith('.png'):
        shutil.move(out_path, 'compression_theft.png')
        out_path = 'compression_theft.png'
    print(f"✓ {out_path} created.")


# ================================================================
# Main Execution
# ================================================================

def main():
    print("Generating figures...\n")
    plot_subsidy_gradient()
    simulate_rsvp()        # diagnostic-scale run
    plot_compression_theft()
    print(f"\nAll figures generated at {datetime.now()}:\n"
          "  - subsidy_gradient.png\n"
          "  - rsvp_free.png\n"
          "  - rsvp_attractor.png\n"
          "  - compression_theft.png\n")


if __name__ == '__main__':
    main()
