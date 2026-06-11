import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
from IPython.display import HTML
from smt.problems import Rosenbrock
from surrogate import Surrogate
import scipy.stats as stats
from smt.surrogate_models import KRG

def run_ego_optimization(dimensions=3, initial_samples=10, max_iterations=50, seed=42):
    print(f"--- Starting EGO Optimization ({dimensions}D) ---")

    # 1. Setup SMT Problem and Dynamic Bounds
    smt_rosenbrock = Rosenbrock(ndim=dimensions)
    bounds = [(-2.0, 2.0) for _ in range(dimensions)]

    def ground_truth(x):
        return float(smt_rosenbrock(np.array([x]))[0][0])

    # 2. Generate Random Initial Samples
    rng = np.random.default_rng(seed)
    random_points = [rng.uniform(bounds[i][0], bounds[i][1], initial_samples) for i in range(dimensions)]
    initial_points = np.column_stack(random_points).tolist()

    # 3. Initialize Surrogate
    surrogate = Surrogate(dim=dimensions, bounds=bounds, min_points=5, seed=seed)

    # Load initial points
    for pt in initial_points:
        surrogate.update(pt, ground_truth(pt), run_async=False)

    recorded_points = list(initial_points)
    recorded_c_next = []
    
    # Track the best value found so far (Rosenbrock is a minimization problem)
    current_best_y = min(surrogate.Y)
    print(f"Initial best function value found: {current_best_y:.6f}")

    # 4. Optimization Loop with Dynamic Stopping Criteria
    converged_at_step = max_iterations
    
    for stage in range(max_iterations):
        # Find next point and calculate its Expected Improvement
        c_next, _ = surrogate.optimize(acquisition="ei")
        ei_value = surrogate.expected_improvement(c_next)
        
        # Calculate threshold: 1% of the current best value
        # We use abs() and a tiny floor value (1e-6) to keep things stable if best_y hits 0
        threshold = max(0.01 * abs(current_best_y), 1e-6)

        # Check for convergence
        if ei_value < threshold:
            print(f"\n[Convergence Reached] Stop criteria met at iteration {stage + 1}!")
            print(f"Expected Improvement ({ei_value:.3e}) dropped below 1% of best value ({threshold:.3e})")
            converged_at_step = stage
            break

        # If not converged, update surrogate and keep going
        recorded_c_next.append(c_next)
        z_actual = ground_truth(c_next)
        surrogate.update(c_next, z_actual, run_async=False)
        recorded_points.append(c_next)
        
        # Update our best tracked value
        current_best_y = min(surrogate.Y)
        if (stage + 1) % 5 == 0:
            print(f"Iteration {stage + 1:02d}: Current Best Z = {current_best_y:.6f} | EI = {ei_value:.3e}")

    plot_ego_diagnostics(surrogate)

    # 5. Handle Outputs Dynamically Based on Dimensions
    print("\n--- Optimization Summary ---")
    print(f"Total Iterations Run: {converged_at_step}")
    print(f"Final Best Function Value: {min(surrogate.Y):.6f}")
    
    best_index = np.argmin(surrogate.Y)
    print(f"Best Coordinates Found: {surrogate.X[best_index]}")

    # ONLY attempt to animate if we are in 2D
    if dimensions == 2:
        print("\nDimensions = 2. Compiling HTML contour animation...")
        return build_2d_animation(recorded_points, recorded_c_next, ground_truth, bounds, converged_at_step, initial_samples, seed)
    else:
        print(f"\nDimensions = {dimensions}. Visual plot skipped (reporting text data only).")
        return None

def plot_ego_diagnostics(surrogate):
    """
    Generates the 3 classical diagnostic plots from Section 5 of 
    the EGO paper to evaluate Kriging model fit and determine if 
    a log transformation is required.
    """
    if len(surrogate.X) < 5:
        print("Diagnostics require at least 5 training points to compute cross-validation.")
        return
        
    # 1. Extract internal history from your surrogate object
    X = np.array(surrogate.X)
    Y = np.array(surrogate.Y).flatten()
    n_samples = len(X)
    
    cv_predictions = []
    cv_standard_errors = []
    
    # 2. Run Leave-One-Out Cross-Validation Loop
    for i in range(n_samples):
        # Exclude the i-th point from training arrays
        X_train = np.delete(X, i, axis=0)
        Y_train = np.delete(Y, i, axis=0)
        X_test = X[i:i+1]  # Slice keeps the 2D shape intact
        
        # Create a fresh temporary Kriging model matching your dimensions
        tmp_model = KRG(print_global=False)
        tmp_model.set_training_values(X_train, Y_train)
        tmp_model.train()
        
        # Predict mean and variance at the omitted coordinate
        y_pred = tmp_model.predict_values(X_test)[0, 0]
        var_pred = tmp_model.predict_variances(X_test)[0, 0]
        
        # Protect against tiny machine underflow negative variances
        std_err = np.sqrt(max(var_pred, 1e-12))
        
        cv_predictions.append(y_pred)
        cv_standard_errors.append(std_err)
        
    cv_predictions = np.array(cv_predictions)
    cv_standard_errors = np.array(cv_standard_errors)
    
    # 3. Compute Residuals
    residuals = Y - cv_predictions
    std_residuals = residuals / cv_standard_errors
    
    # 4. Initialize the 3-panel EGO Dashboard
    fig, (ax1, ax2, ax3) = plt.subplots(1, 3, figsize=(18, 5.5))
    
    # --- Plot 1: Actual vs. Cross-Validated Predicted ---
    axis_min = min(Y.min(), cv_predictions.min())
    axis_max = max(Y.max(), cv_predictions.max())
    
    ax1.scatter(cv_predictions, Y, color='#1f77b4', edgecolors='k', alpha=0.8, s=60, zorder=3)
    ax1.plot([axis_min, axis_max], [axis_min, axis_max], 'r--', linewidth=1.5, label='Ideal 45° Line')
    ax1.set_title("Actual vs. Cross-Validated Predicted", fontsize=11, fontweight='bold')
    ax1.set_xlabel("Cross-Validated Prediction ($\hat{y}_{-i}$)")
    ax1.set_ylabel("Actual Value ($y_i$)")
    ax1.grid(True, linestyle=':', alpha=0.6)
    ax1.legend()
    
    # --- Plot 2: Standardized Residuals vs. Predicted ---
    ax2.scatter(cv_predictions, std_residuals, color='#2ca02c', edgecolors='k', alpha=0.8, s=60, zorder=3)
    ax2.axhline(0, color='black', linestyle='-', linewidth=1)
    ax2.axhline(3, color='red', linestyle=':', linewidth=1.5, label='$\pm$3 Sigma Threshold')
    ax2.axhline(-3, color='red', linestyle=':', linewidth=1.5)
    ax2.set_title("Standardized Residuals vs. Predicted", fontsize=11, fontweight='bold')
    ax2.set_xlabel("Cross-Validated Prediction ($\hat{y}_{-i}$)")
    ax2.set_ylabel("Standardized Residual ($\epsilon_i$)")
    ax2.grid(True, linestyle=':', alpha=0.6)
    ax2.legend()
    
    # --- Plot 3: Normal Q-Q Plot of Residuals ---
    (osm, osr), (slope, intercept, r) = stats.probplot(std_residuals, dist="norm", plot=None)
    ax3.scatter(osm, osr, color='#d62728', edgecolors='k', alpha=0.8, s=60, zorder=3)
    
    qq_min, qq_max = min(osm.min(), osr.min()), max(osm.max(), osr.max())
    ax3.plot([qq_min, qq_max], [qq_min, qq_max], 'r--', linewidth=1.5, label='Normal Distribution')
    ax3.set_title("Normal Q-Q Plot of Residuals", fontsize=11, fontweight='bold')
    ax3.set_xlabel("Theoretical Standard Normal Quantiles")
    ax3.set_ylabel("Ordered Standardized Residuals")
    ax3.grid(True, linestyle=':', alpha=0.6)
    ax3.legend()
    
    plt.suptitle("EGO Model Diagnostics (Jones et al., 1998)", fontsize=14, fontweight='bold', y=1.02)
    plt.tight_layout()
    plt.show()


def build_2d_animation(recorded_points, recorded_c_next, ground_truth, bounds, total_frames, initial_samples, seed):
    """Helper function to build the 2D plot only when appropriate."""
    fig, (ax_true, ax_surr) = plt.subplots(1, 2, figsize=(14, 6))
    resolution = 50
    x1 = np.linspace(bounds[0][0], bounds[0][1], resolution)
    x2 = np.linspace(bounds[1][0], bounds[1][1], resolution)
    X1, X2 = np.meshgrid(x1, x2)
    grid_points = np.c_[X1.ravel(), X2.ravel()]
    
    Z_true = np.array([ground_truth(pt) for pt in grid_points]).reshape(X1.shape)
    levels = np.linspace(Z_true.min(), Z_true.max() * 0.3, 40)

    ax_true.contourf(X1, X2, Z_true, levels=levels, cmap='viridis', alpha=0.9, extend='max')
    ax_true.set_title("Ground Truth (Rosenbrock)")
    ax_true.plot(1.0, 1.0, 'w*', markersize=15, label="Global Min")
    ax_true.legend()

    def update(frame):
        ax_surr.clear()
        frame_surrogate = Surrogate(dim=2, bounds=bounds, min_points=5, seed=seed)
        
        current_points = recorded_points[:initial_samples + frame]
        for pt in current_points:
            frame_surrogate.update(pt, ground_truth(pt), run_async=False)

        grid_scaled = frame_surrogate._scale(grid_points)
        Z_pred = frame_surrogate.model.predict_values(grid_scaled).reshape(X1.shape)

        ax_surr.contourf(X1, X2, Z_pred, levels=levels, cmap='viridis', alpha=0.9, extend='max')
        pts_array = np.array(current_points)
        ax_surr.scatter(pts_array[:, 0], pts_array[:, 1], c='red', edgecolors='white', s=50, label='History')

        if frame > 0 and frame - 1 < len(recorded_c_next):
            newest_pt = recorded_c_next[frame - 1]
            ax_surr.scatter(newest_pt[0], newest_pt[1], c='cyan', edgecolors='black', s=150, marker='*', label='New EI Point')

        ax_surr.set_title(f"Surrogate Model (Frame {frame}/{total_frames})")
        ax_surr.legend(loc="upper left")

    anim = FuncAnimation(fig, update, frames=total_frames + 1, interval=800)
    plt.close(fig)
    return HTML(anim.to_jshtml())


if __name__ == "__main__":
    # Test 3D (Will output text only, no crash!)
    html_out = run_ego_optimization(dimensions=5, initial_samples=12, max_iterations=80)
    
    # If you switch dimensions back to 2, this will save the file:
    if html_out is not None:
        with open("dynamic_rosenbrock_2d.html", "w") as f:
            f.write(html_out.data)