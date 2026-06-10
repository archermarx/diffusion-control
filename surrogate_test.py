import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as animation
from IPython.display import HTML, display

from surrogate import Surrogate
from plotting import plot_1d_on_axis

# -------------------------------------------------
# Progression Testing Functions (Interactive HTML)
# -------------------------------------------------

def run_progression_test(
    name,
    ground_truth,
    bounds,
    control_points,
    min_points=3,
    extension=0.0,
):
    """
    Train a surrogate one control point at a time and generate an interactive JS HTML animation.
    """
    print(f"Simulating progression for {name}...")

    fig, ax = plt.subplots(figsize=(8, 6))
    fig.subplots_adjust(bottom=0.25)
    ax_ei = ax.twinx()

    def update(frame):
        ax.clear()
        ax_ei.clear()
        
        # Build a fresh surrogate up to the current frame to keep rendering stateless
        frame_surrogate = Surrogate(
            dim=1,
            bounds=[bounds],
            min_points=min_points,
            optimize_restarts=20,
            acquisition="mean",
            xi=0.0,
            seed=1,
        )

        for i in range(frame + 1):
            control = control_points[i]
            metric = ground_truth(control)
            frame_surrogate.update([control], metric)

        if frame_surrogate.is_trained:
            plot_1d_on_axis(
                frame_surrogate,
                ax=ax,
                ax_ei=ax_ei,
                ground_truth=ground_truth,
                xlabel="Control c",
                ylabel="Function value z",
                title=f"{name}: {frame + 1} Points Added",
                extension=extension,
            )
        else:
            ax.set_title(f"Initializing {name}... ({frame + 1}/{min_points} points)")
            ax.set_xlim(bounds[0], bounds[1])

    # Generate the animation
    anim = animation.FuncAnimation(
        fig, 
        update, 
        frames=len(control_points), 
        interval=500, 
        blit=False
    )
    
    # Close the figure so the static plot doesn't render inline in the notebook
    plt.close(fig)

    print(f"Animation ready for {name}.")
    return HTML(anim.to_jshtml())


def run_ei_progression_test(
    name,
    ground_truth,
    bounds,
    initial_points,
    ei_iterations,
    min_points=3,
    extension=0.0,
    optimize_restarts=20,
    xi=0.0,
    seed=1,
):
    """
    Start with a small initial design, let Expected Improvement choose
    each subsequent point, and generate an interactive JS HTML animation.
    """
    if len(initial_points) < min_points:
        raise ValueError("The number of initial points must be at least min_points.")

    print(f"Simulating EI optimization for {name}...")

    # Pass 1: Run the full EI logic and record the history
    surrogate = Surrogate(
        dim=1, bounds=[bounds], min_points=min_points,
        optimize_restarts=optimize_restarts, acquisition="ei", xi=xi, seed=seed,
    )

    for control in initial_points:
        surrogate.update([control], ground_truth(control))

    recorded_points = list(initial_points)
    recorded_c_next = []
    recorded_ei_values = []

    for stage in range(ei_iterations):
        c_next, _ = surrogate.optimize(acquisition="ei")
        ei_value = surrogate.expected_improvement(c_next)

        recorded_c_next.append(c_next)
        recorded_ei_values.append(ei_value)

        z_actual = ground_truth(c_next[0])
        surrogate.update(c_next, z_actual)
        recorded_points.append(float(c_next[0]))

    # Dummy values for the final visual frame where no next point is predicted
    recorded_c_next.append(None)
    recorded_ei_values.append(None)

    # Pass 2: Animate the recorded history
    fig, ax = plt.subplots(figsize=(8, 6))
    fig.subplots_adjust(bottom=0.25)
    ax_ei = ax.twinx()

    def update(frame):
        ax.clear()
        ax_ei.clear()

        # Rebuild surrogate up to the current frame's known history
        frame_surrogate = Surrogate(
            dim=1, bounds=[bounds], min_points=min_points,
            optimize_restarts=optimize_restarts, acquisition="ei", xi=xi, seed=seed,
        )

        points_to_add = recorded_points[:len(initial_points) + frame]
        for pt in points_to_add:
            frame_surrogate.update([pt], ground_truth(pt))

        c_next = recorded_c_next[frame]
        ei_value = recorded_ei_values[frame]

        plot_1d_on_axis(
            frame_surrogate,
            ax=ax,
            ax_ei=ax_ei,
            ground_truth=ground_truth,
            xlabel="Control c",
            ylabel="Function value z",
            title=f"{name} (EI): Stage {frame} ({len(frame_surrogate.Y)} points)",
            extension=extension,
            ei_point=c_next,
        )

        if c_next is not None:
            ax.text(
                0.03, 0.97,
                f"Next c = {c_next[0]:.4f}\nEI = {ei_value:.3e}",
                transform=ax.transAxes,
                verticalalignment="top",
                bbox={"boxstyle": "round", "alpha": 0.75, "facecolor": "white"},
            )

    anim = animation.FuncAnimation(
        fig, 
        update, 
        frames=ei_iterations + 1, 
        interval=1000, 
        blit=False
    )
    
    plt.close(fig)

    print(f"Animation ready for {name}.")
    return HTML(anim.to_jshtml())


def generate_control_points(bounds, num_points, seed=1, shuffle=True):
    lower, upper = bounds
    points = np.linspace(lower, upper, num_points)
    if shuffle:
        rng = np.random.default_rng(seed)
        points = rng.permutation(points)
    return points.tolist()

# -------------------------------------------------
# Optimization benchmark functions
# -------------------------------------------------
def quartic(c):
    return ((c - 1.05) ** 2) * ((c + 1.05) ** 2)

def ackley(c):
    a, b = 20.0, 0.2
    return -a * np.exp(-b * np.abs(c)) - np.exp(np.cos(2.0 * np.pi * c)) + a + np.e

def rastrigin(c):
    return c**2 - 10.0 * np.cos(2.0 * np.pi * c) + 10.0

def forrester(c):
    return ((6.0 * c - 2.0) ** 2) * np.sin(12.0 * c - 4.0)

# -------------------------------------------------
# Execution Block
# -------------------------------------------------
if __name__ == "__main__":
    # If you run this in a standard Jupyter Notebook cell, it will render all the HTML widgets inline.
    try:
        test_cases = [
            {
                "name": "Quartic double well",
                "function": quartic,
                "bounds": (-1.5, 1.5),
                "control_points": generate_control_points((-1.5, 1.5), 15, seed=1),
                "extension": 0.1,
            },
        ]

        ei_test_cases = [
            {
                "name": "Quartic double well",
                "function": quartic,
                "bounds": (-1.5, 1.5),
                "initial_points": [-1.5, -0.5, 0.5, 1.5],
                "ei_iterations": 12,
                "extension": 0.1,
                "seed": 1,
            },
        ]

        # Generate and save standard progression animation
        for test in test_cases:
            html_anim = run_progression_test(
                name=test["name"],
                ground_truth=test["function"],
                bounds=test["bounds"],
                control_points=test["control_points"],
                min_points=3,
                extension=test["extension"],
            )
            
            # NEW: Save to file instead of display()
            filename = f"{test['name'].replace(' ', '_').lower()}_progression.html"
            with open(filename, "w") as f:
                f.write(html_anim.data)
            print(f"Saved HTML animation to: {filename}")

        # Generate and save Expected Improvement animation
        for test in ei_test_cases:
            html_ei_anim = run_ei_progression_test(
                name=test["name"],
                ground_truth=test["function"],
                bounds=test["bounds"],
                initial_points=test["initial_points"],
                ei_iterations=test["ei_iterations"],
                min_points=4,
                extension=test["extension"],
                optimize_restarts=25,
                xi=0.0,
                seed=test["seed"],
            )
            
            # NEW: Save to file instead of display()
            filename = f"{test['name'].replace(' ', '_').lower()}_ei_progression.html"
            with open(filename, "w") as f:
                f.write(html_ei_anim.data)
            print(f"Saved HTML animation to: {filename}")

    except KeyboardInterrupt:
        plt.close("all")
        print("\nStopped by user.")