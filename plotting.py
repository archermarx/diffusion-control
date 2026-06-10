import numpy as np
import matplotlib.pyplot as plt

def plot_1d_on_axis(
    surrogate,
    ax,
    ground_truth=None,
    num_points=400,
    errorbar_points=15,
    confidence=2.0,
    show_band=True,
    xlabel="Control c",
    ylabel="Metric z",
    title=None,
    ground_truth_label="Ground truth",
    extension=0.0,
    ei_point=None,
    ax_ei=None,  # NEW: Pass a twin axis to plot the Expected Improvement curve
):
    """
    Draw the current one-dimensional surrogate on an existing axis.
    """

    if surrogate.dim != 1:
        raise ValueError(
            "plot_1d_on_axis() only works for a one-dimensional surrogate."
        )

    if not surrogate.is_trained:
        raise RuntimeError(
            "The surrogate must be trained before it can be plotted."
        )

    observed_c = np.vstack(surrogate.X)[:, 0]
    observed_z = np.asarray(surrogate.Y, dtype=float)

    if surrogate.bounds is not None:
        lower = min(surrogate.bounds[0][0], float(np.min(observed_c)))
        upper = max(surrogate.bounds[0][1], float(np.max(observed_c)))
    else:
        lower = float(np.min(observed_c))
        upper = float(np.max(observed_c))

    c_grid = np.linspace(
        lower - extension,
        upper + extension,
        num_points,
    )

    c_grid_scaled = surrogate._scale(c_grid.reshape(-1, 1))

    predicted_mean = surrogate.model.predict_values(c_grid_scaled).reshape(-1)
    predicted_variance = surrogate.model.predict_variances(c_grid_scaled).reshape(-1)
    predicted_variance = np.maximum(predicted_variance, 0.0)

    predicted_std = np.sqrt(predicted_variance)
    uncertainty = confidence * predicted_std

    ax.plot(
        c_grid,
        predicted_mean,
        linewidth=2,
        color="green",
        linestyle="--",
        label="GPR prediction",
    )

    if show_band:
        ax.fill_between(
            c_grid,
            predicted_mean - uncertainty,
            predicted_mean + uncertainty,
            alpha=0.2,
            color="green",
            label=f"{confidence:g} std confidence",
        )

    ax.plot(
        observed_c,
        observed_z,
        linestyle="",
        marker="s",
        markersize=8,
        color="blue",
        label="DOE (Samples)",
        zorder=4,
    )

    if ground_truth is not None:
        if callable(ground_truth):
            ground_truth_c = c_grid
            ground_truth_z = np.asarray(
                [ground_truth(c) for c in ground_truth_c],
                dtype=float,
            )
        else:
            ground_truth_c = np.asarray(ground_truth[0], dtype=float).reshape(-1)
            ground_truth_z = np.asarray(ground_truth[1], dtype=float).reshape(-1)

        ax.plot(
            ground_truth_c,
            ground_truth_z,
            linestyle="-",
            linewidth=1.5,
            color="black",
            label=ground_truth_label,
        )

    # Use the plotted grid to mark the approximate minimum
    best_index = int(np.argmin(predicted_mean))

    ax.plot(
        c_grid[best_index],
        predicted_mean[best_index],
        marker="*",
        linestyle="",
        markersize=15,
        color="orange",
        label="Predicted minimum",
        zorder=5,
    )

    if ei_point is not None:
        ei_point = surrogate._as_vector(ei_point)
        ei_predicted_z = surrogate(ei_point)

        ax.plot(
            ei_point[0],
            ei_predicted_z,
            marker="*",
            linestyle="",
            markersize=18,
            color="magenta",
            label="Intended Opt Point",
            zorder=6,
        )

    # Plot the Expected Improvement curve on the secondary axis
    if ax_ei is not None:
        # Utilize the surrogate's built in EI function so the plot matches exactly
        ei_values = np.array([surrogate.expected_improvement([c]) for c in c_grid])
        ax_ei.plot(c_grid, ei_values, color="red", label="Expected Improvement")
        ax_ei.set_ylabel("Expected Improvement (EI)")
        ax_ei.set_ylim(bottom=0.0)

    ax.set_xlim(c_grid[0], c_grid[-1])
    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)

    if title is not None:
        ax.set_title(title)

    ax.grid(True, alpha=0.3)

    # Consolidate legends cleanly below the plot
    lines, labels = ax.get_legend_handles_labels()
    if ax_ei is not None:
        lines2, labels2 = ax_ei.get_legend_handles_labels()
        lines.extend(lines2)
        labels.extend(labels2)

    ax.legend(lines, labels, loc="upper center", bbox_to_anchor=(0.5, -0.15), ncol=2)


def _fit_snapshot_model(surrogate, X_snapshot, Y_snapshot):
    """
    Helper function to fit a temporary model using historical snapshots.
    """
    from smt.surrogate_models import KRG

    X = np.vstack(X_snapshot)
    Y = np.array(Y_snapshot, dtype=float)

    X_scaled = surrogate._scale_with_reference(X, X_snapshot)
    X_unique, Y_unique = surrogate._remove_duplicate_points(X_scaled, Y)

    if len(Y_unique) < surrogate.min_points:
        return None

    model = KRG(
        theta0=surrogate.theta0,
        corr=surrogate.corr,
        print_global=False,
    )

    model.set_training_values(X_unique, Y_unique.reshape(-1, 1))
    model.train()

    return model