import numpy as np
from tensorboard.backend.event_processing.event_accumulator import EventAccumulator
import matplotlib.pyplot as plt
import argparse


def smooth(scalars: list[float] | np.ndarray, weight: float) -> list[float]:
    """
    EMA implementation according to
    https://github.com/tensorflow/tensorboard/blob/34877f15153e1a2087316b9952c931807a122aa7/tensorboard/components/vz_line_chart2/line-chart.ts#L699
    """
    last = 0
    smoothed = []
    num_acc = 0
    for next_val in scalars:
        last = last * weight + (1 - weight) * next_val
        num_acc += 1
        # de-bias
        debias_weight = 1
        if weight != 1:
            debias_weight = 1 - (weight**num_acc)
        smoothed_val = last / debias_weight
        smoothed.append(smoothed_val)

    return smoothed


def plot_tensorboard_logs(log_dirs, tags_to_plot, smooth_weight=0.5):
    """
    Plot TensorBoard logs for specified tags with optional smoothing.

    Args:
    - log_dir (str): Path to the directory containing TensorBoard logs.
    - tags_to_plot (list): List of tags to plot.
    - smooth_weight (int, optional): Window size for moving average smoothing. Default is 0.5.

    Returns:
    - None
    """

    # Determine colors for lines
    colors = plt.cm.Set2(np.linspace(0, 1, len(tags_to_plot)))  # type: ignore
    line_width = 0.6

    for log_dir in log_dirs:
        # Load TensorBoard logs
        event_acc = EventAccumulator(log_dir)
        event_acc.Reload()

        # Get all scalar events
        scalar_tags = event_acc.Tags()["scalars"]

        # Load TensorBoard logs
        event_acc = EventAccumulator(log_dir)
        event_acc.Reload()

        # Plot specified tags
        for i, tag in enumerate(tags_to_plot):
            if tag in scalar_tags:
                events = event_acc.Scalars(tag)
                steps = np.array([event.step for event in events])
                values = np.array([event.value for event in events])

                if smooth_weight > 0:
                    # Apply moving average smoothing
                    smoothed_values = smooth(values, smooth_weight)

                    # Plot smoothed data with custom color
                    plt.plot(
                        steps,
                        smoothed_values,
                        label=tag + f" ({smooth_weight} smoothing)",
                        color=colors[i],
                        linewidth=line_width,
                    )

                    # Plot original data with lower opacity using the same color
                    plt.plot(steps, values, alpha=0.3, color=colors[i], label=tag, linewidth=line_width)
                else:
                    plt.plot(steps, values, color=colors[i], label=tag, linewidth=line_width)
            else:
                print(f"Tag '{tag}' not found in TensorBoard logs.")

    plt.grid(True, alpha=0.1)
    plt.xlabel("Step")
    plt.ylabel("Value")
    plt.legend()
    plot_name = "plot-" + "-".join(tags_to_plot).replace("/", "_") + ".pdf"
    plt.savefig(plot_name)
    print(f"Plot saved as '{plot_name}'")
    plt.show()


def main():
    parser = argparse.ArgumentParser(description="Plot TensorBoard logs.")
    parser.add_argument("log_dirs", nargs="+", help="Path(s) to the directory containing TensorBoard logs")
    parser.add_argument("-t", "--tags", nargs="+", help="Scalar tags to plot", required=True)
    parser.add_argument("-s", "--smoothing", type=float, default=0, help="Window size for moving average smoothing")

    args = parser.parse_args()

    plot_tensorboard_logs(args.log_dirs, args.tags, smooth_weight=args.smoothing)


if __name__ == "__main__":
    main()
