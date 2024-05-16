import numpy as np
from tensorboard.backend.event_processing.event_accumulator import EventAccumulator
import matplotlib.pyplot as plt
import argparse

# Setup for Latex rendering
plt.rcParams.update(
    {
        "text.usetex": True,
        "font.family": "serif",  # Use serif font
        "font.serif": ["Computer Modern Roman"],  # Use Computer Modern Roman font
    }
)


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


def plot_tensorboard_logs(log_dirs, tags_to_plot, smooth_weight=0.5, show=True) -> None:
    """
    Plot TensorBoard logs for specified tags with optional smoothing.

    Args:
    - log_dirs (str): Path to the directory containing TensorBoard logs.
    - tags_to_plot (list): List of tags to plot.
    - smooth_weight (int, optional): Window size for moving average smoothing. Default is 0.5.
    - show (bool, optional): Whether to display the plot. Default is True.

    Returns:
    - None
    """

    # Determine colors for lines
    # colors = plt.cm.tab10(np.linspace(0, 1, len(tags_to_plot)))  # type: ignore
    cmap = plt.get_cmap("tab10")
    colors = [cmap(int(i * 3.25 % 10)) for i in np.linspace(0, 1, len(tags_to_plot) * len(log_dirs))]

    line_width = 0.5

    for j, log_dir in enumerate(log_dirs):
        # Load TensorBoard logs
        event_acc = EventAccumulator(log_dir)
        event_acc.Reload()

        # Get all scalar events
        scalar_tags = event_acc.Tags()["scalars"]

        graph_name = log_dir.split("/")[-1]
        plot_name = ", ".join(tags_to_plot)

        # Load TensorBoard logs
        event_acc = EventAccumulator(log_dir)
        event_acc.Reload()

        # Plot specified tags
        for i, tag in enumerate(tags_to_plot):
            color = colors[j * len(tags_to_plot) + i]
            if tag in scalar_tags:
                events = event_acc.Scalars(tag)
                steps = np.array([event.step for event in events])
                values = np.array([event.value for event in events])

                # Custom action space labels
                def action_space(n: int):
                    return r"$\mathcal{A}_" + f"{n}" + r"$"

                label = graph_name.replace("_", ", ").replace("-", " ")
                if "cart" in graph_name.lower():
                    label = label.split(", ")[0] + f", {action_space(2)}"
                elif "reg" in graph_name.lower():
                    label = label.split(", ")[0] + f", {action_space(1)}"
                elif "random" in graph_name.lower():
                    label = label.split(", ")[0] + ", random policy"

                if smooth_weight > 0:
                    # Apply moving average smoothing
                    smoothed_values = smooth(values, smooth_weight)

                    # Plot smoothed data with custom color
                    plt.plot(
                        steps,
                        smoothed_values,
                        label=label + f" ({smooth_weight} EMA)",
                        color=color,
                        linewidth=line_width,
                    )

                    # Plot original data with lower opacity using the same color
                    plt.plot(steps, values, alpha=0.25, color=color, label=None, linewidth=line_width)
                else:
                    plt.plot(steps, values, color=color, label=label, linewidth=line_width)
            else:
                print(f"Tag '{tag}' not found in TensorBoard logs.")

    plt.grid(True, alpha=0.1)
    plt.xlabel("Step")
    plt.ylabel(
        tags_to_plot[0]
        .split("/")[-1]
        .replace("_", " ")
        .replace("rew", "reward")
        .replace("ep", "episode")
        .replace("len", "length")
        .title()
    )
    plt.legend(fontsize="small")

    plot_name = "plot-" + "-".join(tags_to_plot).replace("/", "_") + ".pdf"
    plt.savefig(plot_name)
    print(f"Plot saved as '{plot_name}'")
    if show:
        plt.show()
    plt.clf()


def main():
    parser = argparse.ArgumentParser(description="Plot TensorBoard logs.")
    parser.add_argument("log_dirs", nargs="+", help="Path(s) to the directory containing TensorBoard logs")
    parser.add_argument("-t", "--tags", nargs="+", help="Scalar tags to plot")
    parser.add_argument("-s", "--smoothing", type=float, default=0, help="Window size for moving average smoothing")
    parser.add_argument("-a", "--all", action="store_true", help="Plot all tags in the log directory")
    args = parser.parse_args()

    if args.all:
        tags_to_plot = [
            "eval/success_rate",
            "eval/mean_reward",
            "eval/mean_ep_length",
            "rollout/ep_rew_mean",
            "rollout/ep_len_mean",
        ]
        for tag in tags_to_plot:
            smoothing = 0
            if tag.startswith("rollout"):
                smoothing = args.smoothing or 0.5
            plot_tensorboard_logs(args.log_dirs, [tag], smooth_weight=smoothing, show=False)
    else:
        plot_tensorboard_logs(args.log_dirs, args.tags, smooth_weight=args.smoothing)


if __name__ == "__main__":
    main()
