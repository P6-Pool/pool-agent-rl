import argparse
import os
import re

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from tensorboard.backend.event_processing.event_accumulator import EventAccumulator

# LaTeX rendering
plt.rcParams.update(
    {
        "text.usetex": True,
        "font.family": "serif",
        "font.serif": ["Computer Modern Roman"],
    }
)

TB_TAGS = [
    "eval/success_rate",
    "eval/mean_reward",
    "eval/mean_ep_length",
    "rollout/ep_rew_mean",
    "rollout/ep_len_mean",
]


def get_colors(log_dirs: list[str]):
    color_map = {}

    prefixes = []
    policies = []
    for log_dir in log_dirs:
        log_dir = log_dir if log_dir.endswith("/") else log_dir + "/"
        name = log_dir.split("/")[-2]

        prefix = name.split("_", maxsplit=1)[0]
        if "__" in name:
            policy = int(name.split("__")[-1])
        else:
            policy = name.split("_", maxsplit=1)[-1].replace("-", " ")

        if prefix not in prefixes:
            prefixes.append(prefix)
        if policy not in policies:
            policies.append(policy)

    n_colors = len(prefixes)
    n_shades = len(policies)
    color_starts = np.linspace(0, 3, n_colors + 1)[:-1] * 1.3

    for i, prefix in enumerate(prefixes):
        colors = sns.cubehelix_palette(
            start=color_starts[i], rot=-0.15, dark=0.3, light=0.7, n_colors=n_shades, hue=1.8, gamma=1.2, reverse=True
        )

        color_map[prefix] = {}
        for j, policy in enumerate(policies):
            color_map[prefix][policy] = colors[j]
    return color_map


def smooth(scalars: list[float] | np.ndarray, weight: float) -> list[float]:
    """
    EMA implementation according to: https://github.com/tensorflow/tensorboard/blob/34877f15153e1a2087316b9952c931807a122aa7/tensorboard/components/vz_line_chart2/line-chart.ts#L699
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


def plot_tensorboard(log_dirs: list[str], tags: list[str], ema_weight=0.5, show=True, plot_dir=".") -> None:
    """
    Plot TensorBoard logs for specified tags with optional smoothing.

    Args:
    - log_dirs (str): Path to the directory containing TensorBoard logs.
    - tags (list): List of tags to plot.
    - ema_weight (int, optional): Window size for moving average smoothing. Default is 0.5.
    - show (bool, optional): Whether to display the plot. Default is True.

    Returns:
    - None
    """

    colors = get_colors(log_dirs)

    line_width = 0.5

    for log_dir in log_dirs:
        # Load TensorBoard logs
        event_acc = EventAccumulator(log_dir)
        event_acc.Reload()

        # Get all scalar events
        scalar_tags = event_acc.Tags()["scalars"]
        if log_dir.endswith("/"):
            log_dir = log_dir.rsplit("/", 1)[0]
        graph_name = log_dir.split("/")[-1]
        plot_name = ", ".join(tags)

        # Load TensorBoard logs
        event_acc = EventAccumulator(log_dir)
        event_acc.Reload()

        # Plot specified tags
        for i, tag in enumerate(tags):
            # get color by n-balls and policy

            log_dir = log_dir if log_dir.endswith("/") else log_dir + "/"
            name = log_dir.split("/")[-2]

            prefix = name.split("_", maxsplit=1)[0]
            if "__" in name:
                policy = int(name.split("__")[-1])
            else:
                policy = name.split("_", maxsplit=1)[-1].replace("-", " ")
            # if tag == "rollout/ep_rew_mean" and "reg" in policy:
            #     color = colors[prefix]["random-policy"]
            color = colors[prefix][policy]

            if tag not in scalar_tags:
                print(f"Tag '{tag}' not found in '{log_dir}'.")
                continue
            events = event_acc.Scalars(tag)
            steps = np.array([event.step for event in events])
            values = np.array([event.value for event in events])

            # Custom action space labels
            def action_space(n: int):
                return r"$\mathcal{A}_" + f"{n}" + r"$"

            if "__" in graph_name:
                suffix = action_space(int(graph_name.split("__")[-1]))
            else:
                suffix = graph_name.split("_", maxsplit=1)[-1].replace("-", " ")

            label = graph_name.split("_", maxsplit=1)[0] + ", " + suffix

            if ema_weight > 0:
                # Plot original data with lower opacity
                plt.plot(
                    steps,
                    values,
                    alpha=0.10,
                    color=color,
                    label=None,
                    linewidth=line_width,
                )

                values = smooth(values, ema_weight)

            plt.plot(
                steps,
                values,
                color=color,
                label=label,
                linewidth=line_width,
            )

            if tag == "rollout/ep_rew_mean":
                plt.ylim(bottom=-1)

    plt.grid(True, alpha=0.1)
    plt.xlabel("Step")
    plt.ylabel(
        tags[0]
        .split("/")[-1]
        .replace("_", " ")
        .replace("rew", "reward")
        .replace("ep", "episode")
        .replace("len", "length")
        .title()
    )
    plt.legend(fontsize="small")
    plt.subplots_adjust(top=0.95)

    plot_name = "plot-" + "-".join(tags).replace("/", "_") + ".pdf"
    plot_path = os.path.join(plot_dir, plot_name)
    os.makedirs(plot_dir, exist_ok=True)
    # plt.savefig(plot_path)

    # save tight layout
    plt.savefig(plot_path, bbox_inches="tight", pad_inches=0)

    print(f"Plot saved as '{plot_path}'")
    if show:
        plt.show()
    plt.clf()


def parse_agent_settings(row):
    parts = row["agentName"].split("-")
    row["shot_depth"] = int(parts[1])
    row["vel_samples"] = int(parts[2])
    row["mc_tree_depth"] = int(parts[3])
    row["mc_samples"] = int(parts[4])
    return row


def plot_benchmarks(log_dir, show=True, z_axis: str = "offBreakWin", save_dir=".") -> None:
    """
    Plots a 3D graph.
    With the x-axis as the shot depth, y-axis as the velocity samples, and z-axis as the win rate.
    """

    df = pd.read_csv(log_dir, delimiter=",")
    df = df.apply(parse_agent_settings, axis=1)

    fig, ax = plt.subplots(subplot_kw={"projection": "3d"}, figsize=(10, 6))
    ax.set_proj_type("ortho")

    df["offBreakWinTime"] = df.apply(lambda row: row["offBreakWin"] / row["avgTimePerGame"], axis=1)

    print(df.head())
    # Create data
    x = df["vel_samples"]
    y = df["mc_samples"]
    z = df[z_axis]
    # Plot data

    surf = ax.plot_trisurf(x, y, z, cmap="viridis", edgecolor="none")

    # Customize label
    ax.set_xlabel("Velocity Samples")
    ax.set_ylabel("Monte Carlo Samples")
    z_label = re.sub(r"(?<!^)(?=[A-Z])", " ", z_axis).title()

    ax.set_zlabel(z_label)

    ax.set_xticks(x.unique())
    ax.set_yticks(y.unique())
    ax.set_xlim(x.min(), x.max())
    ax.set_ylim(y.min(), y.max())

    # Color fix
    ax.xaxis.set_pane_color((1.0, 1.0, 1.0, 0.0))
    ax.yaxis.set_pane_color((1.0, 1.0, 1.0, 0.0))
    ax.zaxis.set_pane_color((1.0, 1.0, 1.0, 0.0))
    ax.xaxis._axinfo["grid"]["color"] = (0, 0, 0, 0.1)
    ax.yaxis._axinfo["grid"]["color"] = (0, 0, 0, 0.1)
    ax.zaxis._axinfo["grid"]["color"] = (0, 0, 0, 0.1)

    ax.view_init(30, -45)

    cbar = fig.colorbar(surf, shrink=0.5, aspect=10)

    # Move colorbar
    cbar.ax.yaxis.set_ticks_position("right")
    pos = cbar.ax.get_position()
    cbar.ax.set_position((pos.x0 + 0.04, pos.y0, pos.width, pos.height))

    os.makedirs(save_dir, exist_ok=True)
    save_path = os.path.join(save_dir, f"plot-{z_label.replace(' ', '_').lower()}.pdf")
    print(f"Saving plot to {save_path}")
    plt.savefig(save_path, format="pdf")
    if show:
        plt.show()


def main():
    parser = argparse.ArgumentParser(description="Plot TensorBoard and benchmark logs")

    subparsers = parser.add_subparsers(dest="command")
    tensorboard = subparsers.add_parser("tensorboard", help="Plot TensorBoard logs")
    benchmark = subparsers.add_parser("benchmark", help="Plot benchmark results")

    tensorboard.add_argument("logs", nargs="+", help="Path(s) to the directory containing the logs")
    tensorboard.add_argument("-s", "--show", action="store_true", help="Display the plot(s)")
    tensorboard.add_argument("-w", "--ema-weight", type=float, default=0, help="Window size for moving average smoothing")
    tensor_tags = tensorboard.add_mutually_exclusive_group(required=True)
    tensor_tags.add_argument("-a", "--all", action="store_true", help="Plot all tags in the log directory")
    tensor_tags.add_argument("-t", "--tags", nargs="+", help="Scalar tags to plot")
    tensorboard.add_argument("-o", "--save-dir", type=str, default=".", help="Directory to save the plots")

    benchmark.add_argument("log", help="Path to .csv benchmark log")
    benchmark.add_argument("-s", "--show", action="store_true", help="Display the plot(s)")
    benchmark.add_argument(
        "-z",
        "--z-axis",
        type=str,
        choices=["offBreakWin", "offBreakWinTime"],
        default="offBreakWin",
        help="Z-axis for the plot",
    )
    benchmark.add_argument("-o", "--save-dir", type=str, default=".", help="Directory to save the plots")

    args = parser.parse_args()

    match args.command:
        case "tensorboard":
            assert 0 <= args.ema_weight <= 1, "EMA weight must be between 0 and 1"

            tags = TB_TAGS if args.all else args.tags
            for tag in tags:
                plot_tensorboard(args.logs, [tag], args.ema_weight, args.show, args.save_dir)

        case "benchmark":
            plot_benchmarks(args.log, args.show, args.z_axis, args.save_dir)


if __name__ == "__main__":
    main()
