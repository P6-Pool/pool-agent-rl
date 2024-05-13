import csv
import matplotlib.pyplot as plt
import argparse
import os
from glob import glob
import pandas as pd
import numpy as np
from scipy.interpolate import griddata
from matplotlib import cbook, cm
from matplotlib.colors import LightSource
import re

# Setup for Latex rendering
plt.rcParams.update(
    {
        "text.usetex": True,
        "font.family": "serif",  # Use serif font
        "font.serif": ["Computer Modern Roman"],  # Use Computer Modern Roman font
    }
)


def split_agent_name(row):
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

    Example format of 1 file:

    agentName,offBreakWin,avgTimePerGame
    striker-2-10-1-20,0.0,0.27713334560394287
    striker-2-10-1-40,0.0,0.27545201778411865
    striker-2-20-1-20,50.0,0.25802910327911377
    striker-2-20-1-40,0.0,0.3649400472640991


    """

    df = pd.read_csv(log_dir, delimiter=",")
    df = df.apply(split_agent_name, axis=1)

    # Custom hillshading in a 3D surface plot

    # Set up plot
    # fig = plt.figure()
    # ax = fig.add_subplot(111, projection="3d")
    fig, ax = plt.subplots(subplot_kw={"projection": "3d"}, figsize=(10, 6))

    df["offBreakWinTime"] = df.apply(lambda row: row["offBreakWin"] / row["avgTimePerGame"], axis=1)

    print(df.head())
    # Create data
    x = df["vel_samples"]
    y = df["mc_samples"]
    z = df[z_axis]
    # Plot data
    # ax.scatter(x, y, z, c="r", marker="o")
    # ax.plot_trisurf(x, y, z, cmap="viridis", edgecolor="none")

    # xi = np.linspace(min(x), max(x), 100)
    # yi = np.linspace(min(y), max(y), 100)
    # zi = griddata((x, y), z, (xi[None, :], yi[:, None]), method="cubic")

    surf = ax.plot_trisurf(x, y, z, cmap="viridis", edgecolor="none")
    # surf = ax.plot_trisurf(x, y, z, cmap="viridis", edgecolor="none", subdivide=12)

    # Customize label
    ax.set_xlabel("Velocity Samples")
    ax.set_ylabel("Monte-Carlo Samples")

    z_label = re.sub(r"(?<!^)(?=[A-Z])", " ", z_axis).title()

    ax.set_zlabel(z_label)

    # Color fix
    ax.xaxis.set_pane_color((1.0, 1.0, 1.0, 0.0))
    ax.yaxis.set_pane_color((1.0, 1.0, 1.0, 0.0))
    ax.zaxis.set_pane_color((1.0, 1.0, 1.0, 0.0))
    ax.xaxis._axinfo["grid"]["color"] = (0, 0, 0, 0.1)
    ax.yaxis._axinfo["grid"]["color"] = (0, 0, 0, 0.1)
    ax.zaxis._axinfo["grid"]["color"] = (0, 0, 0, 0.1)

    ax.view_init(30, -45)

    cbar = fig.colorbar(surf, shrink=0.5, aspect=10)
    # move cbar more to the right
    cbar.ax.yaxis.set_ticks_position("right")
    pos = cbar.ax.get_position()
    cbar.ax.set_position([pos.x0 + 0.04, pos.y0, pos.width, pos.height])

    # Remove whitespace

    os.makedirs(save_dir, exist_ok=True)
    save_path = os.path.join(save_dir, f"plot-{z_label.replace(' ', '_').lower()}.pdf")
    print(f"Saving plot to {save_path}")
    plt.savefig(save_path, format="pdf")
    if show:
        plt.show()


def main():
    parser = argparse.ArgumentParser(description="Plot TensorBoard logs.")
    parser.add_argument("log_dir", type=str, help="Path(s) to the directory containing TensorBoard logs")
    parser.add_argument("-s", "--show", action="store_true", help="Show the plot")
    parser.add_argument("-z", "--z-axis", type=str, help="The z-axis to plot", default="offBreakWin")
    args = parser.parse_args()

    save_dir = "plots/benchmarks/"

    plot_benchmarks(args.log_dir, args.show, z_axis=args.z_axis, save_dir=save_dir)


if __name__ == "__main__":
    main()
