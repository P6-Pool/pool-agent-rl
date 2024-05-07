import csv
import matplotlib.pyplot as plt
import argparse
import os
from glob import glob
import pandas as pd
import numpy as np

from matplotlib import cbook, cm
from matplotlib.colors import LightSource


def split_agent_name(row):
    parts = row["agentName"].split("-")
    row["shot_depth"] = int(parts[1])
    row["vel_samples"] = int(parts[2])
    row["mc_tree_depth"] = int(parts[3])
    row["mc_samples"] = int(parts[4])
    return row


def plot_benchmarks(log_dir, show=True, save_dir=".") -> None:
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
    print(df.head())
    df = df.apply(split_agent_name, axis=1)
    print(df.head())

    # Custom hillshading in a 3D surface plot

    # Set up plot
    fig = plt.figure()
    ax = fig.add_subplot(111, projection="3d")

    # Create data
    x = df["vel_samples"]
    y = df["mc_samples"]
    z = df["offBreakWin"]

    # Plot data
    ax.scatter(x, y, z, c="r", marker="o")

    # Customize ticks
    ax.set_xticks([2, 4, 6, 8, 10])
    ax.set_yticks([10, 20, 30, 40, 50])
    ax.set_zticks([0, 25, 50, 75, 100])

    # Customize label
    ax.set_xlabel("Velocity Samples")
    ax.set_ylabel("Monte-Carlo Samples")
    ax.set_zlabel("Off Break Win")

    plt.show()


def main():
    parser = argparse.ArgumentParser(description="Plot TensorBoard logs.")
    parser.add_argument("log_dir", type=str, help="Path(s) to the directory containing TensorBoard logs")
    parser.add_argument("-s", "--show", action="store_true", help="Show the plot")
    args = parser.parse_args()

    save_dir = "plots/"

    name = "asd-1-10-55-100"

    plot_benchmarks(args.log_dir, args.show, save_dir=save_dir)


if __name__ == "__main__":
    main()
