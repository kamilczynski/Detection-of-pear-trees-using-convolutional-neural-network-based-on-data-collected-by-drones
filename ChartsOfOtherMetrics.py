import pandas as pd
import matplotlib.pyplot as plt
import matplotlib as mpl
import os

# Global minimal font size setting (will be overridden locally)
mpl.rcParams['font.size'] = 10

# Define individual font sizes
title_fontsize = 30   # used for the chart title
axis_labelsize = 30   # font size for axis labels (e.g., "Epoch")
ticks_labelsize = 30  # font size for axis tick labels
legend_fontsize = 22  # font size for legend text

# Style and background color settings
plt.style.use("dark_background")
bg_color = "#031867"

# Folder path where plots will be saved
save_dir = r"C:\Users\topgu\Desktop\Art\POSTER 2025\wykresy"
os.makedirs(save_dir, exist_ok=True)

# Load data from both CSV files
df_ipt = pd.read_csv(r"C:\Users\topgu\Desktop\Art\POSTER 2025\resultsold.csv")
df_rot = pd.read_csv(r"C:\Users\topgu\Desktop\Art\POSTER 2025\resultsnew.csv")

# We assume that the 'epoch' column is the same for both models
epochs_ipt = df_ipt['epoch']
epochs_rot = df_rot['epoch']

# We extract the metrics that interest us
metrics = {
    "val/box_loss": {
        "Single tree": df_ipt['val/box_loss'],
        "Row of trees": df_rot['val/box_loss']
    },
    "val/cls_loss": {
        "Single tree": df_ipt['val/cls_loss'],
        "Row of trees": df_rot['val/cls_loss']
    },
    "val/dfl_loss": {
        "Single tree": df_ipt['val/dfl_loss'],
        "Row of trees": df_rot['val/dfl_loss']
    }
}

# (Optional) dictionary with colors for metric titles
metric_title_colors = {
    "val/box_loss": "white",
    "val/cls_loss": "white",
    "val/dfl_loss": "white"
}

# We create a separate chart for each metric
for metric, data in metrics.items():
    fig = plt.figure(figsize=(8, 6))
    fig.patch.set_facecolor(bg_color)  # figure background color

    ax = fig.gca()
    ax.set_facecolor(bg_color)  # chart background color

    # Drawing "Single tree" series (IPT)
    ax.plot(epochs_ipt, data["Single tree"], color='lime', marker='o', label='Single tree')
    # Drawing series "Row of trees" (ROT)
    ax.plot(epochs_rot, data["Row of trees"], color='yellow', marker='o', label='Row of trees')

    # >>> AUTOMATICALLY ADDING A CHART TITLE BASED ON THE METRIC NAME <<<
    ax.set_title(metric, fontsize=title_fontsize, color=metric_title_colors.get(metric, "white"))

    # Set axis labels to white
    ax.set_xlabel("Epoch", fontsize=axis_labelsize, color="white")
    ax.set_ylabel("", fontsize=axis_labelsize, color="white")  # Brak etykiety dla osi Y

    # Automatic adjustment of the axis range
    ax.autoscale()

    # Adding small horizontal margins
    ax.margins(x=0.01)

    # Adding a legend (with semi-transparent background)
    legend = ax.legend(
        loc='best',
        fontsize=legend_fontsize,
        facecolor='black',
        edgecolor='white',
        framealpha=0.6
    )
    # Set legend text color to white
    for text in legend.get_texts():
        text.set_color("white")

    plt.tight_layout()

    # Generating file name based on metric name (no special characters)
    filename = metric.replace("(", "").replace(")", "").replace("/", "_") + ".png"
    file_path = os.path.join(save_dir, filename)

    # Saving the graph to a file with the same background color as the figure
    plt.savefig(file_path, dpi=300, facecolor=fig.get_facecolor())
    plt.close(fig)
