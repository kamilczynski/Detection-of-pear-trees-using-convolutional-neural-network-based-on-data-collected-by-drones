import pandas as pd
import matplotlib.pyplot as plt
import matplotlib as mpl
import os

# Global minimal font size setting (will be overridden locally)
mpl.rcParams['font.size'] = 10

# Define individual font sizes
title_fontsize = 30  # (Not used now because we remove titles)
axis_labelsize = 30  # font size for axis labels (e.g., "Epoch")
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

# We assume that the 'epoch' column is the same for both models:
epochs_ipt = df_ipt['epoch']
epochs_rot = df_rot['epoch']

# We extract the metrics we are interested in – now for each of them the data is downloaded from both files:
metrics = {
    "Precision(B)": {
        "Single tree": df_ipt['metrics/precision(B)'],
        "Row of trees": df_rot['metrics/precision(B)']
    },
    "Recall(B)": {
        "Single tree": df_ipt['metrics/recall(B)'],
        "Row of trees": df_rot['metrics/recall(B)']
    },
    "mAP50(B)": {
        "Single tree": df_ipt['metrics/mAP50(B)'],
        "Row of trees": df_rot['metrics/mAP50(B)']
    },
    "mAP50-95(B)": {
        "Single tree": df_ipt['metrics/mAP50-95(B)'],
        "Row of trees": df_rot['metrics/mAP50-95(B)']
    }
}

# Optional dictionary with colors for metric titles (if a title is used)
metric_title_colors = {
    "Precision(B)": "darkorange",
    "Recall(B)": "red",
    "mAP50(B)": "blueviolet",
    "mAP50-95(B)": "orchid"
}

# We create a separate chart for each metric
for metric, data in metrics.items():
    fig = plt.figure(figsize=(8, 6))
    fig.patch.set_facecolor(bg_color)  # Figure background color

    ax = fig.gca()
    ax.set_facecolor(bg_color)  # Chart background color

    # Drawing "Single tree" series (IPT data)
    ax.plot(epochs_ipt, data["Single tree"], color='lime', marker='o', label='Single tree')
    # Drawing the "Row of trees" series (ROT data)
    ax.plot(epochs_rot, data["Row of trees"], color='yellow', marker='o', label='Row of trees')

    # If you want to add a metric title, uncomment the following line:
    ax.set_title(metric, fontsize=title_fontsize, color=metric_title_colors.get(metric, "white"))

    # Set axis labels to white
    ax.set_xlabel("Epoch", fontsize=axis_labelsize, color="white")
    ax.set_ylabel("", fontsize=axis_labelsize, color="white")  # Brak etykiety dla osi Y

    # Setting common y-ticks values ​​and y-axis range
    ax.set_yticks([0.0, 0.2, 0.4, 0.6, 0.8, 1.0])
    ax.set_ylim(0, 1)
    ax.tick_params(axis='both', labelsize=ticks_labelsize, colors="white")

    # Adding small horizontal margins
    ax.margins(x=0.01)

    # Added legend in the lower right corner
    legend = ax.legend(loc='lower right', fontsize=legend_fontsize)
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

