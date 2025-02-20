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

# Zakładamy, że kolumna 'epoch' jest taka sama dla obu modeli
epochs_ipt = df_ipt['epoch']
epochs_rot = df_rot['epoch']

# Wyodrębniamy interesujące nas metryki – teraz dla każdej z nich dane pobierane są z obu plików:
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

# Opcjonalny słownik z kolorami tytułów metryk (przy ewentualnym użyciu tytułu)
metric_title_colors = {
    "Precision(B)": "darkorange",
    "Recall(B)": "red",
    "mAP50(B)": "blueviolet",
    "mAP50-95(B)": "orchid"
}

# Dla każdej metryki tworzymy osobny wykres
for metric, data in metrics.items():
    fig = plt.figure(figsize=(8, 6))
    fig.patch.set_facecolor(bg_color)  # Kolor tła figury

    ax = fig.gca()
    ax.set_facecolor(bg_color)  # Kolor tła wykresu

    # Rysowanie serii "Single tree" (dane IPT)
    ax.plot(epochs_ipt, data["Single tree"], color='lime', marker='o', label='Single tree')
    # Rysowanie serii "Row of trees" (dane ROT)
    ax.plot(epochs_rot, data["Row of trees"], color='yellow', marker='o', label='Row of trees')

    # Jeśli chcesz dodać tytuł metryki, odkomentuj poniższą linię:
    ax.set_title(metric, fontsize=title_fontsize, color=metric_title_colors.get(metric, "white"))

    # Ustawienie etykiet osi z kolorem białym
    ax.set_xlabel("Epoch", fontsize=axis_labelsize, color="white")
    ax.set_ylabel("", fontsize=axis_labelsize, color="white")  # Brak etykiety dla osi Y

    # Ustawienie wspólnych wartości y-ticks oraz zakresu osi y
    ax.set_yticks([0.0, 0.2, 0.4, 0.6, 0.8, 1.0])
    ax.set_ylim(0, 1)
    ax.tick_params(axis='both', labelsize=ticks_labelsize, colors="white")

    # Dodanie niewielkich marginesów horyzontalnych
    ax.margins(x=0.01)

    # Dodanie legendy w prawym dolnym rogu
    legend = ax.legend(loc='lower right', fontsize=legend_fontsize)
    # Ustawienie koloru tekstu legendy na biały
    for text in legend.get_texts():
        text.set_color("white")

    plt.tight_layout()

    # Generowanie nazwy pliku na podstawie nazwy metryki (bez znaków specjalnych)
    filename = metric.replace("(", "").replace(")", "").replace("/", "_") + ".png"
    file_path = os.path.join(save_dir, filename)

    # Zapisanie wykresu do pliku z tą samą barwą tła, co figura
    plt.savefig(file_path, dpi=300, facecolor=fig.get_facecolor())
    plt.close(fig)

