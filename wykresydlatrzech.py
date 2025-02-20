import pandas as pd
import matplotlib.pyplot as plt
import matplotlib as mpl
import os

# Global minimal font size setting (will be overridden locally)
mpl.rcParams['font.size'] = 10

# Define individual font sizes
title_fontsize = 30   # używane dla tytułu wykresu
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

# Zakładamy, że kolumna 'epoch' jest taka sama dla obu modeli
epochs_ipt = df_ipt['epoch']
epochs_rot = df_rot['epoch']

# Wyodrębniamy interesujące nas metryki
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

# (Opcjonalny) słownik z kolorami tytułów metryk
metric_title_colors = {
    "val/box_loss": "white",
    "val/cls_loss": "white",
    "val/dfl_loss": "white"
}

# Dla każdej metryki tworzymy osobny wykres
for metric, data in metrics.items():
    fig = plt.figure(figsize=(8, 6))
    fig.patch.set_facecolor(bg_color)  # kolor tła figury

    ax = fig.gca()
    ax.set_facecolor(bg_color)  # kolor tła wykresu

    # Rysowanie serii "Single tree" (IPT)
    ax.plot(epochs_ipt, data["Single tree"], color='lime', marker='o', label='Single tree')
    # Rysowanie serii "Row of trees" (ROT)
    ax.plot(epochs_rot, data["Row of trees"], color='yellow', marker='o', label='Row of trees')

    # >>> AUTOMATYCZNE DODANIE TYTUŁU WYKRESU NA PODSTAWIE NAZWY METRYKI <<<
    #ax.set_title(metric, fontsize=title_fontsize, color=metric_title_colors.get(metric, "white"))

    # Ustawienie etykiet osi z kolorem białym
    ax.set_xlabel("Epoch", fontsize=axis_labelsize, color="white")
    ax.set_ylabel("", fontsize=axis_labelsize, color="white")  # Brak etykiety dla osi Y

    # Automatyczne dopasowanie zakresu osi
    ax.autoscale()

    # Dodanie niewielkich marginesów horyzontalnych
    ax.margins(x=0.01)

    # Dodanie legendy (z półprzezroczystym tłem)
    legend = ax.legend(
        loc='best',
        fontsize=legend_fontsize,
        facecolor='black',
        edgecolor='white',
        framealpha=0.6
    )
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
