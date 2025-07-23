import pandas as pd
import matplotlib.pyplot as plt

OR_TOOLS_OPT_COST_PATH = "res/Vrp-Set-XML100_res.csv"
MODEL_RES_PATH = "res/models_res_on_bdd.csv"

# Noms lisibles pour la légende
NAMES = [
    "opt_sol",
    "or_tools_60",
    "Échantillonnage par rejet",
    "Reconstruction heuristique",
    "Baseline SA",
]

DISPLAY_NAMES = [
    "Solution optimale",
    "OR-Tools 60sec",
    "Échant. par rejet",
    "Reconstruction heur.",
    "SA baseline",
]

COLORS = [
    "#1f77b4",
    "#d62728",
    "#2ca02c",
    "#ff7f0e",
    "#9467bd",
]  # Couleurs pour chaque méthode


def read_csv_file(file_path):
    try:
        return pd.read_csv(file_path)
    except Exception as e:
        print(f"Erreur lors de la lecture du fichier CSV : {e}")
        return None


if __name__ == "__main__":
    df1 = read_csv_file(OR_TOOLS_OPT_COST_PATH)
    if "or_tools_1" in df1.columns:
        df1 = df1.drop(columns=["or_tools_1"])
    df2 = read_csv_file(MODEL_RES_PATH)
    NAMES = df1.columns.tolist()[1:] + df2.columns.tolist()[1:]

    if df1 is not None and df2 is not None:
        df = pd.merge(df1, df2, on="name", how="outer")

        data = [df[col].dropna().astype(float) for col in NAMES]
        means = [d.mean() for d in data]

        fig, ax = plt.subplots(figsize=(10, 6))

        # Création du boxplot
        box = ax.boxplot(
            data,
            patch_artist=True,
            tick_labels=DISPLAY_NAMES,
            showfliers=False,
            widths=0.6,
        )

        # Couleur de chaque boîte
        for patch, color in zip(box["boxes"], COLORS):
            patch.set_facecolor(color)
            patch.set_alpha(0.6)
            patch.set_linewidth(1.5)

        # Couleurs des lignes (médiane, whiskers, caps)
        for element in ["medians", "whiskers", "caps"]:
            for line in box[element]:
                line.set_color("black")
                line.set_linewidth(1.2)

        # Moyennes en ligne pointillée
        for i, mean in enumerate(means):
            ax.hlines(
                y=mean,
                xmin=i + 0.8,
                xmax=i + 1.2,
                colors=COLORS[i],
                linestyles="dashed",
                linewidth=2,
                label=f"{DISPLAY_NAMES[i]} (moy: {mean:.2f})",
            )

        # Axe et titre
        ax.set_ylabel("Coût")
        ax.set_title("Comparaison des coûts : Optimal, OR-Tools, modèles")
        ax.grid(True, linestyle="--", alpha=0.5)
        plt.xticks(rotation=20)
        plt.tight_layout()

        # Légende propre
        handles, labels = ax.get_legend_handles_labels()
        by_label = dict(zip(labels, handles))
        ax.legend(
            by_label.values(),
            by_label.keys(),
            loc="upper left",
            fontsize="small",
            frameon=True,
        )

        # Sauvegarde
        plt.savefig("res/boxplot_comparison.png", dpi=300)
        plt.show()

    else:
        print("Erreur : les données sont manquantes ou incomplètes.")
