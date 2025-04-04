import json
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import os
from scipy import stats

# Configuration
ORTools_PATH = "./bdd/results_ortools.json"
CSV_PATH = "./res/model_on_bdd_swap_vs_2opt.csv"
BDD_PATH = "./bdd/bdd.json"
OUTPUT_DIR = "./res/"

# Créer le dossier de sortie
os.makedirs(OUTPUT_DIR, exist_ok=True)


# 1. Chargement des données
def load_data():
    # Charger les résultats OR-Tools
    with open(ORTools_PATH) as f:
        ortools_data = json.load(f)
    ortools_df = pd.DataFrame.from_dict(ortools_data, orient="index").reset_index()
    ortools_df.columns = ["problem_name", "ortools_distance"]
    ortools_df["problem_name"] = ortools_df["problem_name"].str.upper()

    # Charger les données des modèles
    models_df = pd.read_csv(CSV_PATH)
    models_df["problem_name"] = models_df["problem_name"].str.upper()

    # Charger les métadonnées des instances
    with open(BDD_PATH) as f:
        bdd_data = json.load(f)
    bdd_df = pd.DataFrame(bdd_data)
    bdd_df["name"] = bdd_df["name"].str.upper()

    return ortools_df, models_df, bdd_df


ortools_df, models_df, bdd_df = load_data()

# 2. Fusion des données
merged_df = pd.merge(models_df, ortools_df, on="problem_name", how="left")
merged_df = pd.merge(
    merged_df, bdd_df, left_on="problem_name", right_on="name", how="left"
)

# 3. Calcul des métriques
merged_df["Diff_Model_vs_Best"] = merged_df["final_cost"] - merged_df["best_real_cost"]
merged_df["Diff_ORtools_vs_Best"] = (
    merged_df["ortools_distance"] - merged_df["best_real_cost"]
)
# Supprimer les lignes où "Diff_Model_vs_Best" ou "Diff_ORtools_vs_Best" sont négatifs
merged_df = merged_df[
    (merged_df["Diff_Model_vs_Best"] >= 0) & (merged_df["Diff_ORtools_vs_Best"] >= 0)
]
merged_df["Diff_Model_vs_ORtools"] = (
    merged_df["final_cost"] - merged_df["ortools_distance"]
)
merged_df["Relative_Model_Improvement"] = (
    (merged_df["ortools_distance"] - merged_df["final_cost"])
    / merged_df["ortools_distance"]
    * 100
)

glob_stats = {
    "Global Statistics": {
        "Number of instances": len(merged_df["problem_name"].unique()),
        "Number of model evaluations": len(merged_df),
        "Number of different models": len(merged_df["model"].unique()),
    }
}

# Statistiques par modèle
model_stats = (
    merged_df.groupby("model")
    .agg(
        {
            "Diff_Model_vs_Best": ["mean", "median", "std", "min", "max"],
            "Diff_Model_vs_ORtools": ["mean", "median", "std", "min", "max"],
            "final_cost": ["mean", "median", "std"],
        }
    )
    .round(2)
)

# Statistiques OR-Tools
ortools_stats = (
    merged_df.groupby("problem_name")
    .first()
    .agg(
        {
            "Diff_ORtools_vs_Best": ["mean", "median", "std", "min", "max"],
            "ortools_distance": ["mean", "median", "std"],
        }
    )
    .round(2)
)

# Sauvegarder les statistiques
with open(f"{OUTPUT_DIR}statistics.txt", "w") as f:
    f.write("=== Global Statistics ===\n")
    for k, v in glob_stats["Global Statistics"].items():
        f.write(f"{k}: {v}\n")

    f.write("\n=== Model Statistics ===\n")
    f.write(model_stats.to_string())

    f.write("\n\n=== OR-Tools Statistics ===\n")
    f.write(ortools_stats.to_string())


# 4. Classification par difficulté
def classify_difficulty(row):
    # Score basé sur dimension (nombre de points) et capacité
    score = np.log(row["dimension"]) * (
        100 / row["capacity"] if row["capacity"] > 0 else 1
    )

    if score < 20:
        return "Facile"
    elif score < 40:
        return "Moyen"
    elif score < 60:
        return "Difficile"
    else:
        return "Très difficile"


merged_df["difficulty"] = merged_df.apply(classify_difficulty, axis=1)
difficulty_order = ["Facile", "Moyen", "Difficile", "Très difficile"]


# 5. Analyse statistique
def perform_statistical_analysis(df):
    # Tests par paire entre modèles
    models = df["model"].unique()
    model_results = {}

    for model in models:
        model_data = df[df["model"] == model]["Diff_Model_vs_Best"]
        model_results[model] = {
            "mean": model_data.mean(),
            "median": model_data.median(),
            "pvalue_vs_ortools": stats.ttest_ind(
                model_data, df["Diff_ORtools_vs_Best"].dropna()
            ).pvalue,
        }

    # Classement des modèles
    ranked_models = sorted(models, key=lambda x: model_results[x]["median"])

    return model_results, ranked_models


model_results, ranked_models = perform_statistical_analysis(merged_df)

# 6. Visualisations
plt.style.use("seaborn-v0_8")

# Boxplot par difficulté
plt.figure(figsize=(14, 8))
sns.boxplot(
    data=merged_df,
    x="model",
    y="Diff_Model_vs_Best",
    hue="difficulty",
    hue_order=difficulty_order,
    palette="viridis",
)
plt.title("Performance des modèles par niveau de difficulté")
plt.xticks(rotation=45)
plt.tight_layout()
plt.savefig(f"{OUTPUT_DIR}model_performance_by_difficulty.png", dpi=300)
plt.close()

# Heatmap de performance comparative
heatmap_data = (
    merged_df.groupby(["model", "difficulty"])
    .agg({"Relative_Model_Improvement": "median"})
    .unstack()
    .reindex(difficulty_order, axis=1, level=1)
)
heatmap_data.columns = heatmap_data.columns.droplevel()

plt.figure(figsize=(12, 8))
sns.heatmap(
    heatmap_data.T, annot=True, fmt=".1f", cmap="RdYlGn", center=0, linewidths=0.5
)
plt.title("Amélioration relative médiane (%) par rapport à OR-Tools")
plt.tight_layout()
plt.savefig(f"{OUTPUT_DIR}performance_heatmap.png", dpi=300)
plt.close()

# Distribution des différences OR-Tools vs Best
plt.figure(figsize=(12, 6))
sns.histplot(
    data=merged_df.groupby("problem_name").first(), x="Diff_ORtools_vs_Best", bins=50
)
plt.title("Distribution des différences OR-Tools vs Meilleure solution")
plt.savefig(f"{OUTPUT_DIR}ortools_vs_best_hist.png", dpi=300)
plt.close()

palette = sns.color_palette("husl", len(merged_df["model"].unique()))
# Scatter plot Model vs OR-Tools (échantillonnage)
plt.figure(figsize=(10, 10))
sample = merged_df.sample(min(5000, len(merged_df)))
sns.scatterplot(
    data=sample,
    x="ortools_distance",
    y="final_cost",
    hue="model",
    alpha=0.6,
    palette=palette,
)
plt.plot(
    [sample["ortools_distance"].min(), sample["ortools_distance"].max()],
    [sample["ortools_distance"].min(), sample["ortools_distance"].max()],
    "r--",
)
plt.title("Comparaison des modèles vs OR-Tools (échantillon)")
plt.savefig(f"{OUTPUT_DIR}model_vs_ortools_scatter.png", dpi=300)
plt.close()

sample_df = merged_df.groupby("model", group_keys=False).apply(
    lambda x: x.sample(min(1000, len(x)))
)
plt.figure(figsize=(15, 8))
sns.boxplot(
    data=sample_df,
    x="model",
    y="Diff_Model_vs_Best",
    hue="model",
    palette=palette,
    legend=False,
)
plt.xticks(rotation=45)
plt.title(
    "Distribution des différences entre modèles et meilleure solution (échantillon)"
)
plt.tight_layout()
plt.savefig(f"{OUTPUT_DIR}model_vs_best_boxplot.png", dpi=300)
plt.close()

# KDE Plot des différences
plt.figure(figsize=(12, 6))
for model in merged_df["model"].unique():
    subset = merged_df[merged_df["model"] == model]
    if len(subset) > 1000:  # On échantillonne pour les grands datasets
        subset = subset.sample(1000)
    sns.kdeplot(data=subset, x="Diff_Model_vs_Best", label=model, alpha=0.5)
plt.legend()
plt.title("Densité des différences par modèle vs meilleure solution")
plt.savefig(f"{OUTPUT_DIR}model_diff_kde.png", dpi=300)
plt.close()

# KDE Plot OR-Tools vs Modèles
plt.figure(figsize=(12, 6))
sns.kdeplot(
    data=merged_df.groupby("problem_name").first(),
    x="Diff_ORtools_vs_Best",
    label="OR-Tools",
    color="red",
)
for model in merged_df["model"].unique():
    subset = merged_df[merged_df["model"] == model]
    if len(subset) > 1000:
        subset = subset.sample(1000)
    sns.kdeplot(data=subset, x="Diff_Model_vs_Best", label=model, alpha=0.3)
plt.legend()
plt.title("Comparaison des distributions OR-Tools et modèles")
plt.savefig(f"{OUTPUT_DIR}comparison_kde.png", dpi=300)
plt.close()

# 7. Sauvegarde des résultats
with open(f"{OUTPUT_DIR}statistical_results.txt", "w") as f:
    f.write("=== Résultats statistiques ===\n\n")
    f.write("Classement des modèles (du meilleur au moins bon):\n")
    for i, model in enumerate(ranked_models, 1):
        f.write(f"{i}. {model} (médiane: {model_results[model]['median']:.2f})\n")

    f.write("\n=== Tests statistiques ===\n")
    for model in model_results:
        f.write(f"\nModèle: {model}\n")
        f.write(f"- Différence moyenne vs best: {model_results[model]['mean']:.2f}\n")
        f.write(
            f"- p-value vs OR-Tools: {model_results[model]['pvalue_vs_ortools']:.4f}\n"
        )
        if model_results[model]["pvalue_vs_ortools"] < 0.05:
            f.write("  → Différence statistiquement significative\n")
        else:
            f.write("  → Différence non significative\n")


# 8. Analyse complémentaire
def additional_analysis(df):
    # Meilleur modèle par catégorie de difficulté
    best_by_difficulty = df.groupby("difficulty").apply(
        lambda x: x.groupby("model")["Diff_Model_vs_Best"].median().idxmin()
    )

    # Sauvegarde
    with open(f"{OUTPUT_DIR}best_models_by_difficulty.txt", "w") as f:
        f.write("Meilleur modèle par niveau de difficulté:\n")
        for difficulty, model in best_by_difficulty.items():
            f.write(f"- {difficulty}: {model}\n")


additional_analysis(merged_df)


print("Analyse terminée. Résultats sauvegardés dans:", OUTPUT_DIR)
