import plotly.express as px
import pandas as pd

params = {"dim": [20, 50, 100], "clustering": [True, False]}

baseline_df = pd.read_csv("src/res_baseline.csv")
# Création de la colonne "Config" avec les valeurs formatées
baseline_df["Config"] = baseline_df.apply(
    lambda row: f"{row['Step']}_{row['Scheduler']}_{row['Initial Temp']}",
    axis=1,
)
for dim in params["dim"]:
    for cluster in params["clustering"]:
        or_df = baseline_df[
            (baseline_df["Type"] == "OR_TOOLS 1 sec")
            & (baseline_df["Dimension"] == dim)
            & (baseline_df["Clustering"] == cluster)
        ]
        # Filtrage du DataFrame
        filtered_df = baseline_df[
            (baseline_df["Type"] == "Baseline")
            & (baseline_df["Dimension"] == dim)
            & (baseline_df["Clustering"] == cluster)
        ]

        filtered_df = filtered_df.drop_duplicates(subset=["Config"])

        # Sélection des 30 plus petits "Final Cost"
        filtered_df = filtered_df.nsmallest(30, "Final Cost")

        # Création du graphique en barres avec le nouvel index comme axe X
        fig = px.bar(
            filtered_df,
            x="Config",  # Utilisation du nouvel index en X
            y="Final Cost",  # Coût final en Y
            color="Final Cost",  # Couleur par scheduler d'entraînement
            title=f"Final Cost pour Dimension={dim} et Clustering={cluster}, Baseline",
            labels={
                "Final Cost": "Coût Final",
                "Config": "Step_Scheduler_InitialTemp",
            },
            text_auto=True,  # Affichage des valeurs sur les barres
        )
        if or_df["Final Cost"].notna().any():
            # Ajouter une ligne rouge horizontale correspondant à or_df["Final Cost"]
            fig.add_shape(
                type="line",
                x0=0,
                x1=1,
                y0=or_df["Final Cost"].values[0],
                y1=or_df["Final Cost"].values[0],
                line=dict(color="Red", width=2),
                xref="paper",
                yref="y",
                name=or_df["Type"].values[0],
            )

        # fig.show()
        # Save the plot in the specified directory
        fig.write_image(f"pic/final_cost_dim_{dim}_cluster_{cluster}.png")

trained_df = pd.read_csv("src/res_model.csv")

trained_df["Model Name"] = trained_df["Model"].apply(
    lambda x: x.split("/")[-2]  # if x != "/" else "/"
)

# Création de la colonne "Config" avec les valeurs formatées
trained_df["Config_test"] = trained_df.apply(
    lambda row: f"{row['Step']}_{row['Scheduler']}_{row['Initial Temp']}",
    axis=1,
)

trained_df["Config_train"] = trained_df.apply(
    lambda row: f"{row['Train Dimension']}_{row['Train Steps']}_{row['Train Scheduler']}_{row['Train Clustering']}_{row['Train Init Temp']}",
    axis=1,
)
trained_df["Config"] = trained_df.apply(
    lambda row: f"{row["Model Name"]}_{row['Train Steps']}/{row['Step']}_{row['Scheduler']}_{row['Train Clustering']}_{row['Train Init Temp']}/{row['Initial Temp']}",
    axis=1,
)

for dim in params["dim"]:
    for cluster in params["clustering"]:
        or_df = baseline_df[
            (baseline_df["Type"] == "OR_TOOLS 1 sec")
            & (baseline_df["Dimension"] == dim)
            & (baseline_df["Clustering"] == cluster)
        ]
        # Filtrage du DataFrame
        filtered_df = trained_df[
            (trained_df["Type"] == "Trained")
            & (trained_df["Dimension"] == dim)
            & (trained_df["Clustering"] == cluster)
        ]

        filtered_df = filtered_df.drop_duplicates(subset=["Config"])

        # Sélection des 30 plus petits "Final Cost"
        filtered_df = (
            filtered_df.groupby("Step", group_keys=False)
            .apply(lambda x: x.nsmallest(10, "Final Cost"))
            .reset_index(drop=True)
        )
        filtered_df = filtered_df.sort_values(by="Final Cost")

        # Filtrage du DataFrame
        baseline_filtered_df = baseline_df[
            (baseline_df["Type"] == "Baseline")
            & (baseline_df["Dimension"] == dim)
            & (baseline_df["Clustering"] == cluster)
        ]

        baseline_filtered_df = baseline_filtered_df.drop_duplicates(subset=["Config"])
        baseline_filtered_df = baseline_filtered_df.sort_values(by="Final Cost")

        # Sélection des 30 plus petits "Final Cost"
        baseline_filtered_df = (
            baseline_filtered_df.groupby("Step")
            .apply(lambda x: x.nsmallest(10, "Final Cost"))
            .reset_index(drop=True)
        )

        # Création du graphique en barres avec le nouvel index comme axe X
        fig = px.bar(
            filtered_df,
            x="Config",  # Utilisation du nouvel index en X
            y="Final Cost",  # Coût final en Y
            color="Step",  # Couleur par scheduler d'entraînement
            title=f"Final Cost pour Dimension={dim} et Clustering={cluster}, trained models",
            labels={
                "Final Cost": "Coût Final",
                "Config": "Model_TrainSteps/Step_Scheduler_TrainClustering_TrainInitTemp/InitialTemp",
            },
            text_auto=True,  # Affichage des valeurs sur les barres
        )

        # Ajouter une ligne rouge horizontale correspondant à or_df["Final Cost"]
        if or_df["Final Cost"].notna().any():
            fig.add_shape(
                type="line",
                x0=0,
                x1=1,
                y0=or_df["Final Cost"].values[0],
                y1=or_df["Final Cost"].values[0],
                line=dict(color="Red", width=2),
                xref="paper",
                yref="y",
                name=or_df["Type"].values[0],
            )

        # # Ajouter une ligne reliant les sommets des barres
        # fig.add_trace(
        #     px.line(
        #         baseline_filtered_df,
        #         x=filtered_df["Config"],
        #         y="Final Cost",
        #         markers=True,
        #     ).data[0]
        # )

        # fig.show()
        fig.write_image(f"pic/final_cost_dim_{dim}_cluster_{cluster}_Trained.png")
