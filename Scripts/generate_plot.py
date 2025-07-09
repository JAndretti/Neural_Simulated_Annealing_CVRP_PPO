import pandas as pd
import plotly.graph_objects as go

OR_TOOLS_OPT_COST_PATH = "res/Vrp-Set-XML100_res.csv"
MODEL_RES_PATH = "res/models_res_on_bdd.csv"

NAMES = [
    "opt_sol",
    "or_tools_cost",
    "Échantillonnage par rejet",
    "Reconstruction heuristique",
]
colors = ["blue", "red", "green", "orange"]  # Colors for the boxplots


def read_csv_file(file_path):
    """
    Reads a CSV file and returns a DataFrame.

    :param file_path: Path to the CSV file.
    :return: DataFrame containing the CSV data.
    """
    try:
        df = pd.read_csv(file_path)
        return df
    except Exception as e:
        print(f"Error reading the CSV file: {e}")
        return None


if __name__ == "__main__":
    df = read_csv_file(OR_TOOLS_OPT_COST_PATH)
    models_df = read_csv_file(MODEL_RES_PATH)
    df = pd.merge(df, models_df, on="name", how="outer")
    df_names = df.columns.tolist()[1:]  # Exclude the first column 'name'

    # Check if the DataFrame is not None and contains the required columns
    if df is not None and "opt_sol" in df.columns and "or_tools_cost" in df.columns:
        # Create figure
        fig = go.Figure()

        # Calculate means
        means = df[df_names].mean()

        # Add boxplots
        for i, name in enumerate(NAMES):
            fig.add_trace(
                go.Box(
                    y=df[df_names[i]].astype(float),
                    name=name,
                    boxpoints=False,  # Show all points
                    jitter=0.3,  # Add some jitter to the points
                    pointpos=-1.8,  # Position of points relative to the box
                    line=dict(color=colors[i]),  # Set color for the boxplot line
                )
            )
            fig.add_shape(
                type="line",
                x0=-0.5,  # Start slightly before the boxplot
                x1=len(NAMES) - 0.5,  # End slightly after the boxplot
                y0=means.iloc[i],
                y1=means.iloc[i],
                line=dict(
                    width=2,
                    dash="dash",  # Dashed line
                    color=colors[i],  # Use the same color as the boxplot
                ),
            )
            fig.add_trace(
                go.Scatter(
                    x=[None],  # Dummy x value for legend
                    y=[None],  # Dummy y value for legend
                    mode="lines",
                    line=dict(
                        width=2,
                        dash="dash",
                        color=colors[i],
                    ),
                    name=f"{name} mean : {means.iloc[i]:.2f}",  # Legend entry for the
                    # mean line
                )
            )

        # Update layout
        fig.update_layout(
            title="Comparison of Optimal Solution vs OR-Tools Cost vs Model",
            yaxis_title="Cost",
            boxmode="group",
        )

        # Save and show the plot
        fig.write_image("res/boxplot_comparison.png")
        # Open and display the saved PNG image using matplotlib
        import matplotlib.pyplot as plt
        import matplotlib.image as mpimg

        # Read the image
        img = mpimg.imread("res/boxplot_comparison.png")

        # Display the image
        plt.figure(figsize=(10, 8))
        plt.imshow(img)
        plt.axis("off")  # Hide axes
        plt.tight_layout()
        plt.show()
    else:
        print("DataFrame is None or required columns are missing.")
