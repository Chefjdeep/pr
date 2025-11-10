import os, numpy as np, pandas as pd, seaborn as sns, matplotlib.pyplot as plt, plotly.express as px, networkx as nx
from scipy.cluster.hierarchy import dendrogram, linkage

OUTDIR = "outputs"
os.makedirs(OUTDIR, exist_ok=True)

def save(fig, name):
    path = os.path.join(OUTDIR, name)
    fig.savefig(path, bbox_inches="tight", dpi=150)
    plt.close(fig)
    print("Saved:", path)

# --- Datasets ---
iris = sns.load_dataset('iris')
iris['species'] = iris['species'].astype(str)  # seaborn iris already has species names

np.random.seed(42)
df_adult = pd.DataFrame({
    "age": np.random.randint(18, 70, 200),
    "hours_per_week": np.random.randint(20, 60, 200),
    "education_num": np.random.randint(1, 16, 200),
    "income": np.random.choice([">50K", "<=50K"], 200),
    "workclass": np.random.choice(["Private", "Self-emp", "Gov"], 200),
    "year": np.random.choice(range(2000, 2020), 200)
})

# --- 1D Histogram ---
fig = sns.histplot(df_adult['age'], bins=20, kde=True, color="skyblue").get_figure()
plt.title("Adult Age Distribution")
save(fig, "adult_1d_hist_age.png")

# --- 2D Scatter ---
fig = sns.scatterplot(data=iris, x="sepal_length", y="sepal_width", hue="species", palette="Set2").get_figure()
plt.title("Iris Sepal Scatter")
save(fig, "iris_2d_scatter.png")

# --- 3D Scatter ---
fig3d = px.scatter_3d(iris, x="sepal_length", y="sepal_width", z="petal_length", color="species",
                      title="Iris 3D Scatter")
fig3d.write_image(os.path.join(OUTDIR, "iris_3d.png"))
print("Saved:", os.path.join(OUTDIR, "iris_3d.png"))

# --- Temporal Trend ---
df_yearly = df_adult.groupby("year")["hours_per_week"].mean().reset_index()
fig = sns.lineplot(data=df_yearly, x="year", y="hours_per_week", marker="o").get_figure()
plt.title("Average Hours per Week by Year")
save(fig, "adult_temporal_year_hours.png")

# --- Pairplot ---
pair = sns.pairplot(iris, hue="species", diag_kind="kde")
pair.fig.suptitle("Iris Pairplot", y=1.02)
pair.savefig(os.path.join(OUTDIR, "iris_pairplot.png"))
plt.close("all")
print("Saved:", os.path.join(OUTDIR, "iris_pairplot.png"))

# --- Dendrogram ---
X = iris[["sepal_length","sepal_width","petal_length","petal_width"]]
fig = plt.figure(figsize=(10,5))
dendrogram(linkage(X, method="ward"), labels=iris['species'].values, leaf_rotation=90, leaf_font_size=8)
plt.title("Iris Dendrogram")
save(fig, "iris_dendrogram.png")

# --- Network graph ---
G = nx.Graph()
for wc in df_adult['workclass'].unique():
    for inc in df_adult['income'].unique():
        G.add_edge(wc, inc, weight=np.random.randint(1,10))
fig = plt.figure(figsize=(6,6))
nx.draw(G, nx.spring_layout(G), with_labels=True, node_size=2000, node_color="lightgreen", font_size=12, width=2)
plt.title("Workclass vs Income Network")
save(fig, "adult_network.png")

print("✅ All visualizations saved in", OUTDIR)
