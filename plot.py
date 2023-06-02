import matplotlib
import pandas as pd
import matplotlib.pyplot as plt

python_codex_protein = pd.read_csv("python_codex_protein.csv", index_col=0, header=0)
python_codex_protein = python_codex_protein.apply(pd.to_numeric)

r_codex_protein = pd.read_csv("R_codex_protein.csv", index_col=0, header=0)
r_codex_protein = r_codex_protein.apply(pd.to_numeric)

fig, ax = plt.subplots(figsize=(12, 12))  # Creates a larger figure

for i, column in enumerate(r_codex_protein.columns):
    x = r_codex_protein[[column]]
    y = python_codex_protein[[column]]
    ax.scatter(x, y, label=column)

ax.set_xlabel("R")
ax.set_ylabel("Python")
ax.set_title("Scatter Plot")

ax.legend()
plt.show()

fig, ax = plt.subplots(figsize=(12, 12))
for i, column in enumerate(r_codex_protein.columns):
    if(column not in ["CD11b", "CD4", "IgD"]):
        continue
    x = r_codex_protein[[column]]
    y = python_codex_protein[[column]]
    ax.scatter(x, y, label=column)

ax.set_xlabel("R")
ax.set_ylabel("Python")
ax.set_title("Scatter Plot with Only CD11b, CD4, and IgD")

ax.legend()
plt.show()

