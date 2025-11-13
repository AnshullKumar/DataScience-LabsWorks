# DataScience-LabsWorks

This repository contains short, focused notebooks and examples for learning foundational data-science tasks using NumPy, Pandas, Matplotlib/Seaborn, scikit-learn, and statsmodels. The current notebook covers basic NumPy arrays and indexing. Below is a tracker of topics covered and planned, plus notes, example formats, and run instructions to keep the project organized.


## Planned topics (near-future)
Use this list as a checklist and learning roadmap. Each entry includes a short description and example APIs that will be demonstrated in accompanying notebooks.

1. Convert multi-dimensional arrays into 1D arrays
	- Techniques: `np.flatten()`, `np.ravel()`, `arr.reshape(-1)`

2. Concatenating two arrays
	- Techniques: `np.concatenate`, `np.vstack`, `np.hstack`, `np.stack`

3. Basic descriptive statistics: mean, median, mode, standard deviation
	- `np.mean`, `np.median`, `np.std`, `pandas.Series.mode()` or `scipy.stats.mode`

4. Probability distributions
	- Use `numpy.random` for sampling; `scipy.stats` for distribution objects, pdf/pmf/cdf and hypothesis tests

5. File handling and Pandas basics
	- `pd.read_csv`, `pd.to_csv`, using `open()` for raw file ops, `zipfile` for zip handling

6. CSV file handling and ZIP files
	- Examples showing reading CSV from disk and CSV inside ZIP archives

7. Clean and preprocess data
	- Missing value handling (`dropna`, `fillna`), type conversions, scaling (`StandardScaler` / manual), encoding categorical variables

8. Example dataset: exercise duration and heartbeat
	- CSV columns: `participant_id, exercise_duration_minutes, heartbeat_bpm, height_cm`
	- Demonstrate loading to a `DataFrame`, cleaning, EDA, and visualization

9. Histogram of a dataset (height distribution)
	- Use Matplotlib / Seaborn to plot histograms showing counts (frequency) per bin

10. Scatter plot: exercise vs heartbeat
	 - Plot `exercise_duration_minutes` (x) against `heartbeat_bpm` (y) to visualize correlation; optionally color by `height_cm` or `age`

11. Statistical analysis: 1-way ANOVA & ANCOVA (1D examples)
	 - Use `statsmodels.formula.api` `ols()` + `anova_lm()` for ANOVA; ANCOVA examples with a covariate (e.g., height)

12. IRIS dataset classification
	 - K-NN classification with `sklearn` (train/test split, metrics, confusion matrix, simple pipeline)

13. K-NN (from-scratch) and K-Means clustering (from-scratch)
	 - Pedagogical implementations: manually implement algorithms (no helper functions), then compare with `sklearn` versions

## Libraries to be used
- numpy
- pandas
- matplotlib
- seaborn
- scikit-learn
- statsmodels
- scipy (recommended for distributions and some statistics helpers)

## Suggested project structure

- `README.md`  (this file)
- `Numpy_array.ipynb`  (already present)
- `notebooks/`
  - `01_numpy_basics.ipynb`
  - `02_data_io_and_pandas.ipynb`
  - `03_visualization_and_eda.ipynb`
  - `04_stats_anova_ancova.ipynb`
  - `05_iris_classification_knn.ipynb`
  - `06_clustering_from_scratch.ipynb`
- `data/`
  - `exercise_heartbeat_example.csv`

These are suggestions to keep the repo organized as you add content.

## Example CSV format (exercise / heartbeat example)

The example CSV (`exercise_heartbeat_example.csv`) will have rows like:

participant_id,exercise_duration_minutes,heartbeat_bpm,height_cm
1,30,120,175
2,45,135,168
3,20,110,180

Load it with `pd.read_csv('data/exercise_heartbeat_example.csv')`.

## How to run / install dependencies (Windows PowerShell)

Create a virtual environment, activate it, and install required packages:

```powershell
# Create venv and activate (PowerShell)
python -m venv .venv; .\.venv\Scripts\Activate.ps1
# Install packages
pip install --upgrade pip; pip install numpy pandas matplotlib seaborn scikit-learn statsmodels scipy
```

Or create a `requirements.txt` and run `pip install -r requirements.txt`.

Open notebooks with Jupyter Lab or VS Code's notebook editor.

## Quick code snippets to include in notebooks
- Flatten array: `flat = arr.reshape(-1)`
- Concatenate: `combined = np.concatenate([a, b])`
- Mean/median/std: `np.mean(df['heartbeat'])`, `np.median(...)`, `np.std(...)`
- Mode: `df['col'].mode()` or `scipy.stats.mode(df['col'])`
- Histogram: `sns.histplot(df['height_cm'], bins=20)`
- Scatter: `plt.scatter(df['exercise_duration_minutes'], df['heartbeat_bpm'])` then `plt.xlabel()`/`plt.ylabel()` and `plt.show()`
- ANOVA (statsmodels):
  - `from statsmodels.formula.api import ols`
  - `model = ols('heartbeat_bpm ~ C(group)', data=df).fit()`
  - `import statsmodels.api as sm; anova_table = sm.stats.anova_lm(model, typ=2)`

## Assumptions & notes
- NumPy has no direct `mode` function: use `pandas.Series.mode()` or `scipy.stats.mode`. A pure-NumPy mode implementation can be added if desired.
- "H^s Frequency" was interpreted as histogram frequency (counts per bin). If you meant something else, tell me and I'll adapt.
- ANOVA/ANCOVA examples use `statsmodels`; let me know if you'd prefer alternate libraries.

## Next actionable steps (you can ask me to run any of these)
1. Add a sample CSV under `data/exercise_heartbeat_example.csv` (I can create one for you).
2. Add a `requirements.txt` with pinned versions.
3. Create starter notebook `notebooks/03_visualization_and_eda.ipynb` that loads the CSV, draws the histogram and scatter, and runs ANOVA/ANCOVA.
4. Add the IRIS classification and clustering notebooks.

If you'd like, I can create the sample CSV and starter notebook now — tell me preferred sample size and whether to include extra columns (age, gender, activity_type).

---
This README now serves as a tracker and roadmap for the planned notebooks and examples.
