# ==========================================================
# Project: Employee Salary Analysis with Seaborn
# Dataset: 32 employees (hand-made sample data for practice)
# Tip: each plot opens in its own window. Close it to see the next one.
# ==========================================================
import pandas as p
import seaborn as s
import matplotlib.pyplot as m

# ----------------------------------------------------------
# SECTION 0: BUILD THE DATA
# ----------------------------------------------------------
data = {
    "Employee": [f"E{str(i).zfill(2)}" for i in range(1, 33)],
    "Company": [
        "Google", "Microsoft", "Amazon", "Google", "Microsoft", "Amazon", "Google", "Amazon",
        "Microsoft", "Google", "Amazon", "Microsoft", "Amazon", "Google", "Microsoft", "Amazon",
        "Google", "Microsoft", "Amazon", "Google", "Microsoft", "Amazon", "Google", "Microsoft",
        "Amazon", "Google", "Microsoft", "Amazon", "Google", "Microsoft", "Amazon", "Google"
    ],
    "Department": [
        "Engineering", "Engineering", "Sales", "HR", "Engineering", "Finance", "Sales", "Engineering",
        "HR", "Finance", "Engineering", "Sales", "Engineering", "Engineering", "Finance", "HR",
        "Sales", "Engineering", "Finance", "Engineering", "Sales", "HR", "Finance", "Engineering",
        "Sales", "HR", "Engineering", "Finance", "Engineering", "Sales", "Engineering", "Finance"
    ],
    "Job_Level": [
        "Junior", "Mid", "Senior", "Junior", "Senior", "Mid", "Mid", "Junior",
        "Mid", "Senior", "Junior", "Senior", "Mid", "Junior", "Senior", "Junior",
        "Senior", "Mid", "Junior", "Senior", "Mid", "Senior", "Junior", "Mid",
        "Senior", "Mid", "Junior", "Senior", "Mid", "Junior", "Senior", "Mid"
    ],
    "Age": [
        23, 29, 35, 24, 38, 31, 28, 25, 30, 34, 26, 37, 32, 22, 39, 27,
        36, 33, 24, 41, 29, 34, 25, 31, 40, 28, 26, 37, 30, 23, 42, 32
    ],
    "Experience": [
        1, 5, 10, 2, 12, 7, 4, 2, 6, 9, 3, 11, 8, 1, 14, 3,
        13, 7, 2, 16, 5, 10, 2, 6, 15, 4, 3, 12, 7, 1, 17, 8
    ],
    "Salary_LPA": [
        8, 18, 32, 7, 35, 20, 16, 9, 14, 28, 10, 34, 22, 7, 38, 8,
        30, 19, 9, 42, 17, 33, 8, 21, 40, 15, 10, 36, 20, 7, 45, 23
    ],
    "Performance": [
        7, 8, 9, 6, 9, 7, 8, 6, 7, 9, 6, 8, 8, 7, 10, 6,
        9, 8, 5, 10, 7, 9, 6, 8, 9, 7, 6, 9, 8, 5, 10, 8
    ],
}
df = p.DataFrame(data)

num_cols = ["Age", "Experience", "Salary_LPA", "Performance"]  # numeric columns used often
levels = ["Junior", "Mid", "Senior"]                           # logical order for Job_Level

print(df.head())
print(df.shape)
df.info()
print(df.describe())

# ----------------------------------------------------------
# SECTION 1: HISTPLOT (distribution of one column)
# ----------------------------------------------------------
s.histplot(data=df, x="Salary_LPA", bins=8)
m.title("Salary distribution")
m.show()
# Conclusion: look at the shape. Is it symmetric or skewed?

s.histplot(data=df, x="Age", bins=8, kde=True)
m.title("Age distribution")
m.show()
# Conclusion: read the peak on the Age (x) axis.

s.histplot(data=df, x="Salary_LPA", hue="Company", bins=8)   # hue needs data=df
m.title("Salary by company")
m.show()

s.histplot(data=df, x="Salary_LPA", hue="Company", bins=8, multiple="stack")
m.title("Salary by company (stacked)")
m.show()

s.histplot(data=df, x="Experience", stat="percent", bins=8)
m.title("Experience (percent of employees)")
m.show()
print("Percent with under 5 years experience:", (df["Experience"] < 5).mean() * 100)

# ----------------------------------------------------------
# SECTION 2: JOINTPLOT (two columns + their distributions)
# ----------------------------------------------------------
s.jointplot(data=df, x="Experience", y="Salary_LPA", kind="scatter")
m.show()

s.jointplot(data=df, x="Experience", y="Salary_LPA", kind="reg")
m.show()
# Conclusion: upward line = more experience, higher salary.

s.jointplot(data=df, x="Age", y="Salary_LPA", kind="kde")
m.show()

s.jointplot(data=df, x="Experience", y="Performance", kind="hex")
m.show()
# Note: hex looks sparse because only 32 points and Performance has few distinct values.

s.jointplot(data=df, x="Experience", y="Salary_LPA", hue="Department")
m.show()

# ----------------------------------------------------------
# SECTION 3: PAIRPLOT (all numeric pairs at once)
# ----------------------------------------------------------
s.pairplot(df)                       # Employee is text, so it is skipped automatically
m.show()

s.pairplot(df, vars=["Age", "Experience", "Salary_LPA"])
m.show()

s.pairplot(df, vars=num_cols, hue="Company")
m.show()
# Conclusion: companies overlap a lot, so no clear company-wise pattern.

s.pairplot(df, vars=num_cols, diag_kind="kde", kind="reg")
m.show()

print(df[num_cols].corr())           # confirms which pairs are most linear

# ----------------------------------------------------------
# SECTION 4: BARPLOT AND BOXPLOT
# ----------------------------------------------------------
s.barplot(data=df, x="Department", y="Salary_LPA")           # average salary per department
m.title("Average salary by department")
m.show()
print(df.groupby("Department")["Salary_LPA"].mean().sort_values(ascending=False))

s.barplot(data=df, x="Job_Level", y="Salary_LPA", hue="Company", order=levels)
m.title("Average salary by level and company")
m.show()

s.boxplot(data=df, x="Job_Level", y="Salary_LPA", order=levels)
m.title("Salary by job level")
m.show()

s.boxplot(data=df, x="Company", y="Performance")
m.title("Performance spread by company")
m.show()

s.boxplot(data=df, x="Department", y="Salary_LPA", hue="Company")
m.title("Salary by department and company")
m.show()
# Note: some Company x Department groups have only 1-2 rows, so boxes look odd.

# ----------------------------------------------------------
# SECTION 5: VIOLIN, STRIP, SWARM
# ----------------------------------------------------------
s.violinplot(data=df, x="Job_Level", y="Salary_LPA", order=levels)
m.show()

s.violinplot(data=df, x="Job_Level", y="Salary_LPA", order=levels, inner="stick")
m.show()   # inner="stick" draws one line per employee inside the violin

s.stripplot(data=df, x="Department", y="Salary_LPA", jitter=True)
m.show()

s.swarmplot(data=df, x="Company", y="Performance")
m.show()   # swarm never overlaps points, so every point is visible

# Overlay: violin (shape) + swarm (real points)
s.violinplot(data=df, x="Job_Level", y="Salary_LPA", order=levels, inner=None, color="lightgray")
s.swarmplot(data=df, x="Job_Level", y="Salary_LPA", order=levels, color="black")
m.title("Salary by level: violin + swarm")
m.show()

# ----------------------------------------------------------
# SECTION 6: LMPLOT AND REGPLOT (regression lines)
# ----------------------------------------------------------
s.lmplot(data=df, x="Experience", y="Salary_LPA")
m.show()

s.lmplot(data=df, x="Experience", y="Salary_LPA", hue="Company")
m.show()

s.lmplot(data=df, x="Experience", y="Salary_LPA", col="Company")
m.show()

s.regplot(data=df, x="Experience", y="Salary_LPA", order=2)
m.show()   # curve barely differs from a straight line, so linear is enough

s.regplot(data=df, x="Performance", y="Salary_LPA", x_jitter=0.1)
m.show()   # jitter nudges overlapping points sideways; the line does not change

s.regplot(data=df, x="Age", y="Salary_LPA")
m.xlabel("Age")
m.ylabel("Salary_LPA")
m.show()

# ----------------------------------------------------------
# SECTION 7: HEATMAPS
# ----------------------------------------------------------
corr = df[num_cols].corr()
s.heatmap(corr, annot=True)
m.title("Correlation heatmap")
m.show()

s.heatmap(corr, annot=True, cmap="coolwarm", vmin=-1, vmax=1, fmt=".2f")
m.title("Correlation heatmap (coolwarm)")
m.show()

# Heatmap of a pivot table (the table itself, not .corr() of it)
pt1 = p.pivot_table(df, values="Salary_LPA", index="Department",
                    columns="Job_Level", aggfunc="mean")[levels]
s.heatmap(pt1, annot=True, fmt=".1f")
m.title("Average salary: Department x Job level")
m.show()

pt2 = p.pivot_table(df, values="Performance", index="Company",
                    columns="Department", aggfunc="mean")
s.heatmap(pt2, annot=True, fmt=".1f")
m.title("Average performance: Company x Department")
m.show()

# ----------------------------------------------------------
# SECTION 8: CLUSTERMAP (optional)
# ----------------------------------------------------------
s.clustermap(corr, annot=True, cmap="coolwarm", vmin=-1, vmax=1)
m.show()

s.clustermap(pt1.fillna(0), annot=True, standard_scale=1, cmap="viridis", figsize=(7, 6))
m.show()   # standard_scale=1 rescales each column to 0-1 so scales are comparable

# ----------------------------------------------------------
# FINAL FINDINGS (numbers are printed so your text matches the output)
# ----------------------------------------------------------
print("\nAverage salary by company:\n", df.groupby("Company")["Salary_LPA"].mean().round(1))
print("\nAverage salary by department:\n", df.groupby("Department")["Salary_LPA"].mean().round(1))
print("\nCorrelation with salary:\n", corr["Salary_LPA"].round(2))

# 1. Experience, Age and Performance all show strong positive correlation with salary.
#    Age and Experience carry almost the same information, so we cannot say which
#    matters more. Correlation does not prove cause.
# 2. Amazon has the highest average salary, and Sales is the top-paying department
#    (check the printed tables above).
# 3. Senior level earns clearly more than Mid and Junior (see box and violin plots).
# 4. Caveat: only 32 rows of made-up sample data, and some groups have 1-2 rows,
#    so conclusions are indicative only.