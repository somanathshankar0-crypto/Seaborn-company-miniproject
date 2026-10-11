
# ============================================================
# LAPTOP LISTINGS ANALYSIS - INDIA
# Mini-project using Pandas, Matplotlib and Seaborn
# ============================================================

import pandas as p
import seaborn as s
import matplotlib.pyplot as m

# ------------------------------------------------------------
# 1. LOAD THE DATASET
# ------------------------------------------------------------

df = p.read_csv("laptop_data.csv")

# Display the first five rows
print("\nFIRST FIVE ROWS")
print(df.head())

# Check dataset size and column data types
print("\nDATASET INFORMATION")
print(df.info())

# Display summary statistics
print("\nSUMMARY STATISTICS")
print(df.describe())

# ------------------------------------------------------------
# 2. MISSING-VALUE ANALYSIS
# ------------------------------------------------------------

print("\nMISSING VALUES IN EACH COLUMN")
print(df.isna().sum())

print("\nMISSING VALUES AS PERCENTAGES")
print((df.isna().mean() * 100).round(2))

# ------------------------------------------------------------
# 3. NUMBER OF LAPTOP MODELS PER BRAND
# Countplot: compares the number of listings for each brand
# ------------------------------------------------------------

m.figure(figsize=(9, 5))
s.countplot(data=df, x="Brand", hue="Brand", palette="rainbow", legend=False)
m.xlabel("Brand")
m.ylabel("Number of Laptop Listings")
m.title("Number of Laptop Listings by Brand")
m.xticks(rotation=45)
m.tight_layout()
m.show()

# ------------------------------------------------------------
# 4. PRICE DISTRIBUTION
# Histogram: shows the distribution of laptop prices
# ------------------------------------------------------------

price_df = df.dropna(subset=["Price_INR"])

m.figure(figsize=(8, 5))
s.histplot(data=price_df, x="Price_INR", bins=8, kde=True)
m.xlabel("Price (INR)")
m.ylabel("Number of Laptops")
m.title("Distribution of Laptop Prices")
m.tight_layout()
m.show()

# ------------------------------------------------------------
# 5. AVERAGE PRICE BY BRAND
# Barplot: compares average prices across brands
# ------------------------------------------------------------

m.figure(figsize=(9, 5))
s.barplot(data=price_df, x="Brand", y="Price_INR", hue="Brand",
          palette="rainbow", legend=False)
m.xlabel("Brand")
m.ylabel("Average Price (INR)")
m.title("Average Laptop Price by Brand")
m.xticks(rotation=45)
m.tight_layout()
m.show()

# ------------------------------------------------------------
# 6. PRICE SUMMARY BY BRAND
# Calculates maximum, minimum and average price
# ------------------------------------------------------------

result = df.groupby("Brand")["Price_INR"].agg(["max", "min", "mean"])

print("\nPRICE SUMMARY BY BRAND")
print(result.round(2))

# Find the cheapest and most expensive listings
if not price_df.empty:
    print("\nCHEAPEST LAPTOP")
    print(price_df.loc[price_df["Price_INR"].idxmin()])

    print("\nMOST EXPENSIVE LAPTOP")
    print(price_df.loc[price_df["Price_INR"].idxmax()])

# ------------------------------------------------------------
# 7. RAM VS PRICE
# Scatterplot: investigates the relationship between RAM
# and laptop price
# ------------------------------------------------------------

ram_df = df.dropna(subset=["RAM_GB", "Price_INR"])

s.jointplot(data=ram_df, x="RAM_GB", y="Price_INR", kind="scatter")
m.show()

# ------------------------------------------------------------
# 8. STORAGE VS PRICE
# Scatterplot: investigates the relationship between storage
# capacity and laptop price
# ------------------------------------------------------------

storage_price_df = df.dropna(subset=["Storage_GB", "Price_INR"])

s.jointplot(
    data=storage_price_df,
    x="Storage_GB",
    y="Price_INR",
    kind="scatter"
)
m.show()

# ------------------------------------------------------------
# 9. DISPLAY SIZE VS PRICE
# Scatterplot: compares display size with laptop price
# ------------------------------------------------------------

display_df = df.dropna(subset=["Display_Inches", "Price_INR"])

m.figure(figsize=(8, 5))
s.scatterplot(data=display_df, x="Display_Inches", y="Price_INR")
m.xlabel("Display Size (inches)")
m.ylabel("Price (INR)")
m.title("Display Size vs Laptop Price")
m.tight_layout()
m.show()

# ------------------------------------------------------------
# 10. PRICE DISTRIBUTION BY BRAND
# Boxplot: compares prices and potential outliers
# ------------------------------------------------------------

m.figure(figsize=(9, 5))
s.boxplot(data=price_df, x="Brand", y="Price_INR", hue="Brand",
          palette="rainbow", legend=False)
m.xlabel("Brand")
m.ylabel("Price (INR)")
m.title("Price Distribution by Brand")
m.xticks(rotation=45)
m.tight_layout()
m.show()

# ------------------------------------------------------------
# 11. CUSTOMER RATING VS PRICE
# Jointplot: investigates whether ratings and prices
# appear to be associated
# ------------------------------------------------------------

rating_price_df = df.dropna(subset=["Rating", "Price_INR"])

s.jointplot(
    data=rating_price_df,
    x="Rating",
    y="Price_INR",
    kind="scatter"
)
m.show()

# ------------------------------------------------------------
# 12. SELLER VS PRICE
# Barplot: compares average listing prices by seller
# ------------------------------------------------------------

seller_df = df.dropna(subset=["Seller", "Price_INR"])

m.figure(figsize=(11, 5))
s.barplot(data=seller_df, x="Seller", y="Price_INR", hue="Seller",
          palette="rainbow", legend=False)
m.xlabel("Seller")
m.ylabel("Average Price (INR)")
m.title("Average Laptop Price by Seller")
m.xticks(rotation=45, ha="right")
m.tight_layout()
m.show()

# ------------------------------------------------------------
# 13. BATTERY CAPACITY BY BRAND
# Stripplot: shows individual battery-capacity observations
# ------------------------------------------------------------

battery_df = df.dropna(subset=["Brand", "Battery_Wh"])

m.figure(figsize=(9, 5))
s.stripplot(data=battery_df, x="Brand", y="Battery_Wh", jitter=True, size=8)
m.xlabel("Brand")
m.ylabel("Battery Capacity (Wh)")
m.title("Battery Capacity by Brand")
m.xticks(rotation=45)
m.tight_layout()
m.show()

# ------------------------------------------------------------
# 14. STORAGE CAPACITY BY BRAND
# Stripplot: displays individual storage capacities
# ------------------------------------------------------------

storage_df = df.dropna(subset=["Brand", "Storage_GB"])

m.figure(figsize=(9, 5))
s.stripplot(data=storage_df, x="Brand", y="Storage_GB", jitter=True, size=8)
m.xlabel("Brand")
m.ylabel("Storage Capacity (GB)")
m.title("Laptop Storage by Brand")
m.xticks(rotation=45)
m.tight_layout()
m.show()

# ------------------------------------------------------------
# 15. CORRELATION MATRIX
# Examines pairwise linear associations between numerical
# variables. Correlation does NOT prove causation.
# ------------------------------------------------------------

numeric_columns = [
    "Price_INR",
    "RAM_GB",
    "Storage_GB",
    "Display_Inches",
    "Battery_Wh",
    "Rating",
    "Review_Count"
]

numeric_df = df[numeric_columns]

correlation = numeric_df.corr()

print("\nCORRELATION MATRIX")
print(correlation.round(2))

m.figure(figsize=(9, 7))
s.heatmap(correlation, annot=True, cmap="coolwarm", fmt=".2f", center=0)
m.title("Correlation Matrix of Laptop Features")
m.tight_layout()
m.show()

# ------------------------------------------------------------
# 16. FINAL CONCLUSION
# Write conclusions based on the actual plots and values.
# Do not assume that higher price guarantees better quality.
# ------------------------------------------------------------

print("\nPROJECT CONCLUSION")
print("""
1. Laptop prices vary across brands and configurations.
2. RAM and storage alone do not fully explain laptop prices.
3. Other factors, including processors, graphics, displays and
   build quality, may also contribute to price differences.
4. Customer ratings and review counts do not directly prove
   service quality.
5. The dataset is small and contains missing values, so the
   results describe this sample rather than the entire market.
""")
