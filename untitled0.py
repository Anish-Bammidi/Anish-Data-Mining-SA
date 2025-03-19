# ------------------------------
# 1. Import Libraries & Load Data
# ------------------------------
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import re
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.decomposition import PCA
from sklearn.cluster import KMeans
from mlxtend.frequent_patterns import apriori, association_rules

# Set plot style
sns.set_theme(style="whitegrid")
plt.rcParams["figure.figsize"] = (10, 6)

# ====== LOAD DATASET ======
# Make sure 'uber_eats_data.csv' is in your working directory.
file_path = "uber-eats-deliveries.csv"
df = pd.read_csv(file_path)
# ------------------------------
# 2. Data Preprocessing
# ------------------------------

# Display initial data overview
print("Initial Data Overview:")
print(df.info())
print(df.head())

# Handle missing values:
# For numerical columns, fill missing values with the median; for categorical, use "Unknown"
num_cols = df.select_dtypes(include=[np.number]).columns
df[num_cols] = df[num_cols].fillna(df[num_cols].median())
df = df.fillna("Unknown")

# Drop duplicate rows
df.drop_duplicates(inplace=True)

# Preprocess 'Time_taken(min)' column (e.g., "(min) 24" -> 24)
def parse_time_taken(value):
    cleaned = re.sub(r"[^\d.]+", "", str(value))
    return float(cleaned) if cleaned != "" else np.nan

df["Time_taken(min)"] = df["Time_taken(min)"].apply(parse_time_taken)

# Convert Order_Date to datetime (assuming format 'dd-mm-yyyy')
df["Order_Date"] = pd.to_datetime(df["Order_Date"], format='%d-%m-%Y', errors='coerce')

# Extract hour from Time_Orderd (format 'HH:MM:SS')
df["Order_Hour"] = pd.to_datetime(df["Time_Orderd"], format='%H:%M:%S', errors='coerce').dt.hour

# Convert Delivery_person_Ratings and Delivery_person_Age to numeric
df["Delivery_person_Ratings"] = pd.to_numeric(df["Delivery_person_Ratings"], errors='coerce')
df["Delivery_person_Age"] = pd.to_numeric(df["Delivery_person_Age"], errors='coerce')

# Encode selected categorical columns for plotting purposes
categorical_cols = ['Weatherconditions', 'Road_traffic_density', 'Type_of_order', 'Type_of_vehicle']
existing_cat_cols = [col for col in categorical_cols if col in df.columns]
encoder = LabelEncoder()
for col in existing_cat_cols:
    df[col] = encoder.fit_transform(df[col])

print("\nData After Preprocessing:")
print(df.info())
print(df.head())

# Save cleaned dataset (optional)
df.to_csv("uber_eats_cleaned.csv", index=False)
print("Cleaned dataset saved as 'uber_eats_cleaned.csv'.")

# ------------------------------
# 3. Exploratory Data Analysis (EDA)
# ------------------------------

# Distribution of Delivery Time
plt.figure()
sns.histplot(df["Time_taken(min)"], kde=True, bins=30)
plt.title("Distribution of Delivery Time (minutes)")
plt.xlabel("Time Taken (min)")
plt.ylabel("Frequency")
plt.show()

# Boxplot of Delivery Time
plt.figure()
sns.boxplot(y=df["Time_taken(min)"])
plt.title("Boxplot of Delivery Time (minutes)")
plt.ylabel("Time Taken (min)")
plt.show()

# Delivery Time vs. Weather Conditions
if "Weatherconditions" in df.columns:
    plt.figure()
    sns.boxplot(x="Weatherconditions", y="Time_taken(min)", data=df)
    plt.title("Delivery Time by Weather Conditions (Encoded)")
    plt.xlabel("Weather Conditions")
    plt.ylabel("Time Taken (min)")
    plt.show()

# Delivery Time vs. Road Traffic Density
if "Road_traffic_density" in df.columns:
    plt.figure()
    sns.boxplot(x="Road_traffic_density", y="Time_taken(min)", data=df)
    plt.title("Delivery Time by Road Traffic Density (Encoded)")
    plt.xlabel("Road Traffic Density")
    plt.ylabel("Time Taken (min)")
    plt.show()

# Orders by City
plt.figure()
sns.countplot(x="City", data=df, order=df["City"].value_counts().index)
plt.title("Number of Orders by City")
plt.xlabel("City")
plt.ylabel("Number of Orders")
plt.xticks(rotation=45)
plt.show()

# Orders by Hour of Day
plt.figure()
sns.countplot(x="Order_Hour", data=df)
plt.title("Number of Orders by Hour of Day")
plt.xlabel("Hour of Day")
plt.ylabel("Number of Orders")
plt.show()

# Delivery Time vs. Multiple Deliveries
plt.figure()
sns.boxplot(x="multiple_deliveries", y="Time_taken(min)", data=df)
plt.title("Delivery Time by Number of Deliveries")
plt.xlabel("Multiple Deliveries")
plt.ylabel("Time Taken (min)")
plt.show()

# Correlation Heatmap (numeric features)
numeric_features = df.select_dtypes(include=[np.number])
plt.figure()
sns.heatmap(numeric_features.corr(), annot=True, cmap='coolwarm', fmt=".2f")
plt.title("Correlation Heatmap of Numeric Features")
plt.show()

# ------------------------------
# 4. K-Means Clustering & PCA Visualization
# ------------------------------

# Select features for clustering (adjust as needed)
cluster_features = ["Time_taken(min)", "Delivery_person_Ratings", "Order_Hour", "Vehicle_condition"]

# Drop rows with missing values in these features
df_cluster = df[cluster_features].dropna()

# Scale features
scaler_cluster = StandardScaler()
X_scaled = scaler_cluster.fit_transform(df_cluster)

# Apply K-Means clustering (using k=4 as an example)
kmeans = KMeans(n_clusters=4, random_state=42)
clusters = kmeans.fit_predict(X_scaled)
df_cluster["Cluster"] = clusters

# Use PCA to reduce to 2 dimensions for visualization
pca = PCA(n_components=2, random_state=42)
X_pca = pca.fit_transform(X_scaled)
df_cluster["PCA1"] = X_pca[:, 0]
df_cluster["PCA2"] = X_pca[:, 1]

# Plot clusters
plt.figure()
sns.scatterplot(x="PCA1", y="PCA2", hue="Cluster", data=df_cluster, palette="viridis")
plt.title("K-Means Clustering (PCA Visualization)")
plt.xlabel("PCA Component 1")
plt.ylabel("PCA Component 2")
plt.legend(title="Cluster")
plt.show()

# ------------------------------
# 5. Association Rule Mining
# ------------------------------

# Select categorical columns for association rules.
cols_for_assoc = ['Weatherconditions', 'Road_traffic_density', 'Festival', 'Type_of_order', 'Type_of_vehicle', 'City']
df_assoc = df[cols_for_assoc].copy()

# One-hot encode the selected columns.
df_assoc_encoded = pd.get_dummies(df_assoc, prefix=cols_for_assoc)

# Generate frequent itemsets using Apriori (adjust min_support as needed)
frequent_itemsets = apriori(df_assoc_encoded, min_support=0.05, use_colnames=True)

# Generate association rules based on a confidence threshold (adjust as needed)
rules = association_rules(frequent_itemsets, metric="confidence", min_threshold=0.6)
print("\nAssociation Rules (first 5):")
print(rules.head())

# ------------------------------
# 6. User Behavior Analysis
# ------------------------------

# a. Average Delivery Person Ratings by City
avg_rating_city = df.groupby("City")["Delivery_person_Ratings"].mean().reset_index()
plt.figure()
sns.barplot(x="City", y="Delivery_person_Ratings", data=avg_rating_city)
plt.title("Average Delivery Person Ratings by City")
plt.xlabel("City")
plt.ylabel("Average Rating")
plt.xticks(rotation=45)
plt.show()

# b. Average Delivery Time by Hour of Day
avg_time_hour = df.groupby("Order_Hour")["Time_taken(min)"].mean().reset_index()
plt.figure()
sns.lineplot(x="Order_Hour", y="Time_taken(min)", data=avg_time_hour, marker="o")
plt.title("Average Delivery Time by Hour of Day")
plt.xlabel("Hour of Day")
plt.ylabel("Average Delivery Time (min)")
plt.show()
