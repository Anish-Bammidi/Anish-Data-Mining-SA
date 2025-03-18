# Import necessary libraries
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import re

# Set plot style
sns.set_theme(style="whitegrid")
plt.rcParams["figure.figsize"] = (10, 6)

# ====== UPLOAD DATASET ======
df = pd.read_csv("uber-eats-deliveries.csv")
df = pd.read_csv(filename)

# ====== DISPLAY INITIAL DATA OVERVIEW ======
print("Initial Data Overview:")
print(df.info())
print(df.head())

# ====== DATA PREPROCESSING ======

# 1. Handle missing values
# For numerical columns, fill missing values with the median; for categorical, use "Unknown"
num_cols = df.select_dtypes(include=[np.number]).columns
df[num_cols] = df[num_cols].fillna(df[num_cols].median())
df = df.fillna("Unknown")

# 2. Drop duplicate rows
df.drop_duplicates(inplace=True)

# 3. Preprocess 'Time_taken(min)' column
# This column contains strings like "(min) 24". Remove non-numeric characters and convert to float.
def parse_time_taken(value):
    # Remove non-digit and non-decimal characters
    cleaned = re.sub(r"[^\d.]+", "", str(value))
    return float(cleaned) if cleaned != "" else np.nan

df["Time_taken(min)"] = df["Time_taken(min)"].apply(parse_time_taken)

# 4. Convert Order_Date to datetime (assuming format 'dd-mm-yyyy')
df["Order_Date"] = pd.to_datetime(df["Order_Date"], format='%d-%m-%Y', errors='coerce')

# 5. Extract hour from Time_Orderd (assumes format 'HH:MM:SS')
df["Order_Hour"] = pd.to_datetime(df["Time_Orderd"], format='%H:%M:%S', errors='coerce').dt.hour

# 6. Encode categorical columns if needed.
# Here, we encode the following columns if they exist: Weatherconditions, Road_traffic_density, Type_of_order, Type_of_vehicle.
# (You can adjust or skip encoding if you want to see the original text in your plots.)
from sklearn.preprocessing import LabelEncoder
categorical_cols = ['Weatherconditions', 'Road_traffic_density', 'Type_of_order', 'Type_of_vehicle']
existing_cat_cols = [col for col in categorical_cols if col in df.columns]
encoder = LabelEncoder()
for col in existing_cat_cols:
    df[col] = encoder.fit_transform(df[col])

# Display preprocessed data overview
print("\nData After Preprocessing:")
print(df.info())
print(df.head())

# ====== SAVE THE CLEANED DATASET (OPTIONAL) ======
df.to_csv("uber_eats_cleaned.csv", index=False)
print("Cleaned dataset saved as 'uber_eats_cleaned.csv'.")

# ====== EXPLORATORY DATA ANALYSIS (EDA) ======

# 1. Distribution of Delivery Time
plt.figure()
sns.histplot(df["Time_taken(min)"], kde=True, bins=30)
plt.title("Distribution of Delivery Time (minutes)")
plt.xlabel("Time Taken (min)")
plt.ylabel("Frequency")
plt.show()

# 2. Boxplot of Delivery Time
plt.figure()
sns.boxplot(y=df["Time_taken(min)"])
plt.title("Boxplot of Delivery Time (minutes)")
plt.ylabel("Time Taken (min)")
plt.show()

# 3. Delivery Time vs. Weather Conditions
if "Weatherconditions" in df.columns:
    plt.figure()
    sns.boxplot(x="Weatherconditions", y="Time_taken(min)", data=df)
    plt.title("Delivery Time by Weather Conditions")
    plt.xlabel("Weather Conditions (Encoded)")
    plt.ylabel("Time Taken (min)")
    plt.show()

# 4. Delivery Time vs. Road Traffic Density
if "Road_traffic_density" in df.columns:
    plt.figure()
    sns.boxplot(x="Road_traffic_density", y="Time_taken(min)", data=df)
    plt.title("Delivery Time by Road Traffic Density")
    plt.xlabel("Road Traffic Density (Encoded)")
    plt.ylabel("Time Taken (min)")
    plt.show()

# 5. Orders by City
plt.figure()
sns.countplot(x="City", data=df, order=df["City"].value_counts().index)
plt.title("Number of Orders by City")
plt.xlabel("City")
plt.ylabel("Number of Orders")
plt.xticks(rotation=45)
plt.show()

# 6. Orders by Time of Day (Order_Hour)
plt.figure()
sns.countplot(x="Order_Hour", data=df)
plt.title("Number of Orders by Hour of Day")
plt.xlabel("Hour of Day")
plt.ylabel("Number of Orders")
plt.show()

# 7. Delivery Time vs. Multiple Deliveries
plt.figure()
sns.boxplot(x="multiple_deliveries", y="Time_taken(min)", data=df)
plt.title("Delivery Time by Number of Deliveries")
plt.xlabel("Multiple Deliveries")
plt.ylabel("Time Taken (min)")
plt.show()

# 8. Correlation Heatmap (for numeric features)
numeric_features = df.select_dtypes(include=[np.number])
plt.figure()
sns.heatmap(numeric_features.corr(), annot=True, cmap='coolwarm', fmt=".2f")
plt.title("Correlation Heatmap of Numeric Features")
plt.show()

from google.colab import drive
drive.mount('/content/drive')
