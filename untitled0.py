import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import re
from sklearn.preprocessing import LabelEncoder
from sklearn.cluster import KMeans
from mlxtend.frequent_patterns import apriori, association_rules

# Set plot style
sns.set_theme(style="whitegrid")
plt.rcParams["figure.figsize"] = (10, 6)

# ====== STREAMLIT UI SETUP ======
st.title("Uber Eats Delivery Data Analysis")
st.sidebar.header("Navigation")

# Load dataset from GitHub
filename = ("uber-eats-deliveries.csv")
df = pd.read_csv(filename)

# ====== DATA PREPROCESSING ======
st.subheader("Data Overview")
st.write(df.head())
st.write(df.info())

# Handle missing values
num_cols = df.select_dtypes(include=[np.number]).columns
df[num_cols] = df[num_cols].fillna(df[num_cols].median())
df = df.fillna("Unknown")

# Drop duplicate rows
df.drop_duplicates(inplace=True)

# Preprocess 'Time_taken(min)' column
def parse_time_taken(value):
    cleaned = re.sub(r"[^\d.]+", "", str(value))
    return float(cleaned) if cleaned != "" else np.nan

df["Time_taken(min)"] = df["Time_taken(min)"].apply(parse_time_taken)

# Convert Order_Date to datetime
df["Order_Date"] = pd.to_datetime(df["Order_Date"], format='%d-%m-%Y', errors='coerce')

# Extract hour from Time_Orderd
df["Order_Hour"] = pd.to_datetime(df["Time_Orderd"], format='%H:%M:%S', errors='coerce').dt.hour

# Encode categorical columns
categorical_cols = ['Weatherconditions', 'Road_traffic_density', 'Type_of_order', 'Type_of_vehicle']
existing_cat_cols = [col for col in categorical_cols if col in df.columns]
encoder = LabelEncoder()
for col in existing_cat_cols:
    df[col] = encoder.fit_transform(df[col])

# Sidebar options for different sections
analysis_type = st.sidebar.selectbox("Select Analysis Type", ["EDA", "Association Rules", "User Behaviour", "K-Means Clustering"])

if analysis_type == "EDA":
    # ====== EXPLORATORY DATA ANALYSIS (EDA) ======
    st.subheader("Exploratory Data Analysis")
    
    def plot_histogram():
        fig, ax = plt.subplots()
        sns.histplot(df["Time_taken(min)"], kde=True, bins=30, ax=ax)
        ax.set_title("Distribution of Delivery Time (minutes)")
        ax.set_xlabel("Time Taken (min)")
        st.pyplot(fig)
    
    def plot_correlation_heatmap():
        fig, ax = plt.subplots()
        sns.heatmap(df.select_dtypes(include=[np.number]).corr(), annot=True, cmap='coolwarm', fmt=".2f", ax=ax)
        ax.set_title("Correlation Heatmap of Numeric Features")
        st.pyplot(fig)
    
    plot_option = st.sidebar.selectbox("Select EDA Analysis", ["Distribution of Delivery Time", "Correlation Heatmap"])
    if plot_option == "Distribution of Delivery Time":
        plot_histogram()
    elif plot_option == "Correlation Heatmap":
        plot_correlation_heatmap()

elif analysis_type == "Association Rules":
    st.subheader("Association Rules Analysis")
    
    # Convert dataset into transactional format for Apriori
    df_apriori = df[['Type_of_order', 'Type_of_vehicle', 'Weatherconditions']].astype(str)
    df_apriori = df_apriori.apply(lambda x: x.str.strip())  # Ensure no extra spaces
    df_apriori = pd.get_dummies(df_apriori)  # Convert categorical data to one-hot encoding
    df_apriori = df_apriori.astype(bool)  # Ensure the values are boolean (True/False)
    
    # Apply Apriori algorithm
    frequent_itemsets = apriori(df_apriori, min_support=0.05, use_colnames=True)
    rules = association_rules(frequent_itemsets, metric="lift", min_threshold=1.0)
    st.write("Frequent Itemsets:", frequent_itemsets)
    st.write("Association Rules:", rules)

elif analysis_type == "User Behaviour":
    st.subheader("User Behaviour Analysis")
    
    # User Behaviour Analysis: Preferred Order Time
    fig, ax = plt.subplots()
    sns.countplot(x="Order_Hour", data=df, ax=ax)
    ax.set_title("Preferred Order Time of Users")
    st.pyplot(fig)

elif analysis_type == "K-Means Clustering":
    st.subheader("K-Means Clustering Analysis")
    
    # Select features for clustering
    features = df[['Time_taken(min)', 'Order_Hour']].dropna()
    kmeans = KMeans(n_clusters=3, random_state=42)
    features['Cluster'] = kmeans.fit_predict(features)
    
    # Plot K-Means clusters
    fig, ax = plt.subplots()
    sns.scatterplot(x=features['Order_Hour'], y=features['Time_taken(min)'], hue=features['Cluster'], palette='viridis', ax=ax)
    ax.set_title("K-Means Clustering of Delivery Time and Order Hour")
    st.pyplot(fig)
