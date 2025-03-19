import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import re
from sklearn.preprocessing import LabelEncoder

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

# ====== EXPLORATORY DATA ANALYSIS (EDA) ======
st.subheader("Exploratory Data Analysis")

def plot_histogram():
    fig, ax = plt.subplots()
    sns.histplot(df["Time_taken(min)"], kde=True, bins=30, ax=ax)
    ax.set_title("Distribution of Delivery Time (minutes)")
    ax.set_xlabel("Time Taken (min)")
    st.pyplot(fig)

def plot_boxplot():
    fig, ax = plt.subplots()
    sns.boxplot(y=df["Time_taken(min)"], ax=ax)
    ax.set_title("Boxplot of Delivery Time (minutes)")
    st.pyplot(fig)

def plot_weather_vs_time():
    if "Weatherconditions" in df.columns:
        fig, ax = plt.subplots()
        sns.boxplot(x="Weatherconditions", y="Time_taken(min)", data=df, ax=ax)
        ax.set_title("Delivery Time by Weather Conditions")
        st.pyplot(fig)

def plot_traffic_vs_time():
    if "Road_traffic_density" in df.columns:
        fig, ax = plt.subplots()
        sns.boxplot(x="Road_traffic_density", y="Time_taken(min)", data=df, ax=ax)
        ax.set_title("Delivery Time by Road Traffic Density")
        st.pyplot(fig)

def plot_orders_by_city():
    fig, ax = plt.subplots()
    sns.countplot(x="City", data=df, order=df["City"].value_counts().index, ax=ax)
    ax.set_title("Number of Orders by City")
    plt.xticks(rotation=45)
    st.pyplot(fig)

def plot_orders_by_hour():
    fig, ax = plt.subplots()
    sns.countplot(x="Order_Hour", data=df, ax=ax)
    ax.set_title("Number of Orders by Hour of Day")
    st.pyplot(fig)

def plot_delivery_vs_multiple():
    fig, ax = plt.subplots()
    sns.boxplot(x="multiple_deliveries", y="Time_taken(min)", data=df, ax=ax)
    ax.set_title("Delivery Time by Number of Deliveries")
    st.pyplot(fig)

def plot_correlation_heatmap():
    fig, ax = plt.subplots()
    sns.heatmap(df.select_dtypes(include=[np.number]).corr(), annot=True, cmap='coolwarm', fmt=".2f", ax=ax)
    ax.set_title("Correlation Heatmap of Numeric Features")
    st.pyplot(fig)

# Sidebar options for different plots
plot_option = st.sidebar.selectbox("Select Analysis", [
    "Distribution of Delivery Time", "Boxplot of Delivery Time", "Delivery Time vs. Weather Conditions",
    "Delivery Time vs. Road Traffic Density", "Orders by City", "Orders by Hour", 
    "Delivery Time vs. Multiple Deliveries", "Correlation Heatmap"
])

if plot_option == "Distribution of Delivery Time":
    plot_histogram()
elif plot_option == "Boxplot of Delivery Time":
    plot_boxplot()
elif plot_option == "Delivery Time vs. Weather Conditions":
    plot_weather_vs_time()
elif plot_option == "Delivery Time vs. Road Traffic Density":
    plot_traffic_vs_time()
elif plot_option == "Orders by City":
    plot_orders_by_city()
elif plot_option == "Orders by Hour":
    plot_orders_by_hour()
elif plot_option == "Delivery Time vs. Multiple Deliveries":
    plot_delivery_vs_multiple()
elif plot_option == "Correlation Heatmap":
    plot_correlation_heatmap()
