# Anish-Data-Mining-SA
# Uber Eats Delivery Data Analysis

## Project Overview

This project is a data analysis and visualization application built with Streamlit that analyzes Uber Eats delivery data. The app provides insights into delivery performance, user behavior, and hidden associations within the data through Exploratory Data Analysis (EDA), Association Rules Mining, and K-Means Clustering.

## Features

- **Exploratory Data Analysis (EDA):**  
  Visualize the distribution of delivery times, examine correlations between numeric features, and analyze various aspects of the delivery data.

- **Association Rules Mining:**  
  Generate and display association rules to uncover relationships between different delivery attributes such as order type, vehicle type, and weather conditions.

- **User Behavior Analysis:**  
  Identify peak ordering hours and understand customer ordering patterns.

- **K-Means Clustering:**  
  Apply K-Means clustering to group deliveries based on key features like delivery time and order hour, and visualize clusters using a scatterplot.

## Prerequisites

- Python 3.x
- Required Python packages:
  - `streamlit`
  - `pandas`
  - `numpy`
  - `matplotlib`
  - `seaborn`
  - `scikit-learn`
  - `mlxtend`

- Dataset file: `uber-eats-deliveries.csv` (ensure this file is in your working directory)

## Installation

1. **Clone the repository:**

   ```bash
   git clone https://github.com/YourUsername/UberEatsDeliveryAnalysis.git
   cd UberEatsDeliveryAnalysis
