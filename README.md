# Uber Eats Delivery Data Analysis

**Project by:** [Your Name]  
**CRS:** Artificial Intelligence  
**Course:** Data Mining  
**School:** [Your School Name]

---

## Project Title

Optimizing Uber Eats Delivery Through Data Mining

---

## Project Overview

This project analyzes Uber Eats delivery data to uncover the factors influencing delivery times and identify potential delays. By leveraging clustering, association rule mining, and exploratory data analysis (EDA), the project provides actionable insights to improve operational efficiency and customer satisfaction. An interactive dashboard is deployed using Streamlit, enabling real-time data exploration.

---

## Project Scope

- **Data Analysis:**  
  Examine the Uber Eats dataset to identify key trends and relationships that affect delivery performance.
  
- **Customer Insights:**  
  Understand the impact of external factors such as weather conditions, road traffic density, and order type on delivery times.
  
- **Clustering:**  
  Apply K-Means clustering to segment delivery data and highlight distinct operational patterns.
  
- **Association Rule Mining:**  
  Discover frequent associations between delivery attributes to support cross-selling and resource optimization.
  
- **Interactive Dashboard:**  
  Deploy a Streamlit dashboard to visualize data, interact with analysis results, and facilitate decision-making.

---

## Key Preprocessing Steps

- **Data Cleaning & Formatting:**  
  - Handled missing values by filling numerical columns with the median and categorical columns with "Unknown".
  - Removed duplicate records to ensure data accuracy.
  - Parsed and cleaned the `Time_taken(min)` column (e.g., converting "(min) 24" to `24.0`).

- **Date & Time Conversion:**  
  - Converted `Order_Date` to a datetime object.
  - Extracted the order hour from the `Time_Orderd` column for temporal analysis.

- **Categorical Encoding:**  
  - Encoded variables such as `Weatherconditions`, `Road_traffic_density`, `Type_of_order`, and `Type_of_vehicle` for use in clustering and association rule mining.

---

## Analysis Sections

1. **Exploratory Data Analysis (EDA):**  
   - **Visualization:** Histograms, boxplots, and correlation heatmaps to explore the distribution of delivery times and relationships between features.
   
2. **K-Means Clustering:**  
   - Applied K-Means on key features (e.g., delivery time, order hour) and visualized clusters using PCA.
   
3. **Association Rule Mining:**  
   - One-hot encoded selected categorical data.
   - Employed the Apriori algorithm to extract frequent itemsets and generate association rules.
   
4. **User Behaviour Analysis:**  
   - Analyzed order frequency by hour to identify peak periods.
   - Explored additional metrics such as delivery ratings and vehicle conditions.

---

## Streamlit Deployment & Functionality

The analysis is deployed on a Streamlit dashboard, allowing users to:
- View interactive visualizations (e.g., histograms, scatter plots, heatmaps).
- Navigate between different analysis sections (EDA, Association Rules, User Behaviour, and Clustering) using a sidebar.
- Interact with the clustering results and association rules to derive actionable insights.

**Live Streamlit App Link:**  
[Insert Live App URL here]

---

## Repository Structure


---

## References

- **Data Visualization:** Data-to-Viz and Seaborn documentation.
- **Clustering:** scikit-learn's K-Means and PCA documentation.
- **Association Rule Mining:** mlxtend documentation on the Apriori algorithm.
- **Streamlit:** Streamlit official documentation.

---

## Contact

For any questions or feedback, please contact Anish Bammidi at anish.bammidi@gmail.com
