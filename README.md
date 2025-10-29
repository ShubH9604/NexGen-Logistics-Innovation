# 🚚 NexGen Logistics Innovation Challenge

### 🧠 Predictive Delivery Optimization using Data Analytics and Streamlit

---

## 📘 Project Overview

**NexGen Logistics Pvt. Ltd.** is a mid-sized logistics company operating across India with international connections to Singapore, Dubai, Hong Kong, and Bangkok.  
The company manages 200+ monthly shipments across multiple product categories such as Electronics, Fashion, Food & Beverage, Healthcare, Industrial Goods, Books, and Home Goods.

Despite steady growth, the company faces challenges like:
- Delivery delays and inconsistencies  
- Rising operational costs  
- Inefficient fleet utilization  
- Limited innovation in logistics decision-making  

This project aims to **transform NexGen Logistics into a data-driven organization** by leveraging analytics and predictive modeling.

---

## 🎯 Objective

To design and build an **interactive Streamlit application** that:
- Predicts **delivery delays** before they occur  
- Identifies **key contributing factors**  
- Suggests **corrective actions** for improvement  
- Optimizes routes and vehicle assignment  
- Tracks **sustainability metrics (CO₂ emissions)**  

---

## ⚙️ Tech Stack

| Component | Technology Used |
|------------|----------------|
| **Language** | Python |
| **Framework** | Streamlit |
| **Libraries** | Pandas, NumPy, Scikit-learn, Matplotlib, Seaborn, Plotly, Joblib |
| **Visualization** | Plotly & Matplotlib |
| **ML Model** | Random Forest Classifier |
| **Deployment** | Streamlit Local App |

---

## 📊 Datasets Used

| File Name | Description |
|------------|-------------|
| `orders.csv` | Order-level data (IDs, dates, priorities, categories) |
| `delivery_performance.csv` | Carrier data with delivery times and ratings |
| `routes_distance.csv` | Route details like distance, tolls, and traffic |
| `vehicle_fleet.csv` | Vehicle specifications and fuel efficiency |
| `warehouse_inventory.csv` | Inventory and stock management data |
| `customer_feedback.csv` | Customer ratings and feedback |
| `cost_breakdown.csv` | Fuel, maintenance, and operational cost data |

---

## 🧠 Data Analysis Summary

- Cleaned and merged 7 datasets using unique identifiers.  
- Created new metrics such as:
  - `Delay_Duration`
  - `Fuel_Efficiency_Index`
  - `Cost_per_km`
  - `Customer_Satisfaction_Score`
- Built predictive model to forecast delivery delays.
- Achieved **~85% model accuracy** using Random Forest.  
- Derived actionable insights:
  - Bad weather & traffic are top delay causes.
  - Refrigerated trucks have highest delays.
  - Express delivery works best with mid-sized vans.
  - Fuel and tolls contribute heavily to total cost.

---

## 💡 Streamlit App Features

1. 📂 **Multi-dataset Integration** – Upload and merge all 7 CSV files  
2. 📊 **Analytics Dashboard** – Explore insights interactively  
3. 🔮 **Predictive Delay Analyzer** – Forecast order delays using ML  
4. 🚛 **Optimization Suggestions** – Route & vehicle recommendations  
5. ♻️ **Sustainability Tracker** – Track CO₂ emissions per route  
6. 📥 **Download Reports** – Export analytics as CSV or PDF  

---

## 💻 Installation & Setup (Mac)

```bash
# 1. Clone this repository
git clone https://github.com/ShubH9604/NexGen-Logistics-Innovation.git

# 2. Create a virtual environment
python3 -m venv venv
source venv/bin/activate

# 3. Install dependencies
pip install -r requirements.txt

# 4. Run the Streamlit app
streamlit run app.py
