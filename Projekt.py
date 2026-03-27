import streamlit as st
import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestRegressor
from supabase import create_client, Client
import os

# --- 1. Cloud Database Setup (Supabase) ---
# You will get these URLs and Keys from your free Supabase account
SUPABASE_URL = "YOUR_SUPABASE_URL"
SUPABASE_KEY = "YOUR_SUPABASE_ANON_KEY"

# Initialize Cloud Connection
@st.cache_resource
def init_connection():
    return create_client(SUPABASE_URL, SUPABASE_KEY)

supabase: Client = init_connection()

# --- 2. App UI & Data Fetching ---
st.set_page_config(page_title="Cloud AI Finance", layout="wide")
st.title("☁️ Advanced Cloud-Based AI Finance Tracker")

# Fetch data from Cloud PostgreSQL
def load_data():
    response = supabase.table("expenses").select("*").execute()
    if response.data:
        return pd.DataFrame(response.data)
    else:
        # Fallback if cloud is empty
        return pd.DataFrame(columns=['id', 'month', 'category', 'amount'])

df = load_data()

# --- 3. Sidebar: Add New Expenses to the Cloud ---
st.sidebar.header("➕ Add New Expense")
new_month = st.sidebar.number_input("Month (e.g., 7)", min_value=1, max_value=12, step=1)
new_category = st.sidebar.selectbox("Category", ["Food", "Transport", "Utilities", "Entertainment"])
new_amount = st.sidebar.number_input("Amount (₹)", min_value=0)

if st.sidebar.button("Save to Cloud Database"):
    # Insert directly into Supabase
    new_data = {"month": new_month, "category": new_category, "amount": new_amount}
    supabase.table("expenses").insert(new_data).execute()
    st.sidebar.success("Saved successfully! Refreshing data...")
    st.rerun() # Refresh app to show new data

# --- 4. Main Dashboard ---
if not df.empty:
    st.subheader("📊 Live Cloud Data")
    
    # Group by month for ML training
    monthly_data = df.groupby('month')['amount'].sum().reset_index()
    monthly_data.rename(columns={'amount': 'total_expenses'}, inplace=True)
    
    col1, col2 = st.columns(2)
    with col1:
        st.dataframe(df, use_container_width=True)
    with col2:
        st.bar_chart(monthly_data.set_index('month'))

    # --- 5. Advanced Machine Learning (Random Forest) ---
    if len(monthly_data) >= 3: # Need at least a few months to predict
        st.subheader("🤖 Advanced AI Forecasting (Random Forest)")
        
        X = monthly_data[['month']]
        y = monthly_data['total_expenses']
        
        # Using Random Forest for non-linear, advanced predictions
        model = RandomForestRegressor(n_estimators=100, random_state=42)
        model.fit(X, y)
        
        next_month_num = monthly_data['month'].max() + 1
        predicted_expense = model.predict([[next_month_num]])[0]
        
        st.success(f"**Predicted Total for Month {next_month_num}:** ₹{predicted_expense:.2f}")
        st.info("This prediction is powered by an ensemble learning method (Random Forest), which builds multiple decision trees to ensure higher accuracy.")
    else:
        st.warning("Add at least 3 months of data to unlock AI predictions!")
else:
    st.info("No data in the cloud yet. Use the sidebar to add some expenses!")