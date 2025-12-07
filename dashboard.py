"""Safety Compliance Dashboard using Streamlit (Free)"""
import streamlit as st
import sqlite3
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from datetime import datetime
import json

st.set_page_config(page_title="PPE Compliance Dashboard", layout="wide")

class ComplianceDashboard:
    def __init__(self, db_path="ppe_detections.db"):
        self.db_path = db_path
    
    def get_data(self, days=30):
        conn = sqlite3.connect(self.db_path)
        query = f"""SELECT * FROM detections 
                    WHERE timestamp >= datetime('now', '-{days} days')
                    ORDER BY timestamp DESC"""
        df = pd.read_sql_query(query, conn)
        conn.close()
        return df
    
    def render(self):
        st.title("🦺 PPE Safety Compliance Dashboard")
        
        st.sidebar.header("Filters")
        days = st.sidebar.slider("Days to show", 1, 90, 30)
        
        df = self.get_data(days)
        
        if df.empty:
            st.warning("No data available. Run inference pipeline first.")
            return
        
        df['timestamp'] = pd.to_datetime(df['timestamp'])
        df['date'] = df['timestamp'].dt.date
        
        # Key Metrics
        st.header("📊 Key Metrics")
        col1, col2, col3, col4 = st.columns(4)
        
        total = len(df)
        compliant = df['compliant'].sum()
        compliance_rate = (compliant / total * 100) if total > 0 else 0
        
        col1.metric("Total Detections", f"{total:,}")
        col2.metric("Compliance Rate", f"{compliance_rate:.1f}%")
        col3.metric("Avg Confidence", f"{df['confidence_avg'].mean():.2f}")
        col4.metric("Avg Processing", f"{df['processing_time'].mean():.3f}s")
        
        # Compliance Trend
        st.header("📈 Compliance Trend")
        daily = df.groupby('date').agg({'compliant': ['sum', 'count']}).reset_index()
        daily.columns = ['date', 'compliant', 'total']
        daily['rate'] = (daily['compliant'] / daily['total'] * 100)
        
        fig = go.Figure()
        fig.add_trace(go.Scatter(x=daily['date'], y=daily['rate'],
                                mode='lines+markers', name='Compliance Rate'))
        fig.add_hline(y=80, line_dash="dash", line_color="red", annotation_text="Target: 80%")
        st.plotly_chart(fig, use_container_width=True)
        
        # Violations
        st.header("⚠️ Recent Violations")
        violations = df[df['compliant'] == 0].head(10)
        for _, row in violations.iterrows():
            with st.expander(f"Violation at {row['timestamp']}"):
                st.write(f"Image: {row['image_path']}")
                st.write(f"Confidence: {row['confidence_avg']:.2f}")

if __name__ == "__main__":
    dashboard = ComplianceDashboard()
    dashboard.render()
