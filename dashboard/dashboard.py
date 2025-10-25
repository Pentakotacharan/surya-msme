import streamlit as st
import pandas as pd
import sqlite3
import plotly.express as px

st.set_page_config(page_title="Chat Analytics", layout="wide")

conn = sqlite3.connect("../backend/chat_history.db")
df = pd.read_sql_query("SELECT * FROM chats", conn)

st.title("📊 Chatbot Usage Dashboard")

st.metric("Total Chats", len(df))
st.metric("Unique Languages", df['lang'].nunique())

col1, col2 = st.columns(2)
with col1:
    st.subheader("Language Distribution")
    lang_count = df['lang'].value_counts().reset_index()
    st.plotly_chart(px.pie(lang_count, names='index', values='lang'))

with col2:
    st.subheader("Chat Volume Over Time")
    df['timestamp'] = pd.to_datetime(df['timestamp'])
    df['date'] = df['timestamp'].dt.date
    date_count = df.groupby('date').size().reset_index(name='count')
    st.plotly_chart(px.line(date_count, x='date', y='count'))

