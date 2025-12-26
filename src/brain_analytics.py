import streamlit as st
import duckdb
import pandas as pd
import plotly.express as px
from pathlib import Path
import os

# Page config
st.set_page_config(
    page_title="Brain Analytics",
    page_icon="🧠",
    layout="wide"
)

# Constants
DB_PATH = Path(__file__).parent.parent / "brain.duckdb"

def load_data():
    if not DB_PATH.exists():
        st.error(f"Database not found at {DB_PATH}")
        return None
        
    try:
        conn = duckdb.connect(str(DB_PATH), read_only=True)
        
        # Get basic stats
        files_df = conn.execute("""
            SELECT 
                filename,
                path,
                created_at,
                full_text, 
                length(full_text) as text_length,
                embedding IS NOT NULL as is_indexed
            FROM files_index
        """).fetchdf()
        
        conn.close()
        return files_df
    except Exception as e:
        st.error(f"Error loading data: {e}")
        return None

def main():
    st.title("🧠 Brain Indexer Analytics")
    
    df = load_data()
    
    if df is None or df.empty:
        st.warning("No data found in the index.")
        return

    # Add file extension column
    df['extension'] = df['filename'].apply(lambda x: os.path.splitext(x)[1].lower() if os.path.splitext(x)[1] else 'No Ext')

    # Top Level Metrics
    col1, col2, col3 = st.columns(3)
    
    with col1:
        st.metric("Total Files", len(df))
    
    with col2:
        indexed_count = df['is_indexed'].sum()
        st.metric("Indexed Files", int(indexed_count))
        
    with col3:
        avg_size = df['text_length'].mean()
        st.metric("Avg Text Length (chars)", f"{int(avg_size):,}")

    st.divider()

    # Charts
    col_chart1, col_chart2 = st.columns(2)

    with col_chart1:
        st.subheader("File Type Distribution")
        type_counts = df['extension'].value_counts().reset_index()
        type_counts.columns = ['Extension', 'Count']
        fig_pie = px.pie(type_counts, values='Count', names='Extension', hole=0.4)
        st.plotly_chart(fig_pie, use_container_width=True)

    with col_chart2:
        st.subheader("Text Content Size Distribution")
        fig_hist = px.histogram(df, x="text_length", nbins=20, title="Character Count per File")
        st.plotly_chart(fig_hist, use_container_width=True)

    st.divider()

    # Recent Files Table
    st.subheader("Indexed Files")
    st.dataframe(
        df[['filename', 'extension', 'text_length', 'path']],
        use_container_width=True,
        hide_index=True
    )

if __name__ == "__main__":
    main()
