import streamlit as st
import duckdb
import pandas as pd
import plotly.express as px
from pathlib import Path
import os
import numpy as np
import networkx as nx
from sklearn.metrics.pairwise import cosine_similarity
from pyvis.network import Network
import streamlit.components.v1 as components
from matplotlib import cm, colors as mcolors
import matplotlib.pyplot as plt
import tempfile

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
        # Try connecting with explicit READ_ONLY config
        # This is often more robust than read_only=True kwarg in some versions
        conn = duckdb.connect(str(DB_PATH), config={'access_mode': 'READ_ONLY'})
        
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
        import traceback
        st.code(traceback.format_exc())
        return None
        st.error(f"Error loading data: {e}")
        return None

def fetch_embeddings():
    try:
        conn = duckdb.connect(str(DB_PATH), config={'access_mode': 'READ_ONLY'})
        df = conn.execute("""
            SELECT filename, embedding
            FROM files_index
            WHERE embedding IS NOT NULL
        """).fetchdf()
        conn.close()
        return df
    except Exception as e:
        st.error(f"Error fetching embeddings: {e}")
        return pd.DataFrame(columns=["filename", "embedding"])

def sim_to_width(s: float, threshold: float) -> float:
    return float(1 + 7 * max(0.0, (s - threshold)) / max(1e-9, (1.0 - threshold)))

def sim_to_hex(s: float, threshold: float):
    cmap = cm.get_cmap('viridis')
    x = max(0.0, (s - threshold)) / max(1e-9, (1.0 - threshold))
    return mcolors.rgb2hex(cmap(x))

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

    # Similarity Graph
    st.subheader("Similarity Graph (Embeddings)")
    df_embeddings = fetch_embeddings()
    if df_embeddings is None or df_embeddings.empty:
        st.info("No embeddings found. Try indexing some files first.")
        return

    threshold = st.slider("Edge threshold (cosine)", min_value=0.5, max_value=0.95, value=0.6, step=0.001)
    try:
        df_e = df_embeddings.dropna(subset=['embedding']).reset_index(drop=True)
        names = df_e['filename'].astype(str).to_numpy()
        embedding_list = [np.asarray(v, dtype=np.float32) for v in df_e['embedding']]
        embeddings = np.stack(embedding_list)  # (N, D)
        similarity = cosine_similarity(embeddings)  # (N, N)

        # Build graph
        G = nx.Graph()
        G.add_nodes_from(names)
        N = len(names)
        for i in range(N):
            for j in range(i + 1, N):
                score = float(similarity[i, j])
                if score >= threshold:
                    G.add_edge(
                        names[i], names[j],
                        weight=float(score),
                        value=float(sim_to_width(score, threshold)),
                        color=str(sim_to_hex(score, threshold)),
                        title=f"sim={score:.3f}"
                    )

        # Render with PyVis
        net = Network(height="600px", width="100%", bgcolor="#222222", font_color="white")
        net.from_nx(G)
        net.toggle_physics(True)
        net.toggle_drag_nodes(True)

        with tempfile.NamedTemporaryFile(suffix=".html", delete=False) as tmp:
            net.write_html(tmp.name, notebook=False, open_browser=False)
            html = Path(tmp.name).read_text()

        # Inject a CSS/HTML legend overlay directly into the PyVis HTML
        def viridis_stops_hex(n=20, t=threshold):
            cm_ = cm.get_cmap('viridis')
            xs = np.linspace(0.0, 1.0, n)
            # Map [0,1] → [t,1]
            vals = t + xs * max(0.0, 1.0 - t)
            return [mcolors.rgb2hex(cm_(x)) for x in xs], vals

        stops, _ = viridis_stops_hex(24, threshold)
        gradient_css = ", ".join(f"{c} {int(i*100/(len(stops)-1))}%" for i, c in enumerate(stops))
        legend_html = f"""
        <style>
            .legend-overlay {{
                position: fixed; right: 12px; bottom: 12px; z-index: 9999;
                background: rgba(30,30,30,0.85); color: #eee; padding: 10px 12px;
                border-radius: 8px; font-family: system-ui, -apple-system, sans-serif;
                box-shadow: 0 4px 12px rgba(0,0,0,0.4);
            }}
            .legend-bar {{
                width: 220px; height: 12px; border-radius: 6px; margin-top: 6px; margin-bottom: 4px;
                background: linear-gradient(90deg, {gradient_css});
            }}
            .legend-labels {{ display: flex; justify-content: space-between; font-size: 11px; opacity: 0.9; }}
            .legend-title {{ font-size: 12px; font-weight: 600; }}
        </style>
        <div class="legend-overlay">
            <div class="legend-title">Cosine similarity</div>
            <div class="legend-bar"></div>
            <div class="legend-labels"><span>{threshold:.2f}</span><span>1.00</span></div>
        </div>
        """
        # Insert before </body>
        if "</body>" in html:
            html = html.replace("</body>", legend_html + "</body>")
        components.html(html, height=600, scrolling=True)

        st.caption("Hover edges to see sim=..., thickness and color scale with similarity.")
    except Exception as e:
        st.error(f"Graph rendering failed: {e}")

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
