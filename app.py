"""
CARDIOGraph - Interactive Knowledge Graph Explorer

A Streamlit demo for exploring drug-protein-pathway relationships
and viewing GNN-predicted associations for cardiotoxicity research.
"""

import streamlit as st
import pandas as pd
import networkx as nx
import pickle
from pathlib import Path
import sys

# Add src to path
sys.path.insert(0, str(Path(__file__).parent / 'src'))

# Page config
st.set_page_config(
    page_title="CARDIOGraph Explorer",
    page_icon="💊",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS for better readability
st.markdown("""
<style>
    .stApp {
        background: #0e1117;
    }
    .main-header {
        font-size: 2.5rem;
        font-weight: 700;
        background: linear-gradient(90deg, #FF6B6B, #4ECDC4);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        text-align: center;
        padding: 1rem 0;
    }
    .sub-header {
        color: #aaaaaa;
        text-align: center;
        margin-bottom: 2rem;
        font-size: 1.1rem;
    }
    .node-card {
        background: #1e2130;
        border-radius: 10px;
        padding: 1rem;
        margin: 0.5rem 0;
        border-left: 4px solid #4ECDC4;
        color: #ffffff;
    }
    .node-card strong {
        color: #ffffff;
        font-size: 1.1rem;
    }
    .node-card small {
        color: #cccccc;
        font-size: 0.9rem;
    }
    .drug-node { border-left-color: #FF6B6B; }
    .protein-node { border-left-color: #4ECDC4; }
    .pathway-node { border-left-color: #FFEAA7; }
    .prediction-card {
        background: #1a2634;
        border-radius: 10px;
        padding: 1rem;
        margin: 0.5rem 0;
        border: 1px solid #4ECDC4;
        color: #ffffff;
    }
    .prediction-card strong {
        color: #ffffff;
        font-size: 1.05rem;
    }
    .prediction-card small {
        color: #bbbbbb;
    }
    .metric-box {
        background: #1e2130;
        border-radius: 10px;
        padding: 1.5rem;
        text-align: center;
        color: #ffffff;
    }
    .metric-box h3 {
        color: #ffffff;
        font-size: 1.8rem;
        margin: 0.5rem 0;
    }
    .metric-box p {
        color: #aaaaaa;
        font-size: 1rem;
        margin: 0;
    }
    /* Sidebar styling */
    .css-1d391kg, [data-testid="stSidebar"] {
        background: #151922;
    }
    /* Make all text more readable */
    .stMarkdown, .stText, p, span, label {
        color: #e0e0e0 !important;
    }
    h1, h2, h3, h4 {
        color: #ffffff !important;
    }
</style>
""", unsafe_allow_html=True)


@st.cache_data
def load_data():
    """Load all datasets and build the knowledge graph."""
    data_dir = Path(__file__).parent / 'data' / 'raw'
    
    # Load drug-protein data
    drug_protein_path = data_dir / 'clean_drug_protein_targets.csv'
    df_drug_protein = pd.read_csv(drug_protein_path) if drug_protein_path.exists() else pd.DataFrame()
    
    # Load pathway data
    pathway_path = data_dir / 'Pathway_Cardiovascular_Filtered.csv'
    df_pathway = pd.read_csv(pathway_path) if pathway_path.exists() else pd.DataFrame()
    
    # Load PPI data
    ppi_path = data_dir / 'biogrid_ppi_human.csv'
    df_ppi = pd.read_csv(ppi_path) if ppi_path.exists() else pd.DataFrame()
    
    # Build graph
    G = nx.DiGraph()
    
    # Add drug-protein edges
    if not df_drug_protein.empty:
        for _, row in df_drug_protein.iterrows():
            drug = str(row.get('drug_name', '')).strip()
            protein = str(row.get('uniprot_id', '')).strip()
            protein_name = str(row.get('uniprot_name', '')).strip()
            
            if drug and protein:
                G.add_node(drug, node_type='Drug', label=drug)
                G.add_node(protein, node_type='Protein', label=protein_name or protein)
                G.add_edge(drug, protein, relationship='TARGETS')
    
    # Add pathway edges
    if not df_pathway.empty:
        for _, row in df_pathway.iterrows():
            protein = str(row.get('uniprot_id', '')).strip()
            if ':' in protein:
                protein = protein.split(':')[-1].strip()
            pathway = str(row.get('pathway_name', '')).strip()
            
            if protein and pathway:
                if protein not in G:
                    G.add_node(protein, node_type='Protein', label=protein)
                G.add_node(pathway, node_type='Pathway', label=pathway)
                G.add_edge(protein, pathway, relationship='INVOLVED_IN')
    
    # Add PPI edges (limited for performance)
    if not df_ppi.empty:
        ppi_count = 0
        for _, row in df_ppi.iterrows():
            p1 = str(row.get('uniprot_a', '')).strip()
            p2 = str(row.get('uniprot_b', '')).strip()
            
            if p1 and p2 and p1 != '-' and p2 != '-':
                if p1 in G or p2 in G:  # Only add if connected to existing nodes
                    if p1 not in G:
                        G.add_node(p1, node_type='Protein', label=p1)
                    if p2 not in G:
                        G.add_node(p2, node_type='Protein', label=p2)
                    G.add_edge(p1, p2, relationship='INTERACTS_WITH')
                    ppi_count += 1
                    if ppi_count >= 10000:
                        break
    
    # Get drug list
    drugs = sorted([n for n, d in G.nodes(data=True) if d.get('node_type') == 'Drug'])
    
    return G, drugs, df_drug_protein, df_pathway


@st.cache_data
def load_embeddings():
    """Load pre-trained GNN embeddings if available."""
    emb_path = Path(__file__).parent / 'output' / 'dvgae_embeddings.pkl'
    if emb_path.exists():
        with open(emb_path, 'rb') as f:
            return pickle.load(f)
    return None


def get_subgraph(G, center_node, depth=2):
    """Get subgraph around a center node."""
    if center_node not in G:
        return nx.DiGraph()
    
    # BFS to get neighbors up to depth
    nodes = {center_node}
    frontier = {center_node}
    
    for _ in range(depth):
        next_frontier = set()
        for node in frontier:
            next_frontier.update(G.predecessors(node))
            next_frontier.update(G.successors(node))
        nodes.update(next_frontier)
        frontier = next_frontier
    
    return G.subgraph(nodes).copy()


def visualize_subgraph(subgraph, center_node):
    """Create a visualization of the subgraph using Streamlit."""
    import matplotlib.pyplot as plt
    import matplotlib.patches as mpatches
    
    if subgraph.number_of_nodes() == 0:
        return None
    
    fig, ax = plt.subplots(figsize=(14, 9))
    fig.patch.set_facecolor('#0e1117')
    ax.set_facecolor('#0e1117')
    
    # Layout
    pos = nx.spring_layout(subgraph, k=2.5, iterations=50, seed=42)
    
    # Node colors - brighter for visibility
    colors = []
    sizes = []
    for node in subgraph.nodes():
        node_type = subgraph.nodes[node].get('node_type', 'Unknown')
        if node_type == 'Drug':
            colors.append('#FF7B7B')  # Brighter coral
        elif node_type == 'Protein':
            colors.append('#5EDDD4')  # Brighter teal
        elif node_type == 'Pathway':
            colors.append('#FFE066')  # Brighter yellow
        else:
            colors.append('#aaaaaa')
        
        # Larger size for center node
        sizes.append(1200 if node == center_node else 600)
    
    # Draw edges with better visibility
    nx.draw_networkx_edges(subgraph, pos, alpha=0.6, edge_color='#555555', 
                          arrows=True, arrowsize=20, width=1.5, ax=ax,
                          connectionstyle="arc3,rad=0.1")
    
    # Draw nodes with edge outline for better visibility
    nx.draw_networkx_nodes(subgraph, pos, node_color=colors, node_size=sizes, 
                          ax=ax, edgecolors='#ffffff', linewidths=1.5)
    
    # Labels - truncate long names, with black outline for readability
    labels = {}
    for node in subgraph.nodes():
        label = subgraph.nodes[node].get('label', node)
        if len(str(label)) > 18:
            label = str(label)[:15] + '...'
        labels[node] = label
    
    # Draw labels with better visibility
    nx.draw_networkx_labels(subgraph, pos, labels, font_size=9, 
                           font_color='#ffffff', font_weight='bold', ax=ax)
    
    # Legend with better styling
    legend_elements = [
        mpatches.Patch(facecolor='#FF7B7B', edgecolor='white', label='Drug'),
        mpatches.Patch(facecolor='#5EDDD4', edgecolor='white', label='Protein'),
        mpatches.Patch(facecolor='#FFE066', edgecolor='white', label='Pathway')
    ]
    legend = ax.legend(handles=legend_elements, loc='upper left', 
                       facecolor='#1e2130', labelcolor='white', 
                       framealpha=0.95, fontsize=11, edgecolor='#444444')
    legend.get_frame().set_linewidth(1.5)
    
    ax.set_title(f'Knowledge Graph: {center_node}', fontsize=16, color='white', 
                pad=20, fontweight='bold')
    ax.axis('off')
    
    plt.tight_layout()
    return fig


def main():
    # Header
    st.markdown('<h1 class="main-header">💊 CARDIOGraph Explorer</h1>', unsafe_allow_html=True)
    st.markdown('<p class="sub-header">Explore Drug-Protein-Pathway Relationships for Cardiotoxicity Research</p>', unsafe_allow_html=True)
    
    # Load data
    with st.spinner('Loading knowledge graph...'):
        G, drugs, df_drug_protein, df_pathway = load_data()
        embeddings = load_embeddings()
    
    # Sidebar
    st.sidebar.header("🔍 Search")
    
    # Drug search
    search_query = st.sidebar.text_input("Search for a drug:", placeholder="e.g., Aspirin, Metformin...")
    
    # Filter drugs based on search
    if search_query:
        filtered_drugs = [d for d in drugs if search_query.lower() in d.lower()]
    else:
        filtered_drugs = drugs[:100]  # Show first 100 by default
    
    selected_drug = st.sidebar.selectbox(
        "Select a drug:",
        options=filtered_drugs if filtered_drugs else ["No matches found"],
        index=0
    )
    
    # Depth slider
    depth = st.sidebar.slider("Exploration depth:", 1, 3, 2)
    
    # Stats in sidebar
    st.sidebar.markdown("---")
    st.sidebar.header("📊 Database Stats")
    
    num_drugs = len([n for n, d in G.nodes(data=True) if d.get('node_type') == 'Drug'])
    num_proteins = len([n for n, d in G.nodes(data=True) if d.get('node_type') == 'Protein'])
    num_pathways = len([n for n, d in G.nodes(data=True) if d.get('node_type') == 'Pathway'])
    
    st.sidebar.metric("Drugs", f"{num_drugs:,}")
    st.sidebar.metric("Proteins", f"{num_proteins:,}")
    st.sidebar.metric("Pathways", f"{num_pathways:,}")
    st.sidebar.metric("Total Edges", f"{G.number_of_edges():,}")
    
    if embeddings:
        st.sidebar.success("✅ GNN embeddings loaded")
    else:
        st.sidebar.warning("⚠️ Run GNN training for predictions")
    
    # Main content
    if selected_drug and selected_drug != "No matches found" and selected_drug in G:
        col1, col2 = st.columns([2, 1])
        
        with col1:
            st.header(f"💊 {selected_drug}")
            
            # Get subgraph
            subgraph = get_subgraph(G, selected_drug, depth=depth)
            
            # Visualize
            fig = visualize_subgraph(subgraph, selected_drug)
            if fig:
                st.pyplot(fig)
            
        with col2:
            # Drug info
            st.subheader("🎯 Protein Targets")
            
            targets = list(G.successors(selected_drug))
            protein_targets = [t for t in targets if G.nodes[t].get('node_type') == 'Protein']
            
            if protein_targets:
                for protein in protein_targets[:10]:
                    protein_label = G.nodes[protein].get('label', protein)
                    st.markdown(f"""
                    <div class="node-card protein-node">
                        <strong>{protein}</strong><br>
                        <small>{protein_label}</small>
                    </div>
                    """, unsafe_allow_html=True)
                
                if len(protein_targets) > 10:
                    st.info(f"... and {len(protein_targets) - 10} more targets")
            else:
                st.info("No protein targets found")
            
            # Pathways
            st.subheader("🧬 Related Pathways")
            
            pathways = set()
            for protein in protein_targets:
                for neighbor in G.successors(protein):
                    if G.nodes[neighbor].get('node_type') == 'Pathway':
                        pathways.add(neighbor)
            
            if pathways:
                for pathway in list(pathways)[:5]:
                    st.markdown(f"""
                    <div class="node-card pathway-node">
                        <strong>{pathway}</strong>
                    </div>
                    """, unsafe_allow_html=True)
            else:
                st.info("No pathways found")
        
        # Predictions section
        st.markdown("---")
        st.header("🔮 GNN Predictions")
        
        if embeddings and selected_drug in embeddings:
            st.success("Predictions available based on trained GNN model")
            
            # Find similar drugs based on embedding distance
            import numpy as np
            drug_emb = embeddings[selected_drug]
            
            similarities = []
            for node, emb in embeddings.items():
                if node != selected_drug and G.nodes.get(node, {}).get('node_type') == 'Protein':
                    # Check if not already connected
                    if not G.has_edge(selected_drug, node):
                        dist = np.linalg.norm(drug_emb - emb)
                        similarity = 1 / (1 + dist)
                        similarities.append((node, similarity))
            
            # Sort by similarity
            similarities.sort(key=lambda x: x[1], reverse=True)
            
            if similarities:
                st.subheader("Predicted Novel Protein Targets")
                
                pred_cols = st.columns(3)
                for i, (protein, score) in enumerate(similarities[:6]):
                    with pred_cols[i % 3]:
                        protein_label = G.nodes.get(protein, {}).get('label', protein)
                        st.markdown(f"""
                        <div class="prediction-card">
                            <strong>{protein}</strong><br>
                            <small>{protein_label[:30]}...</small><br>
                            <span style="color: #4ECDC4;">Score: {score:.3f}</span>
                        </div>
                        """, unsafe_allow_html=True)
        else:
            st.info("""
            💡 **To see predictions:**
            1. Run the GNN training notebook first
            2. This will generate embeddings for link prediction
            3. Reload this page to see novel drug-target predictions
            """)
        
        # Path exploration
        st.markdown("---")
        st.header("🔗 Explore Connections")
        
        col1, col2 = st.columns(2)
        with col1:
            target_node = st.text_input("Find path to:", placeholder="Enter protein ID or pathway name")
        with col2:
            max_length = st.slider("Max path length:", 2, 5, 3)
        
        if target_node and target_node in G:
            try:
                paths = list(nx.all_simple_paths(G.to_undirected(), selected_drug, target_node, cutoff=max_length))
                if paths:
                    st.success(f"Found {len(paths)} path(s)!")
                    for i, path in enumerate(paths[:3]):
                        st.write(f"**Path {i+1}:** {' → '.join(path)}")
                else:
                    st.warning("No paths found within the specified length")
            except nx.NetworkXError:
                st.error("Could not find path between these nodes")
    
    else:
        # Welcome message
        st.info("👆 Select a drug from the sidebar to explore its connections")
        
        # Quick stats
        st.header("📈 Knowledge Graph Overview")
        
        col1, col2, col3, col4 = st.columns(4)
        
        with col1:
            st.markdown("""
            <div class="metric-box">
                <h2 style="color: #FF7B7B; font-size: 2.5rem;">💊</h2>
                <h3 style="color: #ffffff; font-size: 2rem;">{:,}</h3>
                <p style="color: #cccccc; font-size: 1.1rem;">Drugs</p>
            </div>
            """.format(num_drugs), unsafe_allow_html=True)
        
        with col2:
            st.markdown("""
            <div class="metric-box">
                <h2 style="color: #5EDDD4; font-size: 2.5rem;">🧬</h2>
                <h3 style="color: #ffffff; font-size: 2rem;">{:,}</h3>
                <p style="color: #cccccc; font-size: 1.1rem;">Proteins</p>
            </div>
            """.format(num_proteins), unsafe_allow_html=True)
        
        with col3:
            st.markdown("""
            <div class="metric-box">
                <h2 style="color: #FFE066; font-size: 2.5rem;">🔬</h2>
                <h3 style="color: #ffffff; font-size: 2rem;">{:,}</h3>
                <p style="color: #cccccc; font-size: 1.1rem;">Pathways</p>
            </div>
            """.format(num_pathways), unsafe_allow_html=True)
        
        with col4:
            st.markdown("""
            <div class="metric-box">
                <h2 style="color: #DDA0DD; font-size: 2.5rem;">🔗</h2>
                <h3 style="color: #ffffff; font-size: 2rem;">{:,}</h3>
                <p style="color: #cccccc; font-size: 1.1rem;">Connections</p>
            </div>
            """.format(G.number_of_edges()), unsafe_allow_html=True)
        
        # Sample drugs
        st.header("🔥 Popular Drugs to Explore")
        
        sample_drugs = ['Aspirin', 'Metformin', 'Warfarin', 'Lisinopril', 'Atorvastatin', 
                       'Metoprolol', 'Amlodipine', 'Digoxin', 'Amiodarone', 'Furosemide']
        
        available_samples = [d for d in sample_drugs if d in drugs]
        
        if available_samples:
            cols = st.columns(5)
            for i, drug in enumerate(available_samples[:10]):
                with cols[i % 5]:
                    if st.button(f"💊 {drug}", key=f"btn_{drug}"):
                        st.session_state['selected_drug'] = drug
                        st.rerun()


if __name__ == "__main__":
    main()

