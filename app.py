import streamlit as st
from models.model_loader import load_model
import networkx as nx
import matplotlib.pyplot as plt
import torch
import pickle
import json
import numpy as np

# Configuración inicial de la app
st.title("Recomendador de Canciones para Playlists de Spotify")
st.sidebar.title("Opciones")

# Cargar el modelo
model_options = [
    "LGCN_GAT_3_e64_nodes35300__BPR_hard.pt",
    "LGCN_GAT_3_e64_nodes35300__BPR_random.pt",
    "LGCN_LGC_4_e64_nodes35300__BPR_hard.pt",
    "LGCN_LGC_4_e64_nodes35300__BPR_random.pt",
    "LGCN_SAGE_3_e64_nodes35300__BPR_hard.pt",
    "LGCN_SAGE_3_e64_nodes35300__BPR_random.pt"
]
model_name = st.sidebar.selectbox("Selecciona el modelo a cargar", model_options)

try:
    model = load_model(model_name)
    st.sidebar.success(f"Modelo '{model_name}' cargado correctamente.")
except Exception as e:
    st.sidebar.error(f"Error al cargar el modelo: {e}")
    st.stop()

# Cargar el grafo
graph_path = "./data/30core_first_50.pkl"
try:
    with open(graph_path, "rb") as file:
        graph = pickle.load(file)

    st.sidebar.success("Grafo cargado correctamente.")
except Exception as e:
    st.sidebar.error(f"Error al cargar el grafo: {e}")
    st.stop()

# Enriquecer el grafo con nombres de canciones
json_path = "./spotify_million_playlist_dataset/data/mpd.slice.0-999.json"  # Cambiar según la ubicación del archivo JSON
try:
    with open(json_path, "r") as file:
        data = json.load(file)

    # Crear un mapa de track_uri -> track_name
    track_map = {}
    for playlist in data['playlists']:
        for track in playlist['tracks']:
            track_map[track['track_uri']] = track['track_name']

    # Añadir nombres al grafo
    for node, attrs in graph.nodes(data=True):
        if attrs.get("node_type") == "track":
            track_uri = attrs["name"]
            attrs["track_name"] = track_map.get(track_uri, "Desconocido")

    st.sidebar.success("Grafo enriquecido con nombres de canciones.")
except Exception as e:
    st.sidebar.warning(f"No se pudo enriquecer el grafo: {e}")

# Extraer playlists
playlists = [node for node, attrs in graph.nodes(data=True) if attrs["node_type"] == "playlist"]

# Seleccionar una playlist
selected_playlist = st.sidebar.selectbox("Selecciona una Playlist", playlists)

# Nodos de canciones en la playlist seleccionada
playlist_tracks = [neighbor for neighbor in graph.neighbors(selected_playlist) if graph.nodes[neighbor]["node_type"] == "track"]

# Visualizar canciones en la playlist
st.header("Playlist Seleccionada")
st.write(f"Playlist ID: {selected_playlist}")
st.write("Canciones en la Playlist:")
for track in playlist_tracks:
    track_name = graph.nodes[track].get("track_name", "Desconocido")
    st.write(f"- {track}: {track_name}")

# Predicción de canciones recomendadas
st.header("Recomendaciones de Canciones")
if st.button("Generar Recomendaciones"):
    try:
        # Crear un subgrafo para pruebas si el grafo es muy grande
        small_graph = graph.subgraph(list(graph.nodes)[:10000])  # 1000 nodos para pruebas

        # Crear mapeo de nodos
        node_map = {node: i for i, node in enumerate(small_graph.nodes)}

        # Convertir nodos y aristas en tensores
        node_indices = torch.tensor([node_map[n] for n in small_graph.nodes], dtype=torch.long)
        edge_indices = torch.tensor(
            [[node_map[src], node_map[dst]] for src, dst in small_graph.edges],
            dtype=torch.long
        ).t()

        # Generar predicciones con torch.no_grad()
        with torch.no_grad():
            output = model(node_indices, edge_indices).detach().numpy()

        # Recomendar canciones
        playlist_track_set = set(playlist_tracks)  # Convertir a conjunto para comparar eficientemente
        track_scores = {
            track: np.linalg.norm(output[node_map[track]])  # Calcular la norma del vector
            for track in small_graph.nodes
            if small_graph.nodes[track]["node_type"] == "track" and track not in playlist_track_set
        }
        recommended_tracks = sorted(track_scores, key=track_scores.get, reverse=True)[:10]

        st.write("Canciones Recomendadas:")
        for track in recommended_tracks:
            track_name = small_graph.nodes[track].get("track_name", "Desconocido")
            st.write(f"- {track_name}")
    except Exception as e:
        st.error(f"Error al generar recomendaciones: {e}")

# Visualización del grafo
st.header("Visualización de la Playlist y Recomendaciones")
plt.figure(figsize=(12, 12))

# Subgrafo con la playlist y sus canciones
subgraph = graph.subgraph([selected_playlist] + playlist_tracks)

# Dibujar el grafo con nombres
pos = nx.spring_layout(subgraph)
labels = {node: graph.nodes[node].get("track_name", node) for node in subgraph.nodes}
nx.draw(
    subgraph,
    pos,
    with_labels=True,
    labels=labels,
    node_size=700,
    node_color="skyblue"
)
st.pyplot(plt)
