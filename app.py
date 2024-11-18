import streamlit as st
from models.model_loader import load_model
import networkx as nx
import matplotlib.pyplot as plt
import torch
import pickle
import json
import numpy as np
import requests

# Función para obtener el nombre de la canción desde la API de Spotify
def get_track_name(track_uri, access_token):
    """Obtiene el nombre de la canción a partir de su URI usando la API de Spotify."""
    track_id = track_uri.split(':')[-1]
    url = f"https://api.spotify.com/v1/tracks/{track_id}"
    headers = {"Authorization": f"Bearer {access_token}"}
    response = requests.get(url, headers=headers)

    if response.status_code == 200:
        track_info = response.json()
        return track_info.get('name', "Desconocido")
    else:
        return "Desconocido"

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
access_token = "BQAQU6MOlVm5sLE5sHC_hkchsj_gO_YDKRtSHPoB6l473aFU9bou7MmsH10x5UtrhSTKysrcvwtHBBofGiG6nwZRgUN-ZZIdUvS7qwwlR7exLIsIRhs"  # Cambia esto por un token válido

try:
    with open(json_path, "r") as file:
        data = json.load(file)

    # Crear un mapa de track_uri -> track_name y playlist_id -> playlist_name
    track_map = {}
    playlist_name_map = {}
    for playlist in data['playlists']:
        playlist_name_map[f"playlist_{playlist['pid']}"] = playlist['name'].strip()
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

# Extraer playlists y mapear sus nombres desde el JSON
playlists = [node for node, attrs in graph.nodes(data=True) if attrs["node_type"] == "playlist"]
playlist_names = {playlist: playlist_name_map.get(playlist, "Sin Nombre") for playlist in playlists}

# Seleccionar una playlist
selected_playlist = st.sidebar.selectbox("Selecciona una Playlist", playlists, format_func=lambda x: playlist_names[x])

# Verificar si se ha seleccionado una playlist
if selected_playlist:
    # Nodos de canciones en la playlist seleccionada
    playlist_tracks = [neighbor for neighbor in graph.neighbors(selected_playlist) if graph.nodes[neighbor]["node_type"] == "track"]

    # Consultar canciones desconocidas en la playlist
    for track in playlist_tracks:
        track_name = graph.nodes[track].get("track_name", "Desconocido")
        if track_name == "Desconocido" and "spotify:track:" in graph.nodes[track]["name"]:
            graph.nodes[track]["track_name"] = get_track_name(graph.nodes[track]["name"], access_token)

    # Visualizar canciones en la playlist
    st.header(f"Playlist Seleccionada: {playlist_names[selected_playlist]}")
    st.write(f"Playlist ID: {selected_playlist}")
    st.write("Canciones en la Playlist:")
    for track in playlist_tracks:
        track_name = graph.nodes[track].get("track_name", "Desconocido")
        st.write(f"- {track}: {track_name}")

    # Predicción de canciones recomendadas
    st.header("Recomendaciones de Canciones")
    if st.button("Generar Recomendaciones"):
        try:
            small_graph = graph.subgraph(list(graph.nodes)[:10000])  # Subgrafo más manejable

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
            playlist_track_set = set(playlist_tracks)
            track_scores = {
                track: np.linalg.norm(output[node_map[track]])
                for track in small_graph.nodes
                if small_graph.nodes[track]["node_type"] == "track" and track not in playlist_track_set
            }
            recommended_tracks = sorted(track_scores, key=track_scores.get, reverse=True)[:10]

            # Consultar nombres de canciones desconocidas en recomendaciones
            for track in recommended_tracks:
                track_name = small_graph.nodes[track].get("track_name", "Desconocido")
                if track_name == "Desconocido" and "spotify:track:" in track:
                    small_graph.nodes[track]["track_name"] = get_track_name(track, access_token)

            st.write("Canciones Recomendadas:")
            for track in recommended_tracks:
                track_name = small_graph.nodes[track].get("track_name", "Desconocido")
                st.write(f"- {track_name}")
        except Exception as e:
            st.error(f"Error al generar recomendaciones: {e}")

    # Visualización del grafo
    st.header("Visualización de la Playlist y Recomendaciones")
    plt.figure(figsize=(12, 12))
    subgraph = graph.subgraph([selected_playlist] + playlist_tracks)
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
