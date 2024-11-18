import streamlit as st
from models.model_loader import load_model
import networkx as nx
import matplotlib.pyplot as plt
import torch
import pickle
import json
import numpy as np
import requests
import time
import matplotlib.pyplot as plt
import re

# Dictionary to store execution times for each model
if 'model_times' not in st.session_state:
    st.session_state.model_times = {}

# Función para obtener el nombre de la canción desde la API de Spotify
def get_track_name(track_uri, access_token):
    """Obtiene el nombre de la canción a partir de su URI usando la API de Spotify."""
    track_id = track_uri.split(':')[-1]
    url = f"https://api.spotify.com/v1/tracks/{track_id}"
    headers = {"Authorization": f"Bearer {access_token}"}
    response = requests.get(url, headers=headers)

    if response.status_code == 200:
        track_info = response.json()
        return track_info.get('name', "Desconocido"), track_info.get('artists', [{}])[0].get('name', "Desconocido")
    else:
        return "Desconocido", "Desconocido"

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
access_token = "BQBq0BuGdhFZGJkJKtKkShm5rbcNx8KX1C1U_wjX5WQ8Z72jl6XjXJiXpTDKM_Yizyn_BSBZKQCt3VLdVcRfbpVAYe7KQbvXnQ2NN1sfiLT9YWh_u8w"

try:
    with open(json_path, "r") as file:
        data = json.load(file)

    # Crear un mapa de track_uri -> track_name y playlist_id -> playlist_name
    track_map = {}
    track_artist_map = {}
    playlist_name_map = {}
    for playlist in data['playlists']:
        playlist_name = playlist['name'].strip()
        if playlist_name:  # Excluir playlists sin nombre
            playlist_name_map[f"playlist_{playlist['pid']}"] = playlist_name
        for track in playlist['tracks']:
            track_map[track['track_uri']] = track['track_name']
            track_artist_map[track['track_uri']] = track['artist_name']

    # Añadir nombres y autores al grafo
    for node, attrs in graph.nodes(data=True):
        if attrs.get("node_type") == "track":
            track_uri = attrs["name"]
            attrs["track_name"] = track_map.get(track_uri, "Desconocido")
            attrs["artist_name"] = track_artist_map.get(track_uri, "Desconocido")

    st.sidebar.success("Grafo enriquecido con nombres de canciones.")
except Exception as e:
    st.sidebar.warning(f"No se pudo enriquecer el grafo: {e}")

# Extraer playlists y mapear sus nombres desde el JSON
playlists = [node for node, attrs in graph.nodes(data=True) if attrs["node_type"] == "playlist"]
playlist_names = {playlist: playlist_name_map.get(playlist) for playlist in playlists if playlist_name_map.get(playlist)}

# Seleccionar una playlist
selected_playlist = st.sidebar.selectbox("Selecciona una Playlist", playlist_names.keys(), format_func=lambda x: playlist_names[x])

# Verificar si se ha seleccionado una playlist
if selected_playlist:
    # Nodos de canciones en la playlist seleccionada
    playlist_tracks = [neighbor for neighbor in graph.neighbors(selected_playlist) if graph.nodes[neighbor]["node_type"] == "track"]

    # Consultar canciones desconocidas en la playlist
    for track in playlist_tracks:
        track_name = graph.nodes[track].get("track_name", "Desconocido")
        if track_name == "Desconocido" and "spotify:track:" in graph.nodes[track]["name"]:
            track_name, artist_name = get_track_name(graph.nodes[track]["name"], access_token)
            graph.nodes[track]["track_name"] = track_name
            graph.nodes[track]["artist_name"] = artist_name

    # Visualizar canciones en la playlist
    st.header(f"Playlist Seleccionada: {playlist_names[selected_playlist]}")
    st.write("Canciones en la Playlist:")
    for track in playlist_tracks:
        track_name = graph.nodes[track].get("track_name", "Desconocido")
        artist_name = graph.nodes[track].get("artist_name", "Desconocido")
        st.write(f"- {track_name} - {artist_name}")

    # Predicción de canciones recomendadas
    st.header("Recomendaciones de Canciones")
    # Add a section to generate recommendations and record execution times
    if st.button("Generar Recomendaciones"):
        try:
            small_graph = graph.subgraph(list(graph.nodes)[:10000])  # Subgrafo más manejable
            node_map = {node: i for i, node in enumerate(small_graph.nodes)}
            node_indices = torch.tensor([node_map[n] for n in small_graph.nodes], dtype=torch.long)
            edge_indices = torch.tensor(
                [[node_map[src], node_map[dst]] for src, dst in small_graph.edges],
                dtype=torch.long
            ).t()

            # Start timing the recommendation generation
            start_time = time.time()
            
            # Generar predicciones con torch.no_grad()
            with torch.no_grad():
                output = model(node_indices, edge_indices).detach().numpy()

            # Record the time taken for the current model
            elapsed_time = time.time() - start_time
            st.session_state.model_times[model_name] = elapsed_time

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
                    track_name, artist_name = get_track_name(track, access_token)
                    small_graph.nodes[track]["track_name"] = track_name
                    small_graph.nodes[track]["artist_name"] = artist_name
            st.write("Canciones Recomendadas:")
            for track in recommended_tracks:
                track_name = small_graph.nodes[track].get("track_name", "Desconocido")
                artist_name = small_graph.nodes[track].get("artist_name", "Desconocido")
                st.write(f"- {track_name} - {artist_name}")

            # Plot the comparative bar graph if there are at least two models
            if len(st.session_state.model_times) > 1:
                st.header("Comparativa de Tiempos de Recomendación entre Modelos")
                fig, ax = plt.subplots()
                ax.bar([re.sub(r'^(LGCN_(?:GAT|LGC|SAGE))_.*?(BPR_(?:hard|random)\.pt)$', r'\1_\2', name) for name in st.session_state.model_times.keys()], st.session_state.model_times.values(), color='skyblue')
                ax.set_xlabel("Modelos")
                ax.set_ylabel("Tiempo (segundos)")
                ax.set_title("Comparativa de Tiempos de Recomendación")
                ax.tick_params(axis='x', rotation=45)  # Rotate x labels for readability
                st.pyplot(fig)

        except Exception as e:
            st.error(f"Error al generar recomendaciones: {e}")

    # Visualización del grafo
    st.header("Visualización de la Playlist y Recomendaciones")
    plt.figure(figsize=(12, 12))
    subgraph = graph.subgraph([selected_playlist] + playlist_tracks)
    pos = nx.spring_layout(subgraph)
    labels = {
        node: f"{graph.nodes[node].get('track_name', node).replace('$', '\\$')}\n{graph.nodes[node].get('artist_name', '').replace('$', '\\$')}"
        for node in subgraph.nodes
    }
    nx.draw(
        subgraph,
        pos,
        with_labels=True,
        labels=labels,
        node_size=700,
        node_color="skyblue"
    )
    st.pyplot(plt)
