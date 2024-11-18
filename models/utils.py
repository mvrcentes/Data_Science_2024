import requests
import pickle
from tqdm import tqdm

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

def enrich_graph_with_spotify_data(graph_path, access_token):
    """Enriquece el grafo cargado con los nombres correctos de las canciones."""
    with open(graph_path, "rb") as file:
        graph = pickle.load(file)
    for node, attrs in tqdm(graph.nodes(data=True)):
        if attrs.get("node_type") == "track":
            track_uri = attrs.get("name", "")
            if track_uri and "spotify:track:" in track_uri:
                track_name = get_track_name(track_uri, access_token)
                attrs["track_name"] = track_name
            else:
                attrs["track_name"] = "Desconocido"
    enriched_graph_path = graph_path.replace(".pkl", "_enriched.pkl")
    with open(enriched_graph_path, "wb") as file:
        pickle.dump(graph, file)
    return enriched_graph_path
