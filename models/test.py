import pickle

# Ruta al archivo .pkl
pkl_path = "./data/30core_first_50.pkl"

# Cargar el archivo como un grafo
try:
    with open(pkl_path, "rb") as file:
        graph = pickle.load(file)

    # Verificar información de los nodos (contenido completo)
    print("Contenido completo de los nodos en el grafo:")
    for i, (node, attrs) in enumerate(graph.nodes(data=True)):
        print(f"\nNodo {i + 1}: {node}")
        for key, value in attrs.items():
            print(f"  {key}: {value}")
        if i >= 35299:  # Limitar a los primeros 20 nodos para evitar demasiada salida
            break

    print(f"\nTotal de nodos en el grafo: {graph.number_of_nodes()}")
except Exception as e:
    print(f"Error al cargar el archivo: {e}")


