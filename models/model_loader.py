import torch
from torch.nn import Linear
from torch_geometric.nn import GATConv  # Graph Attention Network Layer


class GATGraphModel(torch.nn.Module):
    def __init__(self, input_dim, hidden_dim, output_dim, num_heads=5):
        super(GATGraphModel, self).__init__()
        self.embedding = torch.nn.Embedding(35300, input_dim)  # Clave: embedding.weight

        # Capas GAT (Graph Attention)
        self.gat1 = GATConv(input_dim, hidden_dim, heads=num_heads, concat=True)
        self.gat2 = GATConv(hidden_dim * num_heads, hidden_dim, heads=num_heads, concat=True)
        self.gat3 = GATConv(hidden_dim * num_heads, hidden_dim, heads=num_heads, concat=True)

        # Capas finales (lineales)
        self.fc1 = Linear(hidden_dim * num_heads, hidden_dim)  # Clave: linears.0.*
        self.fc2 = Linear(hidden_dim, hidden_dim)              # Clave: linears.1.*
        self.fc3 = Linear(hidden_dim, output_dim)              # Clave: linears.2.*

    def forward(self, x, edge_index):
        # Paso 1: Embedding
        x = self.embedding(x.long())

        # Paso 2: Capas GAT
        x = self.gat1(x, edge_index).relu()
        x = self.gat2(x, edge_index).relu()
        x = self.gat3(x, edge_index).relu()

        # Paso 3: Capas finales
        x = self.fc1(x).relu()
        x = self.fc2(x).relu()
        x = self.fc3(x)

        return x


def load_model(model_name):
    """
    Cargar el modelo PyTorch con la arquitectura correspondiente y asignar los pesos.
    """
    model_path = f"./models/{model_name}"
    try:
        # Ajusta las dimensiones según los datos en state_dict
        model = GATGraphModel(input_dim=64, hidden_dim=320, output_dim=10)  
        state_dict = torch.load(model_path, weights_only=False)
        model.load_state_dict(state_dict, strict=False)  # Permitir diferencias en claves no críticas
        model.eval()  # Configurar en modo evaluación
        return model
    except FileNotFoundError:
        raise Exception(f"No se encontró el modelo en: {model_path}")
    except Exception as e:
        raise Exception(f"Error al cargar el modelo: {e}")
