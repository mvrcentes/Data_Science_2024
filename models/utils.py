import numpy as np

def preprocess_input(input_data):
    """
    Función para preprocesar la entrada del usuario.
    """
    return np.array(input_data).reshape(1, -1)
