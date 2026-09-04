import pickle

def serialize_and_save(obj: 'any', filepath: str):
    """
    Salva um objeto qualquer em .pkl.

    Args:
        obj: Objeto a ser salvo (qualquer formato);
        filepath (str): Caminho de criação do arquivo, com especificação ".pkl" na extensão.
    """
    with open(filepath, 'wb') as serial_file:
        pickle.dump(obj, serial_file)
    print(f"'{filepath}' saved!")

def deserialize_and_load(filepath: str):
    """
    Carrega um arquivo '.pkl'.

    Args:
        filepath (str): Caminho do arquivo, com especificação ".pkl" na extensão;

    Returns:
        Objeto desserializado.
    """
    with open(filepath, 'rb') as serial_file:
        obj = pickle.load(serial_file)
    return obj