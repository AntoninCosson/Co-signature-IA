import os

def load_txt_files(folder_path):
    """Charge tous les fichiers .txt d'un dossier et retourne un dict {nom: contenu}."""
    texts = {}
    for filename in os.listdir(folder_path):
        if filename.endswith(".txt"):
            with open(os.path.join(folder_path, filename), "r", encoding="utf-8") as f:
                texts[filename] = f.read()
    return texts