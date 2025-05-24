from deepface import DeepFace as df
import os
import pickle
import json
import numpy as np

model = "Facenet"
#print(os.listdir("dataset"))
embeddings = []
labels = []

for person in os.listdir("dataset"):
    if person == 'val':
        continue
    dataset_path = os.path.join("dataset",person)
    imgs = os.listdir(dataset_path)

    # Upewnij się, że folder error_face istnieje
    os.makedirs(os.path.join(dataset_path, "error_face"), exist_ok=True)

    for img in imgs:
        img_path = os.path.join(dataset_path, img)
        if not os.path.isfile(img_path):
            continue
        try:
            embedding = df.represent(img_path, model)
            # DeepFace.represent() zwraca listę embeddingów (nawet dla 1 zdjęcia)
            if isinstance(embedding, list) and len(embedding) > 0:
                embedding_vector = embedding[0]["embedding"]  # Pobieramy embedding
                embeddings.append(embedding_vector)
                labels.append(person)
        except ValueError as e:
            if "Face could not be detected" in str(e):
                os.rename(img_path, os.path.join(dataset_path, 'error_face', 'brak-twarzy-' + img))

# Konwersja embeddingów na numpy array (każdy wektor tej samej długości)
embeddings = np.array(embeddings)

# Zapis do plików
embeddings_file_path = 'data/embeddings.pkl'
labels_file_path = 'data/labels.json'

os.makedirs('data', exist_ok=True)  # Upewnij się, że folder data istnieje

with open(embeddings_file_path, 'wb') as embeddings_file:
    pickle.dump(embeddings, embeddings_file)

with open(labels_file_path, 'w') as labels_file:
    json.dump(labels, labels_file)

print(f"Zapisano {len(embeddings)} embeddingów i {len(labels)} etykiet.")
