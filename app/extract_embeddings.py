from deepface import DeepFace as df
import os
import pickle
import json

model = "Facenet"
dataset_path = "dataset/michal"
imgs = os.listdir(dataset_path)

embeddings = []
labels = []

os.makedirs(os.path.join(dataset_path, "error_face"), exist_ok=True)

for img in imgs:
    if not os.path.isfile(os.path.join(dataset_path, img)):
        continue
    try:
        embedding = df.represent(os.path.join(dataset_path, img), model)
        embeddings.append(embedding)
        labels.append('michal')
    except ValueError as e:
        if "Face could not be detected" in str(e):
            os.rename(os.path.join(dataset_path, img), os.path.join(dataset_path,'error_face','brak-twarzy-' + img))


embeddings_file_path = 'data/embeddings.pkl'
labels_file_path = 'data/labels.json'

with open(labels_file_path, 'w') as labels_file:
    json.dump(labels, labels_file)
with open(embeddings_file_path, 'wb') as embeddings_file:
    pickle.dump(embeddings, embeddings_file)