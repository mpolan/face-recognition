from deepface import DeepFace as df
import os
import numpy as np
import pickle

# Wczytaj wytrenowany model
with open('data/model_knn.pkl', 'rb') as f:
    knn = pickle.load(f)

val_folder = 'dataset/val/'
total = 0 # Wszystgkie
correct = 0 # Poprawne

Y_true = []
Y_pred = []

for person in os.listdir(val_folder):
    person_path = os.path.join(val_folder, person)
    if not os.path.isdir(person_path):
        continue
    for img in os.listdir(person_path):
        img_path = os.path.join(person_path, img)
        if not os.path.isfile(img_path):
            continue
        try:
            # Generowanie embeddingu z enforce_detection=False
            embedding = df.represent(img_path, model_name="Facenet", enforce_detection=False)

            # Debug: pokaż co zwraca represent
            print(f"Embedding dla {img_path}: {embedding}")
            print(f"Type: {type(embedding)}")

            if embedding and isinstance(embedding, list) and 'embedding' in embedding[0]:
                embedding_vector = embedding[0]["embedding"] # Sam wektor
                predicted_label = knn.predict([embedding_vector])[0] # Predykcja labela poprzez KNN
                print(f"Plik: {img_path} | Prawdziwa: {person} | Predykcja: {predicted_label}")

                Y_true.append(person)
                Y_pred.append(predicted_label)

                if predicted_label == person:
                    correct += 1 # Poprawna

                total += 1
            else:
                print(f"Brak embeddingu dla: {img_path}")

        except Exception as e:
            print(f"Nie udało się przetworzyć: {img_path}, błąd: {e}")

accuracy = correct / total if total > 0 else 0
print(f"Dokładność na zbiorze walidacyjnym: {accuracy:.2%}") # Dokładność końcowa

from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay
import matplotlib.pyplot as plt

if total > 0:
    cm = confusion_matrix(Y_true, Y_pred, labels=sorted(set(Y_true)))
    disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=sorted(set(Y_true)))
    disp.plot(cmap=plt.cm.Blues)
    plt.title(f"Confusion Matrix (Accuracy: {accuracy:.2%})")
    plt.xticks(rotation=45)
    plt.show()