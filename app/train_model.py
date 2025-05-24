import pickle
import json
from sklearn.neighbors import KNeighborsClassifier

embeddings_file_path = 'data/embeddings.pkl'
labels_file_path = 'data/labels.json'


with open(embeddings_file_path, 'rb') as embeddings_file:
    X_train = pickle.load(embeddings_file)

with open(labels_file_path, 'r') as labels_file:
    Y_train = json.load(labels_file)

knn = KNeighborsClassifier(n_neighbors=3, weights='distance', algorithm='auto', metric='euclidean')
knn.fit(X_train, Y_train)

with open("data/model_knn.pkl", 'wb') as f:
    pickle.dump(knn, f)

# Test:
y_pred = knn.predict(X_train)

# Porównaj z y_train
from sklearn.metrics import accuracy_score, classification_report

print("Dokładność na danych treningowych:", accuracy_score(Y_train, y_pred))
print("Raport klasyfikacji:\n", classification_report(Y_train, y_pred))