import cv2
import pickle
from deepface import DeepFace as df
import os

def main():
    cap = cv2.VideoCapture(0)

    if not cap.isOpened():
        print("Kamera nie została wykryta.")
        return

    print("Kamera działa. Naciśnij 'q', aby wyjść.")

    persons = os.listdir("dataset")
    with open('data/model_knn.pkl', 'rb') as f:
            knn = pickle.load(f) #Wczytanie wytrenoanego klasyfikatora KNN

    while True:
        ret, frame = cap.read()
        if not ret or frame is None:
            print("Brak klatki.")
            continue

        face_embd = df.represent(frame, model_name="Facenet", enforce_detection=False) #Gen. embeddingu z klatki 
        predicted_label = None
        if face_embd and isinstance(face_embd, list) and 'embedding' in face_embd[0]: # Jeżeli Deepface coś wykrył:
            embedding_vector = face_embd[0]["embedding"] # Wyciągnięcie tylko embeddingów
            predicted_label = knn.predict([embedding_vector])[0] #predykcja za pom. KNN
            proba = knn.predict_proba([embedding_vector])[0] # Prawdopodobieństwo
            class_index = list(knn.classes_).index(predicted_label)
            confidence = proba[class_index] # Pewnośc predyckji

        if predicted_label and predicted_label in persons:
            text = f"{confidence*100:.2f}%"
            cv2.putText(frame, predicted_label, (30, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0,255,0), 2)
            cv2.putText(frame, text, (30, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255,0,0), 2)
        else:
            cv2.putText(frame, "Unknown", (30, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0,255,0), 2)

        cv2.imshow("Test", frame)

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()
