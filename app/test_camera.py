import cv2

def main():
    cap = cv2.VideoCapture(0)

    if not cap.isOpened():
        print("❌ Kamera nie została wykryta.")
        return

    print("✅ Kamera działa. Naciśnij 'q', aby wyjść.")

    while True:
        ret, frame = cap.read()
        if not ret or frame is None:
            print("❌ Brak klatki.")
            continue

        # Lustrzane odbicie – jak w Twoim projekcie
        # frame = cv2.flip(frame, 1)

        # Opcjonalnie: ustaw rozmiar (często naprawia problemy)
        # cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
        # cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)

        cv2.putText(frame, 'Test kamery (q = wyjscie)', (30, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0,255,0), 2)
        cv2.imshow('Test Kamery', frame)

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()
