import cv2
import numpy as np
import urllib.request
import mediapipe as mp

# URL untuk mengakses stream dari ESP32-CAM
url = "http://192.168.18.148/cam-mid.jpg"

# Inisialisasi Mediapipe hands
mp_hands = mp.solutions.hands
hands = mp_hands.Hands(static_image_mode=False, max_num_hands=2, min_detection_confidence=0.7)
mp_drawing = mp.solutions.drawing_utils

cv2.namedWindow("ESP32-CAM Hand Gesture Recognition", cv2.WINDOW_AUTOSIZE)

while True:
    try:
        # Baca gambar dari stream
        img_resp = urllib.request.urlopen(url)
        img_np = np.array(bytearray(img_resp.read()), dtype=np.uint8)
        img = cv2.imdecode(img_np, -1)
        
        # Pengenalan Gerakan Tangan
        if img is not None:
            # Konversi ke RGB karena Mediapipe bekerja dengan gambar RGB
            img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
            # Proses gambar dengan Mediapipe
            results = hands.process(img_rgb)

            # Jika ditemukan tangan dalam frame
            if results.multi_hand_landmarks:
                for hand_landmarks in results.multi_hand_landmarks:
                    # Gambar tanda tangan di gambar asli
                    mp_drawing.draw_landmarks(img, hand_landmarks, mp_hands.HAND_CONNECTIONS)
            
            # Tampilkan gambar dengan deteksi gerakan tangan
            cv2.imshow('ESP32-CAM Hand Gesture Recognition', img)
        else:
            print("Error: Failed to decode image")

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
    except Exception as e:
        print(f"Exception: {e}")
        break

cv2.destroyAllWindows()
hands.close()
