import os
import threading
from tkinter import messagebox
import cv2
import tkinter as tk
from PIL import Image, ImageTk
import numpy as np
from keras.models import load_model
import pyttsx3

# Sesli geribildirim için pyttsx3 kullanımı
engine = pyttsx3.init()
engine.setProperty('rate', 120)

# Model yolu: ortam değişkeni ile özelleştirilebilir, aksi halde bu dosyanın
# yanındaki best_model.h5 kullanılır.
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
MODEL_PATH = os.environ.get('MODEL_PATH', os.path.join(SCRIPT_DIR, 'best_model.h5'))

# Modeli yükle
try:
    model = load_model(MODEL_PATH)
except Exception as exc:  # Model bulunamazsa kullanıcıyı bilgilendir
    raise SystemExit(
        f"Model yüklenemedi ({MODEL_PATH}). MODEL_PATH ortam değişkenini "
        f"ayarlayın veya best_model.h5 dosyasını bu dizine koyun.\nHata: {exc}"
    )

face_haar_cascade = cv2.CascadeClassifier(
    cv2.data.haarcascades + 'haarcascade_frontalface_default.xml'
)

# Duygu etiketleri, eğitimdeki sınıf sırasıyla AYNI olmalıdır:
# class_labels = ["Happy_img", "Sad_img", "Mad_img", "Normal_img"]
EMOTIONS = ('mutlu', 'uzgun', 'kizgin', 'normal')

is_sound_enabled = False  # Sesin etkin olup olmadığını kontrol etmek için
last_detected_emotion = None  # En son tespit edilen duygu
is_running = False  # Kamera döngüsünün çalışıp çalışmadığını tutar
cap = None  # Paylaşılan VideoCapture nesnesi


def _camera_loop():
    global last_detected_emotion, is_running, cap
    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        messagebox.showerror("Kamera Hatası", "Kamera açılamadı.")
        is_running = False
        return

    while is_running:
        ret, test_img = cap.read()
        if not ret:
            break

        # Model 3 kanallı (RGB) giriş bekliyor.
        rgb_img = cv2.cvtColor(test_img, cv2.COLOR_BGR2RGB)
        gray_img = cv2.cvtColor(test_img, cv2.COLOR_BGR2GRAY)

        faces_detected = face_haar_cascade.detectMultiScale(gray_img, 1.32, 5)

        for (x, y, w, h) in faces_detected:
            center_x = x + w // 2
            center_y = y + h // 2
            radius = min(w, h) // 2
            cv2.circle(test_img, (center_x, center_y), radius, (255, 0, 0), thickness=2)

            roi = rgb_img[y:y + h, x:x + w]  # eksenler: yükseklik=h, genişlik=w
            if roi.size == 0:
                continue
            roi = cv2.resize(roi, (224, 224))
            img_pixels = np.expand_dims(roi.astype('float32'), axis=0)
            img_pixels /= 255.0

            predictions = model.predict(img_pixels, verbose=0)
            max_index = int(np.argmax(predictions[0]))
            predicted_emotion = EMOTIONS[max_index]

            cv2.putText(test_img, predicted_emotion, (int(x), int(y)),
                        cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)
            last_detected_emotion = predicted_emotion
            emotion_label.configure(text=f"Tahmin Edilen Duygu: {predicted_emotion}")

        # Görüntüyü Tkinter etiketinde güncelle
        display_img = cv2.cvtColor(test_img, cv2.COLOR_BGR2RGB)
        img_tk = ImageTk.PhotoImage(Image.fromarray(display_img))
        image_label.configure(image=img_tk)
        image_label.image = img_tk

    if cap is not None:
        cap.release()
        cap = None


def start_camera():
    global is_running
    if is_running:
        return
    is_running = True
    # Arayüzün donmaması için kamera döngüsü ayrı bir thread'de çalışır.
    threading.Thread(target=_camera_loop, daemon=True).start()


def stop_camera():
    global is_running
    is_running = False  # Döngünün durmasını ister; thread cap'i serbest bırakır

    # Son tespit edilen duygu üzerinden sesli geribildirim
    if is_sound_enabled and last_detected_emotion is not None:
        if last_detected_emotion == 'mutlu':
            engine.say("You are happy. What a surprise!")
        elif last_detected_emotion == 'kizgin':
            engine.say("Why are you mad? Life is too short to be mad")
        elif last_detected_emotion == 'normal':
            engine.say("You look normal. Do you have any emotions?")
        elif last_detected_emotion == 'uzgun':
            engine.say("You look sad. I hope things get better soon.")
        else:
            engine.say("Son tespit edilen duygu: " + last_detected_emotion)
        engine.runAndWait()


def toggle_sound():
    global is_sound_enabled
    is_sound_enabled = not is_sound_enabled
    sound_button_text = "Sesli Mesajı Aç" if not is_sound_enabled else "Sesli Mesajı Kapat"
    sound_button.configure(text=sound_button_text)


def on_close():
    global is_running
    is_running = False
    window.destroy()


# Tkinter penceresini oluşturma
window = tk.Tk()
window.title("Duygu Tespit Etme Uygulaması")
window.geometry("800x600")
window.configure(bg="black")
window.protocol("WM_DELETE_WINDOW", on_close)

image_label = tk.Label(window)
image_label.pack()

emotion_label = tk.Label(window, text="Tahmin Edilen Duygu: ")
emotion_label.pack()

# Frame oluşturma
button_frame = tk.Frame(window, bg="black")
button_frame.pack(pady=10)

# Başlat düğmesi
start_button = tk.Button(button_frame, text="Kamerayı Başlat", command=start_camera)
start_button.pack(side="left", padx=10)

# Durdur düğmesi (durdurunca sesli mesaj verir)
stop_button = tk.Button(button_frame, text="Kamerayı Durdur", command=stop_camera)
stop_button.pack(side="right", padx=10)

# Sesli geribildirim düğmesi
sound_button_text = "Sesli Mesajı Aç" if not is_sound_enabled else "Sesli Mesajı Kapat"
sound_button = tk.Button(window, text=sound_button_text, command=toggle_sound)
sound_button.pack()

# Tkinter penceresini çalıştırma
window.mainloop()
