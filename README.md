# 😊 RealTimeEmotionDetection

## 📌 Proje Açıklaması

**RealTimeEmotionDetection**, **Convolutional Neural Network (CNN)** tabanlı bir yapay zekâ modeli kullanarak insan yüz ifadelerini **gerçek zamanlı olarak** analiz eden ve tespit edilen duyguya göre **sesli geribildirim** veren bir uygulamadır.

Uygulama, kamera görüntüsünden yüz ifadelerini tanımlar ve **Mutlu**, **Üzgün**, **Kızgın**, **Normal**, **Korkmuş** gibi temel duyguları tespit eder. Aynı zamanda kullanıcıya **sesli mesaj** ile otomatik geri bildirim sağlar.

Veri seti olarak, **Google'dan indirilen yüz ifadesi görselleri** kullanılmıştır.

---

## 🚀 Proje Özellikleri

- 👁‍️ **Gerçek Zamanlı Duygu Analizi:** Yüz ifadelerini anında algılar ve sınıflandırır.
- 🔊 **Sesli Geribildirim:** Tespit edilen duyguya göre İngilizce sesli mesaj verir.
- 🖥️ **Kullanıcı Dostu Arayüz:** Basit ve şık bir **Tkinter** arayüzü ile kullanım kolaylığı.
- 🧠 **CNN Modeli:** Derin öğrenme tabanlı eğitimli model (Keras ile geliştirilmiş).
- 🔄 **Kameradan Canlı Yayın:** Webcam üzerinden analiz.

---

## 📁 Proje Yapısı

```
RealTimeEmotionDetection/
│
├── EmotionDetection/
│   ├── best_model.h5          # Eğitilmiş CNN modeli
│   └── main.py                # Gerçek zamanlı duygu tespiti arayüzü
│
├── EmotionModel/
│   ├── Happy_img/             # Eğitim verisi - Mutlu
│   ├── Sad_img/               # Eğitim verisi - Üzgün
│   ├── Mad_img/               # Eğitim verisi - Kızgın
│   ├── Normal_img/            # Eğitim verisi - Normal
│   └── model_train/
│       └── main.py            # CNN model eğitimi kodu
│
└── README.md
```

---

## 🔧 Kullanılan Teknolojiler

- Python 3.x
- OpenCV
- TensorFlow / Keras
- Tkinter
- pyttsx3 (Metin-konuşma)
- PIL (Görsel işleme)

---

## 🛠 Kurulum Adımları

1️⃣ Projeyi klonlayın:

```bash
git clone https://github.com/kullanici_adi/RealTimeEmotionDetection.git
cd RealTimeEmotionDetection
```

2️⃣ Gereken kütüphaneleri yükleyin:

```bash
pip install tensorflow opencv-python pyttsx3 pillow
```

3️⃣ Modelinizi doğru dizine yerleştirin:

- **EmotionDetection/best\_model.h5**

---

## 🧪 Model Eğitimi (Opsiyonel)

Eğer modeli kendiniz eğitmek isterseniz:

```bash
cd EmotionModel/model_train
python main.py
```

> Bu adımda `EmotionModel` klasöründeki görseller kullanılarak CNN tabanlı bir model (`best_model.h5`) oluşturulur.

---

## 🎯 Uygulama Nasıl Çalıştırılır?

```bash
cd EmotionDetection
python main.py
```

- 🎥 Webcam otomatik olarak açılır.
- 😊 Yüz ifadenize göre tahminler görüntülenir.
- 🔊 Eğer sesli mesaj seçeneğini aktif ederseniz, kapanışta otomatik olarak mesaj oynatılır.
- 📌 Çıkış için `q` tuşuna basabilirsiniz.

---

## 📸 Duygu Sınıfları

| Etiket     | Anlamı   |
| ---------- | -------- |
| `Happy`    | Mutlu    |
| `Sad`      | Üzgün    |
| `Mad`      | Kızgın   |
| `Normal`   | Nötr     |
| `Korkmus`  | Korkmuş  |
| `Igrenmis` | İğrenmiş |
| `Dogal`    | Doğal    |

---

## 🔊 Sesli Mesajlar (Örnekler)

| Duygu   | Sesli Mesaj                                     |
| ------- | ----------------------------------------------- |
| Mutlu   | "You are happy. What a surprise!"               |
| Kızgın  | "Why are you mad? Life is too short to be mad." |
| Normal  | "You look normal. Do you have any emotions?"    |
| Korkmuş | "Oh my god! What did you see?"                  |
| Diğer   | "Son tespit edilen duygu: ..."                  |

---

## 💡 Geliştirme Fikirleri

✅ Daha fazla duygu sınıfı eklenebilir.\
✅ Sesli mesajlar kişiselleştirilebilir veya çok dilli hale getirilebilir.\
✅ Web tabanlı arayüz entegre edilebilir (Streamlit, Flask).\
✅ Derin öğrenme modeli iyileştirilebilir.

---

## 📜 Lisans

MIT Lisansı altında açık kaynak olarak sunulmuştur.

---

