import os
import cv2
import numpy as np
from sklearn.model_selection import train_test_split
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten, Dense, Dropout
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.callbacks import ModelCheckpoint, EarlyStopping

# Veri kümesi dizini: ortam değişkeni ile özelleştirilebilir.
# Varsayılan olarak bu betiğin iki üst dizinindeki sınıf klasörleri kullanılır
# (EmotionModel/Happy_img, Sad_img, ...).
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
DEFAULT_DATA_DIR = os.path.abspath(os.path.join(SCRIPT_DIR, ".."))
data_dir = os.environ.get("DATA_DIR", DEFAULT_DATA_DIR)

# Sınıf etiketleri - bu sıra tespit (EmotionDetection/main.py) tarafında
# kullanılan EMOTIONS sırasıyla AYNI olmalıdır.
class_labels = ["Happy_img", "Sad_img", "Mad_img", "Normal_img"]

IMG_SIZE = 224

# Veri ve etiketlerin depolanacağı listeler
data = []
labels = []

# Veri kümesini dolaşarak resimleri yükleme ve etiketleme
for i, label in enumerate(class_labels):
    folder_path = os.path.join(data_dir, label)
    if not os.path.isdir(folder_path):
        print(f"Uyarı: dizin bulunamadı, atlanıyor -> {folder_path}")
        continue
    for filename in os.listdir(folder_path):
        image_path = os.path.join(folder_path, filename)
        try:
            image = cv2.imread(image_path)
            if image is None:
                print("Okunamayan dosya atlanıyor:", image_path)
                continue
            image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)  # BGR -> RGB
            image = cv2.resize(image, (IMG_SIZE, IMG_SIZE))
            image = image.astype('float32') / 255.0  # 0-1 normalize
            data.append(image)
            labels.append(i)  # Sınıf etiketi
        except Exception as e:
            print("Hata:", str(e))
            continue

if not data:
    raise SystemExit(
        f"Hiç görsel yüklenemedi. DATA_DIR doğru mu? -> {data_dir}"
    )

# Verileri ve etiketleri NumPy dizisine dönüştürün
data = np.array(data, dtype='float32')
labels = np.array(labels)

# Eğitim, doğrulama ve test kümelerine veriyi bölme (sınıf dengesini koruyarak)
train_data, test_data, train_labels, test_labels = train_test_split(
    data, labels, test_size=0.2, random_state=42, stratify=labels
)
train_data, val_data, train_labels, val_labels = train_test_split(
    train_data, train_labels, test_size=0.2, random_state=42, stratify=train_labels
)

# Veri kümesinin boyutunu kontrol edin
print("Eğitim veri kümesi: ", train_data.shape)
print("Doğrulama veri kümesi: ", val_data.shape)
print("Test veri kümesi: ", test_data.shape)

# Küçük veri kümesi için veri artırma (augmentation)
datagen = ImageDataGenerator(
    rotation_range=15,
    width_shift_range=0.1,
    height_shift_range=0.1,
    zoom_range=0.1,
    horizontal_flip=True,
)

# Modelin oluşturulması
model = Sequential()
model.add(Conv2D(32, (3, 3), activation='relu', input_shape=(IMG_SIZE, IMG_SIZE, 3)))
model.add(MaxPooling2D((2, 2)))
model.add(Conv2D(64, (3, 3), activation='relu'))
model.add(MaxPooling2D((2, 2)))
model.add(Conv2D(128, (3, 3), activation='relu'))
model.add(MaxPooling2D((2, 2)))
model.add(Flatten())
model.add(Dense(128, activation='relu'))
model.add(Dropout(0.5))
model.add(Dense(len(class_labels), activation='softmax'))

# Modelin derlenmesi
model.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])

# En iyi modeli (en yüksek doğrulama doğruluğu) kaydet
checkpoint = ModelCheckpoint(
    "best_model.h5", monitor='val_accuracy', save_best_only=True, verbose=1
)
early_stop = EarlyStopping(monitor='val_loss', patience=5, restore_best_weights=True)

# Modelin eğitimi
history = model.fit(
    datagen.flow(train_data, train_labels, batch_size=16),
    validation_data=(val_data, val_labels),
    epochs=10,
    callbacks=[checkpoint, early_stop],
)

# Modelin değerlendirilmesi
test_loss, test_acc = model.evaluate(test_data, test_labels)
print("Test Loss:", test_loss)
print("Test Accuracy:", test_acc)  # Gerçek hesaplanan değer
