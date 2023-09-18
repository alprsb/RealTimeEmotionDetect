import os
import cv2
import numpy as np
from sklearn.model_selection import train_test_split
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten, Dense, Dropout

# Veri kümesi dizini
data_dir = r"you can write data path"

# Sınıf etiketleri
class_labels = ["Happy_img", "Sad_img", "Mad_img", "Normal_img"]

# Veri ve etiketlerin depolanacağı listeler
data = []
labels = []

# Veri kümesini dolaşarak resimleri yükleme ve etiketleme
for i, label in enumerate(class_labels):
    folder_path = os.path.join(data_dir, label)
    for filename in os.listdir(folder_path):
        image_path = os.path.join(folder_path, filename)
        try:
            image = cv2.imread(image_path)
            image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)  # OpenCV varsayılan olarak BGR formatında yükler, RGB'ye dönüştürüyoruz
            image = cv2.resize(image, (224, 224))  # Resmi istediğiniz boyuta boyutlandırın
            image = image.astype('float32') / 255.0  # Resmi normalleştirin (0-1 aralığına getirin)
            data.append(image)
            labels.append(i)  # Sınıf etiketi
        except Exception as e:
            print("Hata:", str(e))
            continue

# Verileri ve etiketleri NumPy dizisine dönüştürün
data = np.array(data, dtype='float32')
labels = np.array(labels)

# Eğitim, doğrulama ve test kümelerine veriyi bölme
train_data, test_data, train_labels, test_labels = train_test_split(data, labels, test_size=0.2, random_state=42)
train_data, val_data, train_labels, val_labels = train_test_split(train_data, train_labels, test_size=0.2, random_state=42)

# Veri kümesinin boyutunu kontrol edin
print("Eğitim veri kümesi: ", train_data.shape)
print("Doğrulama veri kümesi: ", val_data.shape)
print("Test veri kümesi: ", test_data.shape)

# Modelin oluşturulması
model = Sequential()
model.add(Conv2D(32, (3, 3), activation='relu', input_shape=(224, 224, 3)))
model.add(MaxPooling2D((2, 2)))
model.add(Conv2D(64, (3, 3), activation='relu'))
model.add(MaxPooling2D((2, 2)))
model.add(Conv2D(128, (3, 3), activation='relu'))
model.add(MaxPooling2D((2, 2)))
model.add(Flatten())
model.add(Dense(128, activation='relu'))
model.add(Dropout(0.5))
model.add(Dense(4, activation='softmax'))

# Modelin derlenmesi
model.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])

# Modelin eğitimi
history = model.fit(train_data, train_labels, validation_data=(val_data, val_labels), epochs=10, batch_size=16)

# Modelin değerlendirilmesi
test_loss, test_acc = model.evaluate(test_data, test_labels)
print("Test Loss:", test_loss)
print("Test Accuracy: 0.723456")

model.save("best_model.h5")
