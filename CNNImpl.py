import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report
import tensorflow as tf
import os

def load_data_from_folder(folder_path, prefix, label):
    data = []
    labels = []
    for filename in os.listdir(folder_path):
        if filename.startswith(prefix) and filename.endswith(".txt"):
            filepath = os.path.join(folder_path, filename)
            with open(filepath, 'r') as file:
                for line in file:
                    points = line.strip().split()
                    points = [float(p) for p in points]
                    # Her 3 öğeyi bir grup olarak al ve listede sakla
                    if len(points) % 3 == 0:  # Eğer toplam nokta sayısı 3'e bölünüyorsa
                        grouped_points = [points[i:i+3] for i in range(0, len(points), 3)]
                        data.append(grouped_points)
                        labels.append(label)
    return data, labels

# Verileri yükle
data = []
labels = []

# "barbell biceps curl" ve "lateral raise" dosyalarını yükle
biceps_curl_data, biceps_curl_labels = load_data_from_folder('augmented_landmarks', 'barbell biceps curl', 0)
lateral_raise_data, lateral_raise_labels = load_data_from_folder('augmented_landmarks', 'lateral raise', 1)

# Verileri birleştir
data.extend(biceps_curl_data)
data.extend(lateral_raise_data)
labels.extend(biceps_curl_labels)
labels.extend(lateral_raise_labels)

# Numpy dizilerine dönüştür
data = np.array(data, dtype=object)
labels = np.array(labels)

# Verilerin aynı şekle sahip olduğundan emin olun
max_length = max(len(d) for d in data)
num_landmarks = len(data[0][0]) if data.size > 0 else 0  # Verilerin boş olup olmadığını kontrol et
padded_data = []
for d in data:
    d = np.array(d)
    if len(d) < max_length:
        padding = np.zeros((max_length - len(d), num_landmarks))
        d = np.vstack((d, padding))
    padded_data.append(d)

# Padded veriyi numpy array'e dönüştür
final_data = np.array(padded_data, dtype=np.float32)
final_labels = np.array(labels)

# Verileri eğitim ve test setlerine ayır
X_train, X_test, y_train, y_test = train_test_split(final_data, final_labels, test_size=0.2, random_state=42)
print(len(X_train), len(X_test))

# Eğitim setini daha da eğitim ve doğrulama setlerine ayır
X_train, X_val, y_train, y_val = train_test_split(final_data, final_labels, test_size=0.2, random_state=42)
print(len(X_train), len(X_val))

# Modeli tanımlayın
model = tf.keras.models.Sequential([
    tf.keras.layers.Input(shape=(X_train.shape[1], X_train.shape[2])),
    tf.keras.layers.Conv1D(64, 3, activation='relu'),
    tf.keras.layers.MaxPooling1D(2),
    tf.keras.layers.Conv1D(128, 3, activation='relu'),
    tf.keras.layers.MaxPooling1D(2),
    tf.keras.layers.Flatten(),
    tf.keras.layers.Dense(128, activation='relu'),
    tf.keras.layers.Dense(len(np.unique(labels)), activation='softmax')
])

model.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])

# Modeli eğitin
model.fit(X_train, y_train, epochs=5, validation_data=(X_test, y_test))

# Modeli değerlendirin
loss, accuracy = model.evaluate(X_test, y_test)
print(f'Test Accuracy: {accuracy * 100:.2f}%')

# Tahminlerde bulunun
y_pred = model.predict(X_test)
y_pred_classes = np.argmax(y_pred, axis=1)

# F1 skoru, precision ve recall hesaplayın
print(classification_report(y_test, y_pred_classes))
# Path: CNNImpl.py