from tensorflow.keras.models import load_model
import numpy as np

from CNNImpl import X_train

# Modeli yükle
model = load_model('my_model.keras')

# Girdi verisini yükle veya oluştur
input_data = np.random.rand(1, X_train.shape[1], X_train.shape[2])

# Modeli kullanarak tahmin yap
predictions = model.predict(input_data)

# Tahminleri yazdır
print(predictions)