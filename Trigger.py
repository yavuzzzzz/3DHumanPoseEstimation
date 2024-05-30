import os
import subprocess

from model_check import model, input_data

# Tahminleri al
predictions = model.predict(input_data)

# Tahminlerin ikinci sınıf için olan olasılığını al
probability = predictions[0][1]

# Belirli bir eşik değerine göre farklı dosyaları çalıştır
if probability > 0.5:
    subprocess.call(["python", "LateralRaise.py"])
else:
    subprocess.call(["python", "BarbellBicepsCurl.py"])