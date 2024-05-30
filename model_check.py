import subprocess
from tensorflow.keras.models import load_model
import numpy as np

# CNNImpl dosyasından X_train'i import edin
from CNNImpl import X_train
from Trigger import landmarks_file

# Modeli yükle
model = load_model('my_model.keras')

def load_data_from_txt(file_path):
    data = []
    with open(file_path, 'r') as file:
        for line in file:
            points = line.strip().split()
            points = [float(p) for p in points]
            # Group every 3 elements and store in the list
            if len(points) % 3 == 0:  # If the total number of points is divisible by 3
                grouped_points = [points[i:i + 3] for i in range(0, len(points), 3)]
                data.append(grouped_points)
    return data

demo = load_data_from_txt(landmarks_file)
demo = np.array(demo, dtype=np.float32)

# Reshape the data to match the expected input shape for Conv1D
# The new shape is (1, number of frames, number of landmarks * 3)
demo = demo.reshape(-1, X_train.shape[1], X_train.shape[2])

# Make a prediction using the model
probability = model.predict(demo)[0][0]
# Trigger different Python files based on the prediction
if probability > 0.5:
    subprocess.call(["python", "BarbellBicepsCurlSingle.py"])
else:
    subprocess.call(["python", "LateralRaiseSingle.py"])
