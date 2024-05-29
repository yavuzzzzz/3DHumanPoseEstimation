import os
import numpy as np

landmarks_dir = "landmarks/"
augmented_dir = "augmented_landmarks/"

# Augmented klasörü yoksa oluştur
if not os.path.exists(augmented_dir):
    os.makedirs(augmented_dir)

def augment_data(data, method):
    if method == 'time_shift':
        shift = np.random.randint(1, 5)
        return np.roll(data, shift, axis=0)
    elif method == 'noise_addition':
        noise = np.random.normal(0, 0.01, data.shape)
        return data + noise
    elif method == 'random_cropping':
        crop_size = int(data.shape[0] * 0.9)
        start = np.random.randint(0, data.shape[0] - crop_size)
        return data[start:start+crop_size]
    else:
        raise ValueError("Unknown augmentation method: {}".format(method))

# landmarks/ dizinindeki tüm .txt dosyalarını listele
txt_files = [f for f in os.listdir(landmarks_dir) if f.endswith('.txt')]

for txt_file in txt_files:
    file_path = os.path.join(landmarks_dir, txt_file)

    # Dosyayı oku
    data = np.loadtxt(file_path)

    methods = ['time_shift', 'noise_addition', 'random_cropping']

    for i in range(30):
        method = np.random.choice(methods)
        augmented_data = augment_data(data, method)

        augmented_file_path = os.path.join(augmented_dir, f'{os.path.splitext(txt_file)[0]}_aug_{i+1}.txt')

        # Augmented veriyi dosyaya yaz
        np.savetxt(augmented_file_path, augmented_data, fmt='%.6f')

print("Augmentation completed for all files.")
