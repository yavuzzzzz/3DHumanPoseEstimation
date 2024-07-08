# Barbell Biceps Curl Pose Estimation

Bu proje, MediaPipe kütüphanesini kullanarak bir video üzerindeki kişinin vücut duruşunu analiz eder. Özellikle, kişinin omuz, dirsek ve bilek arasındaki açıları hesaplar ve bu açıları video üzerinde görselleştirir. Bu, bir kişinin barbell biceps curl egzersizini doğru bir şekilde yapıp yapmadığını kontrol etmek için kullanılabilir.

## Gereksinimler

- Python 3.7+
- OpenCV
- MediaPipe
- Numpy

Bu gereksinimler `requirements.txt` dosyasında listelenmiştir ve aşağıdaki komut ile kurulabilir:

```bash
pip install -r requirements.txt
```

## Kullanım

Proje, `videos/biceps curl/` dizinindeki tüm .mp4 dosyalarını işler. İşlenmiş videolar, vücut duruşu analizi ile birlikte gösterilir.

```bash
python BarbellBicepsCurl.py
```

