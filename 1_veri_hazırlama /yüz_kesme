# ================================================
# YÜZ KESME — OpenCV DNN ile Face Crop
# Mevcut framelerden yüzleri kesip yeni klasöre kaydeder
# Tespit oranı: %99.9 (22 frame atlandı / 25.963 toplam)
# Kaynak: Xception_Frames → Hedef: Xception_Frames_FaceCrop
# ================================================

import cv2
import os
import numpy as np
import urllib.request

# =====================
# YOLLAR
# =====================
frames_path   = '/content/drive/MyDrive/FaceForensics_Data/Xception_Frames'
facecrop_path = '/content/drive/MyDrive/FaceForensics_Data/Xception_Frames_FaceCrop'
IMG_SIZE      = (299, 299)

# =====================
# OPENCV DNN YÜZ TESPİT MODELİ
# SSD Caffe modeli — hızlı ve güvenilir
# =====================
prototxt_path = '/content/deploy.prototxt'
model_path    = '/content/face_model.caffemodel'

if not os.path.exists(prototxt_path):
    urllib.request.urlretrieve(
        'https://raw.githubusercontent.com/opencv/opencv/master/samples/dnn/face_detector/deploy.prototxt',
        prototxt_path
    )
    print("✅ prototxt indirildi")

if not os.path.exists(model_path):
    urllib.request.urlretrieve(
        'https://github.com/opencv/opencv_3rdparty/raw/dnn_samples_face_detector_20170830/res10_300x300_ssd_iter_140000.caffemodel',
        model_path
    )
    print("✅ model indirildi")

face_net = cv2.dnn.readNetFromCaffe(prototxt_path, model_path)
print("✅ OpenCV DNN yüz dedektörü hazır")

# =====================
# YÜZ KESME FONKSİYONU
# Orijinal FF++ makalesindeki gibi %30 margin eklendi
# =====================
def yuz_kirp(frame_bgr):
    h, w = frame_bgr.shape[:2]
    blob = cv2.dnn.blobFromImage(
        cv2.resize(frame_bgr, (300, 300)), 1.0,
        (300, 300), (104.0, 177.0, 123.0)
    )
    face_net.setInput(blob)
    detections = face_net.forward()

    best_conf = 0
    best_box  = None
    for i in range(detections.shape[2]):
        conf = detections[0, 0, i, 2]
        if conf > best_conf and conf > 0.5:
            best_conf = conf
            box = detections[0, 0, i, 3:7] * np.array([w, h, w, h])
            best_box = box.astype(int)

    if best_box is None:
        return None

    # %15 margin ekle (toplam %30 — orijinal makale gibi)
    x1, y1, x2, y2 = best_box
    margin_x = int((x2 - x1) * 0.15)
    margin_y = int((y2 - y1) * 0.15)

    x1 = max(0, x1 - margin_x)
    y1 = max(0, y1 - margin_y)
    x2 = min(w, x2 + margin_x)
    y2 = min(h, y2 + margin_y)

    yuz = frame_bgr[y1:y2, x1:x2]
    if yuz.size == 0:
        return None
    return cv2.resize(yuz, IMG_SIZE)

# =====================
# ANA DÖNGÜ
# =====================
toplam = {'train': {'real': 0, 'fake': 0},
          'val':   {'real': 0, 'fake': 0},
          'test':  {'real': 0, 'fake': 0}}
atilan = 0

for split in ['train', 'val', 'test']:
    for label in ['real', 'fake']:
        kaynak_klasor = os.path.join(frames_path, split, label)
        hedef_klasor  = os.path.join(facecrop_path, split, label)
        os.makedirs(hedef_klasor, exist_ok=True)

        dosyalar = sorted(os.listdir(kaynak_klasor))
        print(f"\n{split}/{label} işleniyor... ({len(dosyalar)} frame)")

        for i, dosya in enumerate(dosyalar):
            kaynak_yol = os.path.join(kaynak_klasor, dosya)
            hedef_yol  = os.path.join(hedef_klasor, dosya)

            frame = cv2.imread(kaynak_yol)
            if frame is None:
                continue

            yuz = yuz_kirp(frame)
            if yuz is None:
                atilan += 1
                continue

            cv2.imwrite(hedef_yol, yuz)
            toplam[split][label] += 1

            if (i + 1) % 1000 == 0:
                print(f"  {i+1}/{len(dosyalar)} işlendi...")

# =====================
# ÖZET
# =====================
print(f"\n{'='*50}")
print(f"✅ TAMAMLANDI")
print(f"{'='*50}")
print(f"{'Split':<8} | {'Real':<8} | {'Fake':<8} | {'Toplam':<8}")
print(f"{'-'*38}")
genel = 0
for split in ['train', 'val', 'test']:
    r = toplam[split]['real']
    f = toplam[split]['fake']
    t = r + f
    genel += t
    print(f"{split:<8} | {r:<8} | {f:<8} | {t:<8}")
print(f"{'-'*38}")
print(f"TOPLAM: {genel}")
print(f"Yüz bulunamayıp atlanan: {atilan} frame")
print(f"Tespit oranı: %{(genel/(genel+atilan))*100:.2f}")
