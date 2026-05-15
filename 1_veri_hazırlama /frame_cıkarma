# ================================================
# FRAME ÇIKARMA — FaceForensics++ Veri Seti
# Video bazlı bölme ile train/val/test ayrımı
# Real: 10'da 1 frame, Fake (DF+F2F): 20'de 1 frame
# Seed=42, 250 video, 200/25/25 bölme
# ================================================

import cv2
import os
import random

# =====================
# YOLLAR
# =====================
real_src    = '/content/drive/MyDrive/FaceForensics_Data/original_sequences/youtube/c23/videos'
df_src      = '/content/drive/MyDrive/FaceForensics_Data/manipulated_sequences/Deepfakes/c23/videos'
f2f_src     = '/content/drive/MyDrive/FaceForensics_Data/manipulated_sequences/Face2Face/c23/videos'
output_root = '/content/drive/MyDrive/FaceForensics_Data/Xception_Frames'

# =====================
# AYARLAR
# =====================
IMG_SIZE    = (299, 299)
VIDEO_LIMIT = 250
RANDOM_SEED = 42

# =====================
# VIDEO BAZLI BÖLME
# Kural 1: Aynı videonun frameleri tek split'e gider
# Kural 2: Aynı kaynak videonun real+fake'i aynı split'te
# =====================
random.seed(RANDOM_SEED)
all_ids = sorted([
    v.replace('.mp4', '')
    for v in os.listdir(real_src)
    if v.endswith('.mp4')
])[:VIDEO_LIMIT]

random.shuffle(all_ids)
n = len(all_ids)
train_ids = set(all_ids[:int(n * 0.8)])           # 200 video
val_ids   = set(all_ids[int(n * 0.8):int(n * 0.9)])  # 25 video
test_ids  = set(all_ids[int(n * 0.9):])            # 25 video

print(f"Train: {len(train_ids)} | Val: {len(val_ids)} | Test: {len(test_ids)}")

def split_bul(video_adi):
    v_id = video_adi.replace('.mp4', '').split('_')[0]
    if v_id in train_ids: return 'train'
    if v_id in val_ids:   return 'val'
    if v_id in test_ids:  return 'test'
    return None

def frame_cikart(v_yolu, kayit_klasoru, prefix, aralik):
    cap = cv2.VideoCapture(v_yolu)
    if not cap.isOpened():
        return 0
    os.makedirs(kayit_klasoru, exist_ok=True)
    frame_no = kaydedilen = 0
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        if frame_no % aralik == 0:
            frame_resized = cv2.resize(frame, IMG_SIZE)
            dosya = os.path.join(kayit_klasoru, f"{prefix}_f{frame_no:05d}.jpg")
            cv2.imwrite(dosya, frame_resized)
            kaydedilen += 1
        frame_no += 1
    cap.release()
    return kaydedilen

# Real: 10'da 1, Fake: 20'de 1 (denge için)
kaynaklar = [
    ('real', real_src, 'REAL', 10),
    ('fake', df_src,   'DF',   20),
    ('fake', f2f_src,  'F2F',  20),
]

toplam = {'train': {'real': 0, 'fake': 0},
          'val':   {'real': 0, 'fake': 0},
          'test':  {'real': 0, 'fake': 0}}

for label, kaynak_yol, prefix_tag, aralik in kaynaklar:
    videolar = sorted([v for v in os.listdir(kaynak_yol) if v.endswith('.mp4')])
    print(f"\n{prefix_tag} işleniyor... [aralik={aralik}]")
    islenen = 0
    for v_adi in videolar:
        split = split_bul(v_adi)
        if split is None:
            continue
        v_id   = v_adi.replace('.mp4', '')
        v_yolu = os.path.join(kaynak_yol, v_adi)
        klasor = os.path.join(output_root, split, label)
        prefix = f"{prefix_tag}_{v_id}"
        n = frame_cikart(v_yolu, klasor, prefix, aralik)
        toplam[split][label] += n
        islenen += 1
        print(f"  [{islened}] {v_adi} → {split}/{label} → {n} frame")

print(f"\n{'='*50}")
print(f"✅ TAMAMLANDI")
print(f"{'Split':<8} | {'Real':<8} | {'Fake':<8} | {'Toplam':<8}")
print(f"{'-'*38}")
genel = 0
for split in ['train', 'val', 'test']:
    r = toplam[split]['real']
    f = toplam[split]['fake']
    t = r + f
    genel += t
    print(f"{split:<8} | {r:<8} | {f:<8} | {t:<8}")
print(f"TOPLAM: {genel}")
