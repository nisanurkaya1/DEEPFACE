# ================================================
# VİDEO BAZLI MAJORITY VOTING — Xception V6
# Full Frame senaryosu
# Her video için 1, 5, 10, 50 frame alınır
# Frame tahminleri majority voting ile birleştirilir
# Çıktı: Accuracy, F1, Precision, Recall, AUC,
#        Confusion Matrix, ROC Eğrisi
# ================================================

import tensorflow as tf
import numpy as np
import cv2
import os
import random
import matplotlib.pyplot as plt
import seaborn as sns
from tensorflow.keras.applications.xception import preprocess_input
from sklearn.metrics import (classification_report, confusion_matrix,
                              roc_auc_score, roc_curve, accuracy_score,
                              f1_score, precision_score, recall_score)

# =====================
# AYARLAR
# =====================
MODEL_PATH = '/content/drive/MyDrive/FaceForensics_Data/xception_v6_final.keras'
real_src   = '/content/drive/MyDrive/FaceForensics_Data/original_sequences/youtube/c23/videos'
df_src     = '/content/drive/MyDrive/FaceForensics_Data/manipulated_sequences/Deepfakes/c23/videos'
f2f_src    = '/content/drive/MyDrive/FaceForensics_Data/manipulated_sequences/Face2Face/c23/videos'
SAVE_PATH  = '/content/drive/MyDrive/FaceForensics_Data/Grafikler_V6'
os.makedirs(SAVE_PATH, exist_ok=True)

# =====================
# TEST ID'LERİ
# Seed=42 ile 250 video, son 25'i test
# =====================
random.seed(42)
all_ids = sorted([v.replace('.mp4','') for v in os.listdir(real_src) if v.endswith('.mp4')])[:250]
random.shuffle(all_ids)
test_ids = set(all_ids[225:])
print(f"Test video sayısı: {len(test_ids)}")

# =====================
# MODEL YÜKLE
# =====================
model = tf.keras.models.load_model(
    MODEL_PATH,
    custom_objects={'preprocess_input': preprocess_input}
)
print("✅ Model yüklendi!")

# =====================
# VIDEO TAHMİN FONKSİYONU
# Videodan n_frame alır, majority voting yapar
# 1=real, 0=fake
# =====================
def video_tahmin(v_yolu, n_frame):
    cap = cv2.VideoCapture(v_yolu)
    if not cap.isOpened():
        return None, None
    toplam = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    if toplam <= 0:
        cap.release()
        return None, None
    if n_frame >= toplam:
        secilen = list(range(toplam))
    else:
        secilen = [int(i * toplam / n_frame) for i in range(n_frame)]
    tahminler = []
    for idx in secilen:
        cap.set(cv2.CAP_PROP_POS_FRAMES, idx)
        ret, frame = cap.read()
        if not ret:
            continue
        frame_rgb     = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        frame_resized = cv2.resize(frame_rgb, (299, 299))
        img_array     = np.expand_dims(frame_resized.astype(np.float32), axis=0)
        pred = model.predict(img_array, verbose=0)[0][0]
        tahminler.append(pred)
    cap.release()
    if not tahminler:
        return None, None
    ortalama = np.mean(tahminler)
    karar    = 1 if ortalama > 0.5 else 0
    return karar, ortalama

# =====================
# TEST VİDEOLARINI HAZIRLA
# Her ID için: real + deepfakes + face2face
# =====================
test_videolari = []
for v_id in sorted(test_ids):
    real_yol = os.path.join(real_src, f"{v_id}.mp4")
    if os.path.exists(real_yol):
        test_videolari.append((v_id, real_yol, 1, 'real'))
    for dosya in sorted(os.listdir(df_src)):
        if dosya.startswith(f"{v_id}_"):
            test_videolari.append((v_id, os.path.join(df_src, dosya), 0, 'deepfakes'))
            break
    for dosya in sorted(os.listdir(f2f_src)):
        if dosya.startswith(f"{v_id}_"):
            test_videolari.append((v_id, os.path.join(f2f_src, dosya), 0, 'face2face'))
            break

print(f"Toplam test videosu: {len(test_videolari)}")

# =====================
# 1, 5, 10, 50 FRAME İLE DEĞERLENDİR
# =====================
frame_sayilari = [1, 5, 10, 50]
sonuclar = {}

for n_frame in frame_sayilari:
    print(f"\n{'='*50}")
    print(f"  Frame sayısı: {n_frame}")
    print(f"{'='*50}")

    y_true = []
    y_pred = []
    y_prob = []

    for v_id, v_yolu, gercek, kaynak in test_videolari:
        karar, prob = video_tahmin(v_yolu, n_frame)
        if karar is None:
            continue
        y_true.append(gercek)
        y_pred.append(karar)
        y_prob.append(prob)

        dogru      = "✅" if karar == gercek else "❌"
        tahmin_str = "REAL" if karar == 1 else "FAKE"
        gercek_str = "REAL" if gercek == 1 else "FAKE"
        print(f"  {dogru} [{kaynak:<10}] {v_id} → {gercek_str} → {tahmin_str} (p={prob:.3f})")

    y_true = np.array(y_true)
    y_pred = np.array(y_pred)
    y_prob = np.array(y_prob)

    acc  = accuracy_score(y_true, y_pred)
    auc  = roc_auc_score(y_true, y_prob)
    f1   = f1_score(y_true, y_pred, average='macro')
    prec = precision_score(y_true, y_pred, average='macro')
    rec  = recall_score(y_true, y_pred, average='macro')

    print(f"\n  Accuracy:  %{acc*100:.2f}")
    print(f"  AUC:       {auc:.4f}")
    print(f"  F1 Makro:  {f1:.4f}")
    print(f"  Precision: {prec:.4f}")
    print(f"  Recall:    {rec:.4f}")
    print(classification_report(y_true, y_pred, target_names=['Fake','Real'], digits=4))

    sonuclar[n_frame] = {
        'accuracy': acc, 'auc': auc, 'f1': f1,
        'precision': prec, 'recall': rec,
        'y_true': y_true, 'y_pred': y_pred, 'y_prob': y_prob
    }

# =====================
# 10 FRAME İÇİN DETAYLI GRAFİKLER
# =====================
y_true10 = sonuclar[10]['y_true']
y_pred10 = sonuclar[10]['y_pred']
y_prob10 = sonuclar[10]['y_prob']

# Confusion Matrix
cm = confusion_matrix(y_true10, y_pred10)
plt.figure(figsize=(7, 6))
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
            xticklabels=['Fake', 'Real'],
            yticklabels=['Fake', 'Real'])
plt.title('Karmaşıklık Matrisi — Video Bazlı (10 Frame)', fontsize=12)
plt.xlabel('Tahmin Edilen Sınıf')
plt.ylabel('Gerçek Sınıf')
plt.tight_layout()
plt.savefig(f'{SAVE_PATH}/confusion_matrix_video.png', dpi=150)
plt.close()
print("✅ Confusion matrix kaydedildi")

# ROC Eğrisi
fpr, tpr, _ = roc_curve(y_true10, y_prob10)
auc10 = roc_auc_score(y_true10, y_prob10)
plt.figure(figsize=(7, 6))
plt.plot(fpr, tpr, color='darkorange', lw=2, label=f'ROC (AUC = {auc10:.4f})')
plt.plot([0,1], [0,1], color='navy', lw=1, linestyle='--')
plt.xlabel('Yanlış Pozitif Oranı (FPR)')
plt.ylabel('Doğru Pozitif Oranı (TPR)')
plt.title('ROC Eğrisi — Video Bazlı (10 Frame)', fontsize=12)
plt.legend(loc='lower right')
plt.grid(True, alpha=0.3)
plt.tight_layout()
plt.savefig(f'{SAVE_PATH}/roc_curve_video.png', dpi=150)
plt.close()
print("✅ ROC eğrisi kaydedildi")

# Frame sayısı vs metrikler grafiği
plt.figure(figsize=(10, 6))
plt.plot(frame_sayilari, [sonuclar[n]['accuracy']*100 for n in frame_sayilari], 'b-o', label='Accuracy (%)', markersize=8, linewidth=2)
plt.plot(frame_sayilari, [sonuclar[n]['auc']*100     for n in frame_sayilari], 'r-o', label='AUC×100',     markersize=8, linewidth=2)
plt.plot(frame_sayilari, [sonuclar[n]['f1']*100      for n in frame_sayilari], 'g-o', label='F1×100',      markersize=8, linewidth=2)
plt.xlabel('Video Başına Frame Sayısı', fontsize=12)
plt.ylabel('Başarım (%)', fontsize=12)
plt.title('Majority Voting — Frame Sayısının Başarıma Etkisi', fontsize=13)
plt.legend(fontsize=10)
plt.grid(True, alpha=0.3)
plt.xticks(frame_sayilari)
plt.ylim(50, 85)
plt.tight_layout()
plt.savefig(f'{SAVE_PATH}/majority_voting_grafik.png', dpi=150)
plt.close()
print("✅ Majority voting grafiği kaydedildi")

# =====================
# ÖZET TABLO
# =====================
print(f"\n{'='*65}")
print(f"ÖZET — Video Bazlı Majority Voting (Full Frame)")
print(f"{'='*65}")
print(f"{'Frame':<8} | {'Accuracy':<10} | {'F1':<8} | {'Precision':<10} | {'Recall':<8} | {'AUC':<8}")
print(f"{'-'*65}")
for n in frame_sayilari:
    s = sonuclar[n]
    print(f"{n:<8} | %{s['accuracy']*100:<8.2f} | {s['f1']:<8.4f} | {s['precision']:<10.4f} | {s['recall']:<8.4f} | {s['auc']:.4f}")

# Raporu kaydet
with open(f'{SAVE_PATH}/majority_voting_report.txt', 'w') as f:
    f.write("Video Bazlı Majority Voting — Full Frame\n")
    f.write("="*50 + "\n\n")
    for n in frame_sayilari:
        s = sonuclar[n]
        f.write(f"Frame: {n}\n")
        f.write(f"Accuracy:  %{s['accuracy']*100:.2f}\n")
        f.write(f"AUC:       {s['auc']:.4f}\n")
        f.write(f"F1:        {s['f1']:.4f}\n")
        f.write(f"Precision: {s['precision']:.4f}\n")
        f.write(f"Recall:    {s['recall']:.4f}\n")
        f.write("\n" + classification_report(
            s['y_true'], s['y_pred'],
            target_names=['Fake','Real'], digits=4) + "\n")
        f.write("-"*50 + "\n")
print("✅ Rapor kaydedildi")
print(f"\n✅ Tüm dosyalar: {SAVE_PATH}")
