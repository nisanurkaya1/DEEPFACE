# ================================================
# FRAME BAZLI DEĞERLENDİRME — Xception V6
# Test klasöründeki her frame ayrı ayrı değerlendirilir
# Çıktı: Accuracy, Precision, Recall, F1, AUC,
#        Confusion Matrix, ROC Eğrisi, Hata Dağılımı
# ================================================

import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import os
from sklearn.metrics import (classification_report, confusion_matrix,
                              roc_auc_score, roc_curve, f1_score,
                              precision_score, recall_score, accuracy_score)
from tensorflow.keras.applications.xception import preprocess_input

# =====================
# AYARLAR
# =====================
DATA_PATH  = '/content/drive/MyDrive/FaceForensics_Data/Xception_Frames'
MODEL_PATH = '/content/drive/MyDrive/FaceForensics_Data/xception_v6_final.keras'
SAVE_PATH  = '/content/drive/MyDrive/FaceForensics_Data/Grafikler_V6'
os.makedirs(SAVE_PATH, exist_ok=True)

# =====================
# MODEL YÜKLE
# =====================
model = tf.keras.models.load_model(
    MODEL_PATH,
    custom_objects={'preprocess_input': preprocess_input}
)
print("✅ Model yüklendi!")

# =====================
# TEST SETİ
# =====================
with tf.device('/CPU:0'):
    test_ds = tf.keras.utils.image_dataset_from_directory(
        os.path.join(DATA_PATH, 'test'),
        label_mode='binary',
        image_size=(299, 299),
        batch_size=32,
        shuffle=False
    )

print("Sınıflar:", test_ds.class_names)
test_ds_p = test_ds.prefetch(tf.data.AUTOTUNE)

# =====================
# TAHMİNLER
# =====================
y_true, y_pred_prob = [], []
for images, labels in test_ds_p:
    preds = model.predict(images, verbose=0)
    y_pred_prob.extend(preds.flatten())
    y_true.extend(labels.numpy().flatten())

y_true      = np.array(y_true)
y_pred_prob = np.array(y_pred_prob)
y_pred      = (y_pred_prob > 0.5).astype(int)

# =====================
# 1. CONFUSION MATRIX
# =====================
cm = confusion_matrix(y_true, y_pred)
plt.figure(figsize=(7, 6))
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
            xticklabels=['Fake', 'Real'],
            yticklabels=['Fake', 'Real'])
plt.title('Karmaşıklık Matrisi — Frame Bazlı', fontsize=13)
plt.xlabel('Tahmin Edilen Sınıf')
plt.ylabel('Gerçek Sınıf')
plt.tight_layout()
plt.savefig(f'{SAVE_PATH}/confusion_matrix_frame.png', dpi=150)
plt.close()
print("✅ Confusion matrix kaydedildi")

# =====================
# 2. ROC EĞRİSİ
# =====================
fpr, tpr, _ = roc_curve(y_true, y_pred_prob)
auc = roc_auc_score(y_true, y_pred_prob)

plt.figure(figsize=(7, 6))
plt.plot(fpr, tpr, color='darkorange', lw=2, label=f'ROC (AUC = {auc:.4f})')
plt.plot([0, 1], [0, 1], color='navy', lw=1, linestyle='--', label='Rastgele Tahmin')
plt.xlabel('Yanlış Pozitif Oranı (FPR)')
plt.ylabel('Doğru Pozitif Oranı (TPR)')
plt.title('ROC Eğrisi — Frame Bazlı', fontsize=13)
plt.legend(loc='lower right')
plt.grid(True, alpha=0.3)
plt.tight_layout()
plt.savefig(f'{SAVE_PATH}/roc_curve_frame.png', dpi=150)
plt.close()
print("✅ ROC eğrisi kaydedildi")

# =====================
# 3. HATA DAĞILIMI
# =====================
plt.figure(figsize=(10, 5))
fake_probs = y_pred_prob[y_true == 0]
real_probs = y_pred_prob[y_true == 1]
plt.hist(fake_probs, bins=50, alpha=0.6, color='red',  label='Gerçek: Fake')
plt.hist(real_probs, bins=50, alpha=0.6, color='blue', label='Gerçek: Real')
plt.axvline(x=0.5, color='black', linestyle='--', label='Karar Sınırı (0.5)')
plt.xlabel('Tahmin Olasılığı (Real)')
plt.ylabel('Frame Sayısı')
plt.title('Hata Dağılımı — Tahmin Olasılıklarının Dağılımı', fontsize=13)
plt.legend()
plt.grid(True, alpha=0.3)
plt.tight_layout()
plt.savefig(f'{SAVE_PATH}/hata_dagilimi_frame.png', dpi=150)
plt.close()
print("✅ Hata dağılımı kaydedildi")

# =====================
# 4. SINIFLANDIRMA RAPORU
# =====================
report = classification_report(
    y_true, y_pred,
    target_names=['Fake', 'Real'],
    digits=4
)
print("\n📊 SINIFLANDIRMA RAPORU:")
print(report)

with open(f'{SAVE_PATH}/classification_report_frame.txt', 'w') as f:
    f.write("Frame Bazlı Değerlendirme\n")
    f.write("="*50 + "\n\n")
    f.write(report)
print("✅ Rapor kaydedildi")

# =====================
# 5. GENEL ÖZET
# =====================
acc  = accuracy_score(y_true, y_pred)
f1   = f1_score(y_true, y_pred, average='macro')
prec = precision_score(y_true, y_pred, average='macro')
rec  = recall_score(y_true, y_pred, average='macro')
tn, fp, fn, tp = cm.ravel()

print(f"\n{'='*50}")
print(f"✅ FRAME BAZLI DEĞERLENDİRME ÖZETİ")
print(f"{'='*50}")
print(f"Accuracy:   %{acc*100:.2f}")
print(f"Precision:  {prec:.4f}")
print(f"Recall:     {rec:.4f}")
print(f"F1 (Makro): {f1:.4f}")
print(f"ROC-AUC:    {auc:.4f}")
print(f"{'='*50}")
print(f"TP: {tp}  TN: {tn}  FP: {fp}  FN: {fn}")
print(f"{'='*50}")
print(f"\nGrafikler: {SAVE_PATH}")
