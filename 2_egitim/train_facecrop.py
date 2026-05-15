# ================================================
# XCEPTION — FACE CROP EĞİTİM (V7)
# FaceForensics++ | DeepFakes + Face2Face
# Video bazlı bölme: 200/25/25 (train/val/test)
# Yüz kırpmalı frameler kullanıldı (OpenCV DNN)
# Optuna ile bulunan hiperparametreler kullanıldı
# ================================================

import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt
import json
import os
from tensorflow.keras import layers
from tensorflow.keras.applications import Xception
from tensorflow.keras.applications.xception import preprocess_input

# =====================
# AYARLAR
# =====================
BATCH    = 32
IMG_SIZE = (299, 299)
AUTOTUNE = tf.data.AUTOTUNE
DATA_PATH  = '/content/drive/MyDrive/FaceForensics_Data/Xception_Frames_FaceCrop'
MODEL_PATH = '/content/drive/MyDrive/FaceForensics_Data/xception_v7_facecrop.keras'
SAVE_PATH  = '/content/drive/MyDrive/FaceForensics_Data/Grafikler_V7'
os.makedirs(SAVE_PATH, exist_ok=True)

# =====================
# VERİ SETLERİ
# =====================
with tf.device('/CPU:0'):
    train_ds = tf.keras.utils.image_dataset_from_directory(
        os.path.join(DATA_PATH, 'train'),
        label_mode='binary', image_size=IMG_SIZE,
        batch_size=BATCH, shuffle=True, seed=42
    )
    val_ds = tf.keras.utils.image_dataset_from_directory(
        os.path.join(DATA_PATH, 'val'),
        label_mode='binary', image_size=IMG_SIZE,
        batch_size=BATCH, shuffle=False
    )

print("Sınıflar:", train_ds.class_names)
train_ds = train_ds.prefetch(AUTOTUNE)
val_ds   = val_ds.prefetch(AUTOTUNE)

# =====================
# VERİ ARTIRMA
# =====================
data_augmentation = tf.keras.Sequential([
    layers.RandomFlip("horizontal"),
    layers.RandomRotation(0.2),
    layers.RandomZoom(0.2),
    layers.RandomContrast(0.2),
    layers.RandomBrightness(0.2),
    layers.RandomTranslation(0.1, 0.1),
])

# =====================
# MODEL — XCEPTION
# Optuna Trial 0 parametreleri:
# lr=2.74e-05, dropout1=0.454, dropout2=0.219
# fine_tune=80 katman, dense1=128, dense2=256
# =====================
base = Xception(
    include_top=False,
    weights='imagenet',
    input_shape=(299, 299, 3)
)
base.trainable = True
for layer in base.layers[:-80]:
    layer.trainable = False

inputs  = tf.keras.Input(shape=(299, 299, 3))
x       = data_augmentation(inputs)
x       = layers.Lambda(preprocess_input)(x)
x       = base(x, training=True)
x       = layers.GlobalAveragePooling2D()(x)
x       = layers.BatchNormalization()(x)
x       = layers.Dense(128, activation='relu')(x)
x       = layers.Dropout(0.454)(x)
x       = layers.Dense(256, activation='relu')(x)
x       = layers.Dropout(0.219)(x)
outputs = layers.Dense(1, activation='sigmoid')(x)

model = tf.keras.Model(inputs, outputs)
model.compile(
    optimizer=tf.keras.optimizers.Adam(2.74e-05),
    loss=tf.keras.losses.BinaryCrossentropy(label_smoothing=0.1),
    metrics=['accuracy', tf.keras.metrics.AUC(name='auc')]
)

# =====================
# CALLBACKS
# =====================
callbacks = [
    tf.keras.callbacks.EarlyStopping(
        monitor='val_loss', patience=15,
        restore_best_weights=True, verbose=1
    ),
    tf.keras.callbacks.ReduceLROnPlateau(
        monitor='val_loss', factor=0.2,
        patience=5, min_lr=1e-7, verbose=1
    ),
    tf.keras.callbacks.ModelCheckpoint(
        MODEL_PATH, monitor='val_loss',
        save_best_only=True, verbose=1
    )
]

# =====================
# EĞİTİM
# =====================
print("\n🚀 Face Crop eğitim başlıyor...")
history = model.fit(
    train_ds, validation_data=val_ds,
    epochs=200, callbacks=callbacks, verbose=1
)

# =====================
# HISTORY KAYDET
# =====================
history_dict = {k: [float(v) for v in vals] for k, vals in history.history.items()}
with open(f'{SAVE_PATH}/history.json', 'w') as f:
    json.dump(history_dict, f)
print("✅ History kaydedildi!")

# =====================
# GRAFİKLER
# =====================
epochs = range(1, len(history.history['loss']) + 1)

# Loss
plt.figure(figsize=(10, 4))
plt.plot(epochs, history.history['loss'],     'b-o', label='Eğitim Kaybı',    markersize=4)
plt.plot(epochs, history.history['val_loss'], 'r-o', label='Doğrulama Kaybı', markersize=4)
plt.title('Epoch vs. Kayıp (Loss) — Face Crop', fontsize=14)
plt.xlabel('Epoch')
plt.ylabel('Kayıp')
plt.legend()
plt.grid(True, alpha=0.3)
plt.tight_layout()
plt.savefig(f'{SAVE_PATH}/loss_grafigi.png', dpi=150)
plt.close()
print("✅ Loss grafiği kaydedildi")

# Accuracy
plt.figure(figsize=(10, 4))
plt.plot(epochs, history.history['accuracy'],     'b-o', label='Eğitim Doğruluğu',    markersize=4)
plt.plot(epochs, history.history['val_accuracy'], 'r-o', label='Doğrulama Doğruluğu', markersize=4)
plt.title('Epoch vs. Doğruluk (Accuracy) — Face Crop', fontsize=14)
plt.xlabel('Epoch')
plt.ylabel('Doğruluk')
plt.legend()
plt.grid(True, alpha=0.3)
plt.tight_layout()
plt.savefig(f'{SAVE_PATH}/accuracy_grafigi.png', dpi=150)
plt.close()
print("✅ Accuracy grafiği kaydedildi")

# AUC
plt.figure(figsize=(10, 4))
plt.plot(epochs, history.history['auc'],     'b-o', label='Eğitim AUC',    markersize=4)
plt.plot(epochs, history.history['val_auc'], 'r-o', label='Doğrulama AUC', markersize=4)
plt.title('Epoch vs. AUC — Face Crop', fontsize=14)
plt.xlabel('Epoch')
plt.ylabel('AUC')
plt.legend()
plt.grid(True, alpha=0.3)
plt.tight_layout()
plt.savefig(f'{SAVE_PATH}/auc_grafigi.png', dpi=150)
plt.close()
print("✅ AUC grafiği kaydedildi")

print(f"\n✅ Tüm grafikler kaydedildi: {SAVE_PATH}")
