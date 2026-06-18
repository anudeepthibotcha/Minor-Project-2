# Before running the code install python libraries
pip install tensorflow numpy matplotlib pillow
import os
os.environ["TF_ENABLE_ONEDNN_OPTS"] = "0"

import shutil
import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt

from tensorflow.keras import layers, models
from tensorflow.keras.applications import MobileNetV2
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint, ReduceLROnPlateau


# ==================================================
# STEP 1: DATASET PATH
# ==================================================

# Change this path according to your extracted dataset folder
dataset_dir = r"C:\Users\Asus\AI Project\Curated Dataset for COVID-19 Posterior-Anterior Chest Radiography Images (X-Rays)\Curated X-Ray Dataset\Curated X-Ray Dataset"

# Final expected folders:
# Curated X-Ray Dataset/
# ├── NORMAL/
# ├── BACTERIAL_PNEUMONIA/
# └── VIRAL_PNEUMONIA/


# ==================================================
# STEP 2: DELETE UNWANTED FOLDERS
# ==================================================

unwanted_folders = [
    "COVID",
    "COVID-19",
    "LUNG_OPACITY",
    "Lung_Opacity",
    "PNEUMONIA"
]

for folder in unwanted_folders:
    folder_path = os.path.join(dataset_dir, folder)

    if os.path.exists(folder_path):
        shutil.rmtree(folder_path)
        print("Deleted:", folder_path)


# ==================================================
# STEP 3: RENAME FOLDERS IF NEEDED
# ==================================================

rename_map = {
    "Normal": "NORMAL",
    "normal": "NORMAL",
    "Bacterial Pneumonia": "BACTERIAL_PNEUMONIA",
    "bacterial-pneumonia": "BACTERIAL_PNEUMONIA",
    "Bacterial_Pneumonia": "BACTERIAL_PNEUMONIA",
    "Viral Pneumonia": "VIRAL_PNEUMONIA",
    "viral-pneumonia": "VIRAL_PNEUMONIA",
    "Viral_Pneumonia": "VIRAL_PNEUMONIA"
}

for old_name, new_name in rename_map.items():
    old_path = os.path.join(dataset_dir, old_name)
    new_path = os.path.join(dataset_dir, new_name)

    if os.path.exists(old_path) and not os.path.exists(new_path):
        os.rename(old_path, new_path)
        print("Renamed:", old_name, "to", new_name)


# ==================================================
# STEP 4: CHECK FINAL CLASSES
# ==================================================

print("\nFinal Dataset Folders:")

for folder in os.listdir(dataset_dir):
    folder_path = os.path.join(dataset_dir, folder)

    if os.path.isdir(folder_path):
        print(folder, ":", len(os.listdir(folder_path)), "images")


# ==================================================
# STEP 5: LOAD DATASET WITH TRAIN/VAL SPLIT
# ==================================================

img_size = (224, 224)
batch_size = 16
seed = 42

train_data = tf.keras.utils.image_dataset_from_directory(
    dataset_dir,
    validation_split=0.2,
    subset="training",
    seed=seed,
    image_size=img_size,
    batch_size=batch_size,
    label_mode="categorical"
)

val_data = tf.keras.utils.image_dataset_from_directory(
    dataset_dir,
    validation_split=0.2,
    subset="validation",
    seed=seed,
    image_size=img_size,
    batch_size=batch_size,
    label_mode="categorical"
)

class_names = train_data.class_names
num_classes = len(class_names)

print("\nDetected Classes:", class_names)

if num_classes != 3:
    raise ValueError(
        "Dataset must contain only 3 folders: NORMAL, BACTERIAL_PNEUMONIA, VIRAL_PNEUMONIA"
    )


# ==================================================
# STEP 6: OPTIMIZE DATA LOADING
# ==================================================

AUTOTUNE = tf.data.AUTOTUNE

train_data = train_data.prefetch(AUTOTUNE)
val_data = val_data.prefetch(AUTOTUNE)


# ==================================================
# STEP 7: DATA AUGMENTATION
# ==================================================

augmentation = models.Sequential([
    layers.RandomFlip("horizontal"),
    layers.RandomRotation(0.05),
    layers.RandomZoom(0.1),
])


# ==================================================
# STEP 8: BUILD MODEL
# ==================================================

base_model = MobileNetV2(
    input_shape=(224, 224, 3),
    include_top=False,
    weights="imagenet"
)

base_model.trainable = False

model = models.Sequential([
    layers.Input(shape=(224, 224, 3)),

    augmentation,
    layers.Rescaling(1.0 / 255),

    base_model,

    layers.GlobalAveragePooling2D(),

    layers.Dense(128, activation="relu"),
    layers.Dropout(0.4),

    layers.Dense(num_classes, activation="softmax")
])


# ==================================================
# STEP 9: COMPILE MODEL
# ==================================================

model.compile(
    optimizer=tf.keras.optimizers.Adam(learning_rate=0.0001),
    loss="categorical_crossentropy",
    metrics=["accuracy"]
)

model.summary()


# ==================================================
# STEP 10: CALLBACKS
# ==================================================

callbacks = [
    EarlyStopping(
        monitor="val_loss",
        patience=5,
        restore_best_weights=True
    ),

    ModelCheckpoint(
        "best_curated_pneumonia_model.keras",
        monitor="val_accuracy",
        save_best_only=True,
        mode="max"
    ),

    ReduceLROnPlateau(
        monitor="val_loss",
        factor=0.2,
        patience=3,
        min_lr=0.000001
    )
]


# ==================================================
# STEP 11: TRAIN MODEL
# ==================================================

history = model.fit(
    train_data,
    validation_data=val_data,
    epochs=20,
    callbacks=callbacks
)


# ==================================================
# STEP 12: SAVE MODEL
# ==================================================

model.save("curated_pneumonia_3class_model.keras")
print("\nModel saved successfully.")                                                                                                                        # ==================================================
# STEP 13: ACCURACY AND LOSS GRAPH
# ==================================================

plt.figure(figsize=(10, 5))

plt.subplot(1, 2, 1)
plt.plot(history.history["accuracy"], marker="o", label="Training Accuracy")
plt.plot(history.history["val_accuracy"], marker="o", label="Validation Accuracy")
plt.title("Model Accuracy")
plt.xlabel("Epoch")
plt.ylabel("Accuracy")
plt.legend()

plt.subplot(1, 2, 2)
plt.plot(history.history["loss"], marker="o", label="Training Loss")
plt.plot(history.history["val_loss"], marker="o", label="Validation Loss")
plt.title("Model Loss")
plt.xlabel("Epoch")
plt.ylabel("Loss")
plt.legend()

plt.tight_layout()
plt.show()

# ==================================================
# STEP 14: USER IMAGE INPUT
# ==================================================

print("\n===== USER IMAGE PREDICTION =====")

image_path = input("Enter external chest X-ray image path: ").strip().replace('"', "")

if not os.path.exists(image_path):
    print("Image not found. Please check the path.")
else:
    img = tf.keras.utils.load_img(
        image_path,
        target_size=img_size
    )

    plt.imshow(img)
    plt.title("Input Chest X-Ray")
    plt.axis("off")
    plt.show()

    img_array = tf.keras.utils.img_to_array(img)
    img_array = np.expand_dims(img_array, axis=0)

    prediction = model.predict(img_array)[0]

    predicted_index = np.argmax(prediction)
    predicted_class = class_names[predicted_index]
    confidence = prediction[predicted_index] * 100

    print("\n===== PREDICTION RESULT =====")
    print("Predicted Class:", predicted_class)
    print("Confidence:", round(confidence, 2), "%")

    if confidence < 80:
        print("Warning: Confidence is below 80%. Use a clearer X-ray image.")          explain process with flowchart and algorithm with theory
