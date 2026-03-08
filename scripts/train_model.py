import os
import pickle
import numpy as np
import pandas as pd
import tensorflow as tf
import cv2

from tensorflow.keras import layers, models
from tensorflow.keras.applications import MobileNetV2
from tensorflow.keras.applications.mobilenet_v2 import preprocess_input
from tensorflow.keras.callbacks import EarlyStopping
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.utils.class_weight import compute_class_weight

# ================= PERFORMANCE SETTINGS =================
tf.keras.backend.clear_session()
tf.config.optimizer.set_jit(True)  # XLA acceleration

# ================= PATHS =================
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
IMAGE_DIR = os.path.join(BASE_DIR, "data", "processed", "Blood_Cancer_Classified")
CSV_PATH = os.path.join(BASE_DIR, "data", "processed", "blood_cancer_data.csv")
SAVE_DIR = os.path.join(BASE_DIR, "models", "saved")

os.makedirs(SAVE_DIR, exist_ok=True)

IMG_SIZE = (160, 160)   # 🔥 Smaller = Faster
BATCH_SIZE = 16
EPOCHS = 12              # 🔥 Short training

# ================= LOAD CSV =================
df = pd.read_csv(CSV_PATH)

clinical_cols = ["wbc", "rbc", "platelets", "hb", "blasts", "age"]
X_clinical = df[clinical_cols].values.astype(np.float32)

unique_labels = sorted(df["label"].unique())
label_mapping = {label: idx for idx, label in enumerate(unique_labels)}
y = df["label"].map(label_mapping).values

# ================= LOAD IMAGE DATA =================
class_names = sorted(
    [d for d in os.listdir(IMAGE_DIR)
     if os.path.isdir(os.path.join(IMAGE_DIR, d))]
)

class_to_index = {name: idx for idx, name in enumerate(class_names)}

image_paths = []
image_labels = []

valid_ext = (".png", ".jpg", ".jpeg", ".bmp", ".webp", ".jfif", ".tif")

for cls in class_names:
    folder = os.path.join(IMAGE_DIR, cls)
    for file in os.listdir(folder):
        if file.lower().endswith(valid_ext):
            image_paths.append(os.path.join(folder, file))
            image_labels.append(class_to_index[cls])

image_paths = np.array(image_paths)
image_labels = np.array(image_labels)

# ================= PAIR (FAST VERSION - NO REPEAT) =================
paired_img_paths = []
paired_clinical = []
paired_labels = []

for cls_name, cls_index in class_to_index.items():

    img_idx = np.where(image_labels == cls_index)[0]
    cl_idx = np.where(y == cls_index)[0]

    if len(img_idx) == 0 or len(cl_idx) == 0:
        continue

    n = min(len(img_idx), len(cl_idx))

    for i, j in zip(img_idx[:n], cl_idx[:n]):
        paired_img_paths.append(image_paths[i])
        paired_clinical.append(X_clinical[j])
        paired_labels.append(cls_index)

paired_img_paths = np.array(paired_img_paths)
paired_clinical = np.array(paired_clinical)
paired_labels = np.array(paired_labels)

# Reindex labels
unique_after = sorted(np.unique(paired_labels))
label_map = {old: new for new, old in enumerate(unique_after)}
paired_labels = np.array([label_map[l] for l in paired_labels])
NUM_CLASSES = len(unique_after)

print("Training Classes:", unique_after)
print("Dataset Size:", len(paired_labels))

# ================= SPLIT =================
X_img_train, X_img_test, X_cl_train, X_cl_test, y_train, y_test = train_test_split(
    paired_img_paths,
    paired_clinical,
    paired_labels,
    test_size=0.2,
    random_state=42,
    stratify=paired_labels
)

# ================= SCALE CLINICAL =================
scaler = StandardScaler()
X_cl_train = scaler.fit_transform(X_cl_train)
X_cl_test = scaler.transform(X_cl_test)

with open(os.path.join(SAVE_DIR, "scaler.pkl"), "wb") as f:
    pickle.dump(scaler, f)

# ================= IMAGE LOADER =================
def load_sample(img_path, clinical_features, label):
    img_path = img_path.numpy().decode("utf-8")
    img = cv2.imread(img_path)
    img = cv2.resize(img, IMG_SIZE)
    img = preprocess_input(img.astype(np.float32))
    return img, clinical_features, label

def tf_wrapper(img_path, clinical_features, label):
    img, clinical_features, label = tf.py_function(
        load_sample,
        [img_path, clinical_features, label],
        [tf.float32, tf.float32, tf.int32],
    )
    img.set_shape((IMG_SIZE[0], IMG_SIZE[1], 3))
    clinical_features.set_shape((6,))
    label.set_shape(())
    return (img, clinical_features), label

train_ds = tf.data.Dataset.from_tensor_slices((X_img_train, X_cl_train, y_train))
train_ds = train_ds.map(tf_wrapper).batch(BATCH_SIZE).prefetch(tf.data.AUTOTUNE)

test_ds = tf.data.Dataset.from_tensor_slices((X_img_test, X_cl_test, y_test))
test_ds = test_ds.map(tf_wrapper).batch(BATCH_SIZE).prefetch(tf.data.AUTOTUNE)

# ================= MODEL =================
image_input = layers.Input(shape=(IMG_SIZE[0], IMG_SIZE[1], 3))

base_model = MobileNetV2(
    input_shape=(IMG_SIZE[0], IMG_SIZE[1], 3),
    include_top=False,
    weights="imagenet"
)

base_model.trainable = False  # 🔥 No fine-tuning for speed

x = base_model(image_input, training=False)
x = layers.GlobalAveragePooling2D()(x)
x = layers.Dense(128, activation="relu")(x)
x = layers.Dropout(0.4)(x)

clinical_input = layers.Input(shape=(6,))
c = layers.Dense(64, activation="relu")(clinical_input)

combined = layers.concatenate([x, c])
combined = layers.Dense(128, activation="relu")(combined)
output = layers.Dense(NUM_CLASSES, activation="softmax")(combined)

model = models.Model(inputs=[image_input, clinical_input], outputs=output)

model.compile(
    optimizer=tf.keras.optimizers.Adam(learning_rate=1e-4),
    loss="sparse_categorical_crossentropy",
    metrics=["accuracy"]
)

# ================= TRAIN =================
model.fit(
    train_ds,
    validation_data=test_ds,
    epochs=EPOCHS,
    callbacks=[EarlyStopping(patience=3, restore_best_weights=True)]
)

model.save(os.path.join(SAVE_DIR, "combined_model_fast.h5"))

print("🔥 FAST TRAINING COMPLETED (Under 20 Minutes)")