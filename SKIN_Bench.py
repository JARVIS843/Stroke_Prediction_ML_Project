import os
import time
import numpy as np
import pandas as pd
from PIL import Image
from sklearn.preprocessing import StandardScaler, OneHotEncoder, LabelEncoder
from sklearn.compose import ColumnTransformer
from sklearn.metrics import classification_report, confusion_matrix
from rknnlite.api import RKNNLite

# -----------------------------
# Preprocessing Function
# -----------------------------
def preprocess_sample_data(sample_dir="Dataset/SKIN_sample", image_size=224):
    df = pd.read_csv(os.path.join(sample_dir, "SKIN_sample.csv"))
    df['image_path'] = df['image_id'].apply(lambda x: os.path.join(sample_dir, f"{x}.jpg"))

    X_img = []
    for img_id in df['image_id']:
        img = Image.open(os.path.join(sample_dir, f"{img_id}.jpg"))\
                   .convert("RGB").resize((image_size, image_size))
        X_img.append(np.array(img, np.float32) / 255.0)
    X_img = np.stack(X_img, axis=0)

    meta_df = df[['age', 'sex', 'localization']].copy()
    meta_df['age'].fillna(meta_df['age'].mean(), inplace=True)

    sex_categories = ['male', 'female']
    loc_categories = ['abdomen', 'acral', 'back', 'chest', 'ear', 'face', 'foot',
                      'genital', 'hand', 'lower extremity', 'neck', 'scalp',
                      'trunk', 'unknown', 'upper extremity']

    preprocessor = ColumnTransformer([
        ('num', StandardScaler(), ['age']),
        ('cat', OneHotEncoder(categories=[sex_categories, loc_categories],
                              handle_unknown='ignore'),
         ['sex', 'localization'])
    ])
    X_meta = preprocessor.fit_transform(meta_df).astype(np.float32)
    if hasattr(X_meta, "toarray"):
        X_meta = X_meta.toarray()
    X_meta = np.asarray(X_meta, dtype=np.float32)

    y_true = df['label_name'].tolist()
    return X_img, X_meta, y_true

# -----------------------------
# Load Data
# -----------------------------
X_img, X_meta, y_true = preprocess_sample_data()
BATCH_SIZE = 3

pad_len = (-len(X_img)) % BATCH_SIZE
if pad_len > 0:
    X_img = np.concatenate([X_img, np.zeros((pad_len, 224, 224, 3), dtype=np.float32)])
    X_meta = np.concatenate([X_meta, np.zeros((pad_len, X_meta.shape[1]), dtype=np.float32)])
    y_true += ['__pad__'] * pad_len

# -----------------------------
# RKNN Inference
# -----------------------------
rknn_lite = RKNNLite(verbose = False)
print('--> Loading RKNN model')
ret = rknn_lite.load_rknn('./Models/SKIN_3.rknn')
if ret != 0:
    print('Failed to load RKNN model')
    exit(ret)
print('Done.')

print('--> Init runtime on all NPU cores')
ret = rknn_lite.init_runtime(core_mask=RKNNLite.NPU_CORE_ALL)
if ret != 0:
    print('Failed to init runtime')
    exit(ret)
print('Done.')

# -----------------------------
# Inference Loop
# -----------------------------
inference_times = []
y_pred = []

class_names = [
    'Actinic Keratoses',
    'Basal Cell Carcinoma',
    'Benign Keratosis-like Lesions',
    'Dermatofibroma',
    'Melanocytic Nevi',
    'Melanoma',
    'Vascular Lesions'
]

for i in range(0, len(X_img), BATCH_SIZE):
    batch_img = X_img[i:i+BATCH_SIZE]
    batch_meta = X_meta[i:i+BATCH_SIZE]

    start = time.time()
    output = rknn_lite.inference(inputs=[batch_img, batch_meta], data_format = 'nchw')
    inference_times.append(time.time() - start)

    if output is not None:
        preds = np.argmax(output[0], axis=1).tolist()
        y_pred.extend(preds)
    else:
        y_pred.extend([0] * BATCH_SIZE)

# -----------------------------
# Evaluation
# -----------------------------
y_true = [y for y in y_true if y != '__pad__']
y_pred = y_pred[:len(y_true)]

le = LabelEncoder().fit(class_names)
y_true_enc = [class_names.index(lbl) for lbl in y_true]

print('\n=== Classification Report ===')
print(classification_report(
    y_true_enc, y_pred,
    labels=list(range(len(class_names))),
    target_names=class_names,
    zero_division=0
))

print('\n=== Confusion Matrix ===')
cm = confusion_matrix(y_true_enc, y_pred, labels=list(range(len(class_names))))
print(pd.DataFrame(cm, index=class_names, columns=class_names))

# -----------------------------
# Timing
# -----------------------------
avg_time = np.mean(inference_times)
print(f'\nAverage Inference Time per Batch ({BATCH_SIZE} samples): {avg_time:.4f} seconds')
print(f'Average Time per Sample: {avg_time / BATCH_SIZE:.4f} seconds')

# -----------------------------
# Cleanup
# -----------------------------
rknn_lite.release()
print('RKNNLite released.')
