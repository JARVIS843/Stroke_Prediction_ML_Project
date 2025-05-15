import os
import time
import numpy as np
import pandas as pd
from PIL import Image
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import classification_report, confusion_matrix
from rknnlite.api import RKNNLite

# -----------------------------
# Preprocessing Function
# -----------------------------
def preprocess_sample_data(sample_dir="Dataset/EYE_sample", image_size=224):
    df = pd.read_csv(os.path.join(sample_dir, "EYE_sample.csv"))
    df['image_path'] = df['image_id'].apply(lambda x: os.path.join(sample_dir, x))

    X_img = []
    for img_name in df['image_id']:
        img = Image.open(os.path.join(sample_dir, img_name)).convert("RGB").resize((image_size, image_size))
        X_img.append(np.array(img, np.float32) / 255.0)
    X_img = np.stack(X_img, axis=0)

    y_true = df['label'].tolist()
    return X_img, y_true

# -----------------------------
# Load Data
# -----------------------------
X_img, y_true = preprocess_sample_data()
BATCH_SIZE = 3

pad_len = (-len(X_img)) % BATCH_SIZE
if pad_len > 0:
    X_img = np.concatenate([X_img, np.zeros((pad_len, 224, 224, 3), dtype=np.float32)])
    y_true += ['__pad__'] * pad_len

# -----------------------------
# RKNN Inference
# -----------------------------
rknn_lite = RKNNLite(verbose=False)
print('--> Loading RKNN model')
ret = rknn_lite.load_rknn('./Models/EYE_81.rknn')
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

class_names = ['cataract', 'diabetic_retinopathy', 'glaucoma', 'normal']

for i in range(0, len(X_img), BATCH_SIZE):
    batch_img = X_img[i:i+BATCH_SIZE]

    start = time.time()
    output = rknn_lite.inference(inputs=[batch_img], data_format='nchw')
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
