import time
import numpy as np
import pandas as pd
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix
from sklearn.preprocessing import LabelEncoder, MinMaxScaler
from sklearn.utils import resample
from rknn.api import RKNNLite
import os

# ========== Preprocessing Function ==========
def preprocess_sample_data(df_sample):
    df = df_sample.copy()
    df = df.drop(columns="id")

    df["age_group"] = df["age"].apply(lambda x: "Infant" if (x >= 0) & (x <= 2)
        else ("Child" if (x > 2) & (x <= 12)
        else ("Adolescent" if (x > 12) & (x <= 18)
        else ("Young Adults" if (x > 19) & (x <= 35)
        else ("Middle Aged Adults" if (x > 35) & (x <= 60)
        else "Old Aged Adults")))))

    df['bmi'] = df['bmi'].fillna(df.groupby(["gender", "ever_married", "age_group"])["bmi"].transform('mean'))
    df = df[(df["bmi"] < 66) & (df["bmi"] > 12)]
    df = df[(df["avg_glucose_level"] > 56) & (df["avg_glucose_level"] < 250)]
    df = df.drop(df[df["gender"] == "Other"].index)

    had_stroke = df[df["stroke"] == 1]
    no_stroke = df[df["stroke"] == 0]
    upsampled_had_stroke = resample(had_stroke, replace=True, n_samples=no_stroke.shape[0], random_state=123)
    upsampled_data = pd.concat([no_stroke, upsampled_had_stroke])

    cols = ['gender', 'hypertension', 'heart_disease', 'ever_married', 'work_type', 'Residence_type', 'smoking_status']
    dums = pd.get_dummies(upsampled_data[cols], dtype=int)

    expected_dummy_cols = [
        'gender_Female', 'gender_Male',
        'ever_married_No', 'ever_married_Yes',
        'work_type_Govt_job', 'work_type_Never_worked',
        'work_type_Private', 'work_type_Self-employed', 'work_type_children',
        'Residence_type_Rural', 'Residence_type_Urban',
        'smoking_status_Unknown', 'smoking_status_formerly smoked',
        'smoking_status_never smoked', 'smoking_status_smokes'
    ]

    for col in expected_dummy_cols:
        if col not in dums:
            dums[col] = 0
    dums = dums[expected_dummy_cols]

    model_data = pd.concat([upsampled_data.drop(columns=cols), dums], axis=1)
    encoder = LabelEncoder()
    model_data["age_group"] = encoder.fit_transform(model_data["age_group"])

    scaler = MinMaxScaler()
    for col in ['age', 'avg_glucose_level', 'bmi']:
        scaler.fit(model_data[[col]])
        model_data[col] = scaler.transform(model_data[[col]])

    return model_data

# ========== Load and preprocess data ==========
df_raw = pd.read_csv("Dataset/SP_sample.csv")
model_data = preprocess_sample_data(df_raw)

X_input = model_data.drop(columns=["stroke"]).astype(np.float32).values
y_true = model_data["stroke"].values

# ========== RKNN Inference ==========
rknn_lite = RKNNLite(verbose=True, verbose_file='./LOG/rknn_lite_log.txt')

ret = rknn_lite.load_rknn('./Models/SP_91.rknn')
assert ret == 0, 'Failed to load RKNN model.'

ret = rknn_lite.init_runtime(core_mask=RKNNLite.NPU_CORE_ALL)
assert ret == 0, 'Failed to init runtime.'

# Inference timing
inference_times = []
y_pred = []

for i in range(len(X_input)):
    start_time = time.time()
    output = rknn_lite.inference(inputs=[X_input[i:i+1]])
    inference_times.append(time.time() - start_time)

    if output is not None:
        y_pred.append(int(output[0][0][0] > 0.5))
    else:
        y_pred.append(0)  # fallback to 0 if failed

# ========== Metrics ==========
accuracy = accuracy_score(y_true, y_pred)
precision = precision_score(y_true, y_pred, zero_division=0)
recall = recall_score(y_true, y_pred, zero_division=0)
f1 = f1_score(y_true, y_pred, zero_division=0)
conf_mat = confusion_matrix(y_true, y_pred)

avg_inf_time = np.mean(inference_times)
std_inf_time = np.std(inference_times)
max_inf_time = np.max(inference_times)
min_inf_time = np.min(inference_times)

print("\n--- RKNN Model Evaluation ---")
print(f"Accuracy:  {accuracy:.4f}")
print(f"Precision: {precision:.4f}")
print(f"Recall:    {recall:.4f}")
print(f"F1 Score:  {f1:.4f}")
print(f"Confusion Matrix:\n{conf_mat}")
print(f"\n--- Inference Timing ---")
print(f"Average: {avg_inf_time*1000:.2f} ms")
print(f"Std Dev: {std_inf_time*1000:.2f} ms")
print(f"Max:     {max_inf_time*1000:.2f} ms")
print(f"Min:     {min_inf_time*1000:.2f} ms")

rknn_lite.release()
