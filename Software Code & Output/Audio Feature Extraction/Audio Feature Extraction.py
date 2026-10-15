import numpy as np
import pandas as pd
import os
import librosa
import sounddevice as sd
from scipy.fft import fft, fftfreq
from scipy.signal import find_peaks
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score
import joblib
CSV_FILE = "audio_features.csv"
MIN_SAMPLES_TO_TRAIN = 13
choice = input("📥 Record (R) or Load (L)? [R/L]: ").strip().lower()
if choice == 'r':
    duration = 14
    sr = 44100
    print("🎤 Recording...")
    audio = sd.rec(int(duration * sr), samplerate=sr, channels=1, dtype='int16')
    sd.wait()
    audio = audio.flatten().astype(np.float32)
else:
    path = input("🔍 Enter .wav file path: ")
    audio, sr = librosa.load(path, sr=None)

audio = audio / np.max(np.abs(audio))

features = {}
N = len(signal)
yf = np.abs(fft(signal))[:N//2]
xf = fftfreq(N, 1/sr)[:N//2]
features["Dominant Frequency"] = xf[np.argmax(yf)]
peaks, _ = find_peaks(yf, height=np.max(yf)*0.05)
valid_peaks = xf[peaks]
valid_peaks = valid_peaks[(valid_peaks > 70) & (valid_peaks < 400)]
base_freq = valid_peaks[0] if len(valid_peaks) > 0 else 0
features["Base Frequency"] = base_freq
for i in range(1, 7):
        idx = np.argmin(np.abs(xf - i * base_freq))
        features[f"Harmonics {i}"] = xf[idx]
frame_len = int(0.025 * sr)
hop_len = int(0.010 * sr)
features["Average Energy"] = np.mean([
        np.sum(signal[i:i+frame_len]**2)
        for i in range(0, len(signal)-frame_len, hop_len)
    ])
pitch = librosa.yin(signal, fmin=50, fmax=500, sr=sr)
features["Average Pitch"] = np.mean(pitch[pitch > 0])

mfcc = librosa.feature.mfcc(y=signal, sr=sr, n_mfcc=13)
for i in range(13):
        features[f"MFCC {i+1}"] = np.mean(mfcc[i])
return features
features = extract_features(audio, sr)
label = input("👤 Enter speaker/person label: ")
age_group = input(
    "🎂 Enter age : "
)
features["Label"] = label
features["Age"] = age_group
df_new = pd.DataFrame([features])
df_new.to_csv(
    CSV_FILE,
    mode="a",
    header=not os.path.exists(CSV_FILE),
    index=False
)
print("✅ Sample (with age) added to dataset")
df = pd.read_csv(CSV_FILE)
total_samples = len(df)
print(f"📊 Total samples in dataset: {total_samples}")
if total_samples < MIN_SAMPLES_TO_TRAIN:
    print("\n⚠️ NOT TRAINING ML MODEL YET")
    print(f"👉 Minimum {MIN_SAMPLES_TO_TRAIN} samples required")
    exit()
print("\n🤖 Training SPEAKER model...")
X = df.drop(["Label", "Age"], axis=1)
y_speaker = df["Label"]
X_train, X_test, y_train, y_test = train_test_split(
    X, y_speaker, test_size=0.2, random_state=42
)
scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)
speaker_model = SVC(kernel="rbf", probability=True)
speaker_model.fit(X_train, y_train)
speaker_acc = accuracy_score(y_test, speaker_model.predict(X_test)) * 100
print(f"🎯 Speaker Accuracy: {speaker_acc:.2f}%")
joblib.dump(speaker_model, "speaker_model.pkl")
joblib.dump(scaler, "scaler.pkl")
print("\n🤖 Training AGE model...")
y_age = df["Age"]
X_train, X_test, y_train, y_test = train_test_split(
    X, y_age, test_size=0.2, random_state=42
)
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)
age_model = SVC(kernel="rbf", probability=True)
age_model.fit(X_train, y_train)
age_acc = accuracy_score(y_test, age_model.predict(X_test)) * 100
print(f"🎯 Age Accuracy: {age_acc:.2f}%")
joblib.dump(age_model, "age_model.pkl")
X_new = scaler.transform(df_new.drop(["Label", "Age"], axis=1))
speaker_pred = speaker_model.predict(X_new)[0]
speaker_conf = max(speaker_model.predict_proba(X_new)[0]) * 100
age_pred = age_model.predict(X_new)[0]
age_conf = max(age_model.predict_proba(X_new)[0]) * 100
print("\n🎤 FINAL PREDICTION RESULT")
print(f"Speaker   : {speaker_pred} ({speaker_conf:.2f}%)")
print(f"Age : {age_pred} ({age_conf:.2f}%)")
