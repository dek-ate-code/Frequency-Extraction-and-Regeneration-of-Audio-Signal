import numpy as np
import pandas as pd
import librosa
import sounddevice as sd
from scipy.fft import fft, fftfreq
from scipy.signal import find_peaks
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
CSV_FILE = "audio_features.csv"
MIN_SAMPLES = 13
df = pd.read_csv(CSV_FILE)
if len(df) < MIN_SAMPLES:
    print("❌ Not enough data in CSV to train model")
    exit()
X = df.drop(["Label", "Age"], axis=1)
y_speaker = df["Label"]
y_age = df["Age"]
X_train, X_test, y_train, y_test = train_test_split(
    X, y_speaker, test_size=0.2, random_state=42
)
scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)
speaker_model = SVC(kernel="rbf", probability=True)
speaker_model.fit(X_train, y_train)
speaker_acc = accuracy_score(y_test, speaker_model.predict(X_test)) * 100
print(f"✅ Speaker model trained | Accuracy: {speaker_acc:.2f}%")
X_train, X_test, y_train, y_test = train_test_split(
    X, y_age, test_size=0.2, random_state=42
)
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)
age_model = SVC(kernel="rbf", probability=True)
age_model.fit(X_train, y_train)
age_acc = accuracy_score(y_test, age_model.predict(X_test)) * 100
print(f"✅ Age model trained | Accuracy: {age_acc:.2f}%")
choice = input("\n🎤 Record new audio (R) or Load wav (L)? [R/L]: ").strip().lower()
if choice == "r":
    duration = 14
    sr = 44100
    print("Recording...")
    audio = sd.rec(int(duration * sr), samplerate=sr, channels=1, dtype="int16")
    sd.wait()
    audio = audio.flatten().astype(np.float32)
else:
    path = input("Enter test wav path: ")
    audio, sr = librosa.load(path, sr=None)
audio = audio / np.max(np.abs(audio))
def extract_features(signal, sr):
    features = {}
    N = len(signal)
    yf = np.abs(fft(signal))[:N // 2]
    xf = fftfreq(N, 1 / sr)[:N // 2]
    features["Dominant Frequency"] = xf[np.argmax(yf)]
    peaks, _ = find_peaks(yf, height=np.max(yf) * 0.05)
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
test_features = extract_features(audio, sr)
X_new = pd.DataFrame([test_features])
X_new_scaled = scaler.transform(X_new)
speaker_pred = speaker_model.predict(X_new_scaled)[0]
speaker_conf = max(speaker_model.predict_proba(X_new_scaled)[0]) * 100
age_pred = age_model.predict(X_new_scaled)[0]
age_conf = max(age_model.predict_proba(X_new_scaled)[0]) * 100
print("\n🎯 FINAL PREDICTION RESULT")
print(f"Predicted Person : {speaker_pred} ({speaker_conf:.2f}%)")
print(f"Predicted Age : {age_pred} ({age_conf:.2f}%)")
