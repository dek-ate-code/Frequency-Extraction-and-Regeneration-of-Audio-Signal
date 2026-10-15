import numpy as np
import matplotlib.pyplot as plt
from scipy.io import wavfile
from scipy.fft import fft, fftfreq
import sounddevice as sd
import scipy.io.wavfile as wav
from scipy.signal import find_peaks
import os
import librosa
import librosa.display
choice = input("📥 Record (R) or Load (L)? [R/L]: ").strip().lower()
if choice == 'l':
filename = input("🔍 Enter .wav path: ").strip()
if not os.path.exists(filename):
print("❌ File not found")
exit()
else:
duration = 8
sample_rate = 44100
print("🎤 Recording...")
recorded = sd.rec(int(duration * sample_rate),
samplerate=sample_rate,
channels=1,
dtype='int16')
sd.wait()
filename = input("💾 Filename (no .wav): ") + ".wav"
wav.write(filename, sample_rate, recorded)
sample_rate, signal = wavfile.read(filename)
if signal.ndim > 1:
signal = signal[:, 0]
signal = signal / np.max(np.abs(signal))
duration = len(signal) / sample_rate
time = np.linspace(0, duration, len(signal))
N = len(signal)
yf = np.abs(fft(signal))[:N // 2]
xf = fftfreq(N, 1 / sample_rate)[:N // 2]
peaks, _ = find_peaks(yf, height=np.max(yf) * 0.05)
dominant_freq = xf[np.argmax(yf)]
valid_peaks = xf[peaks]
valid_peaks = valid_peaks[(valid_peaks > 70) & (valid_peaks < 400)]
base_freq = valid_peaks[0] if len(valid_peaks) > 0 else 0
print(f"\n🔊 Dominant Frequency: {dominant_freq:.2f} Hz")
print(f"🎵 Base Frequency: {base_freq:.2f} Hz")
harmonics = []
harmonic_magnitudes = []
if base_freq > 0:
for n in range(1, 7):
target = n * base_freq
idx = np.argmin(np.abs(xf - target))
harmonics.append(xf[idx])
harmonic_magnitudes.append(yf[idx])
print("\n🎶 Harmonics (H1–H6):")
for i, (f, m) in enumerate(zip(harmonics, harmonic_magnitudes), start=1):
print(f"H{i}: {f:.2f} Hz | Magnitude: {m:.4f}")
frame_len = int(0.025 * sample_rate)
hop_len = int(0.010 * sample_rate)
energy = np.array([
np.sum(signal[i:i+frame_len]**2)
for i in range(0, len(signal)-frame_len, hop_len)
])
def estimate_pitch(frame, fs):
frame -= np.mean(frame)
corr = np.correlate(frame, frame, mode='full')
corr = corr[len(corr)//2:]
pks, _ = find_peaks(corr)
if len(pks) < 2:
return 0
return fs / pks[1]
pitches = np.array([
estimate_pitch(signal[i:i+frame_len], sample_rate)
for i in range(0, len(signal)-frame_len, hop_len)
])
mfcc = librosa.feature.mfcc(
y=signal,
sr=sample_rate,
n_mfcc=13,
n_fft=1024,
hop_length=hop_len
)
plt.figure("Time Domain Signal")
plt.plot(time, signal)
plt.title("Time Domain Speech Signal")
plt.xlabel("Time (s)")
plt.ylabel("Amplitude")
plt.grid()
plt.figure("FFT with Harmonics")
plt.plot(xf, yf, label="FFT Spectrum")
plt.axvline(dominant_freq, color='red', linestyle='--',
label=f"Dominant: {dominant_freq:.2f} Hz")
if base_freq > 0:
plt.axvline(base_freq, color='green', linestyle='--',
label=f"Base: {base_freq:.2f} Hz")
for i, (f, m) in enumerate(zip(harmonics, harmonic_magnitudes), start=1):
plt.axvline(f, linestyle=':', color='purple',
label=f"H{i}: {f:.2f} Hz ({m:.3f})")
plt.xlim(0, 2000)
plt.xlabel("Frequency (Hz)")
plt.ylabel("Magnitude")
plt.title("FFT Spectrum with Harmonics (H1–H6)")
plt.legend(fontsize=8)
plt.grid()
t = np.linspace(0, 0.02, int(sample_rate * 0.02), endpoint=False)
sine_wave = np.sin(2 * np.pi * dominant_freq * t)
plt.figure("Sine Wave (Dominant Frequency)")
plt.plot(t, sine_wave)
plt.title(f"Sine Wave at {dominant_freq:.2f} Hz")
plt.xlabel("Time (s)")
plt.ylabel("Amplitude")
plt.grid()
plt.figure("Short-Time Energy")
plt.plot(energy)
plt.title("Short-Time Energy")
plt.xlabel("Frame Index")
plt.ylabel("Energy")
plt.grid()
plt.figure("Pitch Contour")
plt.plot(pitches)
plt.title("Pitch Contour (Hz)")
plt.xlabel("Frame Index")
plt.ylabel("Frequency (Hz)")
plt.ylim(0, 500)
plt.grid()
plt.figure("MFCC")
librosa.display.specshow(mfcc, x_axis='time', sr=sample_rate, cmap='coolwarm')
plt.colorbar(label="MFCC Coefficients")
plt.title("MFCC")
plt.figure("Spectrogram")
plt.specgram(signal, NFFT=1024, Fs=sample_rate, noverlap=512, cmap='inferno')
plt.ylim(0, 10000)
plt.xlabel("Time (s)")
plt.ylabel("Frequency (Hz)")
plt.title("Spectrogram")
plt.colorbar(label="Intensity (dB)")
plt.show()
