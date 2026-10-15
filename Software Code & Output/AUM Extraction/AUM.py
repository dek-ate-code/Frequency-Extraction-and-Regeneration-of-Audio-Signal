import numpy as np
import soundfile as sf
sr = 16000
duration = 11  # Total chant duration in seconds
t = np.linspace(0, duration, int(sr * duration), endpoint=False)
a_ratio = 0.2
u_ratio = 0.45
m_ratio = 0.45
a_samples = int(a_ratio * duration * sr)
u_samples = int(u_ratio * duration * sr)
m_samples = len(t) - a_samples - u_samples
t_a = np.linspace(0, a_samples / sr, a_samples, endpoint=False)
t_u = np.linspace(0, u_samples / sr, u_samples, endpoint=False)
t_m = np.linspace(0, m_samples / sr, m_samples, endpoint=False)
f0 = 136.10  # Base pitch
harmonics_a = [
    (0.8, f0),
    (0.6, 2 * f0),
    (0.4, 3 * f0),
    (0.2, 4 * f0),
    (0.15, 5 * f0),
    (0.1, 6 * f0),
]
a_part = np.zeros_like(t_a)
for amp, freq in harmonics_a:
    a_part += amp * np.sin(2 * np.pi * freq * t_a)
harmonics_u = [
    (0.8, f0),
    (0.8, 2 * f0),
    (0.6, 3 * f0),
    (0.4, 4 * f0),
    (0.2, 5 * f0),
    (0.1, 6 * f0),
]
u_part = np.zeros_like(t_u)
for amp, freq in harmonics_u:
    u_part += amp * np.sin(2 * np.pi * freq * t_u)
harmonics_m = [
    (1.0, f0),
    (0.25, 2 * f0),
    (0.1, 3 * f0),
]
m_part = np.zeros_like(t_m)
for amp, freq in harmonics_m:
    m_part += amp * np.sin(2 * np.pi * freq * t_m)
def crossfade_segments(seg1, seg2, overlap=0.05):
    overlap_len = int(len(seg1) * overlap)
    fade_out = np.linspace(1, 0, overlap_len)
    fade_in = np.linspace(0, 1, overlap_len)
    seg1[-overlap_len:] *= fade_out
    seg2[:overlap_len] *= fade_in
    return np.concatenate([seg1[:-overlap_len], seg1[-overlap_len:] + seg2[:overlap_len], seg2[overlap_len:]])
a_u = crossfade_segments(a_part, u_part)
aum = crossfade_segments(a_u, m_part)
fade_in_duration = int(0.3 * sr)
fade_out_duration = int(0.7 * sr)
envelope = np.ones_like(aum)
envelope[:fade_in_duration] *= np.linspace(0, 1, fade_in_duration)
envelope[-fade_out_duration:] *= np.linspace(1, 0, fade_out_duration)
aum *= envelope
aum /= np.max(np.abs(aum))  # Normalize
def simple_reverb(signal, sr, decay=0.5, delay=0.09):
    delay_samples = int(delay * sr)
    reverb = np.pad(signal, (delay_samples, 0))[:len(signal)] * decay
    return (signal + reverb) / (1 + decay)
aom_reverb = simple_reverb(aum, sr)
sf.write("aum_smooth_chant.wav", aom_reverb, sr)
print("✅ Saved: aum_smooth_chant.wav (A → U → M with smooth transition)")
