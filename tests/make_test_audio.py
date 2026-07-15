import numpy as np
import soundfile as sf

rng = np.random.default_rng(0)
sr = 16000
duration_s = 6.0
t = np.linspace(0, duration_s, int(sr * duration_s), endpoint=False)

envelope = 0.3 * (np.sin(2 * np.pi * 3.0 * t) ** 2 + 0.2)
voiced = (
    0.6 * np.sin(2 * np.pi * 180 * t)
    + 0.3 * np.sin(2 * np.pi * 360 * t)
    + 0.1 * np.sin(2 * np.pi * 720 * t)
)
audio = (envelope * voiced + 0.02 * rng.standard_normal(t.shape)).astype(np.float32)

mask = np.ones_like(audio)
mask[: int(0.5 * sr)] = 0.0
mask[int(3.0 * sr) : int(3.4 * sr)] = 0.0
mask[int(5.5 * sr) :] = 0.0
audio = audio * mask

sf.write("tests/sample.wav", audio, sr)
print("written tests/sample.wav", audio.shape, sr)
