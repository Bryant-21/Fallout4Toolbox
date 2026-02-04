# ---------------- CONFIG ----------------

INPUT_FILE = r"I:\Fallout4Mods\B21_PlasmaCaster\Data\Sound\FX\WPN\B21_PlasmaCaster\fire_single.wav"
import random

import librosa
import numpy as np
import soundfile as sf
from scipy.signal import butter, lfilter

# -------------------------
# CONFIG
# -------------------------


RPMS = [300, 450, 540, 660, 780, 900]

SHOT_COUNT = 12

TAIL_THRESHOLD_DB = -35
PITCH_VARIATION = 0.5
GAIN_VARIATION_DB = 2.0
JITTER_MS = 8


# -------------------------
# Helper Functions
# -------------------------

def db_to_linear(db):
    return 10 ** (db / 20)


def butter_highpass(cutoff, fs, order=4):
    nyq = 0.5 * fs
    normal_cutoff = cutoff / nyq
    return butter(order, normal_cutoff, btype='high', analog=False)


def highpass(data, cutoff, fs):
    b, a = butter_highpass(cutoff, fs)
    return lfilter(b, a, data)


def trim_tail(audio, sr, threshold_db):

    envelope = np.abs(audio)
    envelope_db = 20 * np.log10(envelope + 1e-8)

    cutoff_index = len(audio)

    for i in range(len(envelope_db) - 1, 0, -1):
        if envelope_db[i] > threshold_db:
            cutoff_index = i
            break

    return audio[:cutoff_index]


def random_pitch(audio, sr):
    semitone = random.uniform(-PITCH_VARIATION, PITCH_VARIATION)
    return librosa.effects.pitch_shift(audio, sr=sr, n_steps=semitone)


def random_gain(audio):
    gain_db = random.uniform(-GAIN_VARIATION_DB, GAIN_VARIATION_DB)
    return audio * db_to_linear(gain_db)


# -------------------------
# Load Source Once
# -------------------------

audio, sr = librosa.load(INPUT_FILE, sr=None)
base_shot = trim_tail(audio, sr, TAIL_THRESHOLD_DB)


# -------------------------
# Generate Per RPM
# -------------------------

for rpm in RPMS:

    shot_spacing_sec = 60.0 / rpm
    shot_spacing_samples = int(shot_spacing_sec * sr)

    final_length = shot_spacing_samples * SHOT_COUNT + len(audio)
    output = np.zeros(final_length)

    cursor = 0

    for i in range(SHOT_COUNT):

        shot = base_shot.copy()
        shot = random_pitch(shot, sr)
        shot = random_gain(shot)

        # Optional subtle realism
        if random.random() > 0.5:
            cutoff = random.uniform(80, 200)
            shot = highpass(shot, cutoff, sr)

        jitter = int(random.uniform(-JITTER_MS, JITTER_MS) * sr / 1000)
        placement = max(cursor + jitter, 0)

        end = placement + len(shot)
        output[placement:end] += shot

        cursor += shot_spacing_samples

    # Normalize
    output /= np.max(np.abs(output)) + 1e-8

    output_file = f"auto_fire_{rpm}.wav"
    sf.write(output_file, output, sr)

    print("Created:", output_file)