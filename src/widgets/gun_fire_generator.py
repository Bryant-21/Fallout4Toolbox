import os
import random
import struct
import traceback
import wave

import librosa
import numpy as np
from PySide6.QtWidgets import QFileDialog
from qfluentwidgets import FluentIcon as FIF, InfoBar, PrimaryPushButton, PushSettingCard, SwitchSettingCard
from scipy.signal import butter, lfilter

from src.utils.appconfig import cfg
from src.utils.cards import TextSettingCard, DoubleSpinSettingCard, SpinSettingCard
from src.utils.helpers import BaseWidget
from src.utils.logging_utils import logger


def db_to_linear(db):
    return 10 ** (db / 20)

def butter_highpass(cutoff, fs, order=4):
    nyq = 0.5 * fs
    normal_cutoff = cutoff / nyq
    return butter(order, normal_cutoff, btype='high', analog=False)

def highpass(data, cutoff, fs):
    b, a = butter_highpass(cutoff, fs)
    return lfilter(b, a, data)

def lowpass(data, cutoff, fs):
    nyq = 0.5 * fs
    normal = cutoff / nyq
    b, a = butter(4, normal, btype='low')
    return lfilter(b, a, data)

def tilt_eq(audio, amount):
    """
    amount > 0 = brighter
    amount < 0 = bassier
    """
    t = np.linspace(-1, 1, len(audio))
    tilt = 1 + (t * amount * 0.1)
    return audio * tilt

def trim_tail(audio, sr, threshold_db):
    envelope = np.abs(audio)
    envelope_db = 20 * np.log10(envelope + 1e-8)
    cutoff_index = len(audio)
    for i in range(len(envelope_db) - 1, 0, -1):
        if envelope_db[i] > threshold_db:
            cutoff_index = i
            break
    return audio[:cutoff_index]

def random_pitch(audio, sr, pitch_variation):
    if pitch_variation <= 0:
        return audio
    semitone = random.uniform(-pitch_variation, pitch_variation)
    return librosa.effects.pitch_shift(audio, sr=sr, n_steps=semitone)

def random_gain(audio, gain_variation_db):
    if gain_variation_db <= 0:
        return audio
    gain_db = random.uniform(-gain_variation_db, gain_variation_db)
    return audio * db_to_linear(gain_db)


def build_cue_chunk(marker_defs):
    """
    marker_defs: list of (position, label)
    """
    cue_count = len(marker_defs)
    chunk = struct.pack("<4sI", b"cue ", 4 + cue_count * 24)
    chunk += struct.pack("<I", cue_count)

    for i, (pos, label_text) in enumerate(marker_defs):
        chunk += struct.pack(
            "<IIIIII",
            i + 1,  # cue ID
            pos,  # position
            0x64617461,  # 'data'
            0,
            0,
            pos
        )
    return chunk


def build_smpl_chunk(loop_start, loop_end, sample_rate=48000):
    """Build the sampler chunk that actually defines the loop"""
    # smpl chunk structure
    chunk = struct.pack("<4sI", b"smpl", 60)  # 60 bytes for chunk data

    chunk += struct.pack("<I", 0)  # manufacturer
    chunk += struct.pack("<I", 0)  # product
    chunk += struct.pack("<I", int(1e9 / sample_rate))  # sample period (nanoseconds)
    chunk += struct.pack("<I", 60)  # MIDI unity note (middle C)
    chunk += struct.pack("<I", 0)  # MIDI pitch fraction
    chunk += struct.pack("<I", 0)  # SMPTE format
    chunk += struct.pack("<I", 0)  # SMPTE offset
    chunk += struct.pack("<I", 1)  # number of sample loops (1 loop)
    chunk += struct.pack("<I", 0)  # sampler data

    # Loop definition (24 bytes)
    chunk += struct.pack("<I", 0)  # cue point ID (or 0)
    chunk += struct.pack("<I", 0)  # type 0 is forward loop
    chunk += struct.pack("<I", loop_start)  # loop start in samples
    chunk += struct.pack("<I", loop_end)  # loop end in samples
    chunk += struct.pack("<I", 0)  # fraction
    chunk += struct.pack("<I", 0)  # play count (0 = infinite)

    return chunk


def build_label_chunk(marker_defs):
    """
    marker_defs: list of (position, label)
    """
    entries = []

    def label(marker_id, text):
        text_bytes = text.encode("ascii") + b"\x00"
        size = 4 + len(text_bytes)
        return (
                struct.pack("<4sI", b"labl", size)
                + struct.pack("<I", marker_id)
                + text_bytes
        )

    for i, (pos, label_text) in enumerate(marker_defs):
        entries.append(label(i + 1, label_text))

    adtl_data = b"".join(entries)
    chunk = struct.pack("<4sI", b"LIST", len(adtl_data) + 4)
    chunk += b"adtl" + adtl_data

    return chunk

class GunFireGeneratorWidget(BaseWidget):
    def __init__(self, parent, text):
        super().__init__(parent=parent, text=text, vertical=True)

        # --- Base cards ---
        self.input_file_card = PushSettingCard(
            self.tr('Source Audio File'),
            FIF.MUSIC,
            self.tr("Select the single shot WAV file"),
            cfg.gf_input_file.value
        )
        self.output_folder_card = PushSettingCard(
            self.tr('Output Directory'),
            FIF.FOLDER,
            self.tr("Where to save generated RPM files"),
            cfg.gf_output_folder.value
        )
        self.rpms_card = TextSettingCard(
            cfg.gf_rpms,
            FIF.EDIT,
            self.tr("RPMs to Generate"),
            cfg.gf_rpms.value or self.tr('e.g. 300, 450, 600')
        )
        
        self.shot_count_card = SpinSettingCard(
            cfg.gf_shot_count,
            FIF.SCROLL,
            self.tr("Shot Count"),
            self.tr("Number of shots in the automatic fire sequence")
        )

        self.tail_threshold_card = DoubleSpinSettingCard(
            cfg.gf_tail_threshold,
            FIF.CUT,
            self.tr("Tail Threshold (dB)")
        )
        
        self.pitch_variation_card = DoubleSpinSettingCard(
            cfg.gf_pitch_variation,
            FIF.ARROW_DOWN,
            self.tr("Pitch Variation (semitones)")
        )

        self.gain_variation_card = DoubleSpinSettingCard(
            cfg.gf_gain_variation,
            FIF.VOLUME,
            self.tr("Gain Variation (dB)")
        )

        self.jitter_card = SpinSettingCard(
            cfg.gf_jitter_ms,
            FIF.SYNC,
            self.tr("Jitter (ms)"),
            self.tr("Random timing variation between shots")
        )
        
        self.highpass_card = SwitchSettingCard(
            FIF.FILTER,
            self.tr("Random Highpass Filter"),
            self.tr("Apply a subtle highpass filter randomly for realism"),
            cfg.gf_highpass_enabled
        )

        self.tilt_card = DoubleSpinSettingCard(
            cfg.gf_tilt,
            FIF.UP,
            self.tr("Tilt EQ"),
        )

        self.base_reinforcement_card = SwitchSettingCard(
            FIF.PENCIL_INK,
            self.tr("Parallel Base Reinforcement"),
            self.tr("Adds a low-passed version of the audio for more punch"),
            cfg.gf_base_reinforcement
        )

        self.addToFrame(self.input_file_card)
        self.addToFrame(self.output_folder_card)
        self.addToFrame(self.rpms_card)
        self.addToFrame(self.shot_count_card)
        self.addToFrame(self.tail_threshold_card)
        self.addToFrame(self.pitch_variation_card)
        self.addToFrame(self.gain_variation_card)
        self.addToFrame(self.jitter_card)
        self.addToFrame(self.highpass_card)
        self.addToFrame(self.tilt_card)
        self.addToFrame(self.base_reinforcement_card)

        self.boxLayout.addStretch(1)

        self.run_button = PrimaryPushButton(icon=FIF.PLAY, text=self.tr("Generate Fire Sounds"))
        self.run_button.clicked.connect(self.on_run)
        self.input_file_card.clicked.connect(self.on_select_input_file)
        self.output_folder_card.clicked.connect(self.on_select_output_folder)

        self.addButtonBarToBottom(self.run_button)

    def on_select_input_file(self):
        file_path, _ = QFileDialog.getOpenFileName(self, self.tr("Select source audio"), cfg.gf_input_file.value or "", "Audio Files (*.wav)")
        if file_path:
            cfg.gf_input_file.value = file_path
            self.input_file_card.setContent(file_path)
            
            # Automatically use the same directory as the selected file
            directory = os.path.dirname(file_path)
            cfg.gf_output_folder.value = directory
            self.output_folder_card.setContent(directory)

    def on_select_output_folder(self):
        directory = QFileDialog.getExistingDirectory(self, self.tr("Select output directory"), cfg.gf_output_folder.value or "")
        if directory:
            cfg.gf_output_folder.value = directory
            self.output_folder_card.setContent(directory)

    def on_run(self):
        input_file = cfg.gf_input_file.value
        output_dir = cfg.gf_output_folder.value
        rpms_raw = cfg.gf_rpms.value
        
        if not input_file or not os.path.exists(input_file):
            InfoBar.error(title=self.tr("Error"), content=self.tr("Please select a valid input WAV file."), parent=self)
            return
        if not output_dir or not os.path.isdir(output_dir):
            InfoBar.error(title=self.tr("Error"), content=self.tr("Please select a valid output directory."), parent=self)
            return
        
        try:
            rpms = [int(r.strip()) for r in rpms_raw.split(',') if r.strip()]
        except ValueError:
            InfoBar.error(title=self.tr("Error"), content=self.tr("Invalid RPM list. Please use comma-separated numbers."), parent=self)
            return

        if not rpms:
            InfoBar.error(title=self.tr("Error"), content=self.tr("Please enter at least one RPM value."), parent=self)
            return

        self.run_button.setEnabled(False)
        try:
            self.generate_fire_sounds(input_file, output_dir, rpms)
            InfoBar.success(title=self.tr("Success"), content=self.tr("Fire sounds generated successfully!"), parent=self)
        except Exception as e:
            logger.error(traceback.format_exc())
            InfoBar.error(title=self.tr("Error"), content=str(e), parent=self)
        finally:
            self.run_button.setEnabled(True)

    def generate_fire_sounds(self, input_file, output_dir, rpms):
        shot_count = cfg.gf_shot_count.value
        tail_threshold = cfg.gf_tail_threshold.value
        pitch_variation = cfg.gf_pitch_variation.value
        gain_variation = cfg.gf_gain_variation.value
        jitter_ms = cfg.gf_jitter_ms.value
        highpass_enabled = cfg.gf_highpass_enabled.value
        tilt_amount = cfg.gf_tilt.value
        base_reinforcement = cfg.gf_base_reinforcement.value

        audio, sr = librosa.load(input_file, sr=None)
        base_shot = trim_tail(audio, sr, tail_threshold)
        
        # Apply Tilt EQ to the base shot if configured
        if tilt_amount != 0:
            base_shot = tilt_eq(base_shot, tilt_amount)

        # Apply Parallel Base Reinforcement if enabled
        if base_reinforcement:
            # Add a low-passed version (e.g. 150Hz) to the original
            low = lowpass(base_shot, 150, sr)
            base_shot = base_shot + low

        for rpm in rpms:
            shot_spacing_sec = 60.0 / rpm
            shot_spacing_samples = int(shot_spacing_sec * sr)

            final_length = shot_spacing_samples * shot_count + len(audio)
            output = np.zeros(final_length)
            shot_markers = []

            cursor = 0
            for i in range(shot_count):
                shot = base_shot.copy()
                shot = random_pitch(shot, sr, pitch_variation)
                shot = random_gain(shot, gain_variation)

                if highpass_enabled and random.random() > 0.5:
                    cutoff = random.uniform(80, 200)
                    shot = highpass(shot, cutoff, sr)

                jitter = int(random.uniform(-jitter_ms, jitter_ms) * sr / 1000)
                placement = max(cursor + jitter, 0)

                end = placement + len(shot)
                if end > len(output):
                    # Extend output if needed
                    padding = np.zeros(end - len(output))
                    output = np.concatenate([output, padding])
                
                output[placement:end] += shot
                shot_markers.append(placement)
                cursor += shot_spacing_samples

            # Normalize
            max_val = np.max(np.abs(output))
            if max_val > 0:
                output /= max_val + 1e-8

            # Convert to 16-bit PCM
            pcm = (output * 32767).astype(np.int16)

            # Determine output filename
            base_name = os.path.splitext(os.path.basename(input_file))[0]
            if "single" in base_name.lower():
                # Replace "single" with "auto" (preserving case if possible, but standardizing to lowercase is safer for game files)
                import re
                base_name = re.sub(r'single', 'auto', base_name, flags=re.IGNORECASE)
            else:
                base_name = f"{base_name}_auto"
            
            output_file = os.path.join(output_dir, f"{base_name}_{rpm}.wav")
            
            # Write WAV with markers
            with wave.open(output_file, "wb") as wav:
                wav.setnchannels(1)
                wav.setsampwidth(2)
                wav.setframerate(sr)
                wav.writeframes(pcm.tobytes())

            # Define markers and loops
            if len(shot_markers) >= 2:
                loop_start = shot_markers[1]
                loop_end = shot_markers[-1]

                # Group markers by position to combine labels at the same spot
                pos_to_labels = {}

                for i, pos in enumerate(shot_markers):
                    label = f"SHOT_{i+1}"
                    if pos not in pos_to_labels:
                        pos_to_labels[pos] = []
                    pos_to_labels[pos].append(label)
                
                # Add loop markers
                if loop_start not in pos_to_labels:
                    pos_to_labels[loop_start] = []
                pos_to_labels[loop_start].append("LOOP_START")

                if loop_end not in pos_to_labels:
                    pos_to_labels[loop_end] = []
                pos_to_labels[loop_end].append("LOOP_END")

                # Create final marker definitions (pos, combined_label)
                unique_positions = sorted(pos_to_labels.keys())
                marker_defs = []
                for pos in unique_positions:
                    labels = pos_to_labels[pos]
                    # Specific order for labels if combined: SHOT_N, then LOOP_START/END
                    # Use a custom sort or just join
                    sorted_labels = sorted(labels, key=lambda l: (0 if l.startswith("SHOT") else 1, l))
                    combined_label = " + ".join(sorted_labels)
                    marker_defs.append((pos, combined_label))

                # Re-open file to append chunks
                with open(output_file, "r+b") as f:
                    f.seek(0, 2)

                    # Write chunks
                    cue_chunk = build_cue_chunk(marker_defs)
                    smpl_chunk = build_smpl_chunk(loop_start, loop_end, sample_rate=sr)
                    label_chunk = build_label_chunk(marker_defs)

                    f.write(cue_chunk)
                    f.write(smpl_chunk)
                    f.write(label_chunk)

                    # Update RIFF chunk size
                    file_size = f.tell()
                    f.seek(4)
                    f.write(struct.pack("<I", file_size - 8))

                logger.info(f"Created: {output_file}")


            logger.info(f"Created: {output_file}")
