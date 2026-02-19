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
from src.utils.cards import DoubleSpinSettingCard
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
    chunk = struct.pack("<4sI", b"smpl", 60)

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


def find_zero_crossing(audio, start_idx, direction=1, max_search=1000):
    """Find the nearest zero crossing from start_idx."""
    for i in range(max_search):
        idx = start_idx + (i * direction)
        if idx <= 0 or idx >= len(audio) - 1:
            return start_idx
        if audio[idx - 1] <= 0 < audio[idx] or audio[idx - 1] >= 0 > audio[idx]:
            return idx
    return start_idx


def cosine_crossfade(seg_a, seg_b, fade_len):
    """Crossfade two segments using a cosine curve."""
    if fade_len <= 0 or len(seg_a) < fade_len or len(seg_b) < fade_len:
        return np.concatenate([seg_a, seg_b])
    
    fade_out = np.cos(np.linspace(0, np.pi / 2, fade_len)) ** 2
    fade_in = np.sin(np.linspace(0, np.pi / 2, fade_len)) ** 2
    
    result = np.zeros(len(seg_a) + len(seg_b) - fade_len)
    result[:len(seg_a) - fade_len] = seg_a[:-fade_len]
    result[len(seg_a) - fade_len:len(seg_a)] = seg_a[-fade_len:] * fade_out + seg_b[:fade_len] * fade_in
    result[len(seg_a):] = seg_b[fade_len:]
    
    return result


class LaserBeamGeneratorWidget(BaseWidget):
    def __init__(self, parent, text):
        super().__init__(parent=parent, text=text, vertical=True)

        # --- UI Cards ---
        self.input_file_card = PushSettingCard(
            self.tr('Source Audio File'),
            FIF.MUSIC,
            self.tr("Select the single shot WAV file"),
            cfg.lf_input_file.value
        )
        self.output_folder_card = PushSettingCard(
            self.tr('Output Directory'),
            FIF.FOLDER,
            self.tr("Where to save the beam sound"),
            cfg.lf_output_folder.value
        )

        self.loop_duration_card = DoubleSpinSettingCard(
            cfg.lf_loop_duration_sec,
            FIF.SYNC,
            self.tr("Loop Duration (sec)"),
        )

        self.pitch_variation_card = DoubleSpinSettingCard(
            cfg.lf_pitch_variation,
            FIF.ARROW_DOWN,
            self.tr("Pitch Variation (semitones)"),
        )

        self.gain_variation_card = DoubleSpinSettingCard(
            cfg.lf_gain_variation,
            FIF.VOLUME,
            self.tr("Gain Variation (dB)"),
        )

        self.tail_threshold_card = DoubleSpinSettingCard(
            cfg.lf_tail_threshold,
            FIF.CUT,
            self.tr("Tail Threshold (dB)"),
        )

        self.highpass_card = SwitchSettingCard(
            FIF.FILTER,
            self.tr("Highpass Filter"),
            self.tr("Apply highpass to the loop for clarity"),
            cfg.lf_highpass_enabled
        )

        self.tilt_card = DoubleSpinSettingCard(
            cfg.lf_tilt,
            FIF.UP,
            self.tr("Tilt EQ"),
        )

        self.addToFrame(self.input_file_card)
        self.addToFrame(self.output_folder_card)
        self.addToFrame(self.loop_duration_card)
        self.addToFrame(self.pitch_variation_card)
        self.addToFrame(self.gain_variation_card)
        self.addToFrame(self.tail_threshold_card)
        self.addToFrame(self.highpass_card)
        self.addToFrame(self.tilt_card)

        self.boxLayout.addStretch(1)

        self.run_button = PrimaryPushButton(icon=FIF.PLAY, text=self.tr("Generate Beam Sound"))
        self.run_button.clicked.connect(self.on_run)
        self.input_file_card.clicked.connect(self.on_select_input_file)
        self.output_folder_card.clicked.connect(self.on_select_output_folder)

        self.addButtonBarToBottom(self.run_button)

    def on_select_input_file(self):
        file_path, _ = QFileDialog.getOpenFileName(
            self, self.tr("Select source audio"), 
            cfg.lf_input_file.value or "", 
            "Audio Files (*.wav)"
        )
        if file_path:
            cfg.lf_input_file.value = file_path
            self.input_file_card.setContent(file_path)

            directory = os.path.dirname(file_path)
            cfg.lf_output_folder.value = directory
            self.output_folder_card.setContent(directory)

    def on_select_output_folder(self):
        directory = QFileDialog.getExistingDirectory(
            self, self.tr("Select output directory"), 
            cfg.lf_output_folder.value or ""
        )
        if directory:
            cfg.lf_output_folder.value = directory
            self.output_folder_card.setContent(directory)

    def on_run(self):
        input_file = cfg.lf_input_file.value
        output_dir = cfg.lf_output_folder.value

        if not input_file or not os.path.exists(input_file):
            InfoBar.error(title=self.tr("Error"), content=self.tr("Please select a valid input WAV file."), parent=self)
            return
        if not output_dir or not os.path.isdir(output_dir):
            InfoBar.error(title=self.tr("Error"), content=self.tr("Please select a valid output directory."), parent=self)
            return

        self.run_button.setEnabled(False)
        try:
            self.generate_beam_sound(input_file, output_dir)
            InfoBar.success(title=self.tr("Success"), content=self.tr("Beam sound generated successfully!"), parent=self)
        except Exception as e:
            logger.error(traceback.format_exc())
            InfoBar.error(title=self.tr("Error"), content=str(e), parent=self)
        finally:
            self.run_button.setEnabled(True)

    def generate_beam_sound(self, input_file, output_dir):
        loop_duration_sec = cfg.lf_loop_duration_sec.value
        pitch_variation = cfg.lf_pitch_variation.value
        gain_variation = cfg.lf_gain_variation.value
        tail_threshold = cfg.lf_tail_threshold.value
        highpass_enabled = cfg.lf_highpass_enabled.value
        highpass_cutoff = cfg.lf_highpass_cutoff.value
        tilt_amount = cfg.lf_tilt.value

        # Load audio
        audio, sr = librosa.load(input_file, sr=None)
        
        # Attack: first ~100ms (the initial transient)
        attack_samples = int(0.1 * sr)
        attack = audio[:attack_samples].copy()
        
        # Tail: use the last ~400ms of the ORIGINAL audio as the release/ending
        tail_samples = int(0.4 * sr)
        tail_start = max(len(audio) - tail_samples, attack_samples)
        tail = audio[tail_start:].copy()
        # Add fade-in to tail for smooth blending with loop
        tail_fade_in = min(int(0.05 * sr), len(tail) // 4)
        if tail_fade_in > 0:
            tail[:tail_fade_in] *= np.sin(np.linspace(0, np.pi / 2, tail_fade_in)) ** 2
        
        # Body: the middle portion between attack and tail (source for loop)
        body_start = attack_samples
        body_end = tail_start
        body = audio[body_start:body_end].copy()
        
        # Build the loop using LAYERED SEGMENTS with crossfades (original approach)
        # This creates a smooth sustained sound by overlapping multiple varied copies
        loop_samples = int(loop_duration_sec * sr)
        crossfade_len = int(0.05 * sr)  # 50ms crossfades
        
        # Use a segment from the body (~200-500ms)
        segment_len = min(int(0.4 * sr), max(int(0.2 * sr), len(body) // 2))
        
        # Find the most stable segment in the body
        best_start = 0
        best_variance = float('inf')
        for i in range(0, max(1, len(body) - segment_len), segment_len // 8):
            seg = body[i:i + segment_len]
            if len(seg) < segment_len:
                continue
            variance = np.std(np.abs(seg))
            if variance < best_variance and np.mean(np.abs(seg)) > 0.01:
                best_variance = variance
                best_start = i
        
        base_segment = body[best_start:best_start + segment_len].copy()
        if len(base_segment) < segment_len:
            base_segment = body[:segment_len].copy() if len(body) >= segment_len else body.copy()
        
        # Layer multiple copies with slight offsets and variations to fill the loop
        loop_section = np.zeros(loop_samples)
        layer_count = 3  # Number of overlapping layers
        
        for layer in range(layer_count):
            cursor = int(layer * segment_len / layer_count)  # Offset each layer
            
            while cursor < loop_samples:
                seg = base_segment.copy()
                
                # Apply subtle variations for organic feel
                if pitch_variation > 0:
                    seg = random_pitch(seg, sr, pitch_variation * 0.3)
                if gain_variation > 0:
                    seg = random_gain(seg, gain_variation * 0.3)
                if highpass_enabled and random.random() > 0.7:
                    seg = highpass(seg, random.uniform(100, 300), sr)
                if tilt_amount != 0:
                    seg = tilt_eq(seg, tilt_amount * random.uniform(0.5, 1.5))
                
                # Apply fade in/out to segment for smooth overlap
                fade_len = min(crossfade_len, len(seg) // 4)
                if fade_len > 0:
                    seg[:fade_len] *= np.sin(np.linspace(0, np.pi / 2, fade_len)) ** 2
                    seg[-fade_len:] *= np.cos(np.linspace(0, np.pi / 2, fade_len)) ** 2
                
                # Add to loop (overlap-add)
                end_pos = min(cursor + len(seg), loop_samples)
                copy_len = end_pos - cursor
                loop_section[cursor:end_pos] += seg[:copy_len]
                
                cursor += segment_len - crossfade_len  # Overlap by crossfade amount
        
        # Normalize loop section
        loop_rms = np.sqrt(np.mean(loop_section ** 2))
        if loop_rms > 0:
            target_rms = 0.12
            loop_section = loop_section * (target_rms / loop_rms)
        
        # Make the loop seamlessly loopable (crossfade end to start)
        fade_len = int(0.05 * sr)
        if len(loop_section) > fade_len * 2:
            fade_out = np.cos(np.linspace(0, np.pi / 2, fade_len)) ** 2
            fade_in = np.sin(np.linspace(0, np.pi / 2, fade_len)) ** 2
            loop_section[-fade_len:] = loop_section[-fade_len:] * fade_out + loop_section[:fade_len] * fade_in
        
        # Assemble final audio: attack + loop + tail
        # Crossfade attack into loop start
        attack_fade = min(crossfade_len, len(attack), len(loop_section))
        if attack_fade > 0:
            fade_out = np.cos(np.linspace(0, np.pi / 2, attack_fade)) ** 2
            fade_in = np.sin(np.linspace(0, np.pi / 2, attack_fade)) ** 2
            attack[-attack_fade:] *= fade_out
            loop_section[:attack_fade] *= fade_in
        
        # Crossfade loop end into tail
        tail_fade = min(crossfade_len, len(loop_section), len(tail))
        if tail_fade > 0:
            fade_out = np.cos(np.linspace(0, np.pi / 2, tail_fade)) ** 2
            fade_in = np.sin(np.linspace(0, np.pi / 2, tail_fade)) ** 2
            loop_section[-tail_fade:] *= fade_out
            tail[:tail_fade] *= fade_in
        
        # Concatenate
        output = np.concatenate([attack, loop_section, tail])
        
        # Calculate loop markers (in samples)
        loop_start = len(attack)
        loop_end = len(attack) + len(loop_section) - 1
        
        # Normalize
        max_val = np.max(np.abs(output))
        if max_val > 0:
            output /= max_val + 1e-8
        
        # Convert to 16-bit PCM
        pcm = (output * 32767).astype(np.int16)
        
        # Output filename
        base_name = os.path.splitext(os.path.basename(input_file))[0]
        if "single" in base_name.lower():
            import re
            base_name = re.sub(r'single', 'beam', base_name, flags=re.IGNORECASE)
        else:
            base_name = f"{base_name}_beam"
        
        output_file = os.path.join(output_dir, f"{base_name}.wav")
        
        # Write WAV
        with wave.open(output_file, "wb") as wav:
            wav.setnchannels(1)
            wav.setsampwidth(2)
            wav.setframerate(sr)
            wav.writeframes(pcm.tobytes())
        
        # Add markers and loop points
        marker_defs = [
            (loop_start, "LOOP_START"),
            (loop_end, "LOOP_END"),
        ]
        
        with open(output_file, "r+b") as f:
            f.seek(0, 2)
            
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
        
        logger.info(f"Created beam sound: {output_file}")
        logger.info(f"Loop region: {loop_start} - {loop_end} samples ({loop_start/sr:.3f}s - {loop_end/sr:.3f}s)")
