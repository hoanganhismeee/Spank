"""
detector.py — Real-time impact sound detector for Windows
Listens to microphone input and triggers when a sudden loud sound is detected.

Setup:
    pip install numpy sounddevice pygame

Usage:
    python detector.py              # run the detector
    python detector.py --calibrate  # run the calibration wizard first

Sound effects:
    Drop up to 5 audio files (WAV, MP3, OGG, FLAC) into the sounds/ folder.
    Each detection plays one file chosen at random.
    If the folder is empty, falls back to a winsound.Beep() tone.
"""

import os
import sys
import glob
import time
import random
import threading
import collections
import numpy as np
import sounddevice as sd
import winsound

try:
    import pygame
    pygame.mixer.init()
    _PYGAME_OK = True
except Exception:
    _PYGAME_OK = False

# ---------------------------------------------------------------------------
# Configuration constants
# ---------------------------------------------------------------------------

SAMPLE_RATE      = 44100   # Hz — standard microphone sample rate
BLOCK_SIZE       = 512     # Samples per callback block
CHANNELS         = 1       # Mono input
THRESHOLD        = 0.03   # RMS level that counts as an impact (tune as needed)
COOLDOWN_SECONDS = 0.5     # Minimum seconds between detections
SMOOTHING_WINDOW = 10      # Rolling RMS history length (for display)
BEEP_FREQ_HZ     = 1000    # Fallback beep frequency (Hz)
BEEP_DURATION_MS = 120     # Fallback beep duration (ms)
AUDIO_EXTENSIONS = ("*.wav", "*.mp3", "*.ogg", "*.flac")
SOUNDS_DIR       = os.path.join(os.path.dirname(os.path.abspath(__file__)), "sounds")


# ---------------------------------------------------------------------------
# SoundPlayer — Single responsibility: load and play sounds
# ---------------------------------------------------------------------------

class SoundPlayer:
    """Discovers audio files in a directory and plays a random one per impact."""

    POST_PLAY_REST = 1.3  # seconds to keep detection blocked after sound finishes

    def __init__(self, sounds_dir: str) -> None:
        self._sounds_dir = sounds_dir
        self._files: list[str] = []
        self._mute_until = 0.0  # detection blocked until this timestamp

    def load(self) -> None:
        """Scan sounds_dir for supported audio files."""
        found = []
        for pattern in AUDIO_EXTENSIONS:
            found.extend(glob.glob(os.path.join(self._sounds_dir, pattern)))
        self._files = sorted(found)

    @property
    def files(self) -> list[str]:
        return list(self._files)

    def play_random(self) -> None:
        """Play a random loaded file in a background thread. Beeps if none loaded."""
        threading.Thread(target=self._play, daemon=True).start()

    def is_playing(self) -> bool:
        """
        Return True while an alert sound is playing OR for POST_PLAY_REST seconds
        after it finishes, so speaker output doesn't re-trigger the mic.
        """
        if _PYGAME_OK and pygame.mixer.music.get_busy():
            # Keep pushing the mute window forward while sound is active
            self._mute_until = time.time() + self.POST_PLAY_REST
            return True
        return time.time() < self._mute_until

    def _play(self) -> None:
        if self._files and _PYGAME_OK:
            chosen = random.choice(self._files)
            pygame.mixer.music.stop()
            pygame.mixer.music.load(chosen)
            pygame.mixer.music.play()
        else:
            winsound.Beep(BEEP_FREQ_HZ, BEEP_DURATION_MS)


# ---------------------------------------------------------------------------
# AudioDetector — Single responsibility: mic stream + impact detection
# ---------------------------------------------------------------------------

class AudioDetector:
    """
    Opens a microphone input stream and fires on_impact() whenever
    the RMS of an audio block exceeds `threshold` and the cooldown has elapsed.
    """

    WARMUP_SECONDS = 3  # mic settle time before detection is active

    def __init__(
        self,
        threshold: float,
        cooldown: float,
        on_impact,
        is_busy=None,
        sample_rate: int = SAMPLE_RATE,
        block_size: int = BLOCK_SIZE,
    ) -> None:
        self._threshold    = threshold
        self._cooldown     = cooldown
        self._on_impact    = on_impact
        self._is_busy      = is_busy or (lambda: False)  # injected; blocks detection while True
        self._sample_rate  = sample_rate
        self._block_size   = block_size
        self._last_trigger = 0.0
        self._ready_after  = 0.0   # set in start(); detections blocked until then
        self._rms_history  = collections.deque(maxlen=SMOOTHING_WINDOW)
        self._stream       = None

    def start(self) -> None:
        self._ready_after = time.time() + self.WARMUP_SECONDS
        self._stream = sd.InputStream(
            samplerate=self._sample_rate,
            blocksize=self._block_size,
            channels=CHANNELS,
            dtype="float32",
            callback=self._callback,
        )
        self._stream.start()

    def stop(self) -> None:
        if self._stream:
            self._stream.stop()
            self._stream.close()

    def _callback(self, indata: np.ndarray, frames: int, time_info, status) -> None:
        if status:
            print(f"\n[stream] {status}", flush=True)

        rms = self._compute_rms(indata)
        self._rms_history.append(rms)

        bar = "#" * min(int(rms * 80), 80)
        now = time.time()

        # Warmup: mic is settling — show countdown, skip detection
        if now < self._ready_after:
            remaining = self._ready_after - now
            print(f"\r[{bar:<80}] {rms:.5f}  (warming up… {remaining:.1f}s)", end="", flush=True)
            return

        # Sound feedback guard: ignore mic while alert plays and for 2s after it ends
        if self._is_busy():
            print(f"\r[{bar:<80}] {rms:.5f}  (blocked…)", end="", flush=True)
            return

        if rms >= self._threshold and (now - self._last_trigger) >= self._cooldown:
            self._last_trigger = now
            print(f"\r[{bar:<80}] {rms:.5f}  *** IMPACT DETECTED ***", flush=True)
            self._on_impact()
        else:
            print(f"\r[{bar:<80}] {rms:.5f}", end="", flush=True)

    @staticmethod
    def _compute_rms(block: np.ndarray) -> float:
        return float(np.sqrt(np.mean(block.astype(np.float32) ** 2)))


# ---------------------------------------------------------------------------
# Calibrator — Single responsibility: measure ambient and spank levels
# ---------------------------------------------------------------------------

class Calibrator:
    """
    Two-phase measurement wizard.
      Phase 1 — records quiet ambient noise (3 s).
      Phase 2 — records the user's spank sound (5 s).
    Prints a recommended THRESHOLD value.
    """

    def __init__(self, sample_rate: int = SAMPLE_RATE, block_size: int = BLOCK_SIZE) -> None:
        self._sample_rate = sample_rate
        self._block_size  = block_size

    def run(self) -> None:
        print()
        print("=" * 60)
        print("  CALIBRATION WIZARD")
        print("=" * 60)

        print()
        print("STEP 1/2 — Ambient noise")
        print("  Stay quiet. Recording for 3 seconds...")
        print()
        ambient = self._collect(3.0)
        ambient_avg = float(np.mean(ambient))
        ambient_max = float(np.max(ambient))
        ambient_std = float(np.std(ambient))
        print(f"  Average RMS : {ambient_avg:.5f}")
        print(f"  Max RMS     : {ambient_max:.5f}")
        print(f"  Std-dev     : {ambient_std:.5f}")

        print()
        print("STEP 2/2 — Spank sound")
        print("  Press Enter, then SPANK your laptop as hard as you normally would.")
        print("  You have 5 seconds.")
        input("  [Press Enter to start] ")
        print()
        print("  GO!")
        print()
        spank = self._collect(5.0)
        spank_peak = float(np.max(spank))
        spank_avg  = float(np.mean(spank))
        print(f"  Peak RMS    : {spank_peak:.5f}")
        print(f"  Average RMS : {spank_avg:.5f}")

        print()
        print("=" * 60)
        print("  RESULTS")
        print("=" * 60)
        print(f"  Ambient ceiling (max): {ambient_max:.5f}")
        print(f"  Spank peak           : {spank_peak:.5f}")
        print()

        gap = spank_peak - ambient_max
        if gap <= 0.001:
            print("  WARNING: Spank peak is not clearly above ambient noise.")
            print("  Try again in a quieter environment or spank harder.")
            recommended = round(ambient_max * 1.5, 5)
        else:
            # 30% of the way from ambient ceiling to spank peak
            recommended = round(ambient_max + gap * 0.30, 5)

        print(f"  Recommended THRESHOLD = {recommended}")
        print()
        print("  Formula: ambient_max + (spank_peak - ambient_max) * 0.30")
        print()
        print("  Edit detector.py and set:")
        print(f"    THRESHOLD = {recommended}")
        print()
        print("  Raise → fewer false positives  |  Lower → catches softer spanks")
        print("=" * 60)
        print()

    def _collect(self, duration: float) -> list[float]:
        samples: list[float] = []
        end_time = time.time() + duration
        with sd.InputStream(
            samplerate=self._sample_rate,
            blocksize=self._block_size,
            channels=CHANNELS,
            dtype="float32",
        ) as stream:
            while time.time() < end_time:
                block, _ = stream.read(self._block_size)
                rms = float(np.sqrt(np.mean(block.astype(np.float32) ** 2)))
                samples.append(rms)
                bar = "#" * min(int(rms * 80), 80)
                remaining = max(0.0, end_time - time.time())
                print(f"\r  [{bar:<80}] {rms:.5f}  ({remaining:.1f}s)", end="", flush=True)
        print()
        return samples


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

def main() -> None:
    player = SoundPlayer(SOUNDS_DIR)
    player.load()

    detector = AudioDetector(
        threshold=THRESHOLD,
        cooldown=COOLDOWN_SECONDS,
        on_impact=player.play_random,
        is_busy=player.is_playing,
    )

    print("=" * 60)
    print("  Impact Sound Detector  (Ctrl+C to quit)")
    print(f"  Threshold : {THRESHOLD}")
    print(f"  Cooldown  : {COOLDOWN_SECONDS}s")
    print(f"  Sample rate: {SAMPLE_RATE} Hz  |  Block: {BLOCK_SIZE} samples")

    if player.files:
        print(f"  Sounds    : {len(player.files)} file(s) loaded — random pick per hit")
        for f in player.files:
            print(f"              · {os.path.basename(f)}")
    else:
        print("  Sounds    : none in sounds/ — using fallback Beep")

    print("=" * 60)
    print("Listening …\n")

    detector.start()
    try:
        while True:
            sd.sleep(100)
    except KeyboardInterrupt:
        print("\n\nStopped. Goodbye.")
    finally:
        detector.stop()


