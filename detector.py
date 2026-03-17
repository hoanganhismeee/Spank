"""
detector.py — Real-time impact sound detector for Windows
Listens to microphone input and triggers when a sudden loud sound is detected.

Setup:
    pip install numpy sounddevice

Usage:
    python detector.py              # run the detector
    python detector.py --calibrate  # run the calibration wizard first

Sound effects:
    Drop a PCM WAV file at  sounds/impact.wav  to play it on detection.
    If the file is absent, a winsound.Beep() tone is used instead.
    (MP3 is not supported by winsound; use pygame for MP3 support.)
"""

import os
import sys
import time
import threading
import collections
import numpy as np
import sounddevice as sd
import winsound

# ---------------------------------------------------------------------------
# Configuration — tune these to match your environment
# ---------------------------------------------------------------------------

SAMPLE_RATE     = 44100   # Hz — standard microphone sample rate
BLOCK_SIZE      = 512     # Samples per callback block (smaller = faster response)
CHANNELS        = 1       # Mono input

# Impact detection threshold (0.0 – 1.0 scale of normalised RMS)
# Raise this if ambient noise causes false positives.
# Lower this if real impacts are being missed.
THRESHOLD       = 0.15

# Seconds to wait before allowing another detection (debounce / cooldown)
COOLDOWN_SECONDS = 0.5

# Number of recent RMS values to keep for smoothing the baseline display
SMOOTHING_WINDOW = 10

# Beep settings for winsound.Beep(frequency_hz, duration_ms) — fallback only
BEEP_FREQ_HZ     = 1000
BEEP_DURATION_MS = 120

# Path to an optional WAV file to play on detection (relative to this script)
SOUND_FILE = os.path.join(os.path.dirname(__file__), "sounds", "impact.wav")

# ---------------------------------------------------------------------------
# Shared state (written by the audio callback, read by the main thread)
# ---------------------------------------------------------------------------

last_trigger_time = 0.0          # Timestamp of the most recent detection
rms_history = collections.deque(maxlen=SMOOTHING_WINDOW)  # Recent RMS values


def compute_rms(block: np.ndarray) -> float:
    """Return the Root Mean Square amplitude of an audio block (0.0 – 1.0)."""
    return float(np.sqrt(np.mean(block.astype(np.float32) ** 2)))


def _play_sound():
    """Play sounds/impact.wav if it exists, otherwise fall back to a Beep tone."""
    if os.path.isfile(SOUND_FILE):
        winsound.PlaySound(SOUND_FILE, winsound.SND_FILENAME | winsound.SND_ASYNC)
    else:
        winsound.Beep(BEEP_FREQ_HZ, BEEP_DURATION_MS)


def play_alert_async():
    """Trigger the alert sound in a background thread so the callback isn't blocked."""
    threading.Thread(target=_play_sound, daemon=True).start()


def audio_callback(indata: np.ndarray, frames: int, time_info, status):
    """
    Called by sounddevice for every captured audio block.
    All detection logic lives here.
    """
    global last_trigger_time

    if status:
        # Print any stream warnings (overflows, etc.) without crashing
        print(f"[stream status] {status}", flush=True)

    rms = compute_rms(indata)
    rms_history.append(rms)

    # Build a simple ASCII volume bar for the live readout
    bar_len = int(rms * 80)
    bar = "#" * min(bar_len, 80)

    now = time.time()
    cooldown_ok = (now - last_trigger_time) >= COOLDOWN_SECONDS

    if rms >= THRESHOLD and cooldown_ok:
        last_trigger_time = now
        print(f"\r[{bar:<80}] {rms:.4f}  *** IMPACT DETECTED ***", flush=True)
        play_alert_async()
    else:
        # Overwrite the same line with the current volume level
        print(f"\r[{bar:<80}] {rms:.4f}", end="", flush=True)


def _collect_rms_samples(duration_seconds: float) -> list[float]:
    """
    Record microphone input for `duration_seconds` and return a list of
    per-block RMS values, printing a live volume bar while recording.
    """
    samples: list[float] = []
    end_time = time.time() + duration_seconds

    with sd.InputStream(
        samplerate=SAMPLE_RATE,
        blocksize=BLOCK_SIZE,
        channels=CHANNELS,
        dtype="float32",
    ) as stream:
        while time.time() < end_time:
            block, _ = stream.read(BLOCK_SIZE)
            rms = compute_rms(block)
            samples.append(rms)
            bar = "#" * min(int(rms * 80), 80)
            remaining = max(0.0, end_time - time.time())
            print(f"\r  [{bar:<80}] {rms:.4f}  ({remaining:.1f}s left)", end="", flush=True)

    print()  # newline after the live bar
    return samples


def calibrate():
    """
    Two-phase calibration wizard.

    Phase 1 — Ambient baseline: records quiet ambient noise for 3 seconds.
    Phase 2 — Spank measurement: records the loudest spank you can make for 5 seconds.

    Outputs a recommended THRESHOLD value and explains the formula used.
    """
    print()
    print("=" * 60)
    print("  CALIBRATION WIZARD")
    print("=" * 60)

    # ------------------------------------------------------------------
    # Phase 1 — Ambient noise
    # ------------------------------------------------------------------
    print()
    print("STEP 1/2 — Ambient noise measurement")
    print("  Stay quiet. Recording for 3 seconds...")
    print()

    ambient_samples = _collect_rms_samples(3.0)

    ambient_avg = float(np.mean(ambient_samples))
    ambient_max = float(np.max(ambient_samples))
    ambient_std = float(np.std(ambient_samples))

    print(f"  Ambient average RMS : {ambient_avg:.4f}")
    print(f"  Ambient max RMS     : {ambient_max:.4f}")
    print(f"  Ambient std-dev     : {ambient_std:.4f}")

    # ------------------------------------------------------------------
    # Phase 2 — Spank peak
    # ------------------------------------------------------------------
    print()
    print("STEP 2/2 — Spank sound measurement")
    print("  Press Enter, then SPANK your laptop as hard as you normally would.")
    print("  You have 5 seconds to make the sound.")
    input("  [Press Enter to start] ")
    print()
    print("  GO! Make the spank sound now...")
    print()

    spank_samples = _collect_rms_samples(5.0)

    spank_peak = float(np.max(spank_samples))
    spank_avg  = float(np.mean(spank_samples))

    print(f"  Spank peak RMS      : {spank_peak:.4f}")
    print(f"  Spank average RMS   : {spank_avg:.4f}")

    # ------------------------------------------------------------------
    # Results & recommended threshold
    # ------------------------------------------------------------------
    print()
    print("=" * 60)
    print("  RESULTS")
    print("=" * 60)
    print(f"  Ambient floor (avg) : {ambient_avg:.4f}")
    print(f"  Ambient ceiling (max): {ambient_max:.4f}")
    print(f"  Spank peak           : {spank_peak:.4f}")
    print()

    gap = spank_peak - ambient_max

    if gap <= 0.01:
        print("  WARNING: Spank peak is not clearly above ambient noise.")
        print("  Try again in a quieter environment or spank harder.")
        recommended = ambient_max * 1.5  # fallback: 50% above ambient ceiling
    else:
        # Place threshold 30% of the way from ambient ceiling to spank peak.
        # Close enough to ambient to catch most spanks; far enough to ignore noise.
        recommended = ambient_max + gap * 0.30

    recommended = round(recommended, 4)

    print(f"  Recommended THRESHOLD: {recommended}")
    print()
    print("  Formula: ambient_max + (spank_peak - ambient_max) * 0.30")
    print("  (30% of the way from ambient ceiling to spank peak)")
    print()
    print("  To apply this threshold, edit detector.py and set:")
    print(f"    THRESHOLD = {recommended}")
    print()
    print("  Tips for tuning:")
    print("   - Raise THRESHOLD → fewer false positives, may miss soft spanks")
    print("   - Lower THRESHOLD → catches softer spanks, more false positives")
    print("=" * 60)
    print()


def main():
    sound_mode = f"WAV  → {SOUND_FILE}" if os.path.isfile(SOUND_FILE) else "Beep (no sounds/impact.wav found)"
    print("=" * 60)
    print("  Impact Sound Detector  (Ctrl+C to quit)")
    print(f"  Threshold : {THRESHOLD}")
    print(f"  Cooldown  : {COOLDOWN_SECONDS}s")
    print(f"  Sample rate: {SAMPLE_RATE} Hz  |  Block: {BLOCK_SIZE} samples")
    print(f"  Alert     : {sound_mode}")
    print("=" * 60)
    print("Listening …\n")

    with sd.InputStream(
        samplerate=SAMPLE_RATE,
        blocksize=BLOCK_SIZE,
        channels=CHANNELS,
        dtype="float32",
        callback=audio_callback,
    ):
        try:
            # Keep the main thread alive; the callback runs on a background thread
            while True:
                sd.sleep(100)
        except KeyboardInterrupt:
            print("\n\nStopped. Goodbye.")


if __name__ == "__main__":
    if "--calibrate" in sys.argv or "-c" in sys.argv:
        calibrate()
    main()
