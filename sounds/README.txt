Place a WAV file named "impact.wav" in this folder to play it on detection.
If no file is found, the program falls back to a simple winsound.Beep() tone.

Requirements for the WAV file:
- Format: PCM WAV (uncompressed)
- Any sample rate / bit depth is fine
- Keep the filename exactly: impact.wav

MP3 files are NOT supported by winsound. To use MP3 you would need pygame:
    pip install pygame
Then replace the PlaySound call in detector.py with:
    pygame.mixer.Sound("sounds/impact.mp3").play()
