# Spank

### Ever get so frustrated at your laptop that you just... slap it? 

Well now it slaps back. Spank listens to your microphone and waits for that desk smack, table knock, or full-on laptop spank — and the moment it hears one, it fires off a random sound effect of your choice. 

Drop in your own audio files, tune the sensitivity (threshold number), and let your laptop finally have a personality when you lose your patience with it.

### Libraries used

| Library | What it does here |
|---|---|
| `numpy` | Computes the RMS amplitude of each audio block to measure loudness |
| `sounddevice` | Opens the microphone input stream in real time |
| `pygame` | Loads and plays the random sound effect files (WAV, MP3, OGG, FLAC) |
| `winsound` | Fallback beep tone when no sound files are found (Windows built-in) |
| `collections` | `deque` for keeping a rolling history of recent volume levels |
| `threading` | Plays sounds in a background thread so the mic stream never blocks |
