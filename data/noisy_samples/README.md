# Noisy Audio Demonstration Library

This directory contains pre-mixed, calibrated noisy audio files (16 kHz, 16-bit PCM WAV) ready for direct offline listening, A/B comparisons, and DSP filter evaluation.

## Catalog

| Filename | Speaker | Spoken Sentence | Noise Condition | Target SNR | Recommended Filter |
| :--- | :--- | :--- | :--- | :--- | :--- |
| `female01_50hz_mains_hum_5db.wav` | Female (slt) | "For the twentieth time that evening the two men shook hands." | 50 Hz Power-line Hum | 5.0 dB | Harmonic Notch Filter (50 Hz + Harmonics) |
| `female01_hvac_rumble_3db.wav` | Female (slt) | "For the twentieth time that evening the two men shook hands." | HVAC Low-Frequency Rumble | 3.0 dB | Butterworth Highpass (130-150 Hz SOS) |
| `female01_tape_hiss_6db.wav` | Female (slt) | "For the twentieth time that evening the two men shook hands." | Pre-amp & Tape Hiss | 6.0 dB | FIR Lowpass (3800 Hz) or STFT Spectral Gating |
| `female01_ambient_pink_5db.wav` | Female (slt) | "For the twentieth time that evening the two men shook hands." | Ambient Pink Noise | 5.0 dB | Audio Wiener Filter (Ephraim-Malah DD) |
| `female01_harsh_mixed_0db.wav` | Female (slt) | "For the twentieth time that evening the two men shook hands." | Severe Mixed Noise (Challenging) | 0.0 dB | ML-Assisted Multi-stage Pipeline (Notch + Wiener) |
| `female02_50hz_mains_hum_5db.wav` | Female (slt) | "He turned sharply, and faced Gregson across the table." | 50 Hz Power-line Hum | 5.0 dB | Harmonic Notch Filter (50 Hz + Harmonics) |
| `female02_hvac_rumble_3db.wav` | Female (slt) | "He turned sharply, and faced Gregson across the table." | HVAC Low-Frequency Rumble | 3.0 dB | Butterworth Highpass (130-150 Hz SOS) |
| `female02_tape_hiss_6db.wav` | Female (slt) | "He turned sharply, and faced Gregson across the table." | Pre-amp & Tape Hiss | 6.0 dB | FIR Lowpass (3800 Hz) or STFT Spectral Gating |
| `female02_ambient_pink_5db.wav` | Female (slt) | "He turned sharply, and faced Gregson across the table." | Ambient Pink Noise | 5.0 dB | Audio Wiener Filter (Ephraim-Malah DD) |
| `female02_harsh_mixed_0db.wav` | Female (slt) | "He turned sharply, and faced Gregson across the table." | Severe Mixed Noise (Challenging) | 0.0 dB | ML-Assisted Multi-stage Pipeline (Notch + Wiener) |
| `male01_50hz_mains_hum_5db.wav` | Male (bdl) | "For the twentieth time that evening the two men shook hands." | 50 Hz Power-line Hum | 5.0 dB | Harmonic Notch Filter (50 Hz + Harmonics) |
| `male01_hvac_rumble_3db.wav` | Male (bdl) | "For the twentieth time that evening the two men shook hands." | HVAC Low-Frequency Rumble | 3.0 dB | Butterworth Highpass (130-150 Hz SOS) |
| `male01_tape_hiss_6db.wav` | Male (bdl) | "For the twentieth time that evening the two men shook hands." | Pre-amp & Tape Hiss | 6.0 dB | FIR Lowpass (3800 Hz) or STFT Spectral Gating |
| `male01_ambient_pink_5db.wav` | Male (bdl) | "For the twentieth time that evening the two men shook hands." | Ambient Pink Noise | 5.0 dB | Audio Wiener Filter (Ephraim-Malah DD) |
| `male01_harsh_mixed_0db.wav` | Male (bdl) | "For the twentieth time that evening the two men shook hands." | Severe Mixed Noise (Challenging) | 0.0 dB | ML-Assisted Multi-stage Pipeline (Notch + Wiener) |
| `male02_50hz_mains_hum_5db.wav` | Male (bdl) | "He turned sharply, and faced Gregson across the table." | 50 Hz Power-line Hum | 5.0 dB | Harmonic Notch Filter (50 Hz + Harmonics) |
| `male02_hvac_rumble_3db.wav` | Male (bdl) | "He turned sharply, and faced Gregson across the table." | HVAC Low-Frequency Rumble | 3.0 dB | Butterworth Highpass (130-150 Hz SOS) |
| `male02_tape_hiss_6db.wav` | Male (bdl) | "He turned sharply, and faced Gregson across the table." | Pre-amp & Tape Hiss | 6.0 dB | FIR Lowpass (3800 Hz) or STFT Spectral Gating |
| `male02_ambient_pink_5db.wav` | Male (bdl) | "He turned sharply, and faced Gregson across the table." | Ambient Pink Noise | 5.0 dB | Audio Wiener Filter (Ephraim-Malah DD) |
| `male02_harsh_mixed_0db.wav` | Male (bdl) | "He turned sharply, and faced Gregson across the table." | Severe Mixed Noise (Challenging) | 0.0 dB | ML-Assisted Multi-stage Pipeline (Notch + Wiener) |
