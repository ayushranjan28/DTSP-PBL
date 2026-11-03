# Data Directory

This directory houses audio recordings used for demonstration, evaluation, and model training:

## Bundled Clean Audio Reference Library (`data/samples/`)
All audio files are standardized at **16,000 Hz, mono, 16-bit PCM WAV**, normalized with safe dynamic headroom to $[-0.92, 0.92]$.

### 1. Authentic Studio Spoken Sentences (CMU ARCTIC Corpus)
Real human voices reading phonetically balanced, complete English sentences:

#### Female Speaker (`slt` — Crisp Studio Speech, ~210 Hz Pitch):
1. **`clean_female_sentence_01.wav`**:  
   - *Transcript*: *"For the twentieth time that evening the two men shook hands."* (3.21s)  
   - *Acoustic Purpose*: Conversational female pacing with natural inter-phrase pauses; tests spectral gating noise floor tracking and Wiener filter recursion.
2. **`clean_female_sentence_02.wav`**:  
   - *Transcript*: *"He turned sharply, and faced Gregson across the table."* (3.10s)  
   - *Acoustic Purpose*: Rich in sharp unvoiced sibilants (`sh`, `s`) and stop consonants (`t`, `p`, `k`); ideal for testing lowpass filter cutoff and tape hiss suppression without muffling voice clarity.
3. **`clean_female_sentence_03.wav`**:  
   - *Transcript*: *"If I ever needed a fighter in my life I need one now."* (3.06s)  
   - *Acoustic Purpose*: Expressive dynamic intonation contour and vocal emotion; tests speech presence probability under fluctuating speech power.
4. **`clean_female_sentence_04.wav`**:  
   - *Transcript*: *"The girl faced him, her eyes shining with sudden fear."* (3.33s)  
   - *Acoustic Purpose*: Studio narrative cadence with subtle breath pauses; verifies zero-phase transient preservation.

#### Male Speaker (`bdl` — Resonant Studio Baritone, ~115 Hz Pitch):
5. **`clean_male_sentence_01.wav`**:  
   - *Transcript*: *"For the twentieth time that evening the two men shook hands."* (3.37s)  
   - *Acoustic Purpose*: Deep resonant male chest voice; primary benchmark for 50/60 Hz power-line hum notch filters and HVAC low-frequency rumble attenuation.
6. **`clean_male_sentence_02.wav`**:  
   - *Transcript*: *"He turned sharply, and faced Gregson across the table."* (3.21s)  
   - *Acoustic Purpose*: Crisp male articulation with rich harmonic formants across 100–4000 Hz; tests intelligibility and consonant preservation.
7. **`clean_male_sentence_03.wav`**:  
   - *Transcript*: *"Gregson shoved back his chair and rose to his feet."* (2.95s)  
   - *Acoustic Purpose*: Deep narrative cadence with sentence-ending pitch declination; ideal for $\Delta\text{SNR}$ metrics.
8. **`clean_male_sentence_04.wav`**:  
   - *Transcript*: *"Now you're coming down to business, Phil, he exclaimed."* (3.05s)  
   - *Acoustic Purpose*: Lively conversational speech; tests non-stationary speech vs stationary noise suppression.

---

### 2. Instrumental & Baseline Reference Files
- **`clean_acoustic_instrument.wav`**: Plucked acoustic guitar chord arpeggio (Karplus-Strong physical modeling) with sharp attack transients and natural resonance decay, testing phase linearity.
- **`clean_male_speech_01.wav`** & **`clean_synthetic_speech.wav`**: Glottal pulse train with exact mathematical vocal tract formants ($F_1=600\text{ Hz}, F_2=1400\text{ Hz}, F_3=2500\text{ Hz}$) for controlled acoustic benchmarks.
- **`clean_sample.wav`**: Original baseline reference sample.

---

## Pre-Mixed Noisy Demonstration Library (`data/noisy_samples/`)
Contains 20 pre-mixed, calibrated noisy WAV files ready for direct offline playback, blind listening tests, and filter demonstration:
- **50 Hz Power-line Hum** (5 dB SNR)
- **HVAC Low-Frequency Rumble** (3 dB SNR)
- **Preamp & Tape Hiss** (6 dB SNR)
- **Ambient Pink Noise** (5 dB SNR)
- **Severe Mixed Noise** (0 dB SNR)

See `data/noisy_samples/README.md` and `data/noisy_samples/manifest.json` for full metadata. Generated via `scripts/generate_noisy_samples.py`.

---

## External Corpora & Synthetic Data
- `synthetic/`: Directory for generated synthetic noisy files used during controlled offline experiments.
- `voicebank_demand/`: Destination for users wishing to download and prepare the official VoiceBank-DEMAND corpus.
  - Dataset URL: https://datashare.ed.ac.uk/handle/10283/2791
  - DEMAND Noise URL: https://zenodo.org/records/1227121
  - Use `scripts/prepare_dataset.py` to organize downloaded VoiceBank-DEMAND files.
- `scripts/fetch_proper_speech_samples.py`: Helper script to automatically stream and extract additional CMU ARCTIC sentences.
