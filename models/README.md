# Models Directory

Stores trained Machine Learning models and feature configuration metadata for adaptive noise classification.

### Files
- `noise_classifier.joblib`: Serialized scikit-learn Random Forest model pipeline, containing the fitted scaler, classifier, label encoders, feature names, and training metadata provenance.
- `metadata.json`: Training provenance, cross-validation metrics, class distributions, and feature importance.

### Supported Noise Classes
1. `electrical_hum`: Narrowband 50/60 Hz harmonic interference.
2. `low_frequency_rumble`: High acoustic energy <150 Hz (HVAC, wind, handling noise).
3. `high_frequency_hiss`: Elevated high-frequency energy >3 kHz (tape hiss, thermal noise).
4. `broadband_noise`: Uniform or 1/f spectral distribution across voice spectrum (white/pink noise).
5. `clean_speech`: Clean reference speech with nominal background noise.
