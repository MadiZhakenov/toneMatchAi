# NAM Models Directory

This directory contains Neural Amp Modeler (NAM) model files (`.nam` format).

## What is NAM?

NAM (Neural Amp Modeler) is a neural network-based amplifier modeling system that can capture the sound and behavior of real guitar amplifiers with high accuracy.

## Adding NAM Models

1. Download `.nam` model files from:
   - [ToneHunt](https://tonehunt.org/) - Community repository of NAM models
   - [NAM Project](https://github.com/sdatkinson/NeuralAmpModeler) - Official NAM repository
   - Other NAM model sources

2. Place `.nam` files in this directory (`assets/nam_models/`)

3. The system will automatically detect and use them during optimization

## Model Requirements

- File format: `.nam` (PyTorch model format)
- Sample rate: Models should work with 44.1 kHz or 48 kHz audio
- Mono/Stereo: Both are supported (will be converted to mono if needed)

## Testing Without Models

If no `.nam` files are present, the system will automatically use a **mock mode** that simulates NAM processing using distortion effects. This allows you to test the full rig pipeline even without real NAM models.

## Example Models

Popular NAM models to try:
- Mesa Boogie Dual Rectifier
- Fender Deluxe Reverb
- Marshall JCM800
- Vox AC30
- Orange Rockerverb

## Notes

- The system supports fallback strategies:
  1. **PyTorch**: Direct loading of `.nam` files (if PyTorch is installed)
  2. **Subprocess**: External NAM CLI tool (if available in PATH)
  3. **Mock**: Distortion-based simulation (always available)

- For best results, use models trained on similar gain levels to your target tone

