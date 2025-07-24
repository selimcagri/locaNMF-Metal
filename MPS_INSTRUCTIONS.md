# Using LocaNMF with Apple's MPS Backend

These steps describe how to set up an environment that uses PyTorch's Metal Performance Shaders (MPS) backend on Apple Silicon.

1. Create and activate a new conda environment with a modern Python version:
   ```bash
   conda create -n locanmf python=3.10
   conda activate locanmf
   ```
2. Install PyTorch with MPS support via `pip`:
   ```bash
   pip install torch torchvision torchaudio
   ```
3. Install LocaNMF from this repository:
   ```bash
   pip install .
   ```
4. Verify that the MPS backend is available:
   ```bash
   python - <<'PY'
   import torch
   print("MPS available:", torch.backends.mps.is_available())
   PY
   ```

The optional CUDA extension documented in the README is not required on Apple Silicon.
