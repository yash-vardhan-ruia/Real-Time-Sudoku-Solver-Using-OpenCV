# Real-Time Sudoku Solver (CNN + PyTorch)

This project uses a lightweight CNN trained on synthetic printed digits. The model is trained once via `train_model.py` and saved as PyTorch weights, then loaded for real-time inference in `main.py`.

## Architecture

- `train_model.py`: offline training only. Trains a CNN on synthetic printed digits (50 epochs, ~54% val accuracy) and saves weights to `digit_cnn.pth`.
- `main.py`: real-time inference only. Loads the PyTorch model, performs OCR probabilities per cell, and solves Sudoku with probability-guided backtracking.

## Setup

Install runtime dependencies:

```bash
uv sync
```

Install optional development dependencies (for training on different hardware):

```bash
uv sync --extra training
```

## Train the CNN

```bash
uv run train_model.py
```

This generates `digit_cnn.pth` next to the scripts. Training uses CPU by default; if CUDA is available, PyTorch will automatically use GPU.

## Run the solver

```bash
uv run main.py
```

The solver will load the pre-trained model and start real-time webcam capture.

## Notes

- The solver no longer deletes OCR clues by confidence heuristics—instead, it branches on OCR probabilities for each cell during backtracking.
- CLAHE (histogram equalization) is applied on CPU before any optional GPU acceleration to avoid PCIe bottlenecks.
- Model inference uses PyTorch directly, avoiding ONNX compatibility issues and providing native GPU support.

