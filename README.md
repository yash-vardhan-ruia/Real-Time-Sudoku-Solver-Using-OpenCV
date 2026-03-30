# Real-Time Sudoku Solver (CNN + ONNX)

This project now uses a lightweight CNN exported to ONNX for digit recognition, loaded at runtime via OpenCV DNN.

## Architecture

- `train_model.py`: offline training only. Trains a CNN on synthetic printed digits and exports `digit_cnn.onnx`.
- `main.py`: real-time inference only. Loads `digit_cnn.onnx`, performs OCR probabilities per cell, and solves Sudoku with probability-guided backtracking.

## Setup

Install runtime dependencies:

```bash
uv sync
```

Install optional training dependencies:

```bash
uv sync --extra training
```

## Train the CNN

```bash
uv run train_model.py
```

This writes `digit_cnn.onnx` next to the scripts.

## Run the solver

```bash
uv run main.py
```

## Notes

- The solver no longer deletes OCR clues by confidence heuristics.
- The solver now branches on OCR probabilities for each cell.
- CLAHE is applied on CPU before any optional UMat upload to avoid GPU↔CPU ping-pong in preprocessing.
