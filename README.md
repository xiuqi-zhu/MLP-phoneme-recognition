# MLP — Frame-Level Speech Recognition (HW1 P2)

Frame-level phoneme recognition using MFCC features. This repo includes a small custom deep-learning library **mytorch** and MLP models, plus PyTorch-based training scripts for the assignment.

## Project structure

```
MLP/
├── scripts/                 # Train/eval scripts (converted from Jupyter notebook)
│   ├── __init__.py
│   ├── config.py           # Hyperparameters and PHONEMES
│   ├── dataset.py          # AudioDataset, AudioTestDataset
│   ├── model.py            # Network (MLP)
│   ├── train.py            # train(), eval()
│   ├── test.py             # test(), submission.csv generation
│   └── main.py             # CLI entry
├── mytorch/                 # Custom mini deep-learning library
│   ├── nn/                  # activation, batchnorm, linear, loss
│   └── optim/               # sgd
├── models/                  # MLP0, MLP1, MLP4
├── acknowledgement.txt
├── requirements.txt
└── README.md
```

## Requirements

- Python 3.8+
- See `requirements.txt`

## Setup and run

```bash
# Clone
git clone https://github.com/xiuqi-zhu/MLP-phoneme-recognition.git
cd MLP-phoneme-recognition

# Optional: virtual environment
python -m venv venv
# Windows:
venv\Scripts\activate
# Linux/macOS:
# source venv/bin/activate

# Install dependencies
pip install -r requirements.txt

# Train and validate (root must contain train-clean-100, dev-clean, test-clean)
python -m scripts.main --root /path/to/your/data

# Inference only: generate test predictions and submission.csv
python -m scripts.main --root /path/to/data --no-train --submit

# Override epochs, batch size, output path
python -m scripts.main --root /path/to/data --epochs 30 --batch-size 8192 --out-csv ./submission.csv
```

## Dependencies

- **numpy**, **scipy**: Numerics and custom activations (e.g. GELU in mytorch).
- **torch**, **torchaudio**: Data loading and model training in `scripts/`.
- **tqdm**: Progress bars.

## Using mytorch and models

```python
from mytorch.nn import Linear, ReLU
from mytorch.optim import SGD
from models import MLP1

model = MLP1()
optimizer = SGD(model, lr=0.01, momentum=0.9)
# Use forward/backward and optimizer.step() in your training loop.
```

## Data

- MFCC features: 28 dimensions per frame.
- Train/val: `mfcc` and `transcript` subfolders; test: `mfcc` only (phonemes to be predicted).

## License and acknowledgements

Course assignment. Submission and integrity rules are in `acknowledgement.txt`.
