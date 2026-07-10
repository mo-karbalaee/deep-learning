#!/usr/bin/env bash
set -e

# make `conda activate` work inside a non-interactive script
eval "$(conda shell.bash hook)"

# 1. Create an isolated env (don't pollute base anaconda)
conda create -n dl-ex4 python=3.11 -y
conda activate dl-ex4

# 2. Install CPU builds of what the tests need
pip install torch torchvision onnxruntime onnxscript scikit-image pandas scikit-learn tabulate matplotlib tqdm

# 3. Extract the dataset (tests read real image files)
cd /Users/mohammad/Documents/GitHub/deep-learning/ex4/src_to_implement
unzip -q images.zip        # creates images/

# 4. Run the tests
python PytorchChallengeTests.py          # all tests
python PytorchChallengeTests.py Bonus    # scored breakdown (TestDataset + TestModel)
