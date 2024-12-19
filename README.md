# tangram-2025

## Requirements

To install the required dependencies, you can use the `requirements.txt` file provided in this repository.

### Setup

1. **Create a Conda Environment**

   ```bash
   conda create --name tangram python=3.11
   conda activate tangram
   ```

2. **Install Dependencies**

   Install all the required libraries:

   ```bash
   pip install -r requirements.txt
   ```

### Verify Installation

Run the following command in Python to verify the libraries are installed correctly:

```python
try:
    import mediapipe as mp
    import numpy as np
    import cv2

    print("All libraries are working correctly!")
except ImportError as e:
    print(f"An error occurred: {e}")
```

## Usage

1. Activate the Conda environment:

   ```bash
   conda activate tangram
   ```

2. Run tangram script:

   ```bash
   python test_tangram_mediapipe.py
   ```
