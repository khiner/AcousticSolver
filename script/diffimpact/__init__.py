"""DiffImpact reproduction and Apple Silicon port."""

import os

# Keep float32 matrix operations at full precision on M5 neural accelerators.
os.environ["MLX_ENABLE_TF32"] = "0"
