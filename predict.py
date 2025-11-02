# Prediction interface for Cog ⚙️
# https://github.com/replicate/cog/blob/main/docs/python.md
from cog import BasePredictor, Input, Path
import os
import subprocess
from typing import List

MODEL_CACHE = "models"
CHECKPOINTS_CACHE = "/root/.cache/torch/hub/checkpoints/"

class Predictor(BasePredictor):
    def setup(self) -> None:
        """Load the model into memory to make running multiple predictions efficient"""
        # Verify models were downloaded during build
        if not os.path.exists(MODEL_CACHE):
            raise RuntimeError(f"Models not found at {MODEL_CACHE}")
        if not os.path.exists(CHECKPOINTS_CACHE):
            raise RuntimeError(f"Checkpoints not found at {CHECKPOINTS_CACHE}")

    def predict(
        self,
        audio: Path = Input(description="Input Audio File"),
    ) -> List[Path]:
        """Run a single prediction on the model"""
        output_folder = "/tmp/results/"
        
        # Remove output folder if it exists
        if os.path.exists(output_folder):
            os.system("rm -rf " + output_folder)
        
        # Run MVSEP subprocess
        subprocess.run(
            ["python", "inference.py", "--input_audio", str(audio), "--output_folder", output_folder],
            check=True
        )
        
        # Get list of files in the output folder
        files = os.listdir(output_folder)
        output_files = [Path(os.path.join(output_folder, file)) for file in files]
        
        return output_files
