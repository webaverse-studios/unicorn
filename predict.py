# Prediction interface for Cog ⚙️
# https://github.com/replicate/cog/blob/main/docs/python.md

from cog import BasePredictor, BaseModel, File, Input, Path
from reconstruct_cog import reconstruct 

from typing import Any
import torch 

print('cuda status is',torch.cuda.is_available())


# import unicorn here


class Output(BaseModel):
    file: File


class Predictor(BasePredictor):
    def setup(self):
        """Load the model into memory to make running multiple predictions efficient"""

    def predict(
        self,
        model: str,
        input: Path = Input(description="Image to classify")
    ) -> Any:
        """Run a single prediction on the model"""
        try:
            output = []        
            d = reconstruct(model, input)    

            output.append(d['obj'])
            output.append(d['png'])
            
            return output
        except Exception as e:
            return f"Error: {e}"