import os
import sys
import torch
import coremltools as ct

# Add parent directory to sys.path to import src
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from src.model_lib.MultiFTNet import MultiFTNet
from src.model_lib.MiniFASNet import MiniFASNetV2

def convert():
    model_path = "checkpoint/finetuned.pth"
    output_path = "checkpoint/finetuned.mlpackage"
    
    if not os.path.exists(model_path):
        print(f"Error: checkpoint {model_path} not found.")
        return

    # 1. Setup Model Architecture (matching train.py)
    device = torch.device("cpu") # Convert on CPU
    model = MultiFTNet(num_classes=3, img_channel=3, embedding_size=128, conv6_kernel=(5, 5))
    model.model = MiniFASNetV2(embedding_size=128, conv6_kernel=(5,5), num_classes=3, img_channel=3)
    
    # 2. Load Weights
    print(f"Loading weights from {model_path}...")
    state_dict = torch.load(model_path, map_location=device)
    model.load_state_dict(state_dict)
    model.eval()

    # 3. Dummy Input for Tracing
    # Shape is (Batch, Channel, Height, Width)
    example_input = torch.rand(1, 3, 80, 80)
    
    # 4. Trace the model
    print("Tracing model...")
    traced_model = torch.jit.trace(model, example_input)
    
    # 5. Convert using coremltools
    print("Converting to CoreML...")
    
    model_ct = ct.convert(
        traced_model,
        source='pytorch',
        inputs=[ct.ImageType(
            name="image", 
            shape=example_input.shape,
            scale=255.0, # Scale [0, 1] to [0, 255]
            bias=[0, 0, 0],
            color_layout=ct.colorlayout.BGR
        )],
        outputs=[ct.TensorType(name="logits")],
        minimum_deployment_target=ct.target.iOS15
    )

    # 6. Save Model
    print(f"Saving CoreML model to {output_path}...")
    model_ct.save(output_path)
    print("Conversion successful.")

if __name__ == "__main__":
    convert()
