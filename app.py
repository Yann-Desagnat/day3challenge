import io
import torch
import uvicorn
import numpy as np
import cv2
from fastapi import FastAPI, File, UploadFile
from fastapi.middleware.cors import CORSMiddleware

from fastapi.responses import Response
from models.network_swinir import SwinIR as net
from PIL import Image

# Initialize FastAPI
app = FastAPI()

# Autoriser toutes les origines (pour tester)
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Autorise toutes les origines (à restreindre en production)
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Load SwinIR Model
def load_model(model_path='001_classicalSR_DF2K_s64w8_SwinIR-M_x4.pth', scale=4):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = net(upscale=scale, in_chans=3, img_size=64, window_size=8,
                img_range=1., depths=[6, 6, 6, 6, 6, 6], embed_dim=180, 
                num_heads=[6, 6, 6, 6, 6, 6], mlp_ratio=2, upsampler='pixelshuffle',
                resi_connection='1conv')
    
    # Load weights
    pretrained_model = torch.load(model_path, map_location=device)
    model.load_state_dict(pretrained_model['params'], strict=False)
    model.eval().to(device)
    return model, device

# Load the model
model, device = load_model()

# Preprocess image
def preprocess_image(image: Image.Image):
    img = np.array(image).astype(np.float32) / 255.0  # Normalize
    img = np.transpose(img[:, :, [2, 1, 0]], (2, 0, 1))  # Convert to CHW-RGB
    img = torch.from_numpy(img).float().unsqueeze(0).to(device)  # Convert to tensor
    return img

# Upscale Image
def upscale_image(image: Image.Image):
    img_tensor = preprocess_image(image)
    with torch.no_grad():
        output = model(img_tensor)
    output = output.data.squeeze().float().cpu().clamp_(0, 1).numpy()
    
    # Conserver l'ordre RGB sans conversion vers BGR
    output = np.transpose(output, (1, 2, 0))  # Conserver l'ordre RGB
    output = (output * 255.0).astype(np.uint8)
    
    return output

# API Route
@app.post("/upscale")
async def upscale(file: UploadFile = File(...)):
    image = Image.open(io.BytesIO(await file.read())).convert("RGB")
    upscaled_image = upscale_image(image)

    # Convert back to bytes
    _, img_encoded = cv2.imencode(".png", upscaled_image)
    return Response(content=img_encoded.tobytes(), media_type="image/png")

# Run API
if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000)
