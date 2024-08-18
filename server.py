from fastapi import FastAPI, UploadFile, File
import torch
from torchvision import transforms
from PIL import Image
import io

app = FastAPI()

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# Path to the model
MODEL_PATH = './model/model.pth'

def load_model():
    model = torch.load(MODEL_PATH, map_location=device)
    model.to(device)
    model.eval()
    return model

model = load_model()

transform = transforms.Compose([
    transforms.Resize((512, 512)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406],
                         std=[0.229, 0.224, 0.225])
])

@app.post("/predict")
async def predict(file: UploadFile = File(...)):
    """Predict the number of palm trees in the uploaded image."""
    contents = await file.read()
    image = Image.open(io.BytesIO(contents)).convert("RGB")
    image = transform(image).unsqueeze(0).to(device)
    
    with torch.no_grad():
        predictions = model(image)
    
    # Process predictions
    pred = predictions[0] 
    labels = pred['labels']
    
    palm_tree_count = (labels == 1).sum().item()
    
    # Prepare bounding boxes and scores for the response
    boxes = pred['boxes'][labels == 1].cpu().numpy().tolist()
    scores = pred['scores'][labels == 1].cpu().numpy().tolist()
    
    return {
        "palm_tree_count": palm_tree_count,
        "boxes": boxes,
        "scores": scores
    }

@app.get("/health")
def health_check():
    """Check the health of the API."""
    return {"status": "healthy"}
