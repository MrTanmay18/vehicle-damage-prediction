import os
import shutil
import hashlib
from PIL import Image
import torch
from torch import nn
from torchvision import models, transforms
from huggingface_hub import hf_hub_download

trained_model = None
class_names = ['F_Breakage', 'F_Crushed', 'F_Normal', 'R_Breakage', 'R_Crushed', 'R_Normal']

MODEL_PATH = "model/saved_model.pth"


class CarClassifierResNet(nn.Module):
    def __init__(self, num_classes=6, dropout_rate=0.5):
        super().__init__()
        self.model = models.resnet50(weights='DEFAULT')
        # Freeze all layers except the final fully connected layer
        for param in self.model.parameters():
            param.requires_grad = False

        # Unfreeze layer4 and fully connected layer
        for param in self.model.layer4.parameters():
            param.requires_grad = True

        # Replace the final fully connected layer
        self.model.fc = nn.Sequential(
            nn.Dropout(0.5),
            nn.Linear(self.model.fc.in_features, num_classes)
        )

    def forward(self, X):
        X = self.model(X)
        return X


def ensure_model():
    """Download the model from Hugging Face Hub if it isn't already on disk."""
    if not os.path.exists(MODEL_PATH):
        os.makedirs("model", exist_ok=True)
        downloaded_path = hf_hub_download(
            repo_id="MrTanmay18/saved_model.pth",
            filename="saved_model.pth"
        )
        shutil.copy(downloaded_path, MODEL_PATH)


def file_hash(path):
    with open(path, "rb") as f:
        return hashlib.md5(f.read()).hexdigest()


def predict(image_path):
    image = Image.open(image_path).convert('RGB')
    transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])
    image_tensor = transform(image).unsqueeze(0)  # (1,3,224,224)

    global trained_model

    if not trained_model:
        ensure_model()
        print("MODEL HASH:", file_hash(MODEL_PATH))
        trained_model = CarClassifierResNet()
        state_dict = torch.load(
            MODEL_PATH,
            map_location="cpu"
        )

        trained_model.load_state_dict(state_dict)
        trained_model.eval()

    with torch.no_grad():
        output = trained_model(image_tensor)
        _, predicted_class = torch.max(output, 1)
        return class_names[predicted_class.item()]