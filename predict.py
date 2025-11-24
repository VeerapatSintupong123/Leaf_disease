import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import transforms
from torchvision.models import resnet18, ResNet18_Weights

from PIL import Image
import numpy as np
import base64
from io import BytesIO

# Device
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Classes
label_cols = ['healthy', 'multiple_diseases', 'rust', 'scab']

class PlantModel(nn.Module):
    def __init__(self, num_classes=4):
        super().__init__()

        # Load pretrained ResNet18
        weights = ResNet18_Weights.DEFAULT
        self.backbone = resnet18(weights=weights)
        in_features = self.backbone.fc.in_features
        
        # Remove old classifier
        self.backbone.fc = nn.Identity()

        # New classification
        self.classifier = nn.Linear(in_features, num_classes)

        # Freeze backbone
        for p in self.backbone.parameters():
            p.requires_grad = False

    def forward(self, x):
        features = self.backbone(x)
        features = F.dropout(features, 0.25, training=self.training)
        logits = self.classifier(features)
        return logits

class LeafPredictor:
    def __init__(self, model_path):
        self.model = PlantModel(num_classes=len(label_cols))
        state_dict = torch.load(model_path, map_location=device, weights_only=True)
        self.model.load_state_dict(state_dict)
        self.model.eval()
        self.model.to(device)
        self.transform = transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                 std=[0.229, 0.224, 0.225])
        ])

    def predict(self, img_path=None, img=None, show=False):
        """
        Predict probabilities for an image.
        Can pass either img_path or PIL.Image object img.
        Returns predicted class, probability, and full probs dict.
        """
        if img_path:
            img = Image.open(img_path).convert("RGB")
        elif img is None:
            raise ValueError("Either img_path or img must be provided.")

        input_tensor = self.transform(img).unsqueeze(0).to(device)
        with torch.no_grad():
            outputs = self.model(input_tensor)
            probs = F.softmax(outputs, dim=1).cpu().numpy()[0]

        pred_idx = np.argmax(probs)
        predicted_class = label_cols[pred_idx]
        predicted_prob = probs[pred_idx]

        probs_dict = {cls: float(p) for cls, p in zip(label_cols, probs)}

        if show:
            import matplotlib.pyplot as plt
            plt.imshow(img)
            plt.axis("off")
            title = "\n".join([f"{cls}: {p:.2f}" for cls, p in probs_dict.items()])
            plt.title(title, fontsize=12)
            plt.show()

        return (predicted_class, predicted_prob), probs_dict

    @staticmethod
    def encode_image_to_base64(img_path):
        """Encode image to base64 string"""
        with open(img_path, "rb") as f:
            return base64.b64encode(f.read()).decode("utf-8")

    @staticmethod
    def decode_base64_to_image(base64_str):
        """Decode base64 string to PIL image"""
        img_bytes = base64.b64decode(base64_str)
        return Image.open(BytesIO(img_bytes)).convert("RGB")

if __name__ == "__main__":
    predictor = LeafPredictor(r"checkpoints\tf_resnet18_10_2.pth")
    
    # encode image
    image_base64 = LeafPredictor.encode_image_to_base64('rust.jpg')

    # decode image
    image = LeafPredictor.decode_base64_to_image(image_base64)

    pred_class, probs_dict = predictor.predict(img_path=None, img=image, show=False)
    print("Predicted class:", pred_class)
    print("Class probabilities:", probs_dict)