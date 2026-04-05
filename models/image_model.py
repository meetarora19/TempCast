from transformers import CLIPProcessor, CLIPModel
from PIL import Image
import torch


class ImageModel:
    def __init__(self):
        self.model = CLIPModel.from_pretrained("openai/clip-vit-base-patch32")
        self.processor = CLIPProcessor.from_pretrained("openai/clip-vit-base-patch32")

    def encode_image(self, image_path):
        img = Image.open(image_path).convert("RGB")
        inputs = self.processor(images=img, return_tensors="pt")

        with torch.no_grad():
            outputs = self.model.vision_model(pixel_values=inputs["pixel_values"])

        features = outputs.pooler_output
        return features.squeeze(0).cpu().numpy()


if __name__ == "__main__":
    model = ImageModel()

    while True:
        image_path = input("Enter image path (or 'exit'): ").strip()
        if image_path.lower() == "exit":
            break

        try:
            emb = model.encode_image(image_path)
            print("Image encoded successfully")
            print("Embedding shape:", emb.shape)
            print("-" * 60)
        except Exception as e:
            print("Error:", e)
            print("-" * 60)