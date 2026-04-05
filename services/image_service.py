import os

from models.image_model import ImageModel
from utils.data_loader import load_events


def encode_event_image_by_id(event_id):
    data = load_events()
    model = ImageModel()

    sample = next(item for item in data if item["id"] == event_id)
    image_name = os.path.basename(sample["image"])
    image_path = os.path.join("data", "Images", image_name)

    emb = model.encode_image(image_path)

    return {
        "id": sample["id"],
        "event": sample["event"],
        "image_path": image_path,
        "embedding_shape": emb.shape
    }


def encode_custom_image(image_path):
    model = ImageModel()
    emb = model.encode_image(image_path)

    return {
        "image_path": image_path,
        "embedding_shape": emb.shape
    }