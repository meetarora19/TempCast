import json


def load_events(data_path="data/events.json"):
    with open(data_path, "r", encoding="utf-8") as f:
        return json.load(f)