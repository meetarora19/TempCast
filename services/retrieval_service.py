from models.temporal_memory import TemporalMemory
from utils.data_loader import load_events


def retrieve_events(query, k=3):
    data = load_events()

    memory = TemporalMemory()
    memory.events = [item["event"] for item in data]
    memory.build_index()

    return memory.retrieve(query, k=k)