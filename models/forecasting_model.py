import math
from datetime import datetime

from models.temporal_memory import TemporalMemory
from utils.data_loader import load_events


class ForecastingModel:
    def __init__(self, data_path="data/events.json"):
        self.data = load_events(data_path)

        self.memory = TemporalMemory()
        self.memory.events = [item["event"] for item in self.data]
        self.memory.build_index()

    def _parse_date(self, date_str):
        return datetime.strptime(date_str, "%Y-%m-%d")

    def _temporal_decay(self, event_date, query_date, decay_lambda=0.001):
        gap_days = abs((query_date - event_date).days)
        return math.exp(-decay_lambda * gap_days)

    def predict(self, query, query_timestamp=None, k=5):
        matches = self.memory.retrieve(query, k=k)

        if query_timestamp is None:
            query_date = datetime.today()
        else:
            query_date = self._parse_date(query_timestamp)

        scored_results = []

        for match in matches:
            matched_event = match["event"]
            semantic_score = match["similarity"]

            for item in self.data:
                if item["event"] == matched_event:
                    event_date = self._parse_date(item["timestamp"])
                    temporal_score = self._temporal_decay(event_date, query_date)
                    final_score = semantic_score * temporal_score

                    scored_results.append({
                        "query": query,
                        "matched_event": item["event"],
                        "predicted_next_event": item["expected_next_event"],
                        "timestamp": item["timestamp"],
                        "type": item["type"],
                        "image_hint": item["image_hint"],
                        "image": item["image"],
                        "semantic_score": round(semantic_score, 4),
                        "temporal_score": round(temporal_score, 4),
                        "final_score": round(final_score, 4)
                    })

        scored_results.sort(key=lambda x: x["final_score"], reverse=True)
        return scored_results[0] if scored_results else None


if __name__ == "__main__":
    model = ForecastingModel()

    while True:
        query = input("Enter event query (or 'exit'): ").strip()
        if query.lower() == "exit":
            break

        query_timestamp = input("Enter query date (YYYY-MM-DD) or press Enter for today: ").strip()
        if query_timestamp == "":
            query_timestamp = None

        try:
            result = model.predict(query, query_timestamp=query_timestamp)

            print("\n=== Forecast Result ===")
            print("Query:", result["query"])
            print("Matched Event:", result["matched_event"])
            print("Predicted Next Event:", result["predicted_next_event"])
            print("Timestamp:", result["timestamp"])
            print("Type:", result["type"])
            print("Image Hint:", result["image_hint"])
            print("Image Path:", result["image"])
            print("Semantic Score:", result["semantic_score"])
            print("Temporal Score:", result["temporal_score"])
            print("Final Score:", result["final_score"])
            print("=" * 60)

        except Exception as e:
            print("Error:", e)
            print("=" * 60)