from services.forecast_service import run_forecast
from services.retrieval_service import retrieve_events
from services.image_service import encode_event_image_by_id, encode_custom_image
from utils.data_loader import load_events


def show_dataset_summary():
    data = load_events()

    print("\n=== Dataset Summary ===")
    print("Total events:", len(data))

    types = {}
    for item in data:
        t = item.get("type", "unknown")
        types[t] = types.get(t, 0) + 1

    print("Event types:")
    for k, v in types.items():
        print(f"- {k}: {v}")

    print("\nSample entries:")
    for item in data[:3]:
        print(f"{item['id']}. {item['event']} -> {item['expected_next_event']}")
    print("=" * 60)


def forecasting_demo():
    while True:
        query = input("\nEnter event query (or 'back'): ").strip()
        if query.lower() == "back":
            break

        query_timestamp = input("Enter query date (YYYY-MM-DD) or press Enter for today: ").strip()
        if query_timestamp == "":
            query_timestamp = None

        try:
            result = run_forecast(query, query_timestamp=query_timestamp)

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


def retrieval_demo():
    while True:
        query = input("\nEnter retrieval query (or 'back'): ").strip()
        if query.lower() == "back":
            break

        try:
            results = retrieve_events(query, k=3)

            print("\n=== Top 3 Matches ===")
            for i, r in enumerate(results, start=1):
                print(f"{i}. {r['event']}")
                print(f"   Similarity Score: {r['similarity']:.4f}")
            print("=" * 60)
        except Exception as e:
            print("Error:", e)
            print("=" * 60)


def image_demo():
    while True:
        user_input = input("\nEnter event id, image path, or 'back': ").strip()
        if user_input.lower() == "back":
            break

        try:
            if user_input.isdigit():
                result = encode_event_image_by_id(int(user_input))
            else:
                result = encode_custom_image(user_input)

            print("\n=== Image Encoding Result ===")
            for key, value in result.items():
                print(f"{key}: {value}")
            print("=" * 60)
        except Exception as e:
            print("Error:", e)
            print("=" * 60)


def main():
    while True:
        print("\n========== FORECAST DEMO ==========")
        print("1. Forecast next event")
        print("2. Retrieve similar historical events")
        print("3. Test image encoding")
        print("4. Show dataset summary")
        print("5. Exit")
        print("======================================")

        choice = input("Choose an option: ").strip()

        if choice == "1":
            forecasting_demo()
        elif choice == "2":
            retrieval_demo()
        elif choice == "3":
            image_demo()
        elif choice == "4":
            show_dataset_summary()
        elif choice == "5":
            print("Exiting demo.")
            break
        else:
            print("Invalid choice. Please enter 1-5.")


if __name__ == "__main__":
    main()