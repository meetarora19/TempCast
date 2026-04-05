from models.forecasting_model import ForecastingModel


def run_forecast(query, query_timestamp=None):
    model = ForecastingModel()
    return model.predict(query, query_timestamp=query_timestamp)