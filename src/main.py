from src.data_processing.loading import load_data, save_data
from src.data_processing.cleaning import handle_missing_values
from src.data_processing.transformation import scale_features
from src.model_training.evaluation import evaluate_model
from src.model_training.training import train_model
from src.utils.config import load_config, get_config_value
from src.utils.logger import get_logger

logger = get_logger(__name__)

def main():
    config = load_config()

    raw_data_path = get_config_value(config, "data", "raw_path")
    processed_data_path = get_config_value(config, "data", "processed_path")
    model_path = get_config_value(config, "model", "save_path")
    missing_strategy = get_config_value(config, "preprocessing", "missing_value_strategy")
    fill_value = get_config_value(config, "preprocessing", "fillna_value")
    numerical_features = get_config_value(config, "data", "features")["numerical"]

    df = load_data(raw_data_path)
    df = handle_missing_values(df, strategy=missing_strategy, value=fill_value)
    df = scale_features(df, numerical_features)
    save_data(df, processed_data_path)

    train_model(processed_data_path, model_path)
    accuracy = evaluate_model(processed_data_path, model_path)
    logger.info(f"Model accuracy: {accuracy}")

if __name__ == '__main__':
    main()
