import unittest
import pandas as pd
import numpy as np
import os
import joblib  # For model loading

from your_script import (
    load_config,
    get_config_value,
    load_data,
    handle_missing_values,
    scale_features,
    train_model,
    evaluate_model,
    main, # Import the main function
)

class TestDataProcessingIntegration(unittest.TestCase):

    def setUp(self):
        # ... (Existing setup code for config and regular data)
        self.config_path = "test_config.yaml"
        self.data_path = "test_data.csv"
        self.processed_data_path = "processed_test_data.csv"
        self.model_path = "test_model.pkl"
        self.numerical_features = ["col1", "col2"]
        self.target_variable = "target"
        self.config_data["data"]["processed_path"] = self.processed_data_path
        self.config_data["model"]["save_path"] = self.model_path

        with open(self.config_path, "w") as f:
            import yaml
            yaml.dump(self.config_data, f)
        self.test_df.to_csv(self.data_path, index=False)

    def tearDown(self):
        # ... (Existing teardown code)
        pass

    def test_integration_full_pipeline(self):
        # Test the full data processing and model training/evaluation pipeline
        config = load_config(self.config_path)
        raw_data_path = get_config_value(config, "data", "raw_path")
        processed_data_path = get_config_value(config, "data", "processed_path")
        model_path = get_config_value(config, "model", "save_path")
        missing_strategy = get_config_value(config, "preprocessing", "missing_value_strategy")
        fill_value = get_config_value(config, "preprocessing", "fillna_value")
        numerical_features = get_config_value(config, "data", "features")["numerical"]
        target_variable = get_config_value(config, "data", "target")

        df = load_data(raw_data_path)
        df = handle_missing_values(df, missing_strategy, fill_value, numerical_features)
        df = scale_features(df, numerical_features)
        save_data(df, processed_data_path)

        train_model(processed_data_path, model_path, target_variable)
        accuracy = evaluate_model(processed_data_path, model_path, target_variable)

        self.assertTrue(os.path.exists(processed_data_path))
        self.assertTrue(os.path.exists(model_path))
        self.assertIsInstance(accuracy, float)
        self.assertTrue(0 <= accuracy <= 1)
        # Load the model and make a prediction to test if it's usable
        model = joblib.load(model_path)
        try:
            model.predict(df[numerical_features])
        except Exception as e:
            self.fail(f"Model prediction failed: {e}")

    def test_main_function(self):
        # Test the main function directly - simulates running the script from the command line
        main(["-c", self.config_path]) # Pass arguments as a list

        self.assertTrue(os.path.exists(self.processed_data_path))
        self.assertTrue(os.path.exists(self.model_path))
        model = joblib.load(self.model_path)
        try:
            model.predict(self.test_df[self.numerical_features])
        except Exception as e:
            self.fail(f"Model prediction failed: {e}")

    def test_handle_missing_values_no_numerical_features(self):
        df = self.test_df.copy()
        df_handled = handle_missing_values(df, "mean", 0, [])
        pd.testing.assert_frame_equal(df, df_handled)

    def test_scale_features_no_numerical_features(self):
        df = self.test_df.copy()
        scaled_df = scale_features(df, [])
        pd.testing.assert_frame_equal(df, scaled_df)

    def test_train_model_empty_features(self):
        with self.assertRaises(ValueError):
            train_model(self.data_path, self.model_path, self.target_variable)


if __name__ == "__main__":
    unittest.main()