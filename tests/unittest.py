import unittest

import pandas as pd
import numpy as np
from script import load_config, get_config_value, load_data, handle_missing_values
import os

class TestDataProcessing(unittest.TestCase):

    def setUp(self):
        # Create a dummy config file for testing
        self.config_data = {
            "data": {"raw_path": "test_data.csv", "processed_path": "test_data_processed.csv"},
            "model": {"save_path": "test_model.pkl"},
            "preprocessing": {"missing_value_strategy": "mean", "fillna_value": 0},
        }
        with open("test_config.yaml", "w") as f:
            import yaml
            yaml.dump(self.config_data, f)

        # Create a dummy CSV file for testing
        self.test_df = pd.DataFrame({"col1": [1, 2, np.nan, 4], "col2": [5, np.nan, 7, 8], "target": [0, 1, 0, 1]})
        self.test_df.to_csv("test_data.csv", index=False)
        self.numerical_features = ["col1", "col2"]

    def tearDown(self):
        # Clean up test files
        os.remove("test_config.yaml")
        os.remove("test_data.csv")
        try:
            os.remove("processed_test_data.csv")
        except FileNotFoundError:
            pass
        try:
            os.remove("test_model.pkl")
        except FileNotFoundError:
            pass

    def test_load_config(self):
        config = load_config("test_config.yaml")
        self.assertEqual(config, self.config_data)

        with self.assertRaises(ValueError):
            load_config("non_existent_config.yaml")

    def test_get_config_value(self):
        config = self.config_data
        self.assertEqual(get_config_value(config, "data", "raw_path"), "test_data.csv")
        with self.assertRaises(ValueError):
            get_config_value(config, "non_existent_section", "key")

    def test_load_data(self):
        df = load_data("test_data.csv")
        pd.testing.assert_frame_equal(df, self.test_df)
        with self.assertRaises(ValueError):
            load_data("non_existent_data.csv")

    def test_handle_missing_value(self):
        expected_df = self.test_df.fillna(self.test_df.mean())
        df = handle_missing_values(self.test_df.copy(), "mean", 0, self.numerical_features)
        pd.testing.assert_frame_equal(df, expected_df)

        expected_df = self.test_df.fillna(0)
        df = handle_missing_values(self.test_df.copy(), "constant", 0, self.numerical_features)
        pd.testing.assert_frame_equal(df, expected_df)

    def tst_scale_features(self):
        df = self.test_df.copy()
        df = handle_missing_values(df, "mean", 0, self.numerical_features)
        scaled_df = scale_features(df.copy(), self.numerical_features)
        self.assertEqual(scaled_df.shape, df.shape)
        self.assertIsInstance(scaled_df, np.ndarray)

    def test_train_model(self):
        from your_script import train_model
        train_model("test_data.csv", "test_model.pkl", "target")
        self.assertTrue(os.path.exists("test_model.pkl"))

    def test_evaluate_model(self):
        from your_script import train_model, evaluate_model
        train_model("test_data.csv", "test_model.pkl", "target")
        accuracy = evaluate_model("test_data.csv", "test_model.pkl", "target")
        self.assertIsInstance(accuracy, float)
        self.assertTrue(0 <= accuracy <= 1)


if __name__ == "__main__":
    unittest.main()