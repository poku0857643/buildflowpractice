import unittest
import pandas as pd
import numpy as np
from your_script import (
    load_config,
    get_config_value,
    load_data,
    handle_missing_values,
    scale_features,
    train_model,
    evaluate_model
)
import os

class TestDataProcessingEdgeCases(unittest.TestCase): # New test class

    def setUp(self):
        # ... (Existing setup code for config and regular data)

        # Create edge case dataframes
        self.empty_df = pd.DataFrame()
        self.all_nan_df = pd.DataFrame({"col1": [np.nan, np.nan, np.nan], "col2": [np.nan, np.nan, np.nan]})
        self.one_unique_df = pd.DataFrame({"col1": [1, 1, 1], "col2": [2, 2, 2], "target":[0,0,0]})
        self.string_in_num_df = pd.DataFrame({"col1": ["1", "2", "a"], "col2": [4, 5, 6], "target":[0,1,0]})
        self.zero_length_df = pd.DataFrame(columns=["col1", "col2", "target"])

    def tearDown(self):
        # ... (Existing teardown code)
        pass

    def test_load_empty_data(self):
        with open("empty_test.csv", "w") as f:
            pass # Creating an empty file
        df = load_data("empty_test.csv")
        self.assertTrue(df.empty)
        os.remove("empty_test.csv")

    def test_handle_all_nan(self):
        df = handle_missing_values(self.all_nan_df.copy(), "mean", 0, ["col1","col2"])
        self.assertTrue(np.all(np.isnan(df)))  # Check if still all NaN after mean imputation

    def test_scale_one_unique(self):
        df = self.one_unique_df.copy()
        scaled_df = scale_features(df, ["col1", "col2"])
        self.assertTrue(np.all(np.isnan(scaled_df)))

    def test_handle_string_in_numerical(self):
        with self.assertRaises(TypeError):
            handle_missing_values(self.string_in_num_df.copy(), "mean", 0, ["col1", "col2"])

    def test_train_model_zero_length(self):
        with self.assertRaises(ValueError):
            train_model(self.zero_length_df, "test_model.pkl", "target")

    def test_evaluate_model_zero_length(self):
        with self.assertRaises(ValueError):
            evaluate_model(self.zero_length_df, "test_model.pkl", "target")

    def test_invalid_config_value(self):
        config = self.config_data
        with self.assertRaises(ValueError):
            get_config_value(config, "data", "non_existent_key")

    def test_missing_config_file(self):
        with self.assertRaises(ValueError):
            load_config("missing_config.yaml")

    def test_train_model_one_unique_target(self):
        with self.assertRaises(ValueError):
            train_model("test_data.csv", "test_model.pkl", "target") # use the one_unique_df for this test

if __name__ == "__main__":
    unittest.main()