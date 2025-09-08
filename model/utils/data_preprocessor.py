import pandas as pd
pd.set_option('future.no_silent_downcasting', True)
import numpy as np
import torch

from sklearn.compose import make_column_transformer
from sklearn.experimental import enable_iterative_imputer
from sklearn.impute import IterativeImputer, SimpleImputer
from sklearn.preprocessing import StandardScaler, OneHotEncoder, OrdinalEncoder, LabelEncoder, RobustScaler
from sklearn.model_selection import train_test_split
from sklearn.pipeline import make_pipeline

class DataPreprocessor:
    """
    A class to encapsulate the data preprocessing steps for a dataset.
    """
    def __init__(self, numerical_columns, OHE_columns, Ordinal_columns, target_column, seed=None):
        """
        Initializes the DataPreprocessor.

        Args:
            numerical_columns (list): List of numerical column names.
            OHE_columns (list): List of one-hot encoded column names.
            Ordinal_columns (list): List of ordinal column names.
            target_column (str): Name of the target column.
            random_state (int, optional): Random state for reproducibility. Defaults to 42.
        """
        self.numerical_columns = numerical_columns
        self.OHE_columns = OHE_columns
        self.Ordinal_columns = Ordinal_columns
        self.target_column = target_column
        # self.random_state = random_state
        self.preprocessor = self._build_preprocessor()
        # self.label_encoder = LabelEncoder()
        self.column_names = None
        self.categorical_column_indexes = None
        self._X_train_df = None
        self._X_test_df = None
        self._X_test_for_pred_analysis = None
        self._y_test_original = None

    def _build_preprocessor(self):
        """
        Builds the column transformer for preprocessing.

        Returns:
            sklearn.compose._column_transformer.ColumnTransformer: The preprocessor.
        """
        numeric_transformer = make_pipeline(
            RobustScaler(),
            SimpleImputer(
                missing_values=np.nan,
                strategy="median",
            ),
        )

        OHE_transformer = make_pipeline(
            SimpleImputer(strategy="constant", fill_value="missing"),
            OneHotEncoder(handle_unknown='ignore', sparse_output=False)
        )

        ordinal_transformer = make_pipeline(
            SimpleImputer(fill_value=-100),
            OrdinalEncoder(categories = [[1,2,3,4,5]], handle_unknown='use_encoded_value', unknown_value=-100)
        )

        preprocessor = make_column_transformer(
            (numeric_transformer, self.numerical_columns),
            (OHE_transformer, self.OHE_columns),
            (ordinal_transformer, self.Ordinal_columns)
        )
        return preprocessor

    def preprocess(self, df, drop_missing=False):
        """
        Preprocesses the input DataFrame.

        Args:
            df (pd.DataFrame): The input DataFrame containing features and the target column.
            drop_missing (bool, optional): Whether to drop columns ending with '_missing'. Defaults to False.

        Returns:
            tuple: A tuple containing the preprocessed training features (NumPy array),
                   preprocessed testing features (NumPy array), training target (NumPy array),
                   and testing target (NumPy array).
        """
        train_df = df[df["Split"] == "train"].copy()
        val_df = df[df["Split"] == "val"].copy()
        test_df = df[df["Split"] == "test"].copy()

        # if train_ids is not None and test_ids is not None:
        #     train_df = df[df.index.isin(df.index[df["pt #"].astype(str).isin(train_ids)])].copy()
        #     val_df = df[df.index.isin(df.index[df["pt #"].astype(str).isin(val_ids)])].copy() if val_ids is not None else None
        #     test_df = df[df.index.isin(df.index[df["pt #"].astype(str).isin(test_ids)])].copy()
        # else:
        #     if video_split == False:
        #         train_df = df[df["Split"] == "train"].copy()
        #         val_df = df[df["Split"] == "val"].copy()
        #         test_df = df[df["Split"] == "test"].copy()
        #     elif video_split == True:
        #         train_df = df[df["Split_Video"] == "train"].copy()
        #         val_df = df[df["Split_Video"] == "val"].copy()
        #         test_df = df[df["Split_Video"] == "test"].copy()

        drop_cols = [col for col in [self.target_column, "Split", "recom_therapy"] if col in train_df.columns]
        X_train = train_df.drop(columns=drop_cols)
        y_train = train_df[self.target_column]

        drop_cols = [col for col in [self.target_column, "Split", "recom_therapy"] if col in val_df.columns]
        X_val = val_df.drop(columns=drop_cols)
        y_val = val_df[self.target_column]

        drop_cols = [col for col in [self.target_column, "Split", "recom_therapy"] if col in test_df.columns]
        X_test = test_df.drop(columns=drop_cols)
        y_test = test_df[self.target_column]

        self._X_test_for_pred_analysis = pd.concat([X_test.copy(), y_test.copy()], axis=1)
        self._y_test_original = y_test.copy() 

        X_train_transformed = self.preprocessor.fit_transform(X_train)
        X_val_transformed = self.preprocessor.transform(X_val)
        X_test_transformed = self.preprocessor.transform(X_test)

        # y_train_encoded = self.label_encoder.fit_transform(y_train).ravel()
        # if y_val is not None and len(y_val) > 0:
        #     y_val_encoded = self.label_encoder.transform(y_val).ravel()
        # else:
        #     y_val_encoded = None
        # y_test_encoded = self.label_encoder.transform(y_test).ravel()

        label_mapping = {"MANAG": 0, "INTERVENTION": 1}

        y_train_encoded = y_train.map(label_mapping).values
        y_val_encoded = y_val.map(label_mapping).values
        y_test_encoded = y_test.map(label_mapping).values

        print("Label Mapping Used:")
        for k, v in label_mapping.items():
            print(f"{k}: {v}")

        # class_names = self.label_encoder.classes_

        # print("Label Encoding Mapping:")
        # for i, class_name in enumerate(class_names):
        #     print(f"{class_name}: {i}")

        ohe_feature_names = list(
            self.preprocessor.named_transformers_["pipeline-2"].get_feature_names_out(
                self.OHE_columns
            )
        )

        self.column_names_before_drop = self.numerical_columns + ohe_feature_names + self.Ordinal_columns

        self._X_train_df = pd.DataFrame(X_train_transformed, columns=self.column_names_before_drop, index=None)
        self._X_val_df = pd.DataFrame(X_val_transformed, columns=self.column_names_before_drop, index=None)
        self._X_test_df = pd.DataFrame(X_test_transformed, columns=self.column_names_before_drop, index=None)

        if drop_missing:
            self._X_train_df = self._X_train_df.drop(columns=[col for col in self._X_train_df.columns if col.endswith('_missing')])
            self._X_val_df = self._X_val_df.drop(columns=[col for col in self._X_val_df.columns if col.endswith('_missing')])
            self._X_test_df = self._X_test_df.drop(columns=[col for col in self._X_test_df.columns if col.endswith('_missing')])
            # Recalculate categorical column indexes after dropping '_missing' columns
            categorical_columns = [col for col in ohe_feature_names if not col.endswith('_missing')] + self.Ordinal_columns
        else:
            categorical_columns = ohe_feature_names + self.Ordinal_columns

        self._categorical_column_indexes = [self._X_train_df.columns.get_loc(col) for col in categorical_columns]

        X_train_processed = self._X_train_df.values
        X_val_processed = self._X_val_df.values
        X_test_processed = self._X_test_df.values

        return X_train_processed, X_val_processed, X_test_processed, y_train_encoded, y_val_encoded, y_test_encoded


    def get_column_names(self):
        """
        Returns the names of the processed columns after dropping '_missing' columns.

        Returns:
            list: List of column names after preprocessing.
        """
        if self._X_train_df is not None:
            return list(self._X_train_df.columns)
        else:
            return None

    def get_categorical_column_indexes(self):
        """
        Returns the indexes of the categorical columns in the processed data after dropping '_missing' columns.

        Returns:
            list: List of integer indexes corresponding to categorical columns.
        """
        return self._categorical_column_indexes

    def get_dfs(self):
        """
        Returns the processed training and testing DataFrames.

        Returns:
            tuple: A tuple containing the processed training DataFrame and the processed testing DataFrame.
        """
        return self._X_train_df, self._X_test_df
    
    def get_pred(self, model, X_test_processed, filter=True):
        """
        Returns the original test DataFrame concatenated with predictions and probabilities.
        Always saves both the full prediction CSV and the incorrect predictions CSV.

        Args:
            model: The trained machine learning model.
            X_test_processed (np.ndarray): The preprocessed test features.
            filter (bool, optional): If True, returns both the full and incorrect prediction DataFrames. Defaults to True.

        Returns:
            pandas.DataFrame or tuple:
                - If filter is False, returns the full DataFrame.
                - If filter is True, returns a tuple containing (full DataFrame, incorrect predictions DataFrame).
        """
        if self._X_test_for_pred_analysis is None or self._X_test_df is None:
            print("Warning: preprocess method needs to be called before getting prediction analysis.")
            return None

        y_prediction = model.predict(X_test_processed)
        print("y_prediction: ", y_prediction)

        self._X_test_for_pred_analysis['predicted_class'] = self.label_encoder.inverse_transform(y_prediction) # Map back to original class names

        probability_cols = []

        if hasattr(model, "predict_proba"):
            y_class_probabilities = model.predict_proba(X_test_processed)
            print("y_predict_proba: ", y_class_probabilities)

            class_name_mapping = {i: class_name for i, class_name in enumerate(self.label_encoder.classes_)}

            for i in range(y_class_probabilities.shape[1]):
                class_index = i
                if class_index in class_name_mapping:
                    class_name = class_name_mapping[class_index]
                    prob_col_name = f'probability_{class_name}'
                    self._X_test_for_pred_analysis[prob_col_name] = y_class_probabilities[:, i]
                    probability_cols.append(prob_col_name)
                else:
                    prob_col_name = f'probability_class_{i}'
                    self._X_test_for_pred_analysis[prob_col_name] = y_class_probabilities[:, i]
                    probability_cols.append(prob_col_name)
                    print(f"Warning: No mapping found for class index {class_index}. Using default name.")
        else:
            print("Warning: The provided model does not have a predict_proba method.")

        target_col = self.target_column
        prediction_cols = ['predicted_class', target_col] + probability_cols
        original_cols = [col for col in self._X_test_for_pred_analysis.columns if col not in prediction_cols]
        new_column_order = prediction_cols + original_cols
        self._X_test_for_pred_analysis = self._X_test_for_pred_analysis[new_column_order]

        self._X_test_for_pred_analysis.to_csv("all_predictions.csv", index=False)

        incorrect_predictions_df = self._X_test_for_pred_analysis[
            self._X_test_for_pred_analysis['predicted_class'] != self._X_test_for_pred_analysis[target_col]
        ]
        incorrect_predictions_df.to_csv("incorrect_predictions.csv", index=False)

        print("Dataframes saved.")

        # if filter:
        #     return self._X_test_for_pred_analysis, incorrect_predictions_df
        # else:
        #     return self._X_test_for_pred_analysis

def load_data(binary=True, seed=0, ref_df=None):
    
    file = "/home/diane.kim/nature/data/final/dad_cleaned_full_6865_wTTE.csv"
    
    ACS = pd.read_csv(file, header=0)

    if ref_df is not None:
        ref_df = pd.read_csv(ref_df)
        kept_mrn = ACS['mrn_1'][ACS['Split'].isin(['train', 'val'])].unique().tolist()
        test_mrn = ref_df['mrn_1'][ref_df['Split'].isin(['test'])].unique().tolist()
        kept_mrn.extend(test_mrn)
        ACS = ACS.loc[ACS['mrn_1'].isin(kept_mrn)]

    train_ids = ACS[ACS["Split"] == "train"]["mrn_1"].tolist()
    val_ids = ACS[ACS["Split"] == "val"]["mrn_1"].tolist()
    test_ids = ACS[ACS["Split"] == "test"]["mrn_1"].tolist()

    print(ACS.shape)
    
    if binary:
        target_column = "MANAG"
        print("Running MANAG vs INTERVENTION.")

    else:
        target_column = "recom_therapy"
        print("Running MANAG vs PTCA vs SURG.")
    
    print(ACS[target_column].value_counts())
    
    # train_ids = [str(pid).strip() for pid in train_ids]
    # val_ids = [str(pid).strip() for pid in val_ids]
    # test_ids = [str(pid).strip() for pid in test_ids]

    drop_col = ["pt #", "id_1", "mrn_org", "mrn_1", "dob", "id", "AdmitDate", "DischargeDate", "DxCode_1",
                "event_date_str", "event_date", "pt_height", "pt_weight", "id_2", "DischargeDate_str", "DischargeDate",
                "Study Date", "Study Type", "cad", "cad_level", "EF_category", "EF"] 
    
    ACS = ACS.drop(drop_col, axis=1)     

    ACS = ACS.replace({"": np.nan})
    ACS = ACS.replace({"\\N": np.nan})

    OHE_columns = [
        'pt_sex_abbrev',
        'DxDesc_1',
        'DM',
        'diabetes_type',
        'hypertension',
        'hyperlipidemia',
        'smoking',
        'peripheral_vascular',
        'cerebrovascular',
        'pulmonary',
        'malignancy',
        'chf',
        'prior_infarction',
        # 'EF_category',
        # 'CAD_Original'
    ]

    Ordinal_columns = [
        # 'CAD'
    ]

    numerical_columns = [
        'age',
        'BMI',
        # 'EF'
    ]

    preprocessor = DataPreprocessor(
        numerical_columns=numerical_columns,
        OHE_columns=OHE_columns,
        Ordinal_columns=Ordinal_columns,
        target_column=target_column,
        seed=seed
    )
    
    X_train, X_val, X_test, y_train, y_val, y_test = preprocessor.preprocess(ACS, drop_missing=False)
    # X_train_df, X_test_df = preprocessor.get_dfs()

    # columns = preprocessor.get_column_names

    X_train = torch.tensor(X_train, dtype=torch.float32)
    X_val = torch.tensor(X_val, dtype=torch.float32)
    X_test = torch.tensor(X_test, dtype=torch.float32)

    y_train = torch.tensor(y_train, dtype=torch.long)
    y_val = torch.tensor(y_val, dtype=torch.long)
    y_test = torch.tensor(y_test, dtype=torch.long)

    return X_train, X_val, X_test, y_train, y_val, y_test, train_ids, val_ids, test_ids, preprocessor