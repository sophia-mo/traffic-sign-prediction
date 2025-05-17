import pandas as pd
import os
from sklearn.model_selection import train_test_split
from sklearn.feature_selection import SelectKBest, mutual_info_classif, chi2
import numpy as np

TRAIN_FEATURES_PATH = "train/Features"
TEST_FEATURES_PATH = "test/Features"
TRAIN_METADATA_PATH = "train/train_metadata.csv"
IMAGE_COLUMN_NAME = "image_path"

def load_and_merge_features(features_dir, image_column=IMAGE_COLUMN_NAME):
    """
    Loads all CSV files from the specified features directory and merges them
    into a single pandas DataFrame.
    """
    all_feature_files = [os.path.join(features_dir, f) for f in os.listdir(features_dir) if f.endswith('.csv')]
    if not all_feature_files:
        raise ValueError(f"No CSV files found in {features_dir}")

    merged_df = pd.read_csv(all_feature_files[0])

    for i in range(1, len(all_feature_files)):
        df_to_merge = pd.read_csv(all_feature_files[i])
        if image_column not in df_to_merge.columns:
            raise ValueError(f"'{image_column}' not found in {all_feature_files[i]}. Available columns: {df_to_merge.columns.tolist()}")
        if image_column not in merged_df.columns:
            raise ValueError(f"'{image_column}' not found in the base DataFrame from {all_feature_files[0]}. Available columns: {merged_df.columns.tolist()}")
        merged_df = pd.merge(merged_df, df_to_merge, on=image_column, how='inner')

    return merged_df

def select_features_and_prepare_data(
    features_dir_train,
    metadata_path_train,
    features_dir_test,
    image_column='image_path',
    label_column='ClassId',
    n_top_features=50,
    test_size_for_validation=0.2,
    random_state=42,
    feature_selection_method='mutual_info'
):
    """
    Feature loading, merging, selection, and data preparation.
    """
    # Load and merge training features
    print(f"Loading and merging training features from {features_dir_train}...")
    train_features_df = load_and_merge_features(features_dir_train, image_column)
    print(f"Training features loaded. Shape: {train_features_df.shape}")

    # Load training metadata (labels)
    print(f"Loading training metadata from {metadata_path_train}...")
    train_metadata_df = pd.read_csv(metadata_path_train)
    if image_column not in train_metadata_df.columns or label_column not in train_metadata_df.columns:
        raise ValueError(
            f"'{image_column}' or '{label_column}' not found in {metadata_path_train}. "
            f"Available columns: {train_metadata_df.columns.tolist()}"
        )
    print(f"Training metadata loaded. Shape: {train_metadata_df.shape}")

    # Select only necessary columns from metadata before merge
    train_metadata_df = train_metadata_df[[image_column, label_column]]
    merged_train_data = pd.merge(train_features_df, train_metadata_df, on=image_column, how='inner')
    print(f"Merged training data with labels. Shape: {merged_train_data.shape}")

    if merged_train_data.empty:
        raise ValueError("Merging training features and metadata resulted in an empty DataFrame. Check if image_path values match.")

    # Prepare data for feature selection
    X = merged_train_data.drop(columns=[image_column, label_column])
    y = merged_train_data[label_column]

    # Fill NaN values with 0
    X = X.fillna(0)
    
    # Convert any object columns to numeric if possible
    for col in X.columns:
        if X[col].dtype == 'object':
            try:
                X[col] = pd.to_numeric(X[col])
            except ValueError:
                print(f"Warning: Column {col} could not be converted to numeric and will be dropped.")
                X = X.drop(columns=[col])

    # Split into training and validation sets
    X_train_split, X_val_split, y_train_split, y_val_split = train_test_split(
        X, y, test_size=test_size_for_validation, random_state=random_state, stratify=y
    )
    print(f"Split training data: X_train_split shape {X_train_split.shape}, X_val_split shape {X_val_split.shape}")

    # Create index mappings to retrieve image_path for validation data
    train_indices = X.index.difference(X_val_split.index)
    val_indices = X_val_split.index
    
    # Calculate feature importance and select features
    print(f"Calculating feature importance using {feature_selection_method} for top {n_top_features} features...")
    actual_n_top_features = min(n_top_features, X_train_split.shape[1])
    if actual_n_top_features < n_top_features:
        print(f"Warning: Requested {n_top_features} features, but only {actual_n_top_features} are available.")

    # Choose the feature selection method
    if feature_selection_method.lower() == 'chi2':
        # Chi2 requires non-negative features
        if (X_train_split < 0).any().any():
            print("Warning: Chi-square test requires non-negative features. Converting negative values to 0.")
            X_train_split = X_train_split.clip(lower=0)
        selector = SelectKBest(chi2, k=actual_n_top_features)
    else:  # Default to mutual_info
        selector = SelectKBest(mutual_info_classif, k=actual_n_top_features)
    
    selector.fit(X_train_split, y_train_split)

    selected_feature_indices = selector.get_support(indices=True)
    selected_feature_names = X_train_split.columns[selected_feature_indices].tolist()
    
    # Get scores for selected features and create a DataFrame with feature names and scores
    scores = selector.scores_[selected_feature_indices]
    feature_importance = pd.DataFrame({
        'Feature': selected_feature_names,
        'Score': scores
    })
    # Sort by descending score
    feature_importance = feature_importance.sort_values('Score', ascending=False)
    
    # Print with appropriate header based on the method used
    score_type = "Chi-square" if feature_selection_method.lower() == 'chi2' else "Mutual Information"
    print(f"Selected features (sorted by {score_type} score):")
    for idx, row in feature_importance.iterrows():
        print(f"  {row['Feature']}: {row['Score']:.4f}")
    
    # Get the sorted list of feature names
    selected_feature_names = feature_importance['Feature'].tolist()

    # Create selected feature dataframe for training data
    # Only use samples that were in the training split, not the entire dataset
    train_data = merged_train_data.loc[train_indices]
    selected_train_df = train_data[[image_column, label_column] + selected_feature_names].copy()
    print(f"Selected training features DataFrame shape: {selected_train_df.shape}")
    
    # Create selected feature dataframe for validation data
    val_data = merged_train_data.loc[val_indices]
    selected_val_df = val_data[[image_column, label_column] + selected_feature_names].copy()
    print(f"Selected validation features DataFrame shape: {selected_val_df.shape}")

    # Process test data
    print(f"\nLoading and merging testing features from {features_dir_test}...")
    test_features_df = load_and_merge_features(features_dir_test, image_column)
    print(f"Test features loaded. Shape: {test_features_df.shape}")

    # Select only the image_column and the selected features
    final_test_columns = [image_column]
    for feature in selected_feature_names:
        if feature not in test_features_df.columns:
            print(f"Warning: Selected feature '{feature}' not found in test data. It will be added and filled with 0.")
            test_features_df[feature] = 0
        final_test_columns.append(feature)

    test_selected_df = test_features_df[final_test_columns].copy()
    test_selected_df = test_selected_df.fillna(0)
    print(f"Selected testing features DataFrame shape: {test_selected_df.shape}")

    return selected_train_df, selected_val_df, test_selected_df, selected_feature_names

if __name__ == '__main__':
    print("Starting feature selection process...")
    N_FEATURES_TO_SELECT = 50
    # Choose feature selection method - 'mutual_info' or 'chi2'
    FEATURE_SELECTION_METHOD = 'mutual_info'

    try:
        processed_train_df, processed_val_df, processed_test_df, top_features = select_features_and_prepare_data(
            features_dir_train=TRAIN_FEATURES_PATH,
            metadata_path_train=TRAIN_METADATA_PATH,
            features_dir_test=TEST_FEATURES_PATH,
            n_top_features=N_FEATURES_TO_SELECT,
            feature_selection_method=FEATURE_SELECTION_METHOD
        )

        print("\n--- Process Summary ---")
        print(f"\nSelected Top {len(top_features)} Features: {top_features}")

        print(f"\nProcessed Training Data with selected features (+ image_path, label):")
        print(processed_train_df.head())
        print(f"Shape: {processed_train_df.shape}")
        
        print(f"\nProcessed Validation Data with selected features (+ image_path, label):")
        print(processed_val_df.head())
        print(f"Shape: {processed_val_df.shape}")

        print(f"\nProcessed Test Data with selected features (+ image_path):")
        print(processed_test_df.head())
        print(f"Shape: {processed_test_df.shape}")

        # Uncomment to save results
        # processed_train_df.to_csv("train_selected_features.csv", index=False)
        # processed_val_df.to_csv("validation_selected_features.csv", index=False)
        # processed_test_df.to_csv("test_selected_features.csv", index=False)

    except FileNotFoundError as e:
        print(f"Error: {e}. Please ensure all CSV files and directories are correctly named and placed.")
    except ValueError as e:
        print(f"Error: {e}")
    except Exception as e:
        print(f"An unexpected error occurred: {e}")

