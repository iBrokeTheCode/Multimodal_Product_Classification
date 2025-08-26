import os
import warnings
from io import BytesIO

import numpy as np
import pandas as pd
import requests
from PIL import Image
from sklearn.model_selection import train_test_split

# 💬 NOTE: Suppress all warnings
warnings.filterwarnings("ignore")


def process_embeddings(df, col_name):
    """
    Process embeddings in a DataFrame column.

    Args:
    - df (pd.DataFrame): The DataFrame containing the embeddings column.
    - col_name (str): The name of the column containing the embeddings.

    Returns:
    pd.DataFrame: The DataFrame with processed embeddings.

    Steps:
    1. Convert the values in the specified column to lists.
    2. Extract values from lists and create new columns for each element.
    3. Remove the original embeddings column.

    Example:
    df_processed = process_embeddings(df, 'embeddings')
    """
    # Convert the values (eg. "[-0.123, 0.456, ...]") in the column to lists
    df[col_name] = df[col_name].apply(eval)

    # Extract values from lists and create new columns
    """ 🔎 Example
    text_1   text_2   text_3
    0  -0.123   0.456   0.789
    1   0.321  -0.654   0.987
    """
    embeddings_df = pd.DataFrame(
        df[col_name].to_list(),
        columns=[f"text_{i + 1}" for i in range(df[col_name].str.len().max())],
    )
    df = pd.concat([df, embeddings_df], axis=1)

    # Remove the original "embeddings" column
    df = df.drop(columns=[col_name])

    return df


def rename_image_embeddings(df):
    """
    Rename columns in a DataFrame for image embeddings.

    Args:
    - df (pd.DataFrame): The DataFrame containing columns to be renamed.

    Returns:
    pd.DataFrame: The DataFrame with renamed columns.

    Example:
    df_renamed = rename_image_embeddings(df)
    """
    # From 0    1    2   label  ➡️ image_0  image_1  image_2  label
    df.columns = [f"image_{int(col)}" if col.isdigit() else col for col in df.columns]

    return df


def preprocess_data(
    text_data,
    image_data,
    text_id="image_id",
    image_id="ImageName",
    embeddings_col="embeddings",
):
    """
    Preprocess and merge text and image dataframes.

    Args:
    - text_data (pd.DataFrame): DataFrame containing text data.
    - image_data (pd.DataFrame): DataFrame containing image data.
    - text_id (str): Column name for text data identifier.
    - image_id (str): Column name for image data identifier.
    - embeddings_col (str): Column name for embeddings data.

    Returns:
    pd.DataFrame: Merged and preprocessed DataFrame.

    This function:
    Process text and image embeddings.
    Convert image_id and text_id values to integers.
    Merge dataframes using id.
    Drop unnecessary columns.

    Example:
    merged_df = preprocess_data(text_df, image_df)
    """
    # Call previous functions to tune the text and image dataframes
    text_data = process_embeddings(text_data, embeddings_col)
    image_data = rename_image_embeddings(image_data)

    # Drop missing values in image id - Removes rows where the ID (used to join text ↔ image) is missing.
    image_data = image_data.dropna(subset=[image_id])
    text_data = text_data.dropna(subset=[text_id])

    # Cleans up text IDs: if the column contains file paths (like "data/images/123.jpg"), it extracts just the file name ("123.jpg").
    text_data[text_id] = text_data[text_id].apply(lambda x: x.split("/")[-1])

    # Merge dataframes using image_id - Joins text and image embeddings using the IDs (text_id vs image_id).
    df = pd.merge(text_data, image_data, left_on=text_id, right_on=image_id)

    # Drop unnecessary columns - Removes the original ID columns since they’re no longer needed after the merge.
    df.drop([image_id, text_id], axis=1, inplace=True)

    return df


class ImageDownloader:
    """
    Image downloader class to download images from URLs.

    Args:
    - image_dir (str): Directory to save images.
    - image_size (tuple): Size of the images to be saved.
    - override (bool): Whether to override existing images.

    Methods:
    - download_images(df, print_every=1000): Download images from URLs in a DataFrame.
        Args:
        - df (pd.DataFrame): DataFrame containing image URLs.
        - print_every (int): Print progress every n images.
        Returns:
        pd.DataFrame: DataFrame with image paths added.

    Example:
    downloader = ImageDownloader()
    df = downloader.download_images(df)
    """

    def __init__(
        self, image_dir="data/images/", image_size=(224, 224), overwrite=False
    ):
        self.image_dir = image_dir
        self.image_size = image_size
        self.overwrite = overwrite

        # Create the directory if it doesn't exist
        if not os.path.exists(self.image_dir):
            os.makedirs(self.image_dir)

    def download_images(self, df, print_every=1000):
        # Bulk download images from a DataFrame of URLs, resize them to a standard format, and add their local paths back to the DataFrame.
        image_paths = []

        i = 0
        for index, row in df.iterrows():
            if i % print_every == 0:
                print(f"Downloading image {i}/{len(df)}")
                i += 1

            sku = row["sku"]
            image_url = row["image"]
            image_path = os.path.join(self.image_dir, f"{sku}.jpg")

            if os.path.exists(image_path) and not self.overwrite:
                print(f"Image {sku} is already in the path.")
                image_paths.append(image_path)
                continue

            try:
                response = requests.get(image_url)
                response.raise_for_status()
                img = Image.open(BytesIO(response.content))
                img = img.resize(self.image_size, Image.Resampling.LANCZOS)
                img.save(image_path)
                # print(f"Downloaded image for SKU: {sku}")
                image_paths.append(image_path)
            except Exception as e:
                print(f"Could not download image for SKU: {sku}. Error: {e}")
                image_paths.append(np.nan)

        df["image_path"] = image_paths
        return df


def train_test_split_and_feature_extraction(df, test_size=0.3, random_state=42):
    """
    Split the data into train and test sets and extract features and labels.

    Args:
    - df (pd.DataFrame): DataFrame containing the data.

    Keyword Args:
    - test_size (float): Size of the test set.
    - random_state (int): Random state for reproducibility

    Returns:
    pd.DataFrame: Train DataFrame.
    pd.DataFrame: Test DataFrame.
    list: List of columns with text embeddings.
    list: List of columns with image embeddings.
    list: List of columns with class labels.

    Example:
    train_df, test_df, text_columns, image_columns, label_columns = train_test_split_and_feature_extraction(df)
    """

    # Split the data into train and test sets setting using the test_size and random_state parameters
    train_df, test_df = train_test_split(
        df, test_size=test_size, random_state=random_state
    )

    # Select the name of the columns with the text embeddings and return it as a list (Even if there is only one column)
    text_columns = [col for col in df.columns if col.startswith("text_")]

    # Select the name of the columns with the image embeddings and return it as a list (Even if there is only one column)
    image_columns = [col for col in df.columns if col.startswith("image_")]

    # Select the name of the column with the class labels and return it as a list (Even if there is only one column)
    label_columns = ["class_id"]

    return train_df, test_df, text_columns, image_columns, label_columns
