import json
import os

import numpy as np
import pandas as pd
import torch
from transformers import AutoModel, AutoTokenizer


class HuggingFaceEmbeddings:
    """
    A class to handle text embedding generation using a Hugging Face pre-trained transformer model.
    This class loads the model, tokenizes the input text, generates embeddings, and provides an option
    to save the embeddings to a CSV file.

    Args:
        model_name (str, optional): The name of the Hugging Face pre-trained model to use for generating embeddings.
                                    Default is 'sentence-transformers/all-MiniLM-L6-v2'.
        path (str, optional): The path to the CSV file containing the text data. Default is 'data/file.csv'.
        save_path (str, optional): The directory path where the embeddings will be saved. Default is 'Models'.
        device (str, optional): The device to run the model on ('cpu' or 'cuda'). If None, it will automatically detect
                                a GPU if available; otherwise, it defaults to CPU.

    Attributes:
        model_name (str): The name of the Hugging Face model used for embedding generation.
        tokenizer (transformers.AutoTokenizer): The tokenizer corresponding to the chosen model.
        model (transformers.AutoModel): The pre-trained model loaded for embedding generation.
        path (str): Path to the input CSV file.
        save_path (str): Directory where the embeddings CSV will be saved.
        device (torch.device): The device on which the model and data are processed (CPU or GPU).

    Methods:
        get_embedding(text):
            Generates embeddings for a given text input using the pre-trained model.

        get_embedding_df(column, directory, file):
            Reads a CSV file, computes embeddings for a specified text column, and saves the resulting DataFrame
            with embeddings to a new CSV file in the specified directory.

    Example:
        embedding_instance = HuggingFaceEmbeddings(model_name='sentence-transformers/all-MiniLM-L6-v2',
                                                   path='data/products.csv', save_path='output')
        text_embedding = embedding_instance.get_embedding("Sample product description.")
        embedding_instance.get_embedding_df(column='description', directory='output', file='product_embeddings.csv')

    Notes:
        - The Hugging Face model and tokenizer are downloaded from the Hugging Face hub.
        - The function supports large models and can run on either GPU or CPU, depending on device availability.
        - The input text will be truncated and padded to a maximum length of 512 tokens to fit into the model.
    """

    def __init__(
        self,
        model_name="sentence-transformers/all-MiniLM-L6-v2",
        path="data/file.csv",
        save_path=None,
        device=None,
    ):
        """
        Initializes the HuggingFaceEmbeddings class with the specified model and paths.

        Args:
            model_name (str, optional): The name of the Hugging Face pre-trained model. Default is 'sentence-transformers/all-MiniLM-L6-v2'.
            path (str, optional): The path to the CSV file containing text data. Default is 'data/file.csv'.
            save_path (str, optional): Directory path where the embeddings will be saved. Default is 'Models'.
            device (str, optional): Device to use for model processing. Defaults to 'cuda' if available, otherwise 'cpu'.
        """
        self.model_name = model_name
        # Load the Hugging Face tokenizer from a pre-trained model
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)

        # Load the model from the Hugging Face model hub from the specified model name
        self.model = AutoModel.from_pretrained(model_name)
        self.path = path
        self.save_path = save_path or "Models"

        # Define device
        if device is None:
            # Note: If you have a mac, you may want to change 'cuda' to 'mps' to use GPU
            self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        else:
            self.device = torch.device(device)
        print(f"Using device: {self.device}")

        # Move model to the specified device
        self.model.to(self.device)
        print(f"Model moved to device: {self.device}")
        print(f"Model: {model_name}")

    def get_embedding(self, text):
        """
        Generates embeddings for a given text using the Hugging Face model.

        Args:
            text (str): The input text for which embeddings will be generated.

        Returns:
            np.ndarray: A numpy array containing the embedding vector for the input text.
        """
        # Tokenize the input text using the Hugging Face tokenizer
        inputs = self.tokenizer(
            text, return_tensors="pt", truncation=True, padding=True, max_length=512
        )

        # Move the inputs to the device
        inputs = {key: value.to(self.device) for key, value in inputs.items()}

        with torch.no_grad():
            # Generate the embeddings using the Hugging Face model from the tokenized input
            outputs = self.model(**inputs)

        # Extract the embeddings from the model output, send to cpu and return the numpy array
        last_hidden_state = outputs.last_hidden_state

        embeddings = last_hidden_state.mean(dim=1)
        embeddings = embeddings.cpu().numpy()

        return embeddings[0]

    def get_embedding_df(self, column, directory, file):
        # Load the CSV file
        df = pd.read_csv(self.path)
        # Generate embeddings for the specified column using the `get_embedding` method
        df["embeddings"] = df[column].apply(
            lambda x: self.get_embedding(str(x)).tolist() if pd.notnull(x) else None
        )

        os.makedirs(directory, exist_ok=True)

        # Save the DataFrame with the embeddings to a new CSV file in the specified directory
        output_path = os.path.join(directory, file)
        df.to_csv(output_path, index=False)

        print(f"✅ Embeddings saved to {output_path}")


class GPT:
    """
    A class to interact with the OpenAI GPT API for generating text embeddings from a given dataset.
    This class provides methods to retrieve embeddings for text data and save them to a CSV file.

    Args:
        path (str, optional): The path to the CSV file containing the text data. Default is 'data/file.csv'.
        embedding_model (str, optional): The embedding model to use for generating text embeddings.
                                         Default is 'text-embedding-3-small'.

    Attributes:
        path (str): Path to the CSV file.
        embedding_model (str): The embedding model used for generating text embeddings.

    Methods:
        get_embedding(text):
            Generates and returns the embedding vector for the given text using the OpenAI API.

        get_embedding_df(column, directory, file):
            Reads a CSV file, computes the embeddings for a specified text column, and saves the embeddings
            to a new CSV file in the specified directory.

    Example:
        gpt_instance = GPT(path='data/products.csv', embedding_model='text-embedding-ada-002')
        text_embedding = gpt_instance.get_embedding("Sample product description.")
        gpt_instance.get_embedding_df(column='description', directory='output', file='product_embeddings.csv')

    Notes:
        - The OpenAI API key must be stored in a `.env` file with the variable name `OPENAI_API_KEY`.
        - The OpenAI Python package should be installed (`pip install openai`), and an active OpenAI API key is required.
    """

    def __init__(self, path="data/file.csv", embedding_model="text-embedding-3-small"):
        """
        Initializes the GPT class with the provided CSV file path and embedding model.

        Args:
            path (str, optional): The path to the CSV file containing the text data. Default is 'data/file.csv'.
            embedding_model (str, optional): The embedding model to use for generating text embeddings.
                                             Default is 'text-embedding-3-small'.
        """
        import openai
        from dotenv import find_dotenv, load_dotenv

        # Load the OpenAI API key from the .env file
        _ = load_dotenv(find_dotenv())  # read local .env file
        # Set the OpenAI API key
        openai.api_key = os.getenv("OPENAI_API_KEY")

        self.path = path
        self.embedding_model = embedding_model

    def get_embedding(self, text):
        """
        Generates and returns the embedding vector for the given text using the OpenAI API.

        Args:
            text (str): The input text to generate the embedding for.

        Returns:
            list: A list containing the embedding vector for the input text.
        """
        from openai import OpenAI

        # Instantiate the OpenAI client
        client = OpenAI()

        # Optional. Do text preprocessing if needed (e.g., removing newlines)
        text = text.replace("\n", " ").strip()

        # Call the OpenAI API to generate the embeddings and return only the embedding data
        response = client.embeddings.create(model=self.embedding_model, input=text)

        embeddings_np = np.array(response.data[0].embedding, dtype=np.float32)
        return embeddings_np

    def get_embedding_df(self, column, directory, file):
        """
        Reads a CSV file, computes the embeddings for a specified text column, and saves the results in a new CSV file.

        Args:
            column (str): The name of the column in the CSV file that contains the text data.
            directory (str): The directory where the output CSV file will be saved.
            file (str): The name of the output CSV file.

        Side Effects:
            - Saves a new CSV file containing the original data along with the computed embeddings to the specified directory.
        """
        # Load the CSV file
        df = pd.read_csv(self.path)

        if column not in df.columns:
            raise ValueError(f"Column '{column}' not found in CSV")

        # Generate embeddings in a new column 'embeddings', for the specified column using the `get_embedding` method
        df["embeddings"] = df[column].apply(
            lambda x: json.dumps(self.get_embedding(str(x)).tolist())
        )

        os.makedirs(directory, exist_ok=True)

        # Save the DataFrame with the embeddings to a new CSV file in the specified directory
        output_path = os.path.join(directory, file)
        df.to_csv(output_path, index=False)

        print(f"✅ Embeddings saved to {output_path}")
