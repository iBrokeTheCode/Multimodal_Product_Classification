import os
from typing import Any, Dict, Optional

from numpy import array, expand_dims, float32, ndarray, transpose, zeros
from PIL import Image
from sentence_transformers import SentenceTransformer
from tensorflow import constant
from tensorflow.keras.models import load_model
from transformers import TFConvNextV2Model

# TODO: Hardcoded class labels for the output, as discussed
CLASS_LABELS = [
    "abcat0100000",
    "abcat0200000",
    "abcat0300000",
    "abcat0400000",
    "abcat0500000",
]

# 📌 LOAD MODELS
print("💬 Loading embedding models...")
try:
    text_embedding_model = SentenceTransformer("all-MiniLM-L6-v2")
    image_feature_extractor = TFConvNextV2Model.from_pretrained(
        "facebook/convnextv2-tiny-22k-224"
    )
    print("✅ Embedding models loaded successfully!")
except Exception as e:
    print(f"❌ Error loading embedding models: {e}")
    text_embedding_model, image_feature_extractor = None, None

# Load the final classification models (MLP heads)
print("💬 Loading classification models...")
try:
    text_model = load_model("./models/text_model")
    image_model = load_model("./models/image_model")
    multimodal_model = load_model("./models/multimodal_model")
    print("✅ Classification models loaded successfully!")
except Exception as e:
    print(f"❌ Error loading classification models: {e}")
    text_model, image_model, multimodal_model = None, None, None


# 📌 EMBEDDING FUNCTIONS
def get_text_embeddings(text: Optional[str]) -> ndarray:
    """
    Generates a dense embedding vector from a text string.

    Args:
        text (Optional[str]): The input text. Can be None or an empty string.

    Returns:
        np.ndarray: A NumPy array of shape (1, 384) representing the text
                    embedding. Returns a zero vector if the input is empty.
    """
    # Handle cases where no text is provided
    if not text or not text.strip():
        # Returns a zero vector with the correct dimension (384)
        return zeros(
            (1, text_embedding_model.get_sentence_embedding_dimension()), dtype=float32
        )

    # Use the pre-trained SentenceTransformer to encode the text
    embeddings = text_embedding_model.encode([text])
    return array(embeddings, dtype=float32)


def get_image_embeddings(image_path: Optional[str]) -> ndarray:
    """
    Preprocesses an image and generates an embedding vector using a pre-trained model.

    Args:
        image_path (Optional[str]): The file path to the image.

    Returns:
        np.ndarray: A NumPy array of shape (1, 768) representing the image
                    embedding. Returns a zero vector if no image is provided.
    """
    # Handle cases where no image is provided
    if image_path is None:
        return zeros((1, 768), dtype=float32)

    # Load the image and convert to RGB format
    image = Image.open(image_path).convert("RGB")

    # Resize the image to the model's expected input size (224x224)
    image = image.resize((224, 224), Image.Resampling.LANCZOS)

    # Convert to NumPy array and add a batch dimension (1, H, W, C)
    image_array = array(image, dtype=float32)
    image_array = expand_dims(image_array, axis=0)

    # Transpose the array to match the model's channel order (1, C, H, W)
    image_array = transpose(image_array, (0, 3, 1, 2))

    # Normalize the pixel values (not strictly necessary for this model, but good practice)
    image_array = image_array / 255.0

    # Pass the preprocessed image through the feature extractor model
    embeddings_output = image_feature_extractor(constant(image_array))

    # Extract the final embedding from the pooler_output
    embeddings = embeddings_output.pooler_output

    return embeddings.numpy()


# 📌 MAIN PREDICTION FUNCTION
def predict(
    mode: str, text: Optional[str], image_path: Optional[str]
) -> Dict[str, Any]:
    """
    Predicts the category of a product based on the selected mode.

    Args:
        mode (str): The prediction mode ("Multimodal", "Text Only", "Image Only").
        text (Optional[str]): The product description text.
        image_path (Optional[str]): The file path to the product image.

    Returns:
        Dict[str, Any]: A dictionary of class labels and their corresponding
                        prediction probabilities. Returns an empty dictionary
                        if the mode is invalid.
    """
    # Generate embeddings for both inputs
    text_emb = get_text_embeddings(text)
    image_emb = get_image_embeddings(image_path)

    # Get predictions based on the selected mode
    if mode == "Multimodal":
        predictions = multimodal_model.predict([text_emb, image_emb])
    elif mode == "Text Only":
        predictions = text_model.predict(text_emb)
    elif mode == "Image Only":
        predictions = image_model.predict(image_emb)
    else:
        # Return an empty dictionary if the mode is not recognized
        return {}

    # Format the output into a dictionary with labels and probabilities
    # The model's output is a 2D array, so we take the first row (index 0)
    prediction_dict = dict(zip(CLASS_LABELS, predictions[0]))

    return prediction_dict


# 📌 SANITY CHECKS
if __name__ == "__main__":
    print("\n--- Running sanity checks for predictor.py ---")

    # Check text embedding function
    print("\n--- Testing get_text_embeddings ---")
    sample_text = (
        "A sleek silver laptop with a large screen and high-resolution display."
    )
    text_emb = get_text_embeddings(sample_text)
    print(f"Embedding shape for a normal string: {text_emb.shape}")
    empty_text_emb = get_text_embeddings("")
    print(f"Embedding shape for an empty string: {empty_text_emb.shape}")
    spaces_text_emb = get_text_embeddings("   ")
    print(f"Embedding shape for a string with spaces: {spaces_text_emb.shape}")

    # Check image embedding function
    print("\n--- Testing get_image_embeddings ---")
    test_image_path = "test.jpeg"  # Ensure this file exists for the test to pass
    if os.path.exists(test_image_path):
        image_emb = get_image_embeddings(test_image_path)
        print(f"✅ Embedding shape for an image file: {image_emb.shape}")
    else:
        print(
            f"⚠️ Warning: Test image file not found at {test_image_path}. Skipping image embedding test."
        )

    empty_image_emb = get_image_embeddings(None)
    print(f"Embedding shape for a None input: {empty_image_emb.shape}")
    print("--- Sanity checks complete. ---")
