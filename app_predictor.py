from json import load
from typing import Any, Dict, Optional

from numpy import array, expand_dims, float32, ndarray, transpose, zeros
from PIL import Image
from sentence_transformers import SentenceTransformer
from tensorflow import constant
from tensorflow.keras.models import load_model
from transformers import TFConvNextV2Model

# 📌 GLOBAL VARIABLES (categories)
CATEGORY_MAP: Dict[str, str] = {}
CLASS_LABELS = []


def build_category_map(categories_json_path: str):
    """
    Builds a flat dictionary and a list of category labels by traversing the hierarchical categories.json file.
    """
    global CATEGORY_MAP, CLASS_LABELS

    try:
        with open(categories_json_path, "r") as f:
            categories_data = load(f)
    except FileNotFoundError:
        print(
            f"❌ Error: {categories_json_path} not found. Using hardcoded labels as fallback."
        )
        return

    category_map = {}

    model_trained_ids = [
        "abcat0100000",
        "abcat0200000",
        "abcat0207000",
        "abcat0300000",
        "abcat0400000",
        "abcat0500000",
        "abcat0700000",
        "abcat0800000",
        "abcat0900000",
        "cat09000",
        "pcmcat128500050004",
        "pcmcat139900050002",
        "pcmcat242800050021",
        "pcmcat252700050006",
        "pcmcat312300050015",
        "pcmcat332000050000",
    ]

    def traverse_categories(categories):
        for category in categories:
            category_map[category["id"]] = category["name"]
            if "subCategories" in category and category["subCategories"]:
                traverse_categories(category["subCategories"])
            if "path" in category and category["path"]:
                for path_item in category["path"]:
                    category_map[path_item["id"]] = path_item["name"]

    traverse_categories(categories_data)

    CATEGORY_MAP = category_map
    CLASS_LABELS = model_trained_ids


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

# Generate category map and class labels list
build_category_map("./data/raw/categories.json")


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
    prediction_dict_raw = dict(zip(CLASS_LABELS, predictions[0]))

    #  Map the raw IDs to human-readable names
    prediction_dict_mapped = {}
    for class_id, probability in prediction_dict_raw.items():
        # Get the human-readable name, defaulting to the raw ID if not found
        category_name = CATEGORY_MAP.get(class_id, class_id)
        prediction_dict_mapped[category_name] = probability

    # Sort the dictionary by probability in descending order for a cleaner display
    sorted_predictions = dict(
        sorted(prediction_dict_mapped.items(), key=lambda item: item[1], reverse=True)
    )

    return sorted_predictions
