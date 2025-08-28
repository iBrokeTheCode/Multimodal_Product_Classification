from tensorflow.keras.models import load_model

# TODO: Review Code
# Load the models once at the start of the script
print("💬 Loading models...")
try:
    text_model = load_model("./models/text_model")
    image_model = load_model("./models/image_model")
    multimodal_model = load_model("./models/multimodal_model")
    print("✅ Models loaded successfully!")
except Exception as e:
    print(f"❌ Error loading models: {e}")
    text_model = None
    image_model = None
    multimodal_model = None

# A placeholder for your class labels
CLASS_LABELS = [
    "abcat0100000",
    "abcat0200000",
    "abcat0207000",
]  # Add your actual labels


# 📌 FUNCTIONS
def predict(mode, text, image_path):
    """
    This placeholder function now returns a dictionary
    in the format expected by the gr.Label component.
    """
    multimodal_output = {
        "abcat0100000": 0.05,
        "abcat0200000": 0.10,
        "abcat0300000": 0.20,
        "abcat0400000": 0.45,
        "abcat0500000": 0.20,
    }
    text_only_output = {
        "abcat0100000": 0.08,
        "abcat0200000": 0.15,
        "abcat0300000": 0.25,
        "abcat0400000": 0.35,
        "abcat0500000": 0.17,
    }
    image_only_output = {
        "abcat0100000": 0.10,
        "abcat0200000": 0.20,
        "abcat0300000": 0.30,
        "abcat0400000": 0.25,
        "abcat0500000": 0.15,
    }

    if mode == "Multimodal":
        return multimodal_output
    elif mode == "Text Only":
        return text_only_output
    elif mode == "Image Only":
        return image_only_output
    else:
        return {}
