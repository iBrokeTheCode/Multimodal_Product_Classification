import gradio as gr


# Placeholder for the prediction function
def predict(mode, text, image_path):
    """
    This is a placeholder for the final prediction logic.
    It will return a hardcoded dictionary to demonstrate the output format.
    """
    if mode == "Multimodal":
        result_text = "Result for Multimodal input: a category from a real model. Confidence: 0.95"
    elif mode == "Text Only":
        result_text = (
            "Result for Text Only input: a category from a real model. Confidence: 0.92"
        )
    elif mode == "Image Only":
        result_text = "Result for Image Only input: a category from a real model. Confidence: 0.88"
    else:
        result_text = "Please select a classification mode."

    return {
        "label": result_text,
        "confidences": {
            "abcat0100000": 0.05,
            "abcat0200000": 0.10,
            "abcat0300000": 0.20,
            "abcat0400000": 0.45,
            "abcat0500000": 0.20,
        },
    }


# Function to update input visibility based on mode selection
def update_inputs(mode):
    if mode == "Multimodal":
        return gr.Textbox(visible=True), gr.Image(visible=True)
    elif mode == "Text Only":
        return gr.Textbox(visible=True), gr.Image(visible=False)
    elif mode == "Image Only":
        return gr.Textbox(visible=False), gr.Image(visible=True)
    else:  # Default case
        return gr.Textbox(visible=True), gr.Image(visible=True)


# Gradio Interface using Blocks
with gr.Blocks(title="Multimodal Product Classification") as demo:
    with gr.Tabs():
        with gr.TabItem("App"):
            gr.Markdown("# Multimodal Product Classifier")
            gr.Markdown("Classify products using either text, images, or both.")

            with gr.Row():
                with gr.Column(scale=1):
                    with gr.Column(variant="panel"):
                        gr.Markdown("### ⚙️ Classification Inputs")

                        mode_radio = gr.Radio(
                            choices=["Multimodal", "Text Only", "Image Only"],
                            value="Multimodal",
                            label="Choose Classification Mode",
                        )

                        text_input = gr.Textbox(
                            label="Product Description",
                            placeholder="e.g., Apple iPhone 15 Pro Max 256GB",
                        )
                        image_input = gr.Image(
                            label="Product Image", type="filepath", visible=True
                        )

                    classify_btn = gr.Button("🚀 Classify Product", variant="primary")

                with gr.Column(scale=1):
                    with gr.Column(variant="panel"):
                        gr.Markdown("### 📊 Classification Results")

                        output_label = gr.Label(
                            label="Predicted Category", num_top_classes=5
                        )

                    with gr.Accordion("How to use this demo", open=False):
                        gr.Markdown(
                            """
                            This demo classifies a product based on its description and image.
                            - **Multimodal:** Uses both text and image for the most accurate prediction.
                            - **Text Only:** Uses only the product description.
                            - **Image Only:** Uses only the product image.
                            """
                        )

        with gr.TabItem("About"):
            gr.Markdown(
                """
                ### About the Project
                This project demonstrates a multimodal classification system trained on data from Best Buy. It uses a Multilayer Perceptron (MLP) model trained on pre-generated embeddings from a Text-based model (MiniLM-L6) and an Image-based model (ConvNeXtV2).
                """
            )

        with gr.TabItem("Architecture"):
            gr.Markdown(
                """
                ### Model Architecture
                This section would contain details about the MLP architecture, the embedding models used, and a diagram explaining the data flow.
                """
            )

    # Event listeners for conditional rendering
    mode_radio.change(
        fn=update_inputs, inputs=mode_radio, outputs=[text_input, image_input]
    )

    # Event listener for the classify button
    classify_btn.click(
        fn=predict, inputs=[mode_radio, text_input, image_input], outputs=output_label
    )

demo.launch()
