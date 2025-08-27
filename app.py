import gradio as gr


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


def update_inputs(mode: str):
    if mode == "Multimodal":
        return gr.Textbox(visible=True), gr.Image(visible=True)
    elif mode == "Text Only":
        return gr.Textbox(visible=True), gr.Image(visible=False)
    elif mode == "Image Only":
        return gr.Textbox(visible=False), gr.Image(visible=True)
    else:  # Default case
        return gr.Textbox(visible=True), gr.Image(visible=True)


# 📌 USER INTERFACE
with gr.Blocks(
    title="Multimodal Product Classification",
    theme=gr.themes.Ocean(),
) as demo:
    with gr.Tabs():
        # 📌 APP TAB
        with gr.TabItem("App"):
            gr.Markdown("# 🛍️ Multimodal Product Classification")

            with gr.Row(equal_height=True):
                with gr.Column():
                    with gr.Column():
                        gr.Markdown("## ⚙️ Classification Inputs")

                        mode_radio = gr.Radio(
                            choices=["Multimodal", "Text Only", "Image Only"],
                            value="Multimodal",
                            label="Choose Classification Mode:",
                        )

                        text_input = gr.Textbox(
                            label="Product Description:",
                            placeholder="e.g., Apple iPhone 15 Pro Max 256GB",
                        )

                        image_input = gr.Image(
                            label="Product Image",
                            type="filepath",
                            visible=True,
                            height=300,
                            width="100%",
                        )

                        classify_button = gr.Button(
                            "✨ Classify Product", variant="primary"
                        )

                with gr.Column():
                    with gr.Column():
                        gr.Markdown("## 📊 Results")

                        gr.Markdown(
                            """**💡 How to use this app**

                            This app classifies a product based on its description and image.
                            - **Multimodal:** Uses both text and image for the most accurate prediction.
                            - **Text Only:** Uses only the product description.
                            - **Image Only:** Uses only the product image.
                            """
                        )

                        output_label = gr.Label(
                            label="Predict category", num_top_classes=5
                        )

        # 📌 ABOUT TAB
        with gr.TabItem("About"):
            gr.Markdown("""
## About This Project

- This project is an image classification app powered by a Convolutional Neural Network (CNN).
- Simply upload an image, and the app predicts its category from over 1,000 classes using a pre-trained ResNet50 model.
- Originally developed as a multi-service ML system (FastAPI + Redis + Streamlit), this version has been adapted into a single Streamlit app for lightweight, cost-effective deployment on Hugging Face Spaces.

## Model & Description
- Model: ResNet50 (pre-trained on the ImageNet dataset with 1,000+ categories).
- Pipeline: Images are resized, normalized, and passed to the model.
- Output: The app displays the Top prediction with confidence score.
ResNet50 is widely used in both research and production, making it an excellent showcase of deep learning capabilities and transferable ML skills.
""")

        # 📌 MODEL TAB
        with gr.TabItem("Model"):
            gr.Markdown("""
## Original Architecture

- FastAPI → REST API for image processing
- Redis → Message broker for service communication
- Streamlit → Interactive web UI
- TensorFlow → Deep learning inference engine
- Locust → Load testing & benchmarking
- Docker Compose → Service orchestration

## Simplified Version
                        
- Streamlit only → UI and model combined in a single app
- TensorFlow (ResNet50) → Core prediction engine
- Docker → Containerized for Hugging Face Spaces deployment
This evolution demonstrates the ability to design a scalable microservices system and also adapt it into a lightweight single-service solution for cost-effective demos.
""")

    # 📌 FOOTER
    gr.HTML("<hr>")
    with gr.Row():
        gr.Markdown("""
<div style="text-align: center; margin-bottom: 1.5rem;">
        <b>Connect with me:</b> 💼 <a href="https://www.linkedin.com/in/alex-turpo/" target="_blank">LinkedIn</a> • 
        🐱 <a href="https://github.com/iBrokeTheCode" target="_blank">GitHub</a> • 
        🤗 <a href="https://huggingface.co/iBrokeTheCode" target="_blank">Hugging Face</a>
    </div>
""")

    # 📌 EVENT LISTENERS
    mode_radio.change(
        fn=update_inputs,
        inputs=mode_radio,
        outputs=[text_input, image_input],
    )

    classify_button.click(
        fn=predict, inputs=[mode_radio, text_input, image_input], outputs=output_label
    )


demo.launch()
