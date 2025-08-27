import gradio as gr


# 📌 FUNCTIONS
def predict(mode, text, image_path):
    # ... your existing predict function ...
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
    # ... your existing update_inputs function ...
    if mode == "Multimodal":
        return gr.Textbox(visible=True), gr.Image(visible=True)
    elif mode == "Text Only":
        return gr.Textbox(visible=True), gr.Image(visible=False)
    elif mode == "Image Only":
        return gr.Textbox(visible=False), gr.Image(visible=True)
    else:
        return gr.Textbox(visible=True), gr.Image(visible=True)


# 📌 CUSTOM CSS FOR FIXED FOOTER
css_code = """
/* Target the footer container by its ID and apply fixed positioning */
#footer-container {
    position: fixed;
    bottom: 0;
    left: 0;
    right: 0;
    z-index: 1000; /* Ensure it stays on top of other content */
    background-color: var(--background-fill-primary); /* Use a Gradio theme variable */
    padding: var(--spacing-md);
    border-top: 1px solid var(--border-color-primary);
}

/* Add padding to the body to prevent content from being hidden by the footer */
.gradio-container {
    padding-bottom: 70px !important;
}
"""

# 📌 USER INTERFACE
with gr.Blocks(
    title="Multimodal Product Classification",
    theme=gr.themes.Ocean(),
    css=css_code,
) as demo:
    # 📌 TABS
    with gr.Tabs():
        # ... your existing tabs ...
        # 📌 APP TAB
        with gr.TabItem("App"):
            gr.Markdown("# 🛍️ Multimodal Product Classification")

            with gr.Row(equal_height=True):
                with gr.Column(scale=1):
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
                            label="Product Image", type="filepath", visible=True
                        )

                        classify_button = gr.Button(
                            "✨ Classify Product", variant="primary"
                        )

                with gr.Column(scale=2):
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
            gr.Markdown("""...""")

        # 📌 MODEL TAB
        with gr.TabItem("Model"):
            gr.Markdown("""...""")

    # 📌 FOOTER
    with gr.Row(elem_id="footer-container"):
        gr.HTML("""
<div style="text-align: center;">
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
