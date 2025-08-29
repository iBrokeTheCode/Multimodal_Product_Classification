import gradio as gr

from app_predictor import predict

# 📌 CUSTOM CSS
css_code = """
#footer-container {
    position: fixed;
    bottom: 0;
    left: 0;
    right: 0;
    z-index: 1000;
    background-color: var(--background-fill-primary);
    padding: var(--spacing-md);
    border-top: 1px solid var(--border-color-primary);
    text-align: center;
}

.gradio-container {
    padding-bottom: 70px !important;
}

.center {
    text-align: center;
}
"""


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
    css=css_code,
) as demo:
    with gr.Tabs():
        # 📌 APP TAB
        with gr.TabItem("🚀 App"):
            with gr.Row(elem_classes="center"):
                gr.HTML("""
                    <div>
                        <h1>🛍️ Multimodal Product Classification</h1>
                    </div>
                    <br><br>
                    """)

            with gr.Row(equal_height=True):
                # 📌 CLASSIFICATION INPUTS COLUMN
                with gr.Column():
                    with gr.Column():
                        gr.Markdown("## 📝 Classification Inputs")

                        mode_radio = gr.Radio(
                            choices=["Multimodal", "Image Only", "Text Only"],
                            value="Multimodal",
                            label="Choose Classification Mode:",
                        )

                        text_input = gr.Textbox(
                            label="Product Description:",
                            placeholder="e.g., Apple iPhone 15 Pro Max 256GB",
                            lines=1,
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

                # 📌 RESULTS COLUMN
                with gr.Column():
                    with gr.Column():
                        gr.Markdown("## 📊 Results")

                        gr.Markdown(
                            """**💡 How to use this app**

                            This app classifies a product based on its description and image.
                            - **Multimodal:** The most accurate mode, using both the image and a detailed description for prediction.
                            - **Image Only:** Highly effective for visual products, relying solely on the product image.
                            - **Text Only:** Less precise, this mode requires a very descriptive and specific product description to achieve good results.
                            """
                        )

                        gr.HTML("<hr>")

                        output_label = gr.Label(
                            label="Predict category", num_top_classes=5
                        )

            # 📌 EXAMPLES SECTION
            gr.Examples(
                examples=[
                    [
                        "Multimodal",
                        'Laptop Asus - 15.6" / CPU I9 / 2Tb SSD / 32Gb RAM / RTX 2080',
                        "./assets/sample2.jpg",
                    ],
                    [
                        "Multimodal",
                        "Red Electric Guitar – Stratocaster Style, 6-String, White Pickguard, Solid-Body, Ideal for Rock & Roll",
                        "./assets/sample1.jpg",
                    ],
                    [
                        "Multimodal",
                        "Portable Wireless Speaker / JBL / Black / High Quality Sound",
                        "./assets/sample3.jpg",
                    ],
                ],
                label="Select an example to pre-fill the inputs, then click the 'Classify Product' button.",
                inputs=[mode_radio, text_input, image_input],
                # outputs=output_label,
                # fn=predict,
                # cache_examples=True,
            )

        # 📌 ABOUT TAB
        with gr.TabItem("ℹ️ About"):
            gr.Markdown("""
## Project Overview
                        
- This project is a multimodal product classification system for Best Buy products. 
- The core objective is to categorize products using both their text descriptions and images. 
- The system was trained on a dataset of **almost 50,000** products and their corresponding images to generate embeddings and train the classification models.

<br>

## Technical Workflow
                        
1.  **Data Preprocessing:** Product descriptions and images are extracted from the dataset, and a `categories.json` file is used to map product IDs to human-readable category names.
2.  **Embedding Generation:**
    - **Text:** A pre-trained `SentenceTransformer` model (`all-MiniLM-L6-v2`) is used to generate dense vector embeddings from the product descriptions.
    - **Image:** A pre-trained computer vision model from the Hugging Face `transformers` library (`TFConvNextV2Model`) is used to extract image features.
3.  **Model Training:** The generated text and image embeddings are then used to train a multi-layer perceptron (MLP) model for classification. Separate models were trained for text-only, image-only, and multimodal (combined embeddings) classification.
4.  **Deployment:** The trained models are deployed via a Gradio web interface, allowing for live prediction on new product data.

<br>
                                   
> **💡 Want to explore the process in detail?**   
> See the full 👉 [Jupyter notebook](https://huggingface.co/spaces/iBrokeTheCode/Multimodal_Product_Classification/blob/main/notebook_guide.ipynb) 👈️ for an end-to-end walkthrough, including Exploratory Data Analysis, embeddings generation, models training, evaluation, and model selection.
""")

        # 📌 MODEL TAB
        with gr.TabItem("🎯 Model"):
            gr.Markdown("""
## Model Details
The final classification is performed by a Multi-layer Perceptron (MLP) trained on the embeddings. This architecture allows the model to learn the relationships between the textual and visual features.

<br>
                        
## Performance Summary
                        
The following table summarizes the performance of all models trained in this project.
                        
<br>

| Model               | Modality     | Accuracy | Macro Avg F1-Score | Weighted Avg F1-Score |
| :------------------ | :----------- | :------- | :----------------- | :-------------------- |
| Random Forest       | Text         | 0.90     | 0.83               | 0.90                  |
| Logistic Regression | Text         | 0.90     | 0.84               | 0.90                  |
| Random Forest       | Image        | 0.80     | 0.70               | 0.79                  |
| Random Forest       | Combined     | 0.89     | 0.79               | 0.89                  |
| Logistic Regression | Combined     | 0.89     | 0.83               | 0.89                  |
| **MLP** | **Image** | **0.84** | **0.77** | **0.84** |
| **MLP** | **Text** | **0.92** | **0.87** | **0.92** |
| **MLP** | **Combined** | **0.92** | **0.85** | **0.92** |

<br>
                        
## Conclusion
                        
- Based on the overall results, the MLP models consistently outperformed their classical machine learning counterparts, demonstrating their ability to learn intricate, non-linear relationships within the data.
- Both the Text MLP and Combined MLP models achieved the highest accuracy and weighted F1-score, confirming their superior ability to classify the products.
- This modular approach demonstrates the ability to handle various data modalities and evaluate the contribution of each to the final prediction.
""")

    # 📌 FOOTER
    # gr.HTML("<hr>")
    with gr.Row(elem_id="footer-container"):
        gr.HTML("""
<div>
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
