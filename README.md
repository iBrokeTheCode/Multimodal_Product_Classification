---
title: Multimodal Product Classification
emoji: 📈
colorFrom: purple
colorTo: yellow
sdk: gradio
sdk_version: 5.44.0
app_file: app.py
pinned: false
license: mit
short_description: Product classification using image and text
---

# 🛍️Multimodal Product Classification with Gradio

## Table of Contents

1.  [Project Description](#1-project-description)
2.  [Methodology & Key Features](#2-methodology--key-features)
3.  [Technology Stack](#3-technology-stack)
4.  [Model Details](#4-model-details)

## 1. Project Description

This project implements a **multimodal product classification system** for Best Buy products. The core objective is to categorize products using both their text descriptions and images. The system was trained on a dataset of **almost 50,000** items.

The entire system is deployed as a lightweight, web application using **Gradio**. The app allows users to:

- Use both text and an image for the most accurate prediction.
- Run predictions using only text or only an image to understand the contribution of each data modality.

This project showcases the power of combining different data types to build a more robust and intelligent classification system.

> [!IMPORTANT]
>
> - Check out the deployed app here: 👉️ [Multimodal Product Classification App](https://huggingface.co/spaces/iBrokeTheCode/Multimodal_Product_Classification) 👈️
> - Check out the Jupyter Notebook for a detailed walkthrough of the project here: 👉️ [Jupyter Notebook](https://huggingface.co/spaces/iBrokeTheCode/Multimodal_Product_Classification/blob/main/notebook_guide.ipynb) 👈️

<br>

![App](./assets/app-demo.png)

## 2. Methodology & Key Features

- **Core Task:** Multimodal Product Classification on a Best Buy dataset.

- **Pipeline:**

  - **Data:** A dataset of \~50,000 products, each with a text description and an image.
  - **Feature Extraction:** Pre-trained models are used to convert raw text and image data into high-dimensional embedding vectors.
  - **Classification:** A custom-trained **Multilayer Perceptron (MLP)** model performs the final classification based on the embeddings.

- **Key Features:**

  - **Multimodal:** Combines text and image data for a more accurate prediction.
  - **Single-Service Deployment:** The entire application runs as a single, deployable Gradio app.
  - **Flexible Inputs:** The app supports multimodal, text-only, and image-only prediction modes.

## 3. Technology Stack

This project was built using the following technologies:

**Deployment & Hosting:**

- [Gradio](https://gradio.app/) – interactive web app frontend.
- [Hugging Face Spaces](https://huggingface.co/docs/hub/spaces) – for cost-effective deployment.

**Modeling & Training:**

- [TensorFlow / Keras](https://www.tensorflow.org/) – used to train the final MLP classification model.
- [Sentence-Transformers](https://www.sbert.net/) – for generating text embeddings.
- [Hugging Face Transformers](https://huggingface.co/docs/transformers/index) – for the image feature extractor (`TFConvNextV2Model`).

**Development Tools:**

- [Ruff](https://github.com/charliermarsh/ruff) – Python linter and formatter.
- [uv](https://github.com/astral-sh/uv) – fast Python package installer and resolver.

## 4. Model Details

The final classification is performed by a custom-trained **Multilayer Perceptron (MLP)** model that takes the extracted embeddings as input.

- **Text Embedding Model:** `SentenceTransformer` (`all-MiniLM-L6-v2`)
- **Image Embedding Model:** `TFConvNextV2Model` (`convnextv2-tiny-22k-224`)
- **Classifier:** A custom MLP model trained on top of the embeddings.
- **Classes:** The model classifies products into a set of specific Best Buy product categories.

| Model               | Modality     | Accuracy | Macro Avg F1-Score | Weighted Avg F1-Score |
| :------------------ | :----------- | :------- | :----------------- | :-------------------- |
| Random Forest       | Text         | 0.90     | 0.83               | 0.90                  |
| Logistic Regression | Text         | 0.90     | 0.84               | 0.90                  |
| Random Forest       | Image        | 0.80     | 0.70               | 0.79                  |
| Random Forest       | Combined     | 0.89     | 0.79               | 0.89                  |
| Logistic Regression | Combined     | 0.89     | 0.83               | 0.89                  |
| **MLP**             | **Image**    | **0.84** | **0.77**           | **0.84**              |
| **MLP**             | **Text**     | **0.92** | **0.87**           | **0.92**              |
| **MLP**             | **Combined** | **0.92** | **0.85**           | **0.92**              |

> [!TIP]
>
> Based on the evaluation on the test set, the Multimodal MLP model achieved an excellent **92% accuracy** and a **92% weighted F1-score**, confirming its superior performance by leveraging both text and image data.
