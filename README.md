# 🌿 Plant Disease Detection using CNN  
_A Deep Learning based Image Classification System deployed on Streamlit_

---

## 📖 Overview

This project aims to **detect and classify plant leaf diseases** using **Convolutional Neural Networks (CNN)**.  
By simply uploading an image of a plant leaf, the model predicts the disease category and provides **possible solutions or preventive measures**.

The project leverages **Deep Learning with TensorFlow**, and is deployed on a **Streamlit web app** for easy accessibility.

---

## 🧠 Key Features

✅ Built using a **Custom CNN architecture** with TensorFlow  
✅ Supports **38 different plant disease classes**  
✅ Trained on the **PlantVillage Dataset** (from Kaggle)  
✅ **Automated image preprocessing and classification**  
✅ Provides **disease name with confidence score**  
✅ **Deployed on Streamlit** for easy web-based interaction  
✅ Offers **treatment suggestions** for identified diseases  

---

## 📦 Dataset Details

- **Dataset Source:** [PlantVillage Dataset – Abdallah on Kaggle](https://www.kaggle.com/datasets/abdallah/plantvillage-dataset)  
- **Size:** 2.04 GB  
- **Total Images:** 1,62,916  
- **Formats:** Color, Grayscale, and Segmented  
- **Number of Classes:** 38 (Healthy + Diseased categories)

The dataset includes leaves from plants such as tomato, potato, apple, maize, and more.

---

## ⚙️ Data Preprocessing

To ensure consistency and better performance, preprocessing included:

- Resizing all images to **(256, 256, 3)** for uniformity  
- Scaling each image to **224×224 pixels**  
- **Batching** images in groups of **32**  
- Normalization of pixel values  
- Data generators for training and validation  

---

## 🧩 CNN Model Architecture

The model is built using **TensorFlow’s Sequential API** as it allows stacking layers sequentially.

| Layer | Type | Parameters | Activation | Additional |
|--------|------|-------------|-------------|-------------|
| 1 | Conv2D | 32 filters (3×3) | ReLU | MaxPooling (2×2) |
| 2 | Conv2D | 64 filters (3×3) | ReLU | MaxPooling (2×2) |
| 3 | Flatten | - | - | Converts 2D → 1D |
| 4 | Dense | 256 neurons | ReLU | Fully connected layer |
| 5 | Output | 38 classes | Softmax | Multiclass classification |

**Optimizer:** Adam  
**Loss Function:** Categorical Cross-Entropy  

---

## 🧮 Model Training

- **Validation Method:** Stratified K-Fold (80% Train / 20% Validation)  
- **Epochs:** 5  
- **Batch Size:** 32  
- **Steps per Epoch:** `Total Training Samples / Batch Size`  
- **Metrics Stored:** Accuracy & Loss (using History variable)  

### 📊 Performance Metrics:
- High training and validation accuracy  
- Accuracy and loss plotted using Matplotlib for both training and testing phases  

---

## 🧪 Model Evaluation

Although similar plant leaves (e.g., apple vs. tomato) can cause confusion, the model achieved **strong classification performance**.  

**Evaluation Methods:**
- Accuracy  
- Loss value visualization  
- (Optional) Confusion Matrix for deeper analysis  

---

## 🔮 Building the Predictive System

- Used the **Pillow** library to load and preprocess uploaded leaf images.  
- The system:
  1. Accepts an image input.  
  2. Preprocesses it (resize + normalization).  
  3. Predicts the disease using the trained CNN.  
  4. Displays the **disease name**, **confidence score**, and **possible treatment**.

---

## 🌐 Deployment with Streamlit

The project is deployed on **Streamlit**, creating an **interactive web app** where users can:

- Upload a plant leaf image  
- Get instant prediction results  
- View the model’s confidence percentage  
- Read disease description and treatment suggestions  

**Command to run locally:**
```bash
streamlit run app.py

Plant Disease Detection/
│
├── app.py                          # Streamlit application
├── model/
│   ├── cnn_model.h5                # Trained CNN model file
│
├── dataset/
│   ├── color/                      # PlantVillage color images
│   ├── grayscale/
│   ├── segmented/
│
├── static/
│   ├── sample_images/              # Example test images
│
├── requirements.txt                # Dependencies
├── README.md                       # Project documentation
└── utils/
    ├── preprocess.py               # Image preprocessing script
    ├── predict.py                  # Model prediction function
