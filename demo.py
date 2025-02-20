# demo.py
# streamlit run demo.py
import streamlit as st
import numpy as np
import torch
from PIL import Image
from torchvision import models, transforms
import joblib  # For loading saved models
import os
import pandas as pd
import random

# Load pre-trained model and clusterer
@st.cache_resource
def load_model():
    model = models.resnet50(weights=models.ResNet50_Weights.IMAGENET1K_V1)
    model = torch.nn.Sequential(*(list(model.children())[:-1]))  # Remove last layer
    model.eval()
    return model

@st.cache_resource
def load_clusterer():
    return joblib.load("kmeans_model.pkl")  # Replace with your saved model

@st.cache_data
def load_cluster_mapping():
    return pd.read_csv("cluster_assignments.csv").set_index("filename")

def get_cluster_from_filename(filename):
    return cluster_mapping.loc[filename, "cluster"]

# Load cluster mapping at start
cluster_mapping = load_cluster_mapping()

# Preprocessing pipeline
preprocess = transforms.Compose([
    transforms.Resize(256),
    transforms.CenterCrop(224),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
])

# Streamlit UI
st.title("Art Style Explorer 🎨")
uploaded_file = st.file_uploader("Upload an artwork:", type=["jpg", "png", "jpeg"])

if uploaded_file is not None:
    # Process image
    image = Image.open(uploaded_file).convert("RGB")
    st.image(image, caption="Uploaded Image", use_container_width=True)
    st.write(f"### Similar Artworks in Cluster")
     # Add number input widget
    num_images = st.slider(
        "Number of similar artworks to show:",
        min_value=1,
        max_value=100,
        value=5,
        help="Choose how many similar artworks from the cluster you want to see"
    )

    # Extract embedding
    with torch.no_grad():
        img_tensor = preprocess(image).unsqueeze(0)
        embedding = load_model()(img_tensor).squeeze().numpy()
    
    # Predict cluster
    clusterer = load_clusterer()
    cluster = clusterer.predict(embedding.reshape(1, -1))[0]
    
# Get all images in the predicted cluster
    data_dir = "data_sample/preprocessed_images"  # <-- ADD THIS LINE
    
    cluster_images = [f for f in os.listdir(data_dir) 
                    if f.endswith((".jpg", ".png")) and get_cluster_from_filename(f) == cluster] 

    # Handle case where cluster has fewer images than requested
    display_count = min(num_images, len(cluster_images))
    if display_count < num_images:
        st.warning(f"Only {len(cluster_images)} images available in this cluster")
    
    sample_images = np.random.choice(cluster_images, display_count, replace=False)

    # Display images in a grid (5 per row)
    columns_per_row = st.slider(
        "Images per row:",
        min_value=1,
        max_value=8,
        value=5,
        help="Control how many artworks are shown per row" 
    )
    num_images = len(sample_images)
    
    for i in range(0, num_images, columns_per_row):
        # Create a new row of columns
        cols = st.columns(columns_per_row)
        
        # Fill the row with images
        for j in range(columns_per_row):
            idx = i + j
            if idx < num_images:
                img_path = os.path.join(data_dir, sample_images[idx])
                cols[j].image(img_path, use_container_width=True)
            else:
                cols[j].empty()  # Leave empty if no more images
                