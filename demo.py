# demo.py
# streamlit run demo.py
import streamlit as st
import numpy as np
import torch
from PIL import Image
from torchvision import models, transforms
import joblib
import os
import pandas as pd
import random

# Cache the loaded ResNet50 model to avoid reloading it on every run
@st.cache_resource
def load_model():
    model = models.resnet50(weights=models.ResNet50_Weights.IMAGENET1K_V1)
    model = torch.nn.Sequential(*(list(model.children())[:-1]))
    model.eval()
    return model

# Cache the loaded k-means clusterer to avoid reloading it
@st.cache_resource
def load_clusterer():
    return joblib.load("kmeans_model.pkl")

# Cache the loaded cluster assignments mapping to avoid reloading it
@st.cache_data
def load_cluster_mapping():
    return pd.read_csv("cluster_assignments.csv").set_index("filename")

# Function to get the cluster assignment from a filename
def get_cluster_from_filename(filename):
    return cluster_mapping.loc[filename, "cluster"]

# Load the cluster assignments mapping
cluster_mapping = load_cluster_mapping()

# Define the image preprocessing steps
preprocess = transforms.Compose([
    transforms.Resize(256), # Resize the image to 256x256
    transforms.CenterCrop(224), # Crop the center 224x224 pixels
    transforms.ToTensor(), # Convert the image to a PyTorch tensor
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]), # Normalize the tensor
])

st.title("Art Style Explorer 🎨")

# Initialize a session state variable to track if an image is uploaded
if "image_uploaded" not in st.session_state:
    st.session_state.image_uploaded = False

if st.session_state.image_uploaded == False:  # Show the message only if no image is uploaded
    st.write("Upload a piece of art and explore some possible artistic siblings! Find hidden connections between your uploaded artwork and others across styles, colors, themes, and eras!")

# File uploader widget
uploaded_file = st.file_uploader("Upload an artwork:", type=["jpg", "png", "jpeg"])

# If an image is uploaded
if uploaded_file is not None:
    st.session_state.image_uploaded = True
    image = Image.open(uploaded_file).convert("RGB")
    st.image(image, caption="Uploaded Image", use_container_width=True)
    st.write(f"### Similar Artworks")

    # Sidebar widgets
    num_images = st.sidebar.slider(
        "Number of similar artworks to show:",
        min_value=1,
        max_value=100,
        value=5,
        help="Choose how many similar artworks from the cluster you want to see"
    )
    # Extract the feature embedding from the uploaded image using the loaded model
    with torch.no_grad(): # Disable gradient calculation for inference
        img_tensor = preprocess(image).unsqueeze(0) # Preprocess the image and add a batch dimension
        embedding = load_model()(img_tensor).squeeze().numpy() # Get the feature embedding and convert it to a NumPy array
    
    # Load the clusterer and predict the cluster for the uploaded image
    clusterer = load_clusterer()
    cluster = clusterer.predict(embedding.reshape(1, -1))[0]
    
    # Get the list of images belonging to the same cluster
    data_dir = "data_sample/preprocessed_images"
    cluster_images = [f for f in os.listdir(data_dir) 
                    if f.endswith((".jpg", ".png")) and get_cluster_from_filename(f) == cluster]

    display_count = min(num_images, len(cluster_images))
    if display_count < num_images:
        st.warning(f"Only {len(cluster_images)} images available in this cluster")
    
    # Randomly sample images from the cluster
    sample_images = np.random.choice(cluster_images, display_count, replace=False)

    # Second sidebar widget
    columns_per_row = st.sidebar.slider(
        "Images per row:",
        min_value=1,
        max_value=8,
        value=5,
        help="Control how many artworks are shown per row" 
    )

    num_images = len(sample_images)
    
    for i in range(0, num_images, columns_per_row):
        cols = st.columns(columns_per_row)
        for j in range(columns_per_row):
            idx = i + j
            if idx < num_images:
                img_path = os.path.join(data_dir, sample_images[idx])
                cols[j].image(img_path, use_container_width=True)
            else:
                cols[j].empty()

# Footer
st.markdown("""
<style>
.footer {
    position: fixed;
    left: 0;
    bottom: 0;
    width: 100%;
    background-color: #EEF6F2;
    color: black;
    text-align: center;
    padding: .1px;
}
</style>
<div class="footer">
    <p>
            &copy; 2025 Rohan Xavier Gupta |
            <a href="https://github.com/rohanxavier33" target="_blank">GitHub</a> 
            </p>  </div>
""", unsafe_allow_html=True)