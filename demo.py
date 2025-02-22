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
def load_clusterer(model_type):
    """Modified to support multiple models"""
    model_paths = {
        "K-means": "trained_models/kmeans_model.pkl",
        "DBSCAN": "trained_models/dbscan_model.pkl", 
        # "BIRCH": "trained_models/birch_model_compressed.pkl",
    }
    return joblib.load(model_paths[model_type])

# Cache the loaded cluster assignments mapping to avoid reloading it
@st.cache_data
def load_cluster_mapping(model_type):
    """Modified to support multiple models"""
    cluster_files = {
        "K-means": "cluster_assignments/kmeans_cluster_assignments.csv",
        "DBSCAN": "cluster_assignments/dbscan_cluster_assignments.csv",
        # "BIRCH": "cluster_assignments/birch_cluster_assignments.csv",
    }
    return pd.read_csv(cluster_files[model_type]).set_index("filename")


# Function to get the cluster assignment from a filename
def get_cluster_from_filename(filename):
    return cluster_mapping.loc[filename, "cluster"]

# Define the image preprocessing steps
preprocess = transforms.Compose([
    transforms.Resize(256), # Resize the image to 256x256
    transforms.CenterCrop(224), # Crop the center 224x224 pixels
    transforms.ToTensor(), # Convert the image to a PyTorch tensor
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]), # Normalize the tensor
])

st.title("Art Style Explorer 🎨")

# ===== NEW MODEL SELECTION =====
selected_model = st.sidebar.radio(
    "Select Clustering Model:",
    ("K-means", "DBSCAN"), # "BIRCH - Experimental"
    index=0,
    help="Choose which clustering algorithm to use for similarity detection"
)

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
    st.write(f"### Similar Artworks (Using {selected_model})")  # Modified title

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
    clusterer = load_clusterer(selected_model)
    
    # Handle DBSCAN's fit_predict vs others' predict
    if selected_model == "DBSCAN":
        cluster = clusterer.fit_predict(embedding.reshape(1, -1))[0]
    else:
        cluster = clusterer.predict(embedding.reshape(1, -1))[0]
    
    # Get the list of images belonging to the same cluster
    data_dir = "data_sample/preprocessed_images"
    cluster_mapping = load_cluster_mapping(selected_model)

    # Get list of valid image files that exist in cluster_mapping
    valid_images = [f for f in os.listdir(data_dir) 
                if f.endswith((".jpg", ".png")) 
                and f in cluster_mapping.index]
    
    # Handle DBSCAN noise cluster
    if selected_model == "DBSCAN" and cluster == -1:
        st.warning("This artwork is considered noise - showing random artworks")
        cluster_images = os.listdir(data_dir)
    else:
        cluster_images = [f for f in valid_images 
                        if f.endswith((".jpg", ".png")) 
                        and get_cluster_from_filename(f) == cluster]

    display_count = min(num_images, len(cluster_images))
    if display_count < num_images:
        st.warning(f"Only {len(cluster_images)} images available in this cluster")
    
    # Randomly sample images from the cluster
    sample_images = random.sample(cluster_images, display_count) if cluster_images else []

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