# demo.py
# streamlit run demo.py
import streamlit as st
import numpy as np
import torch
from PIL import Image, UnidentifiedImageError
from torchvision import models, transforms
import joblib
import os
import pandas as pd
import random
import gdown
from zipfile import ZipFile 

# Cache the loaded images from Google Drive to avoid redownloading them on every run
@st.cache_data
def get_google_drive_images(output_dir="gdrive_images"):
    if not os.path.exists(output_dir):
        base_dir = "/mount/src/art_style_grouping" # The KNOWN working directory
        download_path = os.path.join(base_dir, output_dir)  # Explicitly join!
        st.write(f"Download path: {download_path}") # Verify!

        os.makedirs(download_path, exist_ok=True)
        os.makedirs(output_dir, exist_ok=True)

        try:
            gdown.download_folder(
                "https://drive.google.com/drive/folders/" + "1VanQf88QtskwH6I1z-Ut8oZst5ovuwKy",
                output=output_dir,
                quiet=False,
                use_cookies=True,
                remaining_ok=True
            )

            st.success("✅ Successfully downloaded files!")

            zip_file_path = os.path.join(output_dir, "preprocessed_images.zip")  # Direct path
            st.write(f"Zip file path: {zip_file_path}")  # Verify path

            if not os.path.exists(zip_file_path):
                raise FileNotFoundError(f"Zip file not found at: {zip_file_path}")

            extracted_dir = os.path.join(output_dir, "preprocessed_images") # Path to extracted images
            st.write(f"Extraction directory: {extracted_dir}") # Verify path

            with st.spinner("Unzipping files"):
                with ZipFile(zip_file_path, "r") as zip_ref:
                    zip_ref.extractall(output_dir)  # Extract to gdrive_images

                st.success("✅ Successfully unzipped files!")
            
            if not os.path.exists(extracted_dir):
                st.error(f"Extraction failed. Directory not found: {extracted_dir}")
                return output_dir, []

            os.remove(zip_file_path) # delete the zip
            downloaded_files = os.listdir(extracted_dir)
            return extracted_dir, downloaded_files  # Return the correct path!

        except Exception as e:
            st.error(f"❌ Download or unzip failed: {str(e)}")
            return output_dir, []
    else:
        downloaded_files = os.listdir(output_dir)
        return output_dir, downloaded_files
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
        "K-means": "trained_models/full_kmeans_model.pkl",
        # "DBSCAN": "trained_models/dbscan_model.pkl",
        # "BIRCH": "trained_models/birch_model_compressed.pkl",
    }
    return joblib.load(model_paths[model_type])

# Cache the loaded cluster assignments mapping to avoid reloading it
@st.cache_data
def load_cluster_mapping(model_type):
    """Modified to support multiple models"""
    cluster_files = {
        "K-means": "cluster_assignments/full_kmeans_cluster_assignments.csv",
        # "DBSCAN": "cluster_assignments/dbscan_cluster_assignments.csv",
        # "BIRCH": "cluster_assignments/birch_cluster_assignments.csv",
    }
    return pd.read_csv(cluster_files[model_type]).set_index("filename")

# Function to get the cluster assignment from a filename
def get_cluster_from_filename(filename):
    return cluster_mapping.loc[filename, "cluster"]

# Function to validate images
def validate_image(uploaded_file):
    """Thoroughly validate an uploaded image file"""
    try:
        # Check 1: Verify file signature (magic numbers)
        if uploaded_file.type not in ["image/jpeg", "image/png"]:
            return False, "Invalid file type (must be JPG/PNG)"
        
        # Check 2: Attempt to open and verify image integrity
        with Image.open(uploaded_file) as img:
            # Check 3: Verify image format without loading pixels
            img.verify()
            
            # Check 4: Reset file pointer after verification
            uploaded_file.seek(0)
            
            # Check 5: Reopen for proper processing
            img = Image.open(uploaded_file)
            
            # Check 6: Ensure RGB mode and valid dimensions
            if img.mode not in ["RGB", "L"]:
                return False, "Image must be RGB or grayscale"
            if min(img.size) < 50:
                return False, "Image too small (min 50px)"
            
            return True, img.convert("RGB")
            
    except UnidentifiedImageError:
        return False, "Corrupted or invalid image file"
    except Exception as e:
        return False, f"Image validation failed: {str(e)}"

# Define the image preprocessing steps
preprocess = transforms.Compose([
    transforms.Resize(256), # Resize the image to 256x256
    transforms.CenterCrop(224), # Crop the center 224x224 pixels
    transforms.ToTensor(), # Convert the image to a PyTorch tensor
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]), # Normalize the tensor
])

st.title('Art Style Explorer 🎨')
st.write(f"Current working directory: {os.getcwd()}")

# Model selection (commented out since only K-means remains)
# selected_model = st.sidebar.radio(
#     "Select Clustering Model:",
#     ("K-means",), # Removed DBSCAN, kept BIRCH commented
#     index=0,
#     help="Choose which clustering algorithm to use for similarity detection"
# )
selected_model = "K-means"  # Default to only available model

# Initialize a session state variable to track if an image is uploaded
if "image_uploaded" not in st.session_state:
    st.session_state.image_uploaded = False

if st.session_state.image_uploaded == False:  # Show the message only if no image is uploaded
    st.write("Upload a piece of art and explore some possible artistic siblings! Find hidden connections between your uploaded artwork and others across styles, colors, themes, and eras!")

# File uploader widget
uploaded_file = st.file_uploader("Upload an artwork:", type=["jpg", "png", "jpeg"])

# If an image is uploaded
if uploaded_file is not None:
    
    is_valid, validation_result = validate_image(uploaded_file)
    
    if not is_valid:
        st.error(f"🚨 Invalid Image: {validation_result}")
        st.session_state.image_uploaded = False
        uploaded_file = None
        st.stop()
    
    image = validation_result  # The converted RGB image
    st.session_state.image_uploaded = True
    st.image(image, caption="Uploaded Image", use_container_width=True)

    # Sidebar widgets
    num_images = st.sidebar.slider(
        "Number of similar artworks to show:",
        min_value=1,
        max_value=25,
        value=5,
        help="Choose how many similar artworks from the cluster you want to see"
    )
    
    # Extract the feature embedding from the uploaded image using the loaded model
    with torch.no_grad(): # Disable gradient calculation for inference
        img_tensor = preprocess(image).unsqueeze(0) # Preprocess the image and add a batch dimension
        embedding = load_model()(img_tensor).squeeze().numpy() # Get the feature embedding and convert it to a NumPy array
    
        
    # Load the clusterer and predict the cluster for the uploaded image
    clusterer = load_clusterer(selected_model)
    cluster = clusterer.predict(embedding.reshape(1, -1))[0]
    
    # Get the list of images belonging to the same cluster
    cluster_mapping = load_cluster_mapping(selected_model)
    data_dir, downloaded_files = get_google_drive_images()
       
    cluster_images = [f for f in downloaded_files
                    if f in cluster_mapping.index
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
        max_value=10,
        value=5,
        help="Control how many artworks are shown per row" 
    )

    num_images = len(sample_images)
    
        # Display the images in a grid with error handling
    if sample_images:
        # Add refresh button to sidebar
        if st.sidebar.button("🔄 Refresh Images", help="Show new random selection from this cluster"):
            # Force new random sample by clearing cached selection
            if "current_sample" in st.session_state:
                del st.session_state.current_sample
        
    # Store/Load current sample in session state
    if "current_sample" not in st.session_state or st.session_state.get("prev_cluster") != cluster:
        st.session_state.current_sample = sample_images
        st.session_state.prev_cluster = cluster  # Track cluster changes
    
    # Display images using stored sample
    num_images = len(st.session_state.current_sample)
    
    # Create grid display
    for i in range(0, num_images, columns_per_row):
        cols = st.columns(columns_per_row)
        for j in range(columns_per_row):
            idx = i + j
            if idx < num_images:
                img_name = st.session_state.current_sample[idx]
                img_path = os.path.join(data_dir, img_name)
                try:
                    cols[j].image(img_path, 
                                 use_container_width=True,
                                 caption=img_name)
                except Exception as e:
                    # Silently replace bad images with new ones
                    new_candidates = [f for f in cluster_images 
                                     if f not in st.session_state.current_sample]
                    if new_candidates:
                        replacement = random.choice(new_candidates)
                        st.session_state.current_sample[idx] = replacement
                        cols[j].image(os.path.join(f'gdrive_images\preprocessed_images', replacement),
                                    use_container_width=True,
                                    caption=replacement)
                    else:
                        cols[j].empty()

# Footer (unchanged)
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