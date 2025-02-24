Art Style Clustering & Exploration (https://artexplorer.streamlit.app/)
==================================

🚀 User Manual: Using the Streamlit App
---------------------------------------

This project includes a **Streamlit-based application** that allows users to upload an artwork and explore similar artworks from a dataset using unsupervised learning clustering techniques.

### **📂 How It Works**

1.  **Upload an image** of an artwork (also works well with photographs!).

2.  The system extracts **deep learning-based embeddings** using a **ResNet50 model**.

3.  The image is clustered using **K-Means** (additional methods like DBSCAN or Birch coming soon).

4.  The app **retrieves similar artworks** from the same cluster and displays them.

5.  You can adjust the number of retrieved images and refine searches using sidebar settings.

🎨 Project Overview: Art Style Clustering with Unsupervised Learning
====================================================================

**1️⃣ Introduction**
--------------------

This project applies **unsupervised learning** to group artworks by style. We use **deep learning** for feature extraction, **dimensionality reduction** to improve clustering accuracy, and an **interactive app** as a real-world application.

### **Objectives**

-   Extract visual features from paintings using **ResNet50**.

-   Apply **PCA/UMAP** for feature compression.

-   Cluster paintings using **K-Means, DBSCAN, and Birch**.

-   Visualize and analyze how different styles relate to each other.

-   Build an application to allow users to **interactively explore** artwork relationships.

**2️⃣ Dataset & Preprocessing**
-------------------------------

-   **Data Source**: WikiArt Dataset (various painting styles and movements).

-   **Data Handling**:

    -   The dataset is **automatically downloaded and structured**.

    -   A preprocessing pipeline extracts **image embeddings** from a **pre-trained ResNet model**.

    -   **Dimensionality Reduction** (PCA & UMAP) reduces high-dimensional feature vectors.

**3️⃣ Exploratory Analysis**
----------------------------

-   **Feature Distributions**: Understanding extracted feature embeddings.

-   **UMAP 2D Projections**: Visualizing the dataset in lower dimensions.

-   **Outlier Analysis**: Identifying any abnormal data points.

**4️⃣ Clustering Methods**
--------------------------

We explore multiple clustering techniques:

-   **K-Means**: Simple, interpretable, but requires optimal `k` selection.

-   **DBSCAN**: Density-based, good for non-uniform clusters.

-   **Birch**: Hierarchical clustering, effective for large datasets.

**5️⃣ Model Implementation & Hyperparameter Tuning**
----------------------------------------------------

-   **Feature Extraction**: Using ResNet50 without the final classification layer.

-   **Cluster Validation**:

    -   **Silhouette Score** evaluates cluster separation.

    -   **Elbow Method & KneeLocator** determine the best `k` for K-Means.

    -   **Nearest Neighbors Distance Estimation** helps tune DBSCAN.

-   **Cluster Assignments**: Matching images to their assigned clusters.

**6️⃣ Results & Analysis**
--------------------------

-   **K-Means produced the most balanced clusters** but required tuning.

-   **DBSCAN was effective at finding dense clusters** but ignored scattered points.

-   **Birch provided hierarchical organization** but needed refinement.

-   **Visualization**:

    -   **Scatter plots of UMAP-reduced features** to see clusters in 2D.

    -   **Example images from each cluster** to assess artistic coherence.

**7️⃣ Discussion & Use Case: Streamlit Application**
----------------------------------------------------

-   **Deep Learning + Unsupervised Learning**: Combining pre-trained models with clustering was effective for discovering artistic relationships.

-   **Importance of Dimensionality Reduction**: Without PCA/UMAP, high-dimensional noise hindered clustering.

-   **No Single Best Clustering Algorithm**: The best method depends on dataset characteristics.

-   **Application Integration**:

    -   A **Streamlit app was created** to enable users to interactively explore artwork clusters.

    -   Users can **upload an image** and find similar artworks, demonstrating a real-world application of this research.

**8️⃣ Future Work**
-------------------

-   **Improve Feature Extraction**: Try **Vision Transformers (ViT)** or **EfficientNet**.

-   **Hybrid Approaches**: Combine clustering with **self-supervised learning**.

-   **Expand Application Features**: Implement more clustering models and user-based preferences.

**9️⃣ References & GitHub Repository**
--------------------------------------

-   **WikiArt Dataset**: https://www.kaggle.com/datasets/steubk/wikiart

-   **Project Repository**: https://github.com/rohanxavier33/artExplorer

* * * * *
