import streamlit as st
import numpy as np
import cv2
import matplotlib.pyplot as plt

def mean_shift_segmentation(img_path, spatial_radius, color_radius, min_density):
    # Load the image
    img = cv2.imread(img_path)

    # Apply mean shift clustering
    shifted = cv2.pyrMeanShiftFiltering(img, spatial_radius, color_radius, min_density)

    return img, shifted

def main():
    st.set_page_config(
        page_title="Image Segmentation App",
        page_icon=":camera:",
        layout="centered",
        initial_sidebar_state="expanded"
    )

    st.title("Image Segmentation using Mean Shift")

    uploaded_file = st.file_uploader("Upload an image", type=["jpg", "jpeg", "png"])

    if uploaded_file is not None:
        spatial_radius = st.slider("Spatial Radius", min_value=1, max_value=100, value=20)
        color_radius = st.slider("Color Radius", min_value=1, max_value=100, value=30)
        min_density = st.slider("Minimum Density", min_value=1, max_value=100, value=10)

        img_path = "uploaded_image.jpg"
        with open(img_path, "wb") as f:
            f.write(uploaded_file.read())

        original_img, segmented_img = mean_shift_segmentation(img_path, spatial_radius, color_radius, min_density)

        image_width = st.slider("Adjust Image Width", min_value=100, max_value=800, value=500)

        st.subheader("Original Image")
        st.image(cv2.cvtColor(original_img, cv2.COLOR_BGR2RGB), width=image_width)

        st.subheader("Segmented Image")
        st.image(cv2.cvtColor(segmented_img, cv2.COLOR_BGR2RGB), width=image_width)

if __name__ == "__main__":
    main()
