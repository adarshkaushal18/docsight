import streamlit as st
import cv2
import numpy as np
from skimage.filters import sobel
from skimage.segmentation import watershed
from skimage.segmentation import felzenszwalb
from skimage.feature import canny
from skimage.color import rgb2gray
from scipy.ndimage import median_filter


def medical_image_enhancement():
    st.title("Medical Image Enhancement")

    uploaded_file = st.file_uploader("Upload a medical image", type=["png", "jpg", "jpeg"])

    if uploaded_file is not None:
        # Convert uploaded file to OpenCV image
        file_bytes = np.asarray(bytearray(uploaded_file.read()), dtype=np.uint8)
        image = cv2.imdecode(file_bytes, cv2.IMREAD_GRAYSCALE)

        st.image(image, caption="Original Image", use_container_width=True, channels="GRAY")

        st.subheader("Choose Enhancement Technique:")

        enhancement_option = st.selectbox(
            "Select an enhancement method",
            [
                "Histogram Equalization",
                "Gray Level Transformation",
                "Image Smoothing",
                "Image Sharpening",
                "Low Pass Filter",
                "High Pass Filter",
                "Median Filter",
                "Point Detection",
                "Line Detection",
                "Edge Detection",
                "Image Segmentation"
            ]
        )

        enhanced_image = None  # Placeholder for processed image

        if enhancement_option == "Histogram Equalization":
            enhanced_image = cv2.equalizeHist(image)

        elif enhancement_option == "Gray Level Transformation":
            gamma = st.slider("Select gamma value:", 0.1, 3.0, 1.0)
            enhanced_image = np.power(image / 255.0, gamma) * 255
            enhanced_image = enhanced_image.astype(np.uint8)

        elif enhancement_option == "Image Smoothing":
            kernel_size = st.slider("Kernel size:", 3, 11, step=2)
            enhanced_image = cv2.blur(image, (kernel_size, kernel_size))
            enhanced_image = cv2.GaussianBlur(image, (kernel_size, kernel_size), 0)

        elif enhancement_option == "Image Sharpening":
            kernel = np.array([[0, -1, 0], [-1, 5, -1], [0, -1, 0]])
            enhanced_image = cv2.filter2D(image, -1, kernel)

        elif enhancement_option == "Low Pass Filter":
            kernel = np.ones((5, 5), np.float32) / 25
            enhanced_image = cv2.filter2D(image, -1, kernel)

        elif enhancement_option == "High Pass Filter":
            kernel = np.array([[-1, -1, -1], [-1, 8, -1], [-1, -1, -1]])
            enhanced_image = cv2.filter2D(image, -1, kernel)

        elif enhancement_option == "Median Filter":
            kernel_size = st.slider("Kernel size:", 3, 11, step=2)
            enhanced_image = median_filter(image, size=kernel_size)

        elif enhancement_option == "Point Detection":
            kernel = np.array([[1, -2, 1], [-2, 4, -2], [1, -2, 1]])
            enhanced_image = cv2.filter2D(image, -1, kernel)

        elif enhancement_option == "Line Detection":
            line_kernel = np.array([[-1, -1, 2], [-1, 2, -1], [2, -1, -1]])
            enhanced_image = cv2.filter2D(image, -1, line_kernel)

        elif enhancement_option == "Edge Detection":
            edge_method = st.selectbox("Select edge detection method", ["Sobel", "Canny"])
            if edge_method == "Sobel":
                enhanced_image = sobel(image)
            elif edge_method == "Canny":
                threshold1 = st.slider("Canny threshold1:", 0, 255, 100)
                threshold2 = st.slider("Canny threshold2:", 0, 255, 200)
                enhanced_image = canny(image, sigma=1.0) * 255  # Convert to uint8 for display

        elif enhancement_option == "Image Segmentation":
            segmentation_method = st.selectbox("Select segmentation method", ["Watershed", "Fuzzy K-Means"])
            if segmentation_method == "Watershed":
                gradient = sobel(image)
                markers = np.zeros_like(image)
                markers[image < 50] = 1
                markers[image > 150] = 2
                enhanced_image = watershed(gradient, markers)
            elif segmentation_method == "Fuzzy K-Means":
                enhanced_image = felzenszwalb(image, scale=100)

        if enhanced_image is not None:
            st.image(enhanced_image, caption=f"Enhanced Image - {enhancement_option}", use_container_width=True, channels="GRAY")

