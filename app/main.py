import streamlit as st
import tensorflow as tf
import numpy as np
from PIL import Image
import cv2
import matplotlib.pyplot as plt
import os

# Set page configuration
st.set_page_config(page_title="Brain Tumor Analysis", layout="wide")

# Define custom metrics for U-Net model
def dice_coef(y_true, y_pred, smooth=100):
    y_true_flatten = tf.keras.backend.flatten(y_true)
    y_pred_flatten = tf.keras.backend.flatten(y_pred)
    intersection = tf.keras.backend.sum(y_true_flatten * y_pred_flatten)
    union = tf.keras.backend.sum(y_true_flatten) + tf.keras.backend.sum(y_pred_flatten)
    return (2 * intersection + smooth) / (union + smooth)

def dice_loss(y_true, y_pred, smooth=100):
    return -dice_coef(y_true, y_pred, smooth)

def iou_coef(y_true, y_pred, smooth=100):
    intersection = tf.keras.backend.sum(y_true * y_pred)
    sum_ = tf.keras.backend.sum(y_true + y_pred)
    iou = (intersection + smooth) / (sum_ - intersection + smooth)
    return iou

# Title and description
st.title("🧠 Brain Tumor Analysis System")
st.markdown("Choose between **Classification** (detect tumor presence) or **Segmentation** (identify tumor boundaries)")

# Sidebar for task selection
st.sidebar.title("Task Selection")
task = st.sidebar.radio(
    "Choose your analysis task:",
    ["Classification", "Segmentation"],
    help="Classification: Detect if tumor is present\nSegmentation: Identify tumor boundaries"
)

# Load models
@st.cache_resource
def load_classification_model():
    """Load the ResNet101 classification model."""
    current_dir = os.path.dirname(os.path.abspath(__file__))
    model_path = os.path.join(current_dir, "..", "models", "best_model.h5")
    
    if not os.path.exists(model_path):
        st.error(f"Classification model file {model_path} not found!")
        return None
    
    try:
        # Create ResNet101 model architecture
        base_model = tf.keras.applications.ResNet101(
            include_top=False,
            weights='imagenet',
            input_shape=(224, 224, 3)
        )
        base_model.trainable = False
        
        model = tf.keras.Sequential([
            base_model,
            tf.keras.layers.GlobalAveragePooling2D(),
            tf.keras.layers.Dropout(0.5),
            tf.keras.layers.Dense(512, activation='relu'),
            tf.keras.layers.Dropout(0.3),
            tf.keras.layers.Dense(2, activation='softmax')
        ])
        
        # Load weights
        model.load_weights(model_path, by_name=True, skip_mismatch=True)
        model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])
        
        return model
        
    except Exception as e:
        st.error(f"Failed to load classification model: {e}")
        return None

@st.cache_resource
def load_segmentation_model():
    """Load the U-Net segmentation model."""
    current_dir = os.path.dirname(os.path.abspath(__file__))
    model_path = os.path.join(current_dir, "..", "models", "model.h5")
    
    if not os.path.exists(model_path):
        st.error(f"Segmentation model file {model_path} not found!")
        return None
    
    try:
        model = tf.keras.models.load_model(model_path, custom_objects={
            'dice_coef': dice_coef,
            'dice_loss': dice_loss,
            'iou_coef': iou_coef
        })
        return model
    except Exception as e:
        st.error(f"Failed to load segmentation model: {e}")
        return None

# Function to preprocess image for segmentation
def preprocess_segmentation_image(image):
    img = cv2.resize(image, (256, 256))
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    img = img / 255.0
    img = np.expand_dims(img, axis=0)
    return img

# Main application logic
if task == "Classification":
    st.header("🔍 Brain Tumor Classification")
    st.markdown("Upload an MRI image to classify whether it indicates a brain tumor.")
    
    # Load classification model
    classification_model = load_classification_model()
    
    # File uploader
    uploaded_file = st.file_uploader("Choose an MRI image...", type=["jpg", "jpeg", "png"])
    
    if uploaded_file is not None:
        # Display the uploaded image
        image = Image.open(uploaded_file).convert('RGB')
        col1, col2 = st.columns([1, 1])
        
        with col1:
            st.image(image, caption="Uploaded MRI Image", width=300)
        
        # Predict button
        if st.button("🔬 Analyze for Tumor", type="primary"):
            if classification_model is None:
                st.error("Classification model could not be loaded.")
            else:
                with st.spinner("Processing classification..."):
                    try:
                        # Preprocess the image
                        img = image.resize((224, 224))
                        img_array = np.array(img) / 255.0
                        img_array = np.expand_dims(img_array, axis=0)
                        
                        # Predict
                        prediction = classification_model.predict(img_array, verbose=0)
                        predicted_class = np.argmax(prediction, axis=1)[0]
                        confidence = prediction[0][predicted_class]
                        
                        # Display results
                        with col2:
                            result = "🟢 **No Tumor Detected**" if predicted_class == 0 else "🔴 **Tumor Detected**"
                            st.success(result)
                            st.metric("Confidence", f"{confidence:.2%}")
                            
                            # Add confidence bar
                            st.progress(float(confidence))
                            
                            if predicted_class == 1:
                                st.warning("⚠️ **Medical Disclaimer**: This is an AI prediction and should not replace professional medical diagnosis. Please consult a healthcare professional.")
                            else:
                                st.info("ℹ️ **Note**: This is an AI prediction and should not replace professional medical diagnosis.")
                        
                    except Exception as e:
                        st.error(f"Error during classification: {e}")
    else:
        st.info("📁 Please upload an MRI image to proceed with classification.")
        
elif task == "Segmentation":
    st.header("🎯 Brain Tumor Segmentation")
    st.markdown("Upload an MRI image to generate a segmentation mask showing tumor boundaries.")
    
    # Load segmentation model
    segmentation_model = load_segmentation_model()
    
    # File uploader
    uploaded_image = st.file_uploader("Upload MRI Image", type=["png", "jpg", "jpeg"])
    
    if uploaded_image is not None and segmentation_model is not None:
        # Read and preprocess the image
        image = cv2.imdecode(np.frombuffer(uploaded_image.read(), np.uint8), cv2.IMREAD_COLOR)
        processed_image = preprocess_segmentation_image(image)
        
        # Predict the mask
        with st.spinner("🎯 Generating segmentation mask..."):
            predicted_mask = segmentation_model.predict(processed_image)
            predicted_mask = (predicted_mask > 0.5).astype(np.uint8)
            predicted_mask = np.squeeze(predicted_mask)
        
        # Create visualization
        fig, axes = plt.subplots(1, 2, figsize=(12, 5))
        
        # Original image
        axes[0].imshow(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))
        axes[0].set_title("Original MRI Image", fontsize=12, fontweight='bold')
        axes[0].axis('off')
        
        # Predicted mask
        axes[1].imshow(predicted_mask, cmap='hot')
        axes[1].set_title("Predicted Tumor Mask", fontsize=12, fontweight='bold')
        axes[1].axis('off')
        
        plt.tight_layout()
        st.pyplot(fig)
        
        # Add analysis Metrics
        st.subheader("📊 Segmentation Analysis")
        col1, col2 = st.columns(2)
        
        with col1:
            st.metric("Predicted Tumor Area", f"{predicted_mask.sum()} pixels")
        with col2:
            tumor_percentage = (predicted_mask.sum() / (predicted_mask.shape[0] * predicted_mask.shape[1])) * 100
            st.metric("Tumor Coverage", f"{tumor_percentage:.2f}%")
        
        st.info("ℹ️ **Medical Disclaimer**: This segmentation is an AI prediction and should not replace professional medical diagnosis.")
        
    elif uploaded_image is None:
        st.info("📁 Please upload an MRI image to proceed with segmentation.")
    else:
        st.error("❌ Segmentation model could not be loaded. Please check the model file.")

# Footer
st.markdown("---")
st.markdown("""
<div style='text-align: center; color: #666;'>
    <p>🧠 <strong>Brain Tumor Analysis System</strong> | Built with Streamlit & TensorFlow</p>
    <p><em>For research and educational purposes only. Not for clinical use.</em></p>
</div>
""", unsafe_allow_html=True) 