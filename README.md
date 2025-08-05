# 🧠 Brain Tumor Classification & Segmentation

A modern, user-friendly web application for brain tumor analysis using deep learning with ResNet101 for classification and U-Net for segmentation.

## 📋 Project Overview

This project implements a dual-purpose brain tumor analysis system using:
- **Deep Learning Framework**: TensorFlow 2.13.0
- **Classification Model**: ResNet101 with transfer learning
- **Segmentation Model**: U-Net architecture with custom metrics
- **Web Interface**: Streamlit for interactive user experience
- **Image Processing**: OpenCV and PIL for image preprocessing
- **Visualization**: Matplotlib and Seaborn for results display

## 🎯 Features

### Classification Features
- **Brain Tumor Detection**: Classify MRI images as "Tumor" or "No Tumor"
- **Confidence Visualization**: Probability charts and confidence scores
- **Real-time Predictions**: Instant classification results

### Segmentation Features
- **Tumor Boundary Detection**: Identify precise tumor boundaries
- **Mask Generation**: Create pixel-level tumor masks
- **Overlay Visualization**: Display segmentation results over original images
- **Dice Coefficient**: Advanced segmentation metrics

### General Features
- **Interactive Web Interface**: User-friendly Streamlit application
- **Dual Task Support**: Switch between classification and segmentation
- **Image Upload**: Support for various image formats
- **Responsive Design**: Modern UI with beautiful styling
- **Professional Medical Interface**: Medical-grade visualization

## 🛠️ Technical Requirements

### Python Version
- **Required**: Python 3.11.0 (recommended)
- **Compatible**: Python 3.7-3.11
- **Not Supported**: Python 3.13+ (TensorFlow compatibility issues)

### System Requirements
- **RAM**: Minimum 4GB (8GB recommended)
- **Storage**: 2GB free space
- **GPU**: Optional (CPU-only mode supported)

## 📦 Installation & Setup

### Step 1: Python Environment Setup

```bash
# Check Python version (should be 3.11.0)
python --version

# If you have multiple Python versions, use:
py -3.11 --version
```

### Step 2: Install Dependencies

```bash
# Navigate to project directory
cd "Sprint_2_Deep_Learning _CNN & Computer_Vision/Brain_Tumor_Classification_Segmentation"

# Install requirements with Python 3.11.0
py -3.11 -m pip install -r requirements.txt
```

### Step 3: Verify Installation

```bash
# Test TensorFlow installation
py -3.11 -c "import tensorflow as tf; print(f'TensorFlow version: {tf.__version__}')"

# Test Streamlit installation
py -3.11 -c "import streamlit as st; print('Streamlit installed successfully')"
```

## 🚀 Running the Application

### Method 1: Using Python 3.11.0 (Recommended)

```bash
# Navigate to app directory
cd app

# Run with Python 3.11.0
py -3.11 -m streamlit run main.py
```

### Method 2: Using Batch Script (Windows)

```bash
# Run the provided batch script
run_app.bat
```

### Method 3: Using Shell Script (Linux/Mac)

```bash
# Make script executable (Linux/Mac only)
chmod +x run_app.sh

# Run the shell script
./run_app.sh
```

## 🌐 Accessing the Application

Once the app is running, you can access it at:

- **Local URL**: `http://localhost:8501`
- **Network URL**: `http://192.168.x.x:8501` (your local IP)

The application will automatically open in your default web browser.

## 📁 Project Structure

```
Brain_Tumor_Classification_Segmentation/
├── app/
│   └── main.py                 # Main Streamlit application
├── models/
│   ├── best_model.h5          # Classification model (ResNet101)
│   └── model.h5               # Segmentation model (U-Net)
├── notebooks/
│   ├── brain-tumor-classification-resnet101.ipynb # Classification training
│   └── brain-tumor-segmentation-unet.ipynb        # Segmentation training
├── requirements.txt           # Python dependencies
└── README.md                 # This file
```

## ⚠️ Important Notes

- **Educational Purpose**: This tool is designed for educational and research purposes
- **Medical Disclaimer**: Always consult healthcare professionals for actual diagnosis
- **Model Limitations**: Performance may vary with different image qualities and formats
- **Data Privacy**: Images are processed locally and not stored
- **Processing Time**: Segmentation may take longer than classification due to complexity

## 🔧 Troubleshooting

### Common Issues & Solutions

#### 1. TensorFlow Import Errors
```bash
# If you get "No module named 'tensorflow.keras'" error:
py -3.11 -m pip uninstall tensorflow tensorflow-intel keras -y
py -3.11 -m pip install tensorflow==2.13.0
```

#### 2. Python Version Issues
```bash
# Ensure you're using Python 3.11.0
py -3.11 --version

# If you have Python 3.13+, downgrade to 3.11.0
```

#### 3. Dependency Conflicts
```bash
# Clean install with specific versions
py -3.11 -m pip install --upgrade pip
py -3.11 -m pip install -r requirements.txt --force-reinstall
```

#### 4. Model Loading Issues
- The app uses relative paths to automatically locate model files
- Check that both `best_model.h5` and `model.h5` exist in the `models/` directory
- The app will automatically find the models regardless of the installation location

### Performance Optimization

- **CPU Mode**: The app runs in CPU-only mode by default
- **Memory Usage**: Close other applications if you experience memory issues
- **Image Size**: Large images are automatically resized to 224x224 pixels for classification
- **Segmentation Processing**: May take longer for high-resolution images

## 📊 Model Information

### Classification Model (ResNet101)
- **Architecture**: ResNet101 with transfer learning
- **Input Size**: 224x224x3 (RGB images)
- **Output**: Binary classification (Tumor/No Tumor)
- **Training Data**: Brain MRI dataset
- **Accuracy**: Model performance varies based on data quality

### Segmentation Model (U-Net)
- **Architecture**: U-Net with custom metrics
- **Input Size**: Variable (automatically resized)
- **Output**: Binary mask (Tumor/Background)
- **Metrics**: Dice coefficient, IoU, custom loss functions
- **Features**: Precise tumor boundary detection

## 🎨 Interface Features

### Main Components:
- **Task Selection Sidebar**: Choose between Classification and Segmentation
- **Image Upload Area**: Drag-and-drop or click to upload
- **Analysis Results**: Clear prediction with confidence scores or segmentation masks
- **Visualization**: Charts for classification, masks for segmentation
- **Professional UI**: Medical-grade interface design

### Results Display:
- **Classification Results**:
  - ✅ **No Tumor**: Green indicator with confidence score
  - ⚠️ **Tumor Detected**: Red warning with confidence score
  - 📈 **Probability Chart**: Visual representation of predictions
  - 📋 **Detailed Metrics**: Exact probability values

- **Segmentation Results**:
  - 🎯 **Tumor Mask**: Binary mask showing tumor regions
  - 🔍 **Overlay Visualization**: Original image with mask overlay
  - 📊 **Metrics Display**: Dice coefficient and IoU scores
  - 🎨 **Color-coded Results**: Clear visual distinction

## 🐛 Troubleshooting

### Common Issues:

1. **Model Loading Error:**
   - Ensure both `best_model.h5` and `model.h5` exist in `models/` directory
   - The app automatically finds the models using relative paths
   - Check file permissions if the models exist but still fail to load

2. **Import Errors:**
   - Verify all dependencies are installed: `py -3.11 -m pip install -r requirements.txt`
   - Check Python version compatibility (use Python 3.11.0)

3. **Image Upload Issues:**
   - Ensure image format is supported (PNG, JPG, JPEG, BMP, TIFF)
   - Check image file size (recommended < 10MB)

4. **Memory Issues:**
   - Close other applications to free up RAM
   - Use smaller images if available
   - Segmentation requires more memory than classification

5. **Segmentation Model Errors:**
   - Ensure U-Net model has been trained with custom metrics
   - Check that `model.h5` contains the segmentation model
   - Verify custom objects (dice_coef, iou_coef) are properly loaded

## 🎯 Usage Guide

### Classification Task
1. Select "Classification" from the sidebar
2. Upload a brain MRI image
3. Click "Analyze Image"
4. View classification results and confidence scores

### Segmentation Task
1. Select "Segmentation" from the sidebar
2. Upload a brain MRI image
3. Click "Analyze Image"
4. View segmentation mask and overlay visualization

## 🤝 Contributing

1. Fork the repository
2. Create a feature branch
3. Make your changes
4. Test with Python 3.11.0
5. Submit a pull request

## 📄 License

This project is part of the INSTANT AI Data Science Training program.


© 2024 Abdelraouf Dahy Abdelraouf.

## 🙏 Acknowledgments

- ResNet101 architecture by Microsoft Research
- U-Net architecture by Olaf Ronneberger et al.
- TensorFlow and Keras for deep learning framework
- Streamlit for the web application framework
- Medical imaging community for datasets and research

## 📞 Contact Me

[![GitHub](https://img.shields.io/badge/GitHub-100000?style=for-the-badge&logo=github&logoColor=white)](https://github.com/Moataz899)
[![LinkedIn](https://img.shields.io/badge/LinkedIn-0077B5?style=for-the-badge&logo=linkedin&logoColor=white)](https://www.linkedin.com/in/moataz-abdelraouf/)
[![Email](https://img.shields.io/badge/Email-0077B5?style=for-the-badge&logo=linkedin&logoColor=white)](mailto:abdelraoufdahy%40gmail.com)


---

**🧠 Brain Tumor Analysis System | Powered by Deep Learning & ResNet101 + U-Net** 

**Last Updated**: August 2024
**Python Version**: 3.11.0
**TensorFlow Version**: 2.13.0
**Features**: Classification + Segmentation