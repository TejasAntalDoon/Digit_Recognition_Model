# Digit Recognition CNN with Streamlit Web App
# Final Year Project - Handwritten Digit Recognition

import streamlit as st
import numpy as np
import cv2
from PIL import Image
import os
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
import pickle
import io

# TensorFlow imports with error handling
try:
    import tensorflow as tf
    from tensorflow import keras
    from tensorflow.keras import layers
    print(f"TensorFlow version: {tf.__version__}")
except ImportError as e:
    st.error("TensorFlow not found. Please install: pip install tensorflow")
    st.stop()
except Exception as e:
    st.error(f"Error importing TensorFlow: {e}")
    st.stop()

# Set page config with professional styling
st.set_page_config(
    page_title="AI Digit Recognition System",
    page_icon="🧠",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS for professional look
st.markdown("""
<style>
    .main-header {
        background: linear-gradient(90deg, #667eea 0%, #764ba2 100%);
        padding: 2rem 1rem;
        border-radius: 10px;
        margin-bottom: 2rem;
        text-align: center;
        color: white;
    }
    .metric-card {
        background: white;
        padding: 1rem;
        border-radius: 10px;
        box-shadow: 0 2px 4px rgba(0,0,0,0.1);
        border-left: 4px solid #667eea;
    }
    .success-card {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        color: white;
        padding: 1.5rem;
        border-radius: 10px;
        text-align: center;
        margin: 1rem 0;
    }
    .sidebar .stSelectbox {
        margin-bottom: 1rem;
    }
    .stProgress .st-bo {
        background-color: #667eea;
    }
</style>
""", unsafe_allow_html=True)

class DigitRecognitionCNN:
    def __init__(self):
        self.model = None
        self.history = None
        
    def load_data(self, data_dir):
        """Load and preprocess images from directory structure"""
        images = []
        labels = []
        
        for digit in range(10):
            digit_dir = os.path.join(data_dir, str(digit))
            if os.path.exists(digit_dir):
                for filename in os.listdir(digit_dir):
                    if filename.lower().endswith(('.jpeg', '.jpg', '.png')):
                        img_path = os.path.join(digit_dir, filename)
                        try:
                            # Load and preprocess image
                            img = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)
                            img = cv2.resize(img, (28, 28))
                            img = img.astype('float32') / 255.0
                            images.append(img)
                            labels.append(digit)
                        except Exception as e:
                            st.warning(f"Error loading {img_path}: {e}")
        
        return np.array(images), np.array(labels)
    
    def create_model(self):
        """Create CNN model architecture"""
        model = keras.Sequential([
            # First Convolutional Block
            layers.Conv2D(32, (3, 3), activation='relu', input_shape=(28, 28, 1)),
            layers.MaxPooling2D((2, 2)),
            layers.BatchNormalization(),
            
            # Second Convolutional Block
            layers.Conv2D(64, (3, 3), activation='relu'),
            layers.MaxPooling2D((2, 2)),
            layers.BatchNormalization(),
            
            # Third Convolutional Block
            layers.Conv2D(128, (3, 3), activation='relu'),
            layers.BatchNormalization(),
            
            # Flatten and Dense layers
            layers.Flatten(),
            layers.Dropout(0.5),
            layers.Dense(128, activation='relu'),
            layers.BatchNormalization(),
            layers.Dropout(0.3),
            layers.Dense(10, activation='softmax')
        ])
        
        model.compile(
            optimizer='adam',
            loss='sparse_categorical_crossentropy',
            metrics=['accuracy']
        )
        
        return model
    
    def train_model(self, X, y, epochs=50, validation_split=0.2):
        """Train the CNN model"""
        # Reshape data for CNN
        X = X.reshape(-1, 28, 28, 1)
        
        # Split data
        X_train, X_val, y_train, y_val = train_test_split(
            X, y, test_size=validation_split, random_state=42, stratify=y
        )
        
        # Create model
        self.model = self.create_model()
        
        # Callbacks
        callbacks = [
            keras.callbacks.EarlyStopping(patience=10, restore_best_weights=True),
            keras.callbacks.ReduceLROnPlateau(factor=0.5, patience=5)
        ]
        
        # Train model
        self.history = self.model.fit(
            X_train, y_train,
            epochs=epochs,
            validation_data=(X_val, y_val),
            callbacks=callbacks,
            verbose=1
        )
        
        return self.history
    
    def predict(self, image):
        """Predict digit from image"""
        if self.model is None:
            return None, None
        
        # Preprocess image
        if len(image.shape) == 3:
            image = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
        
        image = cv2.resize(image, (28, 28))
        image = image.astype('float32') / 255.0
        image = image.reshape(1, 28, 28, 1)
        
        # Predict
        predictions = self.model.predict(image)
        predicted_digit = np.argmax(predictions[0])
        confidence = np.max(predictions[0])
        
        return predicted_digit, confidence, predictions[0]

def plot_training_history(history):
    """Plot training history with professional styling"""
    plt.style.use('seaborn-v0_8-darkgrid')
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 5))
    
    # Plot accuracy
    ax1.plot(history.history['accuracy'], label='Training Accuracy', linewidth=2, color='#667eea')
    ax1.plot(history.history['val_accuracy'], label='Validation Accuracy', linewidth=2, color='#764ba2')
    ax1.set_title('Model Accuracy Over Time', fontsize=14, fontweight='bold')
    ax1.set_xlabel('Epoch')
    ax1.set_ylabel('Accuracy')
    ax1.legend()
    ax1.grid(True, alpha=0.3)
    
    # Plot loss
    ax2.plot(history.history['loss'], label='Training Loss', linewidth=2, color='#667eea')
    ax2.plot(history.history['val_loss'], label='Validation Loss', linewidth=2, color='#764ba2')
    ax2.set_title('Model Loss Over Time', fontsize=14, fontweight='bold')
    ax2.set_xlabel('Epoch')
    ax2.set_ylabel('Loss')
    ax2.legend()
    ax2.grid(True, alpha=0.3)
    
    plt.tight_layout()
    return fig

def display_prediction_result(predicted_digit, confidence):
    """Display prediction result in a professional card"""
    st.markdown(f"""
    <div class="success-card">
        <h1 style="font-size: 3em; margin: 0;">🎯 {predicted_digit}</h1>
        <h3 style="margin: 0.5rem 0;">Predicted Digit</h3>
        <p style="font-size: 1.2em; margin: 0;">Confidence: {confidence:.1%}</p>
    </div>
    """, unsafe_allow_html=True)

def main():
    # Professional header
    st.markdown("""
    <div class="main-header">
        <h1>🧠 AI-Powered Digit Recognition System</h1>
        <p style="font-size: 1.2em; margin: 0;">Advanced CNN Architecture for Handwritten Digit Classification</p>
        <p style="font-size: 0.9em; opacity: 0.8; margin: 0.5rem 0 0 0;">Final Year Project | Deep Learning & Computer Vision</p>
    </div>
    """, unsafe_allow_html=True)
    
    # Initialize session state
    if 'model_trained' not in st.session_state:
        st.session_state.model_trained = False
    if 'cnn_model' not in st.session_state:
        st.session_state.cnn_model = DigitRecognitionCNN()
    
    # Enhanced sidebar
    with st.sidebar:
        st.markdown("### 🎯 Navigation Panel")
        page = st.selectbox("Choose a section:", [
            "🏠 Home Dashboard",
            "🚀 Model Training", 
            "🎯 Digit Recognition", 
            "🎭 Live Demo",
            "📊 Performance Analytics",
            "ℹ️ Project Info"
        ])
        
        st.markdown("---")
        
        # Status indicator
        if st.session_state.model_trained:
            st.success("✅ Model Ready")
        else:
            st.warning("⚠️ Model Not Trained")
        
        st.markdown("---")
        st.markdown("**🎓 Academic Project**")
        st.markdown("*CNN-based Classification*")
    
    if page == "🏠 Home Dashboard":
        st.markdown("## 📊 Project Dashboard")
        
        # Key metrics in cards
        col1, col2, col3, col4 = st.columns(4)
        
        with col1:
            st.markdown("""
            <div class="metric-card">
                <h3>🎯 Accuracy Target</h3>
                <h2 style="color: #667eea;">95%+</h2>
                <p>Expected Performance</p>
            </div>
            """, unsafe_allow_html=True)
        
        with col2:
            st.markdown("""
            <div class="metric-card">
                <h3>⚡ Processing Speed</h3>
                <h2 style="color: #667eea;"><100ms</h2>
                <p>Per Image Inference</p>
            </div>
            """, unsafe_allow_html=True)
        
        with col3:
            st.markdown("""
            <div class="metric-card">
                <h3>🔢 Classes</h3>
                <h2 style="color: #667eea;">10</h2>
                <p>Digit Classifications</p>
            </div>
            """, unsafe_allow_html=True)
        
        with col4:
            model_status = "Ready" if st.session_state.model_trained else "Pending"
            status_color = "#667eea" if st.session_state.model_trained else "#ff6b6b"
            st.markdown(f"""
            <div class="metric-card">
                <h3>🤖 Model Status</h3>
                <h2 style="color: {status_color};">{model_status}</h2>
                <p>Current State</p>
            </div>
            """, unsafe_allow_html=True)
        
        st.markdown("---")
        
        # Quick start guide
        st.markdown("## 🚀 Quick Start Guide")
        col1, col2 = st.columns(2)
        
        with col1:
            st.markdown("""
            ### 📋 **Getting Started**
            1. **📁 Prepare Data** - Ensure your digits_jpeg folder is ready
            2. **🚀 Train Model** - Go to Model Training section  
            3. **🎯 Test Recognition** - Upload images for prediction
            4. **📊 View Analytics** - Check performance metrics
            """)
        
        with col2:
            st.markdown("""
            ### 🏗️ **Technical Stack**
            - **🧠 Deep Learning**: TensorFlow 2.x + Keras
            - **🌐 Web Interface**: Streamlit Framework
            - **🖼️ Image Processing**: OpenCV + PIL
            - **📈 Visualization**: Matplotlib + Charts
            """)
    
    elif page == "🚀 Model Training":
        st.markdown("## 🚀 Model Training Center")
        
        # Data directory input with better styling
        st.markdown("### 📁 Dataset Configuration")
        data_dir = st.text_input("📂 Enter path to digits_jpeg directory:", "digits_jpeg", 
                                help="Folder containing digit subfolders (0-9)")
        
        col1, col2 = st.columns([1, 1])
        
        with col1:
            st.markdown("**⚙️ Training Parameters**")
            epochs = st.slider("🔄 Training Epochs:", 10, 100, 50, help="Number of training iterations")
            validation_split = st.slider("📊 Validation Split:", 0.1, 0.3, 0.2, help="Percentage of data for validation")
        
        with col2:
            st.markdown("**📋 Expected Directory Structure**")
            st.code("""
digits_jpeg/
├── 0/ (digit images)
├── 1/ (digit images)
├── 2/ (digit images)
...
└── 9/ (digit images)
            """)
        
        # Training buttons with better styling
        col1, col2 = st.columns(2)
        
        with col1:
            if st.button("🚀 Start Training", type="primary", use_container_width=True):
                if os.path.exists(data_dir):
                    with st.spinner("🔄 Loading dataset..."):
                        try:
                            X, y = st.session_state.cnn_model.load_data(data_dir)
                            
                            if len(X) > 0:
                                st.success(f"✅ Successfully loaded {len(X)} images")
                                
                                # Data distribution chart
                                unique, counts = np.unique(y, return_counts=True)
                                chart_data = dict(zip(unique.astype(str), counts))
                                st.bar_chart(chart_data)
                                
                                with st.spinner(f"🧠 Training CNN model ({epochs} epochs)..."):
                                    history = st.session_state.cnn_model.train_model(
                                        X, y, epochs=epochs, validation_split=validation_split
                                    )
                                    
                                    st.session_state.model_trained = True
                                    
                                    # Success message
                                    st.markdown("""
                                    <div class="success-card">
                                        <h2>🎉 Training Completed Successfully!</h2>
                                        <p>Your CNN model is now ready for digit recognition</p>
                                    </div>
                                    """, unsafe_allow_html=True)
                                    
                                    # Training history plot
                                    fig = plot_training_history(history)
                                    st.pyplot(fig)
                                    
                                    # Save model
                                    st.session_state.cnn_model.model.save("digit_recognition_model.h5")
                                    st.info("💾 Model automatically saved as 'digit_recognition_model.h5'")
                            else:
                                st.error("❌ No valid images found in the specified directory")
                        except Exception as e:
                            st.error(f"❌ Training failed: {e}")
                else:
                    st.error("❌ Directory not found. Please check the path.")
        
        with col2:
            if st.button("📂 Load Existing Model", use_container_width=True):
                try:
                    st.session_state.cnn_model.model = keras.models.load_model("digit_recognition_model.h5")
                    st.session_state.model_trained = True
                    st.success("✅ Pre-trained model loaded successfully!")
                except:
                    st.error("❌ No saved model found. Please train a model first.")
    
    elif page == "🎯 Digit Recognition":
        st.markdown("## 🎯 Intelligent Digit Recognition")
        
        if not st.session_state.model_trained:
            st.warning("⚠️ Please train or load a model first from the Model Training section!")
            return
        
        # Single image prediction
        st.markdown("### 📸 Single Image Recognition")
        uploaded_file = st.file_uploader("Upload a digit image", type=['jpg', 'jpeg', 'png'],
                                       help="Upload a clear image of a handwritten digit (0-9)")
        
        if uploaded_file is not None:
            col1, col2 = st.columns([1, 1])
            
            with col1:
                image = Image.open(uploaded_file)
                st.image(image, caption="📷 Uploaded Image", use_column_width=True)
            
            with col2:
                img_array = np.array(image)
                predicted_digit, confidence, all_predictions = st.session_state.cnn_model.predict(img_array)
                
                if predicted_digit is not None:
                    display_prediction_result(predicted_digit, confidence)
                    
                    # Prediction breakdown
                    st.markdown("**📊 Confidence Distribution:**")
                    pred_data = {f"Digit {i}": pred for i, pred in enumerate(all_predictions)}
                    st.bar_chart(pred_data)
                else:
                    st.error("❌ Prediction failed. Please try another image.")
        
        st.markdown("---")
        
        # Batch prediction
        st.markdown("### 📊 Batch Image Processing")
        uploaded_files = st.file_uploader("Upload multiple images for batch processing", 
                                        type=['jpg', 'jpeg', 'png'], accept_multiple_files=True)
        
        if uploaded_files:
            st.info(f"📂 Processing {len(uploaded_files)} images...")
            
            results = []
            progress_bar = st.progress(0)
            
            for idx, file in enumerate(uploaded_files):
                image = Image.open(file)
                img_array = np.array(image)
                pred_digit, conf, _ = st.session_state.cnn_model.predict(img_array)
                
                status = "🟢 High" if conf > 0.8 else "🟡 Medium" if conf > 0.5 else "🔴 Low"
                results.append({
                    'Image': file.name,
                    'Predicted Digit': pred_digit,
                    'Confidence': f"{conf:.1%}",
                    'Status': status
                })
                
                progress_bar.progress((idx + 1) / len(uploaded_files))
            
            # Results summary
            st.markdown("### 📈 Batch Processing Results")
            col1, col2, col3 = st.columns(3)
            
            high_conf = sum(1 for r in results if '🟢' in r['Status'])
            avg_conf = np.mean([float(r['Confidence'][:-1])/100 for r in results])
            
            col1.metric("Total Processed", len(results))
            col2.metric("High Confidence", high_conf)
            col3.metric("Average Confidence", f"{avg_conf:.1%}")
            
            st.dataframe(results, use_container_width=True)
    
    elif page == "🎭 Live Demo":
        st.markdown("## 🎭 Interactive Demo Center")
        st.markdown("*Perfect for presentations and live demonstrations*")
        
        if not st.session_state.model_trained:
            st.warning("⚠️ Please train or load a model first!")
            return
        
        # Demo interface with tabs
        demo_tab1, demo_tab2 = st.tabs(["🎯 Quick Demo", "📊 Batch Analysis"])
        
        with demo_tab1:
            st.markdown("### 🚀 Instant Recognition Demo")
            
            col1, col2 = st.columns([1, 1])
            
            with col1:
                test_file = st.file_uploader("Upload demo image", type=['jpg', 'jpeg', 'png'], key="demo")
                
                if test_file:
                    image = Image.open(test_file)
                    st.image(image, caption="Demo Image", use_column_width=True)
                    
                    if st.button("🎯 Predict Now!", type="primary", use_container_width=True):
                        with st.spinner("🤖 AI analyzing..."):
                            import time
                            time.sleep(1)  # Dramatic effect for demo
                            
                            img_array = np.array(image)
                            pred_digit, confidence, all_preds = st.session_state.cnn_model.predict(img_array)
                            
                            display_prediction_result(pred_digit, confidence)
            
            with col2:
                st.markdown("**🎯 Demo Guidelines**")
                st.info("""
                **For Best Results:**
                - Use clear, well-lit digit images
                - Ensure digits are centered
                - Avoid blurry or distorted images
                - Single digit per image works best
                """)
                
                if st.session_state.model_trained:
                    st.success("🤖 **Model Status: Ready**")
                    st.markdown(f"""
                    **Model Info:**
                    - Parameters: {st.session_state.cnn_model.model.count_params():,}
                    - Architecture: CNN with 3 Conv blocks
                    - Classes: 10 digits (0-9)
                    """)
        
        with demo_tab2:
            st.markdown("### 📊 Professional Batch Demo")
            
            demo_images = st.file_uploader("Upload multiple demo images", 
                                         type=['jpg', 'jpeg', 'png'], 
                                         accept_multiple_files=True, key="batch_demo")
            
            if demo_images:
                results = []
                progress_bar = st.progress(0)
                
                for idx, file in enumerate(demo_images):
                    image = Image.open(file)
                    img_array = np.array(image)
                    pred_digit, conf, _ = st.session_state.cnn_model.predict(img_array)
                    
                    results.append({
                        'Image': file.name,
                        'Prediction': pred_digit,
                        'Confidence': f"{conf:.1%}",
                        'Status': '✅' if conf > 0.8 else '⚠️' if conf > 0.5 else '❌'
                    })
                    
                    progress_bar.progress((idx + 1) / len(demo_images))
                
                # Professional results display
                st.markdown("### 🎯 Analysis Results")
                
                # Summary metrics
                col1, col2, col3, col4 = st.columns(4)
                high_conf = sum(1 for r in results if r['Status'] == '✅')
                
                col1.metric("Images Processed", len(results))
                col2.metric("High Confidence", high_conf)
                col3.metric("Success Rate", f"{high_conf/len(results):.0%}")
                col4.metric("Avg Confidence", f"{np.mean([float(r['Confidence'][:-1])/100 for r in results]):.0%}")
                
                st.dataframe(results, use_container_width=True)
    
    elif page == "📊 Performance Analytics":
        st.markdown("## 📊 Advanced Performance Analytics")
        
        if not st.session_state.model_trained:
            st.warning("⚠️ Please train a model to view performance analytics!")
            return
        
        # Analytics tabs
        analytics_tab1, analytics_tab2 = st.tabs(["📈 Training Metrics", "🔍 Model Architecture"])
        
        with analytics_tab1:
            if hasattr(st.session_state.cnn_model, 'history') and st.session_state.cnn_model.history:
                history = st.session_state.cnn_model.history.history
                
                # Training visualization
                col1, col2 = st.columns(2)
                
                with col1:
                    st.markdown("**📊 Accuracy Progression**")
                    acc_data = {
                        'Training': history.get('accuracy', []),
                        'Validation': history.get('val_accuracy', [])
                    }
                    st.line_chart(acc_data)
                
                with col2:
                    st.markdown("**📉 Loss Progression**")
                    loss_data = {
                        'Training': history.get('loss', []),
                        'Validation': history.get('val_loss', [])
                    }
                    st.line_chart(loss_data)
                
                # Final metrics
                st.markdown("### 🎯 Final Performance Metrics")
                metrics_col1, metrics_col2, metrics_col3, metrics_col4 = st.columns(4)
                
                if 'accuracy' in history and history['accuracy']:
                    with metrics_col1:
                        st.metric("Training Accuracy", f"{history['accuracy'][-1]:.3f}")
                if 'val_accuracy' in history and history['val_accuracy']:
                    with metrics_col2:
                        st.metric("Validation Accuracy", f"{history['val_accuracy'][-1]:.3f}")
                if 'loss' in history and history['loss']:
                    with metrics_col3:
                        st.metric("Training Loss", f"{history['loss'][-1]:.4f}")
                if 'val_loss' in history and history['val_loss']:
                    with metrics_col4:
                        st.metric("Validation Loss", f"{history['val_loss'][-1]:.4f}")
            else:
                st.info("📝 Train a model to see detailed performance metrics")
        
        with analytics_tab2:
            if st.session_state.cnn_model.model:
                model = st.session_state.cnn_model.model
                
                col1, col2 = st.columns(2)
                
                with col1:
                    st.markdown("**🏗️ Model Architecture Summary**")
                    st.info(f"""
                    **Model Statistics:**
                    - Total Parameters: {model.count_params():,}
                    - Trainable Parameters: {sum([tf.keras.backend.count_params(w) for w in model.trainable_weights]):,}
                    - Model Layers: {len(model.layers)}
                    - Estimated Size: ~{model.count_params() * 4 / (1024*1024):.1f} MB
                    """)
                
                with col2:
                    st.markdown("**⚙️ Layer Configuration**")
                    layer_data = []
                    for i, layer in enumerate(model.layers):
                        layer_data.append({
                            'Layer': i+1,
                            'Type': type(layer).__name__,
                            'Parameters': layer.count_params()
                        })
                    st.dataframe(layer_data, use_container_width=True)
    
    else:  # Project Info
        st.markdown("## ℹ️ Project Information")
        
        st.markdown("""
        <div class="success-card">
            <h2>🎓 Final Year Project: AI Digit Recognition</h2>
            <p>Advanced Deep Learning Implementation for Handwritten Digit Classification</p>
        </div>
        """, unsafe_allow_html=True)
        
        # Project overview
        col1, col2 = st.columns(2)
        
        with col1:
            st.markdown("""
            ### 🎯 **Project Objectives**
            - Implement CNN architecture for digit recognition
            - Achieve >90% accuracy on validation data
            - Create user-friendly web interface
            - Demonstrate real-time prediction capabilities
            - Provide comprehensive performance analytics
            """)
            
            st.markdown("""
            ### 🔧 **Technical Implementation**
            - **Framework**: TensorFlow 2.x + Keras
            - **Architecture**: Custom CNN with 3 conv blocks
            - **Interface**: Streamlit web application
            - **Processing**: OpenCV + PIL for image handling
            - **Visualization**: Matplotlib for analytics
            """)
        
        with col2:
            st.markdown("""
            ### 📊 **Key Features**
            - Custom dataset training support
            - Real-time digit prediction
            - Batch image processing
            - Interactive training visualization
            - Model persistence (save/load)
            - Professional web interface
            """)
            
            st.markdown("""
            ### 🚀 **Performance Targets**
            - **Accuracy**: >95% (training), >90% (validation)
            - **Speed**: <100ms per image prediction
            - **Memory**: <500MB during training
            - **Compatibility**: Cross-platform deployment
            """)
        
        # Technical specifications
        with st.expander("🔧 **Detailed Technical Specifications**"):
            st.code("""
CNN Architecture Details:
========================
Input Layer:    28x28x1 grayscale images
Conv Block 1:   Conv2D(32, 3x3) → ReLU → MaxPool2D(2x2) → BatchNorm
Conv Block 2:   Conv2D(64, 3x3) → ReLU → MaxPool2D(2x2) → BatchNorm  
Conv Block 3:   Conv2D(128, 3x3) → ReLU → BatchNorm
Dense Block:    Flatten → Dropout(0.5) → Dense(128) → ReLU → 
                BatchNorm → Dropout(0.3) → Dense(10) → Softmax

Training Configuration:
======================
Optimizer:      Adam
Loss Function:  Sparse Categorical Crossentropy
Callbacks:      EarlyStopping(patience=10), ReduceLROnPlateau
Regularization: Dropout layers + Batch Normalization
            """)

if __name__ == "__main__":
    main()