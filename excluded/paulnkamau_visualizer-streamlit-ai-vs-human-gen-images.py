!pip install --upgrade pip -q


!pip install streamlit -q
!pip install kagglehub -q
!npm install localtunnel -q


# IMPORT KAGGLE DATA SOURCES,
import kagglehub
alessandrasala79_ai_vs_human_generated_dataset_path = kagglehub.dataset_download('alessandrasala79/ai-vs-human-generated-dataset')

print('Data source import complete.')


%%writefile visualizer.py
import streamlit as st
import os
import random
import pandas as pd
from PIL import Image
import kagglehub

def main() -> None:
    st.set_page_config(page_title="Dataset Visualizer", layout="wide")
    st.title("🔍 AI vs Human Dataset Explorer")

    # Dataset mode selection
    mode = st.sidebar.radio("Select Dataset", ["Test", "Train"])

    # Download the dataset using kagglehub
    try:
        dataset_path = kagglehub.dataset_download('alessandrasala79/ai-vs-human-generated-dataset')
        st.success(f"Dataset downloaded to: {dataset_path}")
    except Exception as e:
        st.error(f"Failed to download dataset: {str(e)}")
        return

    # Set paths based on mode
    if mode == "Train":
        IMAGE_BASE_PATH = os.path.join(dataset_path, "train_data")
        CSV_PATH = "/kaggle/input/ai-vs-human-generated-dataset/train.csv"  # local CSV path
    else:
        IMAGE_BASE_PATH = os.path.join(dataset_path, "test_data_v2")
        CSV_PATH = None

    # Verify the images directory exists
    if not os.path.exists(IMAGE_BASE_PATH):
        st.error(f"Image directory not found: {IMAGE_BASE_PATH}")
        return

    # Load labels if in train mode
    labels_df = None
    if mode == "Train":
        try:
            labels_df = pd.read_csv(CSV_PATH)
            labels_df['filename'] = labels_df['file_name'].str.split('/').str[-1]
            st.sidebar.success(f"Loaded {len(labels_df)} training labels")
        except Exception as e:
            st.error(f"Failed to load training labels: {str(e)}")
            return

    # Get list of all image files
    try:
        all_images = [
            f for f in os.listdir(IMAGE_BASE_PATH) 
            if f.lower().endswith(('.jpg', '.jpeg', '.png'))
        ]
        if not all_images:
            st.error(f"No images found in directory: {IMAGE_BASE_PATH}")
            return
    except Exception as e:
        st.error(f"Error accessing image directory: {str(e)}")
        return

    # Session state management
    if 'current_idx' not in st.session_state:
        st.session_state.current_idx = 0

    # Sidebar controls
    with st.sidebar:
        st.header("Navigation")
        
        if st.button("🎲 Random Image"):
            st.session_state.current_idx = random.randint(0, len(all_images)-1)
        
        selected_idx = st.slider(
            "Image Index", 
            0, len(all_images)-1, 
            st.session_state.current_idx
        )
        
        selected_file = st.selectbox(
            "Search by Filename",
            options=all_images,
            index=st.session_state.current_idx
        )

    # Update index based on selection
    if selected_idx != st.session_state.current_idx:
        st.session_state.current_idx = selected_idx
    elif selected_file != all_images[st.session_state.current_idx]:
        st.session_state.current_idx = all_images.index(selected_file)

    # Main display
    current_image = all_images[st.session_state.current_idx]
    img_path = os.path.join(IMAGE_BASE_PATH, current_image)
    
    try:
        img = Image.open(img_path)
        col1, col2 = st.columns([3, 1])
        
        with col1:
            st.image(img, use_container_width=True, caption=f"Selected Image: {current_image}")
        
        with col2:
            st.subheader("Image Info")
            st.write(f"**Filename:** {current_image}")
            st.write(f"**Dimensions:** {img.size[0]}x{img.size[1]}")
            st.write(f"**Format:** {img.format}")
            st.write(f"**Mode:** {img.mode}")

            # Display labels for training mode
            if mode == "Train" and labels_df is not None:
                label = labels_df[labels_df['filename'] == current_image]['label'].values
                if len(label) > 0:
                    st.subheader("Dataset Label")
                    label_text = "AI Generated" if label[0] == 1 else "Human Created"
                    st.info(f"**{label_text}**")
                else:
                    st.warning("Label not found for this image")

    except Exception as e:
        st.error(f"Error loading image: {str(e)}")
        st.write(f"Problematic path: {img_path}")

    # Dataset statistics
    with st.sidebar:
        st.markdown("---")
        st.subheader("Dataset Stats")
        st.write(f"**Total Images:** {len(all_images)}")
        if mode == "Train" and labels_df is not None:
            human_count = (labels_df['label'] == 0).sum()
            ai_count = (labels_df['label'] == 1).sum()
            st.write(f"**Human Created:** {human_count}")
            st.write(f"**AI Generated:** {ai_count}")

if __name__ == "__main__":
    main()


## The Password to use appears here
!curl ipv4.icanhazip.com


!streamlit run visualizer.py &>./logs.txt & npx localtunnel --port 8501

