import streamlit as st
import numpy as np
from PIL import Image
import tensorflow as tf

# ===== Config =====
MODEL_PATH = "mnist_cnn.h5"     # change to your file, e.g., "emotion_model.h5"
TARGET_SIZE = (28, 28)          # change to your model’s expected (H, W)
GRAYSCALE = True                # set False if your model expects color (RGB)

@st.cache_resource
def load_model():
    # Works for .h5 and SavedModel dirs
    return tf.keras.models.load_model(MODEL_PATH)

def preprocess(img: Image.Image) -> np.ndarray:
    # Convert to desired mode/size
    if GRAYSCALE:
        img = img.convert("L")
    else:
        img = img.convert("RGB")
    img = img.resize(TARGET_SIZE)

    arr = np.array(img, dtype=np.float32) / 255.0

    # Add channel dim for grayscale
    if GRAYSCALE and arr.ndim == 2:
        arr = np.expand_dims(arr, axis=-1)

    # Add batch dimension
    arr = np.expand_dims(arr, axis=0)
    return arr

def main():
    st.title("TensorFlow Model • Streamlit Demo")
    st.write("Upload an image and I’ll run it through the model.")

    uploaded = st.file_uploader("Choose an image", type=["png","jpg","jpeg"])
    if uploaded is not None:
        image = Image.open(uploaded)
        st.image(image, caption="Uploaded image", use_column_width=True)

        if st.button("Predict"):
            model = load_model()
            x = preprocess(image)
            preds = model.predict(x)
            if preds.ndim == 2 and preds.shape[-1] > 1:
                # classification logits/probs
                cls = int(np.argmax(preds, axis=1)[0])
                prob = float(np.max(preds, axis=1)[0])
                st.success(f"Predicted class: {cls}  |  confidence: {prob:.3f}")
                st.write("Raw output:", preds.tolist())
            else:
                # regression or single-logit
                st.info(f"Model output: {preds.squeeze().tolist()}")

if __name__ == "__main__":
    main()
