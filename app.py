
from pathlib import Path
from datetime import datetime
import csv
import io

import numpy as np
import streamlit as st
import tensorflow as tf
from PIL import Image

# --------------------------------------------------
# 1. Project settings
# --------------------------------------------------

ROOT = Path(__file__).resolve().parent
MODEL_PATH = ROOT / "cat_dog_model.keras"
SAMPLE_DIR = ROOT / "sample_images"

IMAGE_SIZE = (160, 160)
ALLOWED_TYPES = {".jpg", ".jpeg", ".png"}
GALLERY_PAGE_SIZE = 24

st.set_page_config(
    page_title="PawVision | Cat & Dog AI",
    page_icon="🐾",
    layout="wide",
)

# --------------------------------------------------
# 2. Load the existing trained model
# --------------------------------------------------

@st.cache_resource
def load_model():
    if not MODEL_PATH.is_file():
        raise FileNotFoundError(
            f"Trained model not found: {MODEL_PATH}. "
            "Place cat_dog_model.keras beside app.py."
        )

    return tf.keras.models.load_model(MODEL_PATH)


# --------------------------------------------------
# 3. Image classification
# --------------------------------------------------

def classify_image(image, model):
    image = image.convert("RGB")
    image = image.resize(IMAGE_SIZE)

    array = np.asarray(image, dtype=np.float32)[None, ...]

    # The existing model includes MobileNetV2 preprocessing.
    dog_probability = float(
        np.clip(model.predict(array, verbose=0)[0][0], 0.0, 1.0)
    )

    cat_probability = 1.0 - dog_probability

    label = "Dog" if dog_probability >= 0.5 else "Cat"
    confidence = (
        dog_probability if label == "Dog" else cat_probability
    )

    return label, confidence, cat_probability, dog_probability


# --------------------------------------------------
# 4. Prediction history and CSV export
# --------------------------------------------------

def add_history(filename, label, confidence, cat_probability, dog_probability):
    if "history" not in st.session_state:
        st.session_state.history = []

    st.session_state.history.insert(
        0,
        {
            "time": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
            "filename": filename,
            "prediction": label,
            "confidence_percent": round(confidence * 100, 2),
            "cat_probability_percent": round(cat_probability * 100, 2),
            "dog_probability_percent": round(dog_probability * 100, 2),
        },
    )


def history_csv(rows):
    output = io.StringIO()

    fields = [
        "time",
        "filename",
        "prediction",
        "confidence_percent",
        "cat_probability_percent",
        "dog_probability_percent",
    ]

    writer = csv.DictWriter(output, fieldnames=fields)
    writer.writeheader()
    writer.writerows(rows)

    return output.getvalue().encode("utf-8-sig")


# --------------------------------------------------
# 5. Session state
# --------------------------------------------------

if "history" not in st.session_state:
    st.session_state.history = []

if "last_result" not in st.session_state:
    st.session_state.last_result = None


# --------------------------------------------------
# 6. Sidebar navigation
# --------------------------------------------------

with st.sidebar:
    st.markdown("# 🐾 PawVision")
    st.caption("Cat vs Dog • Image Classification")

    page = st.radio(
        "Navigation",
        [
            "Dashboard",
            "Prediction History",
            "Sample Gallery",
            "About Project",
        ],
    )

    st.divider()
    st.caption("Powered by TensorFlow and MobileNetV2")


# --------------------------------------------------
# 7. Website header
# --------------------------------------------------

st.markdown(
    """
    <div style="
        padding: 1.4rem 1.6rem;
        border-radius: 18px;
        background: linear-gradient(120deg, #172554, #1d4ed8);
        color: #ffffff;
        margin-bottom: 1rem;
    ">
        <div style="font-size: .78rem; letter-spacing: .12em; opacity: .85">
            AI IMAGE CLASSIFICATION PROJECT
        </div>
        <h1 style="margin: .3rem 0; color: #ffffff">PawVision</h1>
        <p style="margin: 0; opacity: .92">
            An AI tool that predicts whether an image contains a cat or a dog.
        </p>
    </div>
    """,
    unsafe_allow_html=True,
)


# --------------------------------------------------
# 8. Dashboard information
# --------------------------------------------------

rows = st.session_state.history

cat_count = sum(
    1 for row in rows if row["prediction"] == "Cat"
)

dog_count = sum(
    1 for row in rows if row["prediction"] == "Dog"
)


# --------------------------------------------------
# 9. Dashboard
# --------------------------------------------------

if page == "Dashboard":

    st.subheader("Classification Dashboard")

    col1, col2, col3 = st.columns(3)

    col1.metric("Images classified this session", len(rows))
    col2.metric("Cats predicted", cat_count)
    col3.metric("Dogs predicted", dog_count)

    st.divider()

    left, right = st.columns([1.05, 0.95], gap="large")

    with left:
        st.markdown("### Upload an image")

        uploaded = st.file_uploader(
            "Choose a JPG, JPEG, or PNG image",
            type=["jpg", "jpeg", "png"],
            key="dashboard_upload",
        )

        if uploaded is not None:
            try:
                image = Image.open(uploaded).convert("RGB")

                st.image(
                    image,
                    caption=uploaded.name,
                    width="stretch",
                )

                if st.button(
                    "🔍 Predict Image",
                    type="primary",
                    width="stretch",
                ):
                    with st.spinner("Analyzing image with the trained model..."):
                        label, confidence, cat_prob, dog_prob = (
                            classify_image(image, load_model())
                        )

                    add_history(
                        uploaded.name,
                        label,
                        confidence,
                        cat_prob,
                        dog_prob,
                    )

                    st.session_state.last_result = {
                        "filename": uploaded.name,
                        "label": label,
                        "confidence": confidence,
                        "cat_probability": cat_prob,
                        "dog_probability": dog_prob,
                    }

            except Exception as exc:
                st.error(f"Could not read or classify this image: {exc}")

        else:
            st.info(
                "Upload an image to get started, or open Sample Gallery "
                "to classify an included sample."
            )

    with right:
        st.markdown("### Latest prediction")

        result = st.session_state.last_result

        if result:
            emoji = "🐶" if result["label"] == "Dog" else "🐱"

            st.markdown(f"## {emoji} {result['label']}")

            st.metric(
                "Model confidence",
                f"{result['confidence']:.2%}",
            )

            st.markdown("**Cat probability**")
            st.progress(
                result["cat_probability"],
                text=f"{result['cat_probability']:.2%}",
            )

            st.markdown("**Dog probability**")
            st.progress(
                result["dog_probability"],
                text=f"{result['dog_probability']:.2%}",
            )

            st.caption(
                "Probabilities are model estimates, not guarantees of correctness."
            )

        else:
            st.markdown("### 🐾 Ready when you are")
            st.write(
                "Your prediction and probability bars will appear here "
                "after you classify an image."
            )

    if rows:
        st.divider()
        st.markdown("### Recent predictions")
        st.dataframe(rows[:5], hide_index=True, width="stretch")


# --------------------------------------------------
# 10. Prediction History
# --------------------------------------------------

elif page == "Prediction History":

    st.subheader("Prediction History")

    st.write(
        "History is stored in the current browser session. "
        "It may reset when the session is cleared or restarted."
    )

    if rows:
        col1, col2, col3 = st.columns(3)

        col1.metric("Total predictions", len(rows))
        col2.metric("Cats", cat_count)
        col3.metric("Dogs", dog_count)

        st.dataframe(rows, hide_index=True, width="stretch")

        st.download_button(
            "⬇️ Download history as CSV",
            data=history_csv(rows),
            file_name="pawvision_prediction_history.csv",
            mime="text/csv",
        )

        if st.button("Clear prediction history"):
            st.session_state.history = []
            st.session_state.last_result = None
            st.rerun()

    else:
        st.info(
            "No predictions yet. Go to Dashboard and classify an image first."
        )


# --------------------------------------------------
# 11. Sample Gallery
# Searches inside cats, dogs, and nested folders
# --------------------------------------------------

elif page == "Sample Gallery":

    st.subheader("Sample Image Gallery")

    st.write(
        "Browse sample images from the cats and dogs folders, "
        "then classify a selected image."
    )

    SAMPLE_DIR.mkdir(parents=True, exist_ok=True)

    # Search recursively inside sample_images and its subfolders.
    sample_files = sorted(
        (
            path
            for path in SAMPLE_DIR.rglob("*")
            if path.is_file() and path.suffix.lower() in ALLOWED_TYPES
        ),
        key=lambda path: str(path.relative_to(SAMPLE_DIR)).lower(),
    )

    if not sample_files:

        st.info(
            "No sample images found. Add JPG, JPEG, or PNG images "
            "inside sample_images/cats or sample_images/dogs."
        )

    else:

        total_images = len(sample_files)

        total_pages = (
            total_images + GALLERY_PAGE_SIZE - 1
        ) // GALLERY_PAGE_SIZE

        st.caption(
            f"{total_images:,} sample images available"
        )

        # Show only 24 images per gallery page.
        if total_pages > 1:
            page_number = st.number_input(
                "Gallery page",
                min_value=1,
                max_value=total_pages,
                value=1,
                step=1,
            )
        else:
            page_number = 1

        start_index = (page_number - 1) * GALLERY_PAGE_SIZE
        end_index = start_index + GALLERY_PAGE_SIZE

        page_files = sample_files[start_index:end_index]

        st.caption(
            f"Showing images {start_index + 1}–"
            f"{min(end_index, total_images)} of {total_images}"
        )

        # Display thumbnails for the current page.
        columns = st.columns(4)

        for index, path in enumerate(page_files):
            with columns[index % 4]:
                try:
                    with Image.open(path) as thumbnail:
                        st.image(
                            thumbnail.convert("RGB"),
                            width="stretch",
                        )

                    relative_name = path.relative_to(SAMPLE_DIR).as_posix()
                    st.caption(relative_name)

                except Exception:
                    st.caption(f"Could not preview: {path.name}")

        st.divider()
        st.markdown("### Classify a sample image")

        # Use paths relative to sample_images so cats/dogs filenames
        # remain distinct even if they have the same filename.
        selected_options = [
            path.relative_to(SAMPLE_DIR).as_posix()
            for path in page_files
        ]

        selected_name = st.selectbox(
            "Choose an image from this page",
            selected_options,
        )

        selected_path = SAMPLE_DIR / Path(selected_name)

        try:
            sample = Image.open(selected_path).convert("RGB")

            st.image(
                sample,
                caption=selected_name,
                width="stretch",
            )

            if st.button(
                "Classify selected sample",
                type="primary",
            ):
                with st.spinner("Analyzing sample image..."):
                    label, confidence, cat_prob, dog_prob = (
                        classify_image(sample, load_model())
                    )

                add_history(
                    selected_name,
                    label,
                    confidence,
                    cat_prob,
                    dog_prob,
                )

                st.session_state.last_result = {
                    "filename": selected_name,
                    "label": label,
                    "confidence": confidence,
                    "cat_probability": cat_prob,
                    "dog_probability": dog_prob,
                }

                emoji = "🐶" if label == "Dog" else "🐱"

                st.success(
                    f"Prediction: {emoji} {label} — "
                    f"confidence {confidence:.2%}"
                )

                st.write(
                    f"Cat probability: {cat_prob:.2%} | "
                    f"Dog probability: {dog_prob:.2%}"
                )

        except Exception as exc:
            st.error(f"Could not open or classify this sample: {exc}")


# --------------------------------------------------
# 12. About Project
# --------------------------------------------------

else:

    st.subheader("About PawVision")

    st.markdown(
        """
        **Project goal:** Demonstrate binary image classification by
        predicting whether an image contains a cat or a dog.

        **Model:** MobileNetV2 transfer learning with a binary
        classification output layer.

        **Input size:** 160 × 160 pixels.

        **Training:** This website reuses the existing trained model.
        It does not train the model when the website starts.

        **How to use:** Upload a JPG or PNG image, click Predict Image,
        and review the prediction and probability bars.

        **Limitations:** The model only predicts cats or dogs. It may
        misclassify other animals, drawings, unusual images, or pictures
        with unclear subjects. Confidence is not proof of correctness.

        **Technology:** Python, Streamlit, TensorFlow/Keras, NumPy,
        Pillow, and MobileNetV2.
        """
    )

    st.markdown("### Download a prediction report")

    if rows:
        report = (
            "PawVision Cat vs Dog Classification Report\n"
            f"Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n\n"
            f"Total predictions in this session: {len(rows)}\n"
            f"Cats predicted: {cat_count}\n"
            f"Dogs predicted: {dog_count}\n\n"
            "Prediction history is available from the Prediction History page.\n"
            "Model confidence is not a guarantee of correctness.\n"
        )

        st.download_button(
            "Download summary report (.txt)",
            data=report,
            file_name="pawvision_report.txt",
            mime="text/plain",
        )

    else:
        st.info(
            "Classify at least one image to enable the summary report."
        )


# --------------------------------------------------
# 13. Footer
# --------------------------------------------------

st.divider()

st.caption(
    "PawVision student project • Existing trained model reused • "
    "No retraining performed"
)
