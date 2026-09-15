import streamlit as st
import cv2
import numpy as np
import tempfile
import os
from tensorflow import keras


# =========================================================
# PAGE CONFIG
# =========================================================
st.set_page_config(
    page_title="Driver Behavior AI",
    page_icon="🚗",
    layout="wide"
)


# =========================================================
# CUSTOM CSS
# =========================================================
st.markdown("""
<style>
    /* ================================
       GLOBAL
    ================================= */
    .stApp {
        background:
            radial-gradient(
                circle at 15% 10%,
                rgba(91, 78, 255, 0.16),
                transparent 28%
            ),
            radial-gradient(
                circle at 85% 20%,
                rgba(0, 210, 255, 0.10),
                transparent 25%
            ),
            #070b14;
        color: #f5f7ff;
    }

    /* ================================
       MAIN CONTAINER
    ================================= */
    .block-container {
        max-width: 1200px;
        padding-top: 2.5rem;
        padding-bottom: 3rem;
    }

    /* ================================
       HEADER
    ================================= */
    .hero {
        padding: 2rem 2.2rem;
        border-radius: 24px;
        background:
            linear-gradient(
                135deg,
                rgba(24, 30, 55, 0.96),
                rgba(10, 15, 30, 0.96)
            );
        border: 1px solid rgba(130, 140, 255, 0.20);
        box-shadow:
            0 20px 60px rgba(0, 0, 0, 0.35);
        margin-bottom: 1.8rem;
    }

    .hero-badge {
        display: inline-block;
        padding: 0.35rem 0.8rem;
        border-radius: 999px;
        background: rgba(91, 78, 255, 0.15);
        border: 1px solid rgba(110, 100, 255, 0.30);
        color: #a9a2ff;
        font-size: 0.78rem;
        font-weight: 700;
        letter-spacing: 1.2px;
        text-transform: uppercase;
        margin-bottom: 0.9rem;
    }

    .hero-title {
        font-size: 2.55rem;
        font-weight: 800;
        margin: 0;
        letter-spacing: -1px;
        background: linear-gradient(
            90deg,
            #ffffff,
            #a9a2ff,
            #72dcff
        );
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
    }

    .hero-subtitle {
        color: #a8afc2;
        font-size: 1rem;
        margin-top: 0.6rem;
        line-height: 1.7;
    }

    /* ================================
       SECTION TITLES
    ================================= */
    .section-title {
        font-size: 1.15rem;
        font-weight: 700;
        color: #e9ebff;
        margin-top: 1.4rem;
        margin-bottom: 0.7rem;
    }

    /* ================================
       INFO CARDS
    ================================= */
    .info-card {
        background: rgba(17, 23, 40, 0.88);
        border: 1px solid rgba(130, 140, 255, 0.14);
        border-radius: 18px;
        padding: 1rem 1.2rem;
        min-height: 95px;
        box-shadow:
            0 10px 30px rgba(0, 0, 0, 0.18);
    }

    .info-label {
        color: #858da3;
        font-size: 0.76rem;
        text-transform: uppercase;
        letter-spacing: 1px;
        margin-bottom: 0.35rem;
    }

    .info-value {
        color: #f3f5ff;
        font-size: 1.15rem;
        font-weight: 700;
    }

    .info-accent {
        color: #8f87ff;
    }

    /* ================================
       VIDEO CARD
    ================================= */
    .video-card {
        background: rgba(12, 17, 30, 0.92);
        border: 1px solid rgba(130, 140, 255, 0.15);
        border-radius: 22px;
        padding: 1rem;
        box-shadow:
            0 18px 45px rgba(0, 0, 0, 0.30);
    }

    /* ================================
       UPLOAD AREA
    ================================= */
    [data-testid="stFileUploader"] {
        background: rgba(14, 20, 35, 0.75);
        border: 1px dashed rgba(120, 110, 255, 0.45);
        border-radius: 18px;
        padding: 0.8rem;
    }

    /* ================================
       BUTTON
    ================================= */
    .stButton > button {
        width: 100%;
        border-radius: 13px;
        border: 1px solid rgba(130, 120, 255, 0.35);
        background:
            linear-gradient(
                135deg,
                #635bff,
                #7b61ff
            );
        color: white;
        font-weight: 700;
        padding: 0.7rem 1rem;
        box-shadow:
            0 8px 25px rgba(99, 91, 255, 0.25);
        transition: 0.2s ease;
    }

    .stButton > button:hover {
        transform: translateY(-1px);
        box-shadow:
            0 12px 30px rgba(99, 91, 255, 0.35);
    }

    /* ================================
       STATUS PANEL
    ================================= */
    .status-panel {
        background:
            linear-gradient(
                135deg,
                rgba(20, 27, 48, 0.95),
                rgba(13, 18, 32, 0.95)
            );
        border: 1px solid rgba(130, 140, 255, 0.16);
        border-radius: 18px;
        padding: 1.2rem;
        margin-top: 1rem;
    }

    .status-title {
        color: #858da3;
        font-size: 0.75rem;
        text-transform: uppercase;
        letter-spacing: 1px;
    }

    .status-value {
        color: #ffffff;
        font-size: 1.25rem;
        font-weight: 750;
        margin-top: 0.25rem;
    }

    /* ================================
       FOOTER
    ================================= */
    .footer {
        text-align: center;
        color: #5f6679;
        font-size: 0.78rem;
        margin-top: 2.5rem;
        padding-top: 1.2rem;
        border-top: 1px solid rgba(255,255,255,0.06);
    }
</style>
""", unsafe_allow_html=True)


# =========================================================
# CLASS NAMES
# =========================================================
class_names = {
    "c0": "Safe driving",
    "c1": "Texting - right hand",
    "c2": "Talking on phone - right hand",
    "c3": "Texting - left hand",
    "c4": "Talking on phone - left hand",
    "c5": "Operating radio",
    "c6": "Drinking",
    "c7": "Reaching behind",
    "c8": "Hair and makeup",
    "c9": "Talking to passenger"
}

class_keys = list(class_names.keys())


# =========================================================
# MODEL
# =========================================================
MODEL_PATH = "driver_efficientnet_feature_extraction.keras"


@st.cache_resource
def load_model():
    model = keras.models.load_model(
        MODEL_PATH
    )
    return model


model = load_model()


# =========================================================
# SETTINGS
# =========================================================
CONFIDENCE_THRESHOLD = 0.60
STABLE_FRAMES_REQUIRED = 3

# Process one frame every 3 frames
FRAME_SKIP = 3


# =========================================================
# PREDICTION FUNCTION
# =========================================================
def predict_frame(frame):

    frame_rgb = cv2.cvtColor(
        frame,
        cv2.COLOR_BGR2RGB
    )

    image = cv2.resize(
        frame_rgb,
        (128, 128)
    )

    # Same preprocessing used during evaluation
    image = image.astype("float32")

    image = np.expand_dims(
        image,
        axis=0
    )

    prediction = model.predict(
        image,
        verbose=0
    )

    predicted_index = np.argmax(
        prediction[0]
    )

    confidence = float(
        prediction[0][predicted_index]
    )

    predicted_class = class_keys[
        predicted_index
    ]

    behavior = class_names[
        predicted_class
    ]

    return (
        predicted_class,
        behavior,
        confidence
    )


# =========================================================
# DRAW PREDICTION
# =========================================================
def draw_prediction(
    frame,
    behavior,
    confidence
):

    if confidence < CONFIDENCE_THRESHOLD:

        text = (
            f"UNCERTAIN  |  "
            f"{confidence * 100:.0f}%"
        )

    else:

        text = (
            f"{behavior.upper()}  |  "
            f"{confidence * 100:.0f}%"
        )

    # Prediction panel
    cv2.rectangle(
        frame,
        (15, 15),
        (650, 82),
        (8, 11, 20),
        -1
    )

    # Accent line
    cv2.rectangle(
        frame,
        (15, 15),
        (21, 82),
        (120, 105, 255),
        -1
    )

    # Prediction text
    cv2.putText(
        frame,
        text,
        (35, 59),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.82,
        (255, 255, 255),
        2,
        cv2.LINE_AA
    )

    return frame


# =========================================================
# HERO HEADER
# =========================================================
st.markdown("""
<div class="hero">
    <div class="hero-title">
        Driver Behavior Detection
    </div>
</div>
""", unsafe_allow_html=True)


# =========================================================
# MODEL INFO
# =========================================================
info1, info2, info3, info4 = st.columns(4)

with info1:
    st.markdown("""
    <div class="info-card">
        <div class="info-label">Architecture</div>
        <div class="info-value">EfficientNetB0</div>
    </div>
    """, unsafe_allow_html=True)

with info2:
    st.markdown("""
    <div class="info-card">
        <div class="info-label">Validation Accuracy</div>
        <div class="info-value info-accent">95.5%</div>
    </div>
    """, unsafe_allow_html=True)

with info3:
    st.markdown("""
    <div class="info-card">
        <div class="info-label">Classes</div>
        <div class="info-value">10 Behaviors</div>
    </div>
    """, unsafe_allow_html=True)

with info4:
    st.markdown("""
    <div class="info-card">
        <div class="info-label">Input Size</div>
        <div class="info-value">128 × 128</div>
    </div>
    """, unsafe_allow_html=True)


st.markdown(
    '<div class="section-title">Upload a driving video</div>',
    unsafe_allow_html=True
)


# =========================================================
# VIDEO UPLOAD
# =========================================================
uploaded_video = st.file_uploader(
    "Choose a video",
    type=[
        "mp4",
        "avi",
        "mov",
        "mkv"
    ],
    label_visibility="collapsed"
)


# =========================================================
# PROCESS VIDEO
# =========================================================
if uploaded_video is not None:

    # -----------------------------------------------------
    # Save uploaded video
    # -----------------------------------------------------
    input_file = tempfile.NamedTemporaryFile(
        delete=False,
        suffix=".mp4"
    )

    input_file.write(
        uploaded_video.getvalue()
    )

    input_file.close()

    input_path = input_file.name

    uploaded_video.seek(0)

    # -----------------------------------------------------
    # Original Video
    # -----------------------------------------------------
    st.markdown(
        '<div class="section-title">Original Video</div>',
        unsafe_allow_html=True
    )

    video_col, empty_col = st.columns(
        [2, 1]
    )

    with video_col:

        st.markdown(
            '<div class="video-card">',
            unsafe_allow_html=True
        )

        st.video(
            uploaded_video
        )

        st.markdown(
            '</div>',
            unsafe_allow_html=True
        )

    st.write("")

    # -----------------------------------------------------
    # Analyze button
    # -----------------------------------------------------
    button_col, empty_col = st.columns(
        [1, 2]
    )

    with button_col:

        analyze = st.button(
            "Analyze Video",
            type="primary"
        )

    if analyze:

        cap = cv2.VideoCapture(
            input_path
        )

        if not cap.isOpened():

            st.error(
                "Could not open the uploaded video."
            )

            st.stop()

        # -------------------------------------------------
        # Video information
        # -------------------------------------------------
        fps = cap.get(
            cv2.CAP_PROP_FPS
        )

        width = int(
            cap.get(
                cv2.CAP_PROP_FRAME_WIDTH
            )
        )

        height = int(
            cap.get(
                cv2.CAP_PROP_FRAME_HEIGHT
            )
        )

        total_frames = int(
            cap.get(
                cv2.CAP_PROP_FRAME_COUNT
            )
        )

        if fps <= 0:
            fps = 30

        # -------------------------------------------------
        # Output video
        # -------------------------------------------------
        output_file = tempfile.NamedTemporaryFile(
            delete=False,
            suffix=".mp4"
        )

        output_file.close()

        output_path = output_file.name

        fourcc = cv2.VideoWriter_fourcc(
            *"mp4v"
        )

        writer = cv2.VideoWriter(
            output_path,
            fourcc,
            fps,
            (width, height)
        )

        # -------------------------------------------------
        # Analysis section
        # -------------------------------------------------
        st.markdown(
            '<div class="section-title">AI Analysis</div>',
            unsafe_allow_html=True
        )

        progress_bar = st.progress(
            0
        )

        video_col, info_col = st.columns(
            [2, 1]
        )

        with video_col:

            st.markdown(
                '<div class="video-card">',
                unsafe_allow_html=True
            )

            frame_placeholder = st.empty()

            st.markdown(
                '</div>',
                unsafe_allow_html=True
            )

        with info_col:

            prediction_placeholder = st.empty()
            confidence_placeholder = st.empty()
            frame_placeholder_info = st.empty()

        # -------------------------------------------------
        # Stable prediction state
        # -------------------------------------------------
        current_class = None

        current_behavior = (
            "Analyzing..."
        )

        current_confidence = 0.0

        candidate_class = None

        candidate_count = 0

        # -------------------------------------------------
        # Frame processing
        # -------------------------------------------------
        frame_number = 0

        while True:

            success, frame = cap.read()

            if not success:
                break

            frame_number += 1

            # -------------------------------------------------
            # Prediction
            # -------------------------------------------------
            if (
                frame_number == 1
                or frame_number % FRAME_SKIP == 0
            ):

                (
                    predicted_class,
                    predicted_behavior,
                    predicted_confidence
                ) = predict_frame(
                    frame
                )

                # -------------------------------------------------
                # Confidence filtering
                # -------------------------------------------------
                if predicted_confidence >= CONFIDENCE_THRESHOLD:

                    if predicted_class == current_class:

                        current_confidence = (
                            predicted_confidence
                        )

                        candidate_class = None
                        candidate_count = 0

                    else:

                        if predicted_class == candidate_class:

                            candidate_count += 1

                        else:

                            candidate_class = (
                                predicted_class
                            )

                            candidate_count = 1

                        if candidate_count >= STABLE_FRAMES_REQUIRED:

                            current_class = (
                                predicted_class
                            )

                            current_behavior = (
                                predicted_behavior
                            )

                            current_confidence = (
                                predicted_confidence
                            )

                            candidate_class = None
                            candidate_count = 0

                # -------------------------------------------------
                # First prediction
                # -------------------------------------------------
                if current_class is None:

                    current_class = (
                        predicted_class
                    )

                    current_behavior = (
                        predicted_behavior
                    )

                    current_confidence = (
                        predicted_confidence
                    )

            # -------------------------------------------------
            # Draw prediction
            # -------------------------------------------------
            output_frame = draw_prediction(
                frame,
                current_behavior,
                current_confidence
            )

            # -------------------------------------------------
            # Write frame
            # -------------------------------------------------
            writer.write(
                output_frame
            )

            # -------------------------------------------------
            # Display frame
            # -------------------------------------------------
            frame_rgb = cv2.cvtColor(
                output_frame,
                cv2.COLOR_BGR2RGB
            )

            with video_col:

                frame_placeholder.image(
                    frame_rgb,
                    channels="RGB",
                    width=700
                )

            # -------------------------------------------------
            # Progress
            # -------------------------------------------------
            if total_frames > 0:

                progress = (
                    frame_number /
                    total_frames
                )

                progress_bar.progress(
                    min(progress, 1.0)
                )

            # -------------------------------------------------
            # Information panel
            # -------------------------------------------------
            prediction_placeholder.markdown(
                f"### Current Prediction\n**{current_behavior}**"
            )

            confidence_placeholder.markdown(
                f"### Confidence\n**{current_confidence * 100:.0f}%**"
            )

            frame_placeholder_info.markdown(
                f"### Frame\n**{frame_number} / {total_frames}**"
            )

        # -----------------------------------------------------
        # Release resources
        # -----------------------------------------------------
        cap.release()

        writer.release()

        progress_bar.progress(
            1.0
        )

        st.success(
            "Video analysis completed."
        )

        # -----------------------------------------------------
        # Cleanup
        # -----------------------------------------------------
        try:

            os.remove(
                input_path
            )

            os.remove(
                output_path
            )

        except:

            pass


# =========================================================
# FOOTER
# =========================================================
st.markdown("""
<div class="footer">
    Driver Behavior AI · EfficientNetB0 Transfer Learning ·
    95.5% Validation Accuracy
</div>
""", unsafe_allow_html=True)
