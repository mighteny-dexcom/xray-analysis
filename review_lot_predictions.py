# review_lot_predictions.py
# Lightweight UI to review model predictions against X-ray images

import streamlit as st
import pandas as pd
from pathlib import Path
from PIL import Image, ImageDraw

# ============================================================
# CONFIG DEFAULTS
# ============================================================

DEFAULT_IMAGE_DIR = "lot4_images"
ROI_BOX = (450, 350, 650, 500)  # x1, y1, x2, y2

st.set_page_config(
    page_title="Battery Tab X-ray Review",
    layout="wide"
)

# ============================================================
# HELPERS
# ============================================================

@st.cache_data
def load_predictions(uploaded_csv):
    df = pd.read_csv(uploaded_csv)

    required_cols = {"filename", "probability", "predicted_label"}
    missing = required_cols - set(df.columns)

    if missing:
        raise ValueError(f"Missing required columns: {missing}")

    df = df.copy()
    df["probability"] = pd.to_numeric(df["probability"], errors="coerce")
    df = df.sort_values("probability", ascending=False).reset_index(drop=True)

    return df


def find_image_path(image_dir, filename):
    image_path = Path(image_dir) / filename
    return image_path if image_path.exists() else None


def load_image(image_path):
    # Only loads the selected image, not the whole folder
    return Image.open(image_path).convert("RGB")


def draw_roi_on_image(img, roi_box):
    img_with_roi = img.copy()
    draw = ImageDraw.Draw(img_with_roi)

    x1, y1, x2, y2 = roi_box

    draw.rectangle(
        [x1, y1, x2, y2],
        outline="yellow",
        width=4
    )

    return img_with_roi


def crop_roi(img, roi_box):
    return img.crop(roi_box)


def init_review_state(df):
    if "review_table" not in st.session_state:
        review_df = df.copy()
        review_df["review_label"] = ""
        review_df["review_notes"] = ""
        review_df["reviewed"] = False
        st.session_state.review_table = review_df

    if "current_index" not in st.session_state:
        st.session_state.current_index = 0


def clamp_current_index(num_items):
    if num_items <= 0:
        st.session_state.current_index = 0
    else:
        st.session_state.current_index = max(
            0,
            min(st.session_state.current_index, num_items - 1)
        )


# ============================================================
# SIDEBAR
# ============================================================

st.sidebar.header("Input Data")

uploaded_csv = st.sidebar.file_uploader(
    "Upload prediction CSV",
    type=["csv"]
)

image_dir = st.sidebar.text_input(
    "Image folder path",
    value=DEFAULT_IMAGE_DIR
)

st.sidebar.header("Review Settings")

threshold = st.sidebar.slider(
    "Flag threshold",
    min_value=0.0,
    max_value=1.0,
    value=0.40,
    step=0.01
)

filter_mode = st.sidebar.selectbox(
    "Image filter",
    [
        "Flagged only",
        "All images",
        "Unreviewed only",
        "Reviewed only",
        "Likely good only"
    ]
)

show_roi = st.sidebar.checkbox(
    "Show ROI overlay",
    value=True
)

show_crop = st.sidebar.checkbox(
    "Show ROI crop",
    value=True
)

# ============================================================
# LOAD DATA
# ============================================================

st.title("Battery Tab X-ray Review")

if uploaded_csv is None:
    st.warning("Please upload a prediction CSV to begin.")
    st.stop()

try:
    df = load_predictions(uploaded_csv)
except Exception as e:
    st.error(f"Could not load prediction CSV: {e}")
    st.stop()

init_review_state(df)
review_df = st.session_state.review_table

# ============================================================
# FILTERING
# ============================================================

if filter_mode == "Flagged only":
    display_df = review_df[review_df["probability"] >= threshold]
elif filter_mode == "Unreviewed only":
    display_df = review_df[review_df["reviewed"] == False]
elif filter_mode == "Reviewed only":
    display_df = review_df[review_df["reviewed"] == True]
elif filter_mode == "Likely good only":
    display_df = review_df[review_df["probability"] < threshold]
else:
    display_df = review_df

display_df = display_df.sort_values("probability", ascending=False)

# ============================================================
# SUMMARY METRICS
# ============================================================

total_images = len(review_df)
flagged_images = int((review_df["probability"] >= threshold).sum())
reviewed_images = int(review_df["reviewed"].sum())
remaining_images = total_images - reviewed_images

col1, col2, col3, col4 = st.columns(4)

col1.metric("Total images", total_images)
col2.metric("Flagged images", flagged_images)
col3.metric("Reviewed", reviewed_images)
col4.metric("Remaining", remaining_images)

st.divider()

# ============================================================
# MAIN LAYOUT
# ============================================================

left_col, right_col = st.columns([1, 2])

with left_col:
    st.subheader("Image List")

    search_text = st.text_input("Search filename")

    if search_text:
        display_df = display_df[
            display_df["filename"].str.contains(search_text, case=False, na=False)
        ]

    if display_df.empty:
        st.warning("No images match the current filter.")
        st.stop()

    filenames = display_df["filename"].tolist()

    clamp_current_index(len(filenames))

    nav_col1, nav_col2 = st.columns(2)

    with nav_col1:
        if st.button("◀ Previous", use_container_width=True):
            st.session_state.current_index -= 1
            clamp_current_index(len(filenames))
            st.rerun()

    with nav_col2:
        if st.button("Next ▶", use_container_width=True):
            st.session_state.current_index += 1
            clamp_current_index(len(filenames))
            st.rerun()

    selected_filename = filenames[st.session_state.current_index]

    st.write(
        f"Viewing {st.session_state.current_index + 1} of {len(filenames)}"
    )

    st.dataframe(
        display_df[["filename", "probability", "predicted_label", "reviewed"]],
        use_container_width=True,
        hide_index=True
    )

    selected_idx = review_df.index[
        review_df["filename"] == selected_filename
    ][0]

    selected_row = review_df.loc[selected_idx]

    st.write("Selected prediction")
    st.dataframe(
        selected_row[
            ["filename", "probability", "predicted_label", "reviewed"]
        ].to_frame().T,
        use_container_width=True,
        hide_index=True
    )

with right_col:
    st.subheader("Image Review")

    image_path = find_image_path(image_dir, selected_filename)

    if image_path is None:
        st.error(f"Image not found: {Path(image_dir) / selected_filename}")
        st.stop()

    img = load_image(image_path)

    view_img = draw_roi_on_image(img, ROI_BOX) if show_roi else img

    image_col, crop_col = st.columns([2, 1])

    with image_col:
        st.image(
            view_img,
            caption=f"Full image: {selected_filename}",
            use_container_width=True
        )

    with crop_col:
        if show_crop:
            roi_img = crop_roi(img, ROI_BOX)
            st.image(
                roi_img,
                caption="ROI crop used by model",
                use_container_width=True
            )

        st.write(f"Model probability: `{selected_row['probability']:.4f}`")

        if selected_row["probability"] >= threshold:
            st.warning("Flagged by model")
        else:
            st.success("Below threshold")

        label_options = [
            "",
            "battery_tab_issue",
            "good",
            "uncertain",
            "other_issue"
        ]

        current_label = selected_row["review_label"]
        if current_label not in label_options:
            current_label = ""

        review_label = st.selectbox(
            "Reviewer label",
            label_options,
            index=label_options.index(current_label),
            key=f"review_label_{selected_filename}"
        )

        review_notes = st.text_area(
            "Review notes",
            value=selected_row["review_notes"],
            height=120,
            key=f"review_notes_{selected_filename}"
        )

        mark_reviewed = st.checkbox(
            "Mark as reviewed",
            value=bool(selected_row["reviewed"]),
            key=f"reviewed_{selected_filename}"
        )

        if st.button("Save review", use_container_width=True):
            st.session_state.review_table.loc[selected_idx, "review_label"] = review_label
            st.session_state.review_table.loc[selected_idx, "review_notes"] = review_notes
            st.session_state.review_table.loc[selected_idx, "reviewed"] = mark_reviewed

            st.success("Review saved.")

# ============================================================
# EXPORT
# ============================================================

st.divider()

st.subheader("Export Reviewed Results")

export_df = st.session_state.review_table.copy()

csv_bytes = export_df.to_csv(index=False).encode("utf-8")

st.download_button(
    label="Download reviewed CSV",
    data=csv_bytes,
    file_name="lot_predictions_reviewed.csv",
    mime="text/csv"
)