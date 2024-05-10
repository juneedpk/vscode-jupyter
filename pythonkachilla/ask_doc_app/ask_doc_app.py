import streamlit as st
import cv2
import tempfile
import os

st.title("Parking Space Detection")

# Select camera or uploaded video
source = st.sidebar.selectbox("Select source", ["Upload Video", "Webcam"])

cap = None

if source == "Webcam":
    try:
        # Use webcam
        cap = cv2.VideoCapture(0)
        if not cap.isOpened():
            st.sidebar.error("Could not open video device")
    except Exception as e:
        st.sidebar.error(f"Error accessing webcam: {e}")
elif source == "Upload Video":
    # Upload video file
    uploaded_file = st.sidebar.file_uploader("Upload video file (MP4)", type=["mp4"])
    if uploaded_file is not None:
        # Create a temporary directory
        temp_dir = tempfile.mkdtemp()
        # Save the uploaded file to the temporary directory
        temp_file_path = os.path.join(temp_dir, "uploaded_video.mp4")
        with open(temp_file_path, "wb") as f:
            f.write(uploaded_file.getbuffer())
        # Read the uploaded video
        cap = cv2.VideoCapture(temp_file_path)
        if not cap.isOpened():
            st.sidebar.error("Could not open uploaded video file. Please check if the file format is correct.")

if cap is not None:
    while True:
        ret, frame = cap.read()
        if not ret:
            st.warning("Unable to read frame. Please make sure the video source is accessible.")
            break
        else:
            # Convert the frame from OpenCV's BGR format to RGB format
            frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            # Resize the frame to fit the window
            frame_resized = cv2.resize(frame_rgb, (640, 480))
            # Display the frame
            st.image(frame_resized, caption="Live", use_column_width=True)
            
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break

    # Release the video capture object and close OpenCV windows
    cap.release()

    cv2.destroyAllWindows()






