import os
import sys

import requests
import streamlit as st

st.set_page_config(page_title="Autonomous Vehicle Perception", page_icon="🚗", layout="wide")

API_BASE = os.getenv("API_BASE", "http://127.0.0.1:8000")
TIMEOUT = 600


def check_api_health():
    """Verify the API backend is running and accessible."""
    try:
        response = requests.get(f"{API_BASE}/system/status", timeout=5)
        return response.status_code == 200, response.json()
    except requests.exceptions.ConnectionError:
        return False, {"error": "API unreachable"}
    except Exception as e:
        return False, {"error": str(e)}


def main() -> None:
    st.markdown(
        """
        <style>
            .block-container {padding-top: 1rem;}
            .stApp {background: linear-gradient(135deg, #0f172a, #111827); color: white;}
            div.stButton > button {background: linear-gradient(90deg, #2563eb, #38bdf8); color: white;}
        </style>
        """,
        unsafe_allow_html=True,
    )

    st.title("🚗 Autonomous Vehicle Perception")
    st.caption("Real-time object detection and deployment-ready inference")
    
    # Check API connectivity
    with st.spinner("Checking API connection..."):
        api_ok, api_info = check_api_health()
    
    if not api_ok:
        st.error(f"❌ Cannot connect to API at {API_BASE}")
        st.error(f"Error: {api_info.get('error', 'Unknown error')}")
        st.warning("**Setup Instructions:**")
        st.code("""
# Option 1: Run backend locally
cd c:\\Users\\siddr\\OneDrive\\Documents\\In_project
uvicorn backend.main:app --host 0.0.0.0 --port 8000 --reload

# Option 2: Run with Docker
docker-compose up --build
        """, language="bash")
        st.stop()
    
    st.success(f"✅ Connected to API at {API_BASE}")

    st.success(f"✅ Connected to API at {API_BASE}")

    with st.sidebar:
        st.header("Configuration")
        model_choice = st.selectbox("Model", ["yolov8n", "yolov8s", "best", "last"])
        confidence = st.slider("Confidence", 0.05, 0.95, 0.25)
        iou = st.slider("IoU", 0.1, 0.95, 0.45)
        device = st.selectbox("Device", ["cpu", "cuda"])
        uploaded_file = st.file_uploader("Upload image or video", type=["jpg", "jpeg", "png", "mp4", "avi", "mov"])
        run_button = st.button("Run Detection")

    if uploaded_file is not None and run_button:
        try:
            if uploaded_file.type.startswith("image"):
                with st.spinner("Processing image..."):
                    files = {"file": (uploaded_file.name, uploaded_file.getvalue(), uploaded_file.type)}
                    response = requests.post(
                        f"{API_BASE}/predict/image",
                        files=files,
                        params={"confidence": confidence, "iou": iou},
                        timeout=TIMEOUT,
                    )
                    response.raise_for_status()  # Raise HTTPError for bad status codes
                    
                payload = response.json()
                st.image(payload["image_path"], caption="Processed image")
                st.subheader("Detection statistics")
                st.json(payload["stats"])
                st.subheader("Detections")
                st.dataframe(payload["detections"])
            else:
                with st.spinner("Processing video (this may take a while)..."):
                    files = {"file": (uploaded_file.name, uploaded_file.getvalue(), uploaded_file.type)}
                    response = requests.post(
                        f"{API_BASE}/predict/video",
                        files=files,
                        params={"confidence": confidence, "iou": iou},
                        timeout=TIMEOUT,
                    )
                    response.raise_for_status()
                    
                st.video(response.json()["output_path"])
        except requests.exceptions.Timeout:
            st.error(f"Request timeout after {TIMEOUT}s. The file may be too large or processing is slow.")
        except requests.exceptions.ConnectionError:
            st.error(f"Cannot connect to API at {API_BASE}. Ensure the backend is running.")
        except requests.exceptions.HTTPError as e:
            st.error(f"API Error: {e.response.status_code} - {e.response.text}")
        except Exception as e:
            st.error(f"Unexpected error: {str(e)}")
    else:
        st.info("Upload an image or video and click Run Detection to begin inference.")


if __name__ == "__main__":
    main()
