import streamlit as st
from streamlit_webrtc import webrtc_streamer, VideoProcessorBase
import numpy as np
import cv2
from sahi import AutoDetectionModel
from sahi.predict import get_prediction
from sahi.utils.cv import visualize_object_predictions
import av

# YOLOv8の推論関数
class YOLOv8Processor(VideoProcessorBase):
    def __init__(self):
        self.model = AutoDetectionModel.from_pretrained(
            model_type='yolov8', 
            model_path='models/best.pt',
            confidence_threshold=0.75,  # 一致率がどの程度まで表示するか
            device='cpu'  # GPUを使用しない場合は'cpu'に変更
        )

    def recv(self, frame):
        img = frame.to_ndarray(format="bgr24")

        # YOLOv8を用いて検出を行う
        result = get_prediction(img, self.model)
        result_image = visualize_object_predictions(result, img)

        return av.VideoFrame.from_ndarray(result_image, format="bgr24")

st.title('大熊老師探しリアルタイム検出')
st.subheader('Webカメラを使ってリアルタイムで大熊老師を検出します。')

webrtc_ctx = webrtc_streamer(
    key="example",
    video_processor_factory=YOLOv8Processor,
    media_stream_constraints={"video": True, "audio": False},
    async_processing=True,
)