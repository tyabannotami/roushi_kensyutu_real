import streamlit as st
from streamlit_webrtc import webrtc_streamer, VideoProcessorBase, WebRtcMode, RTCConfiguration
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
            confidence_threshold=0.5,  # 一致率がどの程度まで表示するか
            device='cpu'  # GPUを使用しない場合は'cpu'に変更
        )

    def recv(self, frame):
        img = frame.to_ndarray(format="bgr24")

        # YOLOv8を用いて検出を行う
        result = get_prediction(img, self.model)
        
        # 検出結果をデバッグログに出力
        st.write(f"検出結果: {result}")

        # 検出結果の画像を描画
        object_predictions = result.object_prediction_list
        result_image = visualize_object_predictions(object_predictions, img)
        
        # 検出結果の画像を描画
        result_image = visualize_object_predictions(result.object_prediction_list, img)  # 修正箇所

        return av.VideoFrame.from_ndarray(result_image, format="bgr24")

st.title('大熊老師探しリアルタイム検出')
st.subheader('Webカメラを使ってリアルタイムで大熊老師を検出します。')


# エラーハンドリングの追加
try:
    webrtc_ctx = webrtc_streamer(
        key="example",
        video_processor_factory=YOLOv8Processor,
        media_stream_constraints={"video": True, "audio": False},
        async_processing=True,
        rtc_configuration={  # この設定を足す
            "iceServers": [{"urls": ["stun:stun.l.google.com:19302"]}]
        }
    )
except Exception as e:
    st.error(f"エラーが発生しました: {e}")