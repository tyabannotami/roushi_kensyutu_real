import streamlit as st
from streamlit_webrtc import webrtc_streamer, VideoProcessorBase, WebRtcMode, RTCConfiguration
import numpy as np
import cv2
from sahi import AutoDetectionModel
from sahi.predict import get_prediction
from sahi.utils.cv import visualize_object_predictions
import av
from ultralytics import YOLO
import time
import threading
# YOLOv8の推論関数
model = YOLO('models/best.pt')

# 検出されたかどうかのフラグ
detected_flag = [False]

def callback(frame):
    img = frame.to_ndarray(format="bgr24")

    # YOLOv8を用いて検出を行う
    results = model(img)
    
    kensyutu_flag = False

    # 検出結果の画像を描画
    for result in results:
            for box in result.boxes:
                x1, y1, x2, y2 = map(int, box.xyxy[0])
                conf = box.conf[0]
                cls = int(box.cls[0])
                label = f'{model.names[cls]} {conf:.2f}'
                img = cv2.rectangle(img, (x1, y1), (x2, y2), (255, 0, 0), 2)
                img = cv2.putText(img, label, (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 0, 0), 2)
                kensyutu_flag=True

   # 検出された場合にフラグを設定
    if kensyutu_flag:
        detected_flag[0] = True

    return av.VideoFrame.from_ndarray(img, format="bgr24")


st.title('大熊老師探しリアルタイム検出')
st.subheader('Webカメラを使ってリアルタイムで大熊老師を検出します。')

# 音声ファイルを読み込む
audio_file = open('bonavu.mp3', 'rb').read()

# エラーハンドリングの追加
try:
    webrtc_ctx = webrtc_streamer(
        key="example",
        video_frame_callback=callback,
        media_stream_constraints={"video": True, "audio": False},
        async_processing=True,
        rtc_configuration={  # この設定を足す
            "iceServers": [{"urls": ["stun:stun.l.google.com:19302"]}]
        }
    )
except Exception as e:
    st.error(f"エラーが発生しました: {e}")

# 検出フラグをチェックして音声再生
if detected_flag[0]:
    detected_flag[0] = False
    st.audio(audio_file, format='audio/mp3')