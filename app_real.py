import streamlit as st
from streamlit_webrtc import webrtc_streamer, VideoProcessorBase, WebRtcMode, RTCConfiguration
import numpy as np
import cv2
import av
from ultralytics import YOLO
import threading
import time
import base64
# YOLOv8の検出器を措定
model = YOLO('models/best.pt')
# 検出フラグと時間を管理するための変数とロック
lock = threading.Lock()
img_container = {"img": "FALSE"}
old_time =0

def callback(frame):
    img = frame.to_ndarray(format="bgr24")

    # YOLOv8を用いて検出を行う しきい値は0.6以上を指定
    results = model.predict(img,conf=0.6)
    # 検出結果の画像を描画
    for result in results:
            for box in result.boxes:
                x1, y1, x2, y2 = map(int, box.xyxy[0])
                conf = box.conf[0]
                cls = int(box.cls[0])
                label = f'{model.names[cls]} {conf:.2f}'
                img = cv2.rectangle(img, (x1, y1), (x2, y2), (255, 0, 0), 2)
                img = cv2.putText(img, label, (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 0, 0), 2)
                #コールバックの外にデータを持っていく
                with lock:
                    img_container["img"] = "TRUE"

    return av.VideoFrame.from_ndarray(img, format="bgr24")

st.title('大熊老師探しリアルタイム検出')
st.subheader('Webカメラを使ってリアルタイムで大熊老師を検出します。')
#一応つけた
bonabu = st.button('ボナヴ～!')


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

def talk(path):
    audio_placeholder = st.empty()

    file_ = open(path, "rb")
    contents = file_.read()
    file_.close()

    audio_str = "data:audio/ogg;base64,%s"%(base64.b64encode(contents).decode())
    audio_html = """
                    <audio autoplay=True>
                    <source src="%s" type="audio/ogg" autoplay=True>
                    Your browser does not support the audio element.
                    </audio>
                """ %audio_str

    audio_placeholder.empty()
    time.sleep(0.5) #これがないと上手く再生されません
    audio_placeholder.markdown(audio_html, unsafe_allow_html=True)

if bonabu:

    audio_path1 = 'bonavu.mp3' #入力する音声ファイル
    talk(audio_path1)

#音声出力
while webrtc_ctx.state.playing:
    time.sleep(2)
    genzai_time =time.time()
    with lock:
        if img_container["img"] =="TRUE" and genzai_time-old_time>5 : #前の処理から５秒以上経過の場合
            audio_path1 = 'bonavu.mp3' #入力する音声ファイル
            talk(audio_path1)
            img_container["img"] ="FALSE"
            old_time=time.time()
            
        else:
            continue
