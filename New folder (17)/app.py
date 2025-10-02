from flask import Flask, render_template, request, jsonify
import base64, io, socket
from PIL import Image
import numpy as np
import mediapipe as mp
import tensorflow as tf
import joblib

app = Flask(__name__)

# โหลดโมเดล + labels
model = tf.keras.models.load_model("model.h5")
labels = joblib.load("labels.pkl")

SEQ_LEN = 30   # ความยาว sequence ที่โมเดลเทรน
FEAT_DIM = 63  # 21 landmark × (x,y,z)

mp_hands = mp.solutions.hands

def extract_landmarks(image_np):
    """ดึง landmarks จากภาพ (np.array RGB)"""
    with mp_hands.Hands(
        static_image_mode=True,
        max_num_hands=1,
        min_detection_confidence=0.5
    ) as hands:
        results = hands.process(image_np)
        if results.multi_hand_landmarks:
            coords = []
            for lm in results.multi_hand_landmarks[0].landmark:
                coords.extend([lm.x, lm.y, lm.z])
            return np.array(coords)
    return None

@app.route("/")
def index():
    return render_template("index.html")

@app.route("/predict_video", methods=["POST"])
def predict_video():
    frames_b64 = request.json.get("frames", [])
    seq = []

    for fb64 in frames_b64:
        if "," in fb64:
            fb64 = fb64.split(",")[1]
        img_bytes = base64.b64decode(fb64)
        pil_image = Image.open(io.BytesIO(img_bytes)).convert("RGB")
        np_img = np.array(pil_image)
        coords = extract_landmarks(np_img)
        if coords is not None:
            seq.append(coords)

    if len(seq) == 0:
        return jsonify({"success": False, "error": "no_hand_detected"})

    # ปรับ sequence ให้ตรงกับ SEQ_LEN
    seq = np.array(seq)
    if seq.shape[0] < SEQ_LEN:
        pad = np.zeros((SEQ_LEN - seq.shape[0], FEAT_DIM))
        seq = np.vstack([seq, pad])
    elif seq.shape[0] > SEQ_LEN:
        seq = seq[:SEQ_LEN]

    X = seq.reshape(1, SEQ_LEN, FEAT_DIM)
    preds = model.predict(X)
    idx = np.argmax(preds)
    return jsonify({
        "success": True,
        "label": str(labels[idx]),
        "confidence": float(preds[0][idx])
    })

if __name__ == "__main__":
    # หา IP เครื่องใน network
    hostname = socket.gethostname()
    local_ip = socket.gethostbyname(hostname)
    port = 5000

    print("\n🚀 Server started!")
    print(f"👉 Local:   http://127.0.0.1:{port}")
    print(f"👉 Network: http://{local_ip}:{port}\n")

    app.run(host="0.0.0.0", port=port, debug=True)
