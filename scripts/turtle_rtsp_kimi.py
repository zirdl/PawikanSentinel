#!/usr/bin/env python3
"""
Detect turtles in an RTSP stream with an int8 TFLite model
using ai_edge_litert.Interpreter, and save annotated frames.
"""

import argparse, os, cv2, numpy as np, time
from ai_edge_litert.interpreter import Interpreter   # <-- LiteRT

# -----------------------------------------------------------
# 1. CLI
# -----------------------------------------------------------
parser = argparse.ArgumentParser()
parser.add_argument('--model',   required=True, help='tflite file')
parser.add_argument('--rtsp',    required=True, help='rtsp url')
parser.add_argument('--out_dir', required=True, help='folder to save images')
parser.add_argument('--score',   type=float, default=0.4)
parser.add_argument('--nms',     type=float, default=0.5)
args = parser.parse_args()
os.makedirs(args.out_dir, exist_ok=True)

# -----------------------------------------------------------
# 2. LiteRT Interpreter
# -----------------------------------------------------------
interpreter = Interpreter(model_path=args.model)
interpreter.allocate_tensors()

in_details  = interpreter.get_input_details()[0]
out_details = interpreter.get_output_details()[0]

# Quantization params
scale_in  = in_details['quantization_parameters']['scales'][0]
zp_in     = in_details['quantization_parameters']['zero_points'][0]
scale_out = out_details['quantization_parameters']['scales'][0]
zp_out    = out_details['quantization_parameters']['zero_points'][0]

# -----------------------------------------------------------
# 3. Decode helper
# -----------------------------------------------------------
def decode(raw, score_thr, nms_thr):
    """
    raw shape: [1, 5, 8400] int8 -> list of (x1,y1,x2,y2,score)
    """
    raw = raw.astype(np.float32)
    raw = (raw - zp_out) * scale_out
    raw = raw.reshape(5, 8400)

    boxes = raw[:4].T                       # [8400,4]  cx,cy,w,h
    scores = raw[4]                         # [8400]

    cx, cy, w, h = boxes[:, 0], boxes[:, 1], boxes[:, 2], boxes[:, 3]
    x1, y1, x2, y2 = cx - w/2, cy - h/2, cx + w/2, cy + h/2
    boxes = np.stack([x1, y1, x2, y2], axis=1)

    keep = scores > score_thr
    boxes, scores = boxes[keep], scores[keep]

    idxs = cv2.dnn.NMSBoxes(
        bboxes=boxes.tolist(),
        scores=scores.tolist(),
        score_threshold=score_thr,
        nms_threshold=nms_thr
    )
    if len(idxs) == 0:
        return [], []
    return boxes[idxs.flatten()], scores[idxs.flatten()]

# -----------------------------------------------------------
# 4. RTSP loop
# -----------------------------------------------------------
cap = cv2.VideoCapture(args.rtsp, cv2.CAP_FFMPEG)
if not cap.isOpened():
    raise IOError(f"Cannot open RTSP {args.rtsp}")

frame_id = 0
while True:
    ok, frame = cap.read()
    if not ok:
        print("Reconnecting ...")
        time.sleep(1)
        continue

    img = cv2.resize(frame, (640, 640))
    img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    img_int8 = (img_rgb / 255.0 / scale_in + zp_in).astype(np.int8)

    interpreter.set_tensor(in_details['index'], img_int8[np.newaxis])
    interpreter.invoke()
    raw = interpreter.get_tensor(out_details['index'])  # [1,5,8400]

    boxes, scores = decode(raw[0], args.score, args.nms)

    h0, w0 = frame.shape[:2]
    sx, sy = w0 / 640, h0 / 640
    for (x1, y1, x2, y2), sc in zip(boxes, scores):
        x1, y1, x2, y2 = int(x1*sx), int(y1*sy), int(x2*sx), int(y2*sy)
        cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
        cv2.putText(frame, f"turtle {sc:.2f}", (x1, y1-5),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)

    out_path = os.path.join(args.out_dir, f"{frame_id:08d}.jpg")
    cv2.imwrite(out_path, frame)
    print(f"Saved {out_path} ({len(boxes)} turtles)")
    frame_id += 1
