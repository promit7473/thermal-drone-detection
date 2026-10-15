#!/usr/bin/env python3
"""
Thermal drone detection — side-by-side raw + heat-map with live YOLO detection.

Left panel : raw thermal feed with YOLO detection boxes.
Right panel: heat-colormap (cold -> hot) with the same boxes and hot-spot contours.

Three models available (switch live with 'm'):
  - yolo11s : YOLO11s fine-tuned on HIT-UAV aerial thermal (best accuracy, ~130ms)
  - yolov8n : YOLOv8n fine-tuned on HIT-UAV aerial thermal (faster, ~56ms)
  - coco11n : YOLO11n COCO pre-trained (close-up / general purpose, ~56ms)

Controls:
  q = quit       h = toggle hot-spot contour
  s = snapshot   c = cycle class filter (All / Person / Person+Vehicle)
  m = switch model
"""

import cv2
import numpy as np
import time
import argparse
import subprocess
import os
from ultralytics import YOLO

WIDTH, HEIGHT = 256, 192
MODELS_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "models")
MODELS = {
    "yolo11s":  os.path.join(MODELS_DIR, "thermal_sentinel_yolo11s.pt"),  # thermal aerial, better accuracy, ~130ms
    "yolov8n":  os.path.join(MODELS_DIR, "hituav_thermal_yolov8n.pt"),     # thermal aerial, faster, ~56ms
    "coco11n":  os.path.join(MODELS_DIR, "yolo11n.pt"),                   # COCO 80-class, close-up/general purpose, ~56ms
}

# class colors — thermal model (capitalized) and COCO (lowercase)
CLASS_COLORS = {
    # thermal model classes
    "Person":       ((0, 255, 255), "Person"),
    "Car":          ((0, 0, 255), "Car"),
    "Bicycle":      ((255, 0, 0), "Bicycle"),
    "OtherVehicle": ((204, 0, 204), "Vehicle"),
    "DontCare":     ((128, 128, 128), "?"),
    # COCO model classes (common ones)
    "person":       ((0, 255, 255), "Person"),
    "car":          ((0, 0, 255), "Car"),
    "truck":        ((0, 0, 200), "Truck"),
    "bus":          ((0, 0, 180), "Bus"),
    "bicycle":      ((255, 0, 0), "Bicycle"),
    "motorcycle":   ((255, 0, 100), "Moto"),
    "boat":         ((0, 200, 200), "Boat"),
    "airplane":     ((200, 200, 0), "Plane"),
}

# class filter presets (match both thermal and COCO naming)
FILTER_MODES = ["ALL", "PERSON", "PERSON+VEHICLE"]
FILTER_CLASSES = {
    "ALL":            None,
    "PERSON":         ["Person", "person"],
    "PERSON+VEHICLE": ["Person", "person", "Car", "car", "OtherVehicle", "truck", "bus", "boat"],
}


def find_thermal_camera():
    """Auto-detect the thermal cam by looking for 'Camera: Camera' in v4l2 list."""
    try:
        out = subprocess.check_output(
            ["v4l2-ctl", "--list-devices"], stderr=subprocess.STDOUT
        ).decode()
    except Exception:
        return None

    lines = out.strip().splitlines()
    for i, line in enumerate(lines):
        if "Camera:" in line and "Integrated" not in line:
            # next non-empty line(s) starting with tab have /dev/videoN
            for j in range(i + 1, len(lines)):
                l = lines[j].strip()
                if l.startswith("/dev/video"):
                    return l
                if l and not l.startswith("/dev/video"):
                    break
    return None


def draw_boxes(img, results, allowed, max_area):
    n = 0
    for r in results:
        for box in r.boxes:
            cls = r.names[int(box.cls)]
            if cls == "DontCare":
                continue
            if allowed is not None and cls not in allowed:
                continue
            x1, y1, x2, y2 = map(int, box.xyxy[0])
            bw, bh = x2 - x1, y2 - y1
            area = bw * bh
            # suppress boxes that are too large (close-up false positives like nose=car)
            if area > max_area:
                continue
            color, label = CLASS_COLORS.get(cls, ((0, 255, 0), cls))
            cv2.rectangle(img, (x1, y1), (x2, y2), color, 2)
            txt = f"{label} {float(box.conf):.2f}"
            (tw, th), _ = cv2.getTextSize(txt, cv2.FONT_HERSHEY_SIMPLEX, 0.4, 1)
            cv2.rectangle(img, (x1, y1 - th - 4), (x1 + tw, y1), color, -1)
            cv2.putText(img, txt, (x1, y1 - 2),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.4, (0, 0, 0), 1)
            n += 1
    return n


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--conf", type=float, default=0.35,
                    help="detection confidence threshold (raise to reduce false positives)")
    ap.add_argument("--device", default=None,
                    help="video device (auto-detected if omitted)")
    ap.add_argument("--model", choices=list(MODELS.keys()), default="yolo11s",
                    help="thermal model: yolo11s (better, ~130ms) or yolov8n (faster, ~56ms)")
    ap.add_argument("--imgsz", type=int, default=640)
    ap.add_argument("--max-area", type=float, default=0.6,
                    help="max box area as fraction of frame (0.6 = 60%%) to reject close-up FPs")
    ap.add_argument("--hot-pct", type=float, default=90,
                    help="hot-pixel percentile threshold (default 90 = top 10%%)")
    ap.add_argument("--colormap", type=str, default="inferno",
                    choices=["jet", "inferno", "hot", "turbo", "magma", "viridis"],
                    help="heatmap colormap: inferno (classic thermal), jet, turbo, hot, magma, viridis")
    args = ap.parse_args()

    # map string to cv2 colormap constant
    CMAP = {
        "jet":     cv2.COLORMAP_JET,
        "inferno": cv2.COLORMAP_INFERNO,
        "hot":     cv2.COLORMAP_HOT,
        "turbo":   cv2.COLORMAP_TURBO,
        "magma":   cv2.COLORMAP_MAGMA,
        "viridis": cv2.COLORMAP_VIRIDIS,
    }
    args.colormap = CMAP[args.colormap]

    # auto-detect camera
    dev = args.device or find_thermal_camera()
    if not dev:
        print("ERROR: thermal camera not found. Pass --device /dev/videoN manually.")
        return
    print(f"Using device: {dev}")

    model = YOLO(MODELS[args.model])
    print(f"Model: {args.model} ({MODELS[args.model]})")
    cap = cv2.VideoCapture(dev, cv2.CAP_V4L2)
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, WIDTH)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, HEIGHT)
    cap.set(cv2.CAP_PROP_FOURCC, cv2.VideoWriter_fourcc(*"YUYV"))
    if not cap.isOpened():
        print(f"ERROR: cannot open {dev}")
        return

    # warm up (thermal cam needs ~30+ frames before producing real data)
    print("Warming up camera...")
    for _ in range(40):
        cap.read()

    frame_area = WIDTH * HEIGHT
    max_area_px = args.max_area * frame_area

    show_hot = True
    filter_idx = 1  # start on PERSON filter
    model_key = args.model
    fps_t = time.time()
    fps_n = 0
    fps = 0.0

    print("Running... press 'q' quit, 'h' hot, 's' snap, 'c' filter, 'm' switch model")

    while True:
        ok, frame = cap.read()
        if not ok:
            continue

        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        # normalize to full 0-255 so colormap uses its full range (cold=blue, hot=red)
        gray_norm = cv2.normalize(gray, None, 0, 255, cv2.NORM_MINMAX)
        heat = cv2.applyColorMap(gray_norm, args.colormap)

        results = model(frame, conf=args.conf, imgsz=args.imgsz, verbose=False)

        raw_view = frame.copy()
        heat_view = heat.copy()

        if show_hot:
            # draw contour outline around hottest pixels instead of overwriting with white
            thresh = np.percentile(gray, args.hot_pct)
            hot_mask = cv2.threshold(gray, int(thresh), 255, cv2.THRESH_BINARY)[1]
            hot_mask = cv2.dilate(hot_mask, np.ones((3, 3), np.uint8))
            contours, _ = cv2.findContours(hot_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            cv2.drawContours(heat_view, contours, -1, (255, 255, 255), 1)

        mode = FILTER_MODES[filter_idx]
        allowed = FILTER_CLASSES[mode]
        # COCO model handles close-up objects, so don't apply area filter
        eff_max_area = max_area_px if model_key != "coco11n" else float("inf")
        n_det = draw_boxes(raw_view, results, allowed, eff_max_area)
        draw_boxes(heat_view, results, allowed, eff_max_area)

        # fps
        fps_n += 1
        if time.time() - fps_t >= 1.0:
            fps = fps_n / (time.time() - fps_t)
            fps_t, fps_n = time.time(), 0

        # titles + stats
        cv2.rectangle(raw_view, (0, 0), (WIDTH, 16), (0, 0, 0), -1)
        cv2.putText(raw_view, "RAW THERMAL", (4, 12),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.4, (255, 255, 255), 1)
        cv2.rectangle(heat_view, (0, 0), (WIDTH, 16), (0, 0, 0), -1)
        cv2.putText(heat_view, "HEAT MAP", (4, 12),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.4, (255, 255, 255), 1)

        combined = np.hstack((raw_view, heat_view))
        cv2.rectangle(combined, (0, HEIGHT), (WIDTH * 2, HEIGHT + 18), (0, 0, 0), -1)
        info = f"FPS:{fps:4.1f} det:{n_det} {model_key} {mode} conf:{args.conf} (q h s c m)"
        cv2.putText(combined, info, (4, HEIGHT + 13),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.4, (0, 255, 0), 1)

        cv2.imshow("Thermal Drone Detection", combined)
        key = cv2.waitKey(1) & 0xFF
        if key == ord("q"):
            break
        elif key == ord("h"):
            show_hot = not show_hot
        elif key == ord("c"):
            filter_idx = (filter_idx + 1) % len(FILTER_MODES)
            print("class filter:", FILTER_MODES[filter_idx])
        elif key == ord("m"):
            keys = list(MODELS.keys())
            idx = keys.index(model_key)
            model_key = keys[(idx + 1) % len(keys)]
            model = YOLO(MODELS[model_key])
            print("switched model:", model_key)
        elif key == ord("s"):
            fn = f"snap_{int(time.time())}.png"
            cv2.imwrite(fn, combined)
            print("saved", fn)

    cap.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
