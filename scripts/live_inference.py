"""Real-time webcam / PiCamera Edge inference pipeline."""

from __future__ import annotations

import argparse

import cv2 as cv
import numpy as np

from src.mhi import MHITransformer
from src.utils import FPSMeter, draw_grid, load_config, overlay_info, to_model_input


def main() -> None:
    parser = argparse.ArgumentParser(description="Live MHI-CNN inference")
    parser.add_argument("--config", default="configs/default_config.yaml")
    parser.add_argument("--model", default=None)
    parser.add_argument("--source", default=None, help="Camera index or video path")
    args = parser.parse_args()

    cfg = load_config(args.config)
    cam_cfg = cfg["camera"]
    mhi_cfg = cfg["mhi"]
    source = args.source if args.source is not None else cam_cfg["source"]
    try:
        source = int(source)
    except (ValueError, TypeError):
        pass

    from keras.models import load_model

    model = load_model(args.model or cfg["model"]["weights_path"], compile=False)
    input_shape = tuple(cfg["model"]["input_shape"])
    id_to_name = {v: k for k, v in cfg["gestures"].items()}

    cam = cv.VideoCapture(source)
    cam.set(cv.CAP_PROP_FRAME_WIDTH, cam_cfg["width"])
    cam.set(cv.CAP_PROP_FRAME_HEIGHT, cam_cfg["height"])

    transformer = MHITransformer(tau=mhi_cfg["tau_frames"], threshold=mhi_cfg["threshold"])
    fps_meter = FPSMeter()

    while True:
        ret, frame = cam.read()
        if not ret:
            break
        if cam_cfg.get("mirror", True):
            frame = cv.flip(frame, 1)
        mhi = transformer.update(frame)
        fps = fps_meter.tick()

        preds = model(to_model_input(mhi, input_shape), training=False).numpy()[0]
        label = id_to_name[int(np.argmax(preds))]
        score = float(np.max(preds)) * 100

        vis = cv.resize(mhi, None, fx=mhi_cfg["display_scale"], fy=mhi_cfg["display_scale"])
        cv.imshow("Motion_history_image", vis)
        frame_out = draw_grid(frame, cam_cfg["width"], cam_cfg["height"])
        frame_out = overlay_info(
            frame_out, [f"{label} {score:.0f}%", f"{fps:.1f} FPS", f"thr {mhi_cfg['threshold']}"]
        )
        cv.imshow("Original", frame_out)

        key = cv.waitKey(5) & 0xFF
        if key == 27:
            break

    cam.release()
    cv.destroyAllWindows()


if __name__ == "__main__":
    main()
