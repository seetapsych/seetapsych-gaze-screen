# -*- coding: utf-8 -*-

import os

import cv2
import numpy
from seetapsych_lib.runtime.factory import Factory
from seetapsych_lib.runtime.pipeline import Pipeline
from seetapsych_lib.runtime.runner import Runner

module_roots = [
    os.path.join(os.path.dirname(__file__), "../seetapsych_gaze_screen/modules"),
    os.path.join(os.path.dirname(__file__), "../../seetapsych-face-hub/seetapsych_face_hub/modules"),
]


def deep_merge(base: dict, override: dict) -> dict:
    """Recursively merge override into base, key by key.

    For nested dicts (TOML sections), merge recursively so that only the
    explicitly set keys in override replace those in base — other keys in
    the same section are left untouched.
    """
    result = base.copy()
    for key, value in override.items():
        if key in result and isinstance(result[key], dict) and isinstance(value, dict):
            result[key] = deep_merge(result[key], value)
        else:
            result[key] = value
    return result


def render(dst: numpy.ndarray, src: numpy.ndarray):
    dst_h, dst_w = dst.shape[:2]
    src_h, src_w = src.shape[:2]

    data_h = min(src_h, dst_h)
    data_w = min(src_w, dst_w)

    to_h = max(0, (dst_h - src_h) // 2)
    to_w = max(0, (dst_w - src_w) // 2)

    from_h = max(0, (src_h - dst_h) // 2)
    from_w = max(0, (src_w - dst_w) // 2)

    dst[to_h : to_h + data_h, to_w : to_w + data_w] = src[from_h : from_h + data_h, from_w : from_w + data_w]

    return dst


def main():
    original_screen_pixel_width = 2256
    original_screen_pixel_height = 1504
    screen_pixel_ratio = 1.5

    screen_pixel_width = int(original_screen_pixel_width / screen_pixel_ratio)
    screen_pixel_height = int(original_screen_pixel_height / screen_pixel_ratio)

    screen_config = {
        "camera": {"screen_x_mm": -140.0, "screen_y_mm": -5.0},
        "screen": {
            "h_px": screen_pixel_height,
            "w_px": screen_pixel_width,
            "h_mm": 185.0,
            "w_mm": 280.0,
        },
    }

    factory = Factory()
    for root in module_roots:
        factory.load_dir_modules(root)

    pipeline = Pipeline(
        factory,
        packages=[
            "ee7b0a24-c4b3-470d-b095-9c606384d06e",  # AFFNet
            # '981181f8-86f6-43c0-afea-8b42fc5beb26', # ITrackerPlus
            # '17c5187c-beb1-4e7d-a0a7-da2638d679bd', # TdGazeNet
        ],
        attributes=[
            # 'face/gaze_screen'
        ],
    )

    print(pipeline.problem())
    pipeline.solve()

    print(pipeline.satisfied())
    pipeline.install_requirements()
    pipeline.cache_models()

    package = pipeline.get_package(provide="face/gaze_screen")
    assert package is not None

    parameter_data = next((p for p in (package.parameters or []) if p.name == "data"), None)
    data_value: dict = parameter_data.value if parameter_data is not None else {}

    data_value = deep_merge(data_value, screen_config)

    pipeline.set_parameters(package.uid, {"data": data_value})

    runner = Runner(pipeline)

    fullscreen = False
    window_name = "Gaze"
    cv2.namedWindow(window_name, cv2.WINDOW_NORMAL)

    cap = cv2.VideoCapture(0)

    while cap.isOpened():
        ok = cap.grab()
        if not ok:
            break
        ok, frame = cap.retrieve()
        if not ok:
            break
        report = runner.run(data={"default": frame})

        screen_shape = (screen_pixel_height, screen_pixel_width, 3)
        canvas = numpy.zeros(screen_shape, dtype=numpy.uint8)
        render(canvas, frame)

        face_gaze_screen = report["face_gaze_screen"]
        if face_gaze_screen:
            first_person_gaze = face_gaze_screen[0]["gaze"]
            success = first_person_gaze["success"]
            gaze_screen_px = first_person_gaze["gaze_screen_px"]
            # gaze_cm = first_person_gaze['gaze_cm']

            if success:
                cv2.circle(canvas, list(map(int, gaze_screen_px["left_eye"])), 20, (0, 0, 255), -1)
                cv2.circle(canvas, list(map(int, gaze_screen_px["right_eye"])), 20, (0, 0, 255), -1)

        cv2.imshow(window_name, canvas)
        key = cv2.waitKey(1)
        if key == 27:
            break

        key &= 0xFF
        if key in {ord("q"), ord("x")}:
            break

        if key == ord("f"):
            fullscreen = not fullscreen

            if fullscreen:
                cv2.setWindowProperty(window_name, cv2.WND_PROP_FULLSCREEN, cv2.WINDOW_FULLSCREEN)
            else:
                cv2.setWindowProperty(window_name, cv2.WND_PROP_FULLSCREEN, cv2.WINDOW_NORMAL)

    cv2.destroyWindow(window_name)


if __name__ == "__main__":
    main()
