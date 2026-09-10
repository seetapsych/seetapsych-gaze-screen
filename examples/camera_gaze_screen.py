# -*- coding: utf-8 -*-

import os
import platform

import cv2
import numpy
from seetapsych_lib.runtime.factory import Factory

# from seetapsych_lib.runtime.runner import Runner
from seetapsych_lib.runtime.parallel_runner import ParallelRunner as Runner
from seetapsych_lib.runtime.pipeline import Pipeline

override_modules = [
    os.path.join(os.path.dirname(__file__), "../seetapsych_gaze_screen/modules"),
    os.path.join(os.path.dirname(__file__), "../../seetapsych-face-hub/seetapsych_face_hub/modules"),
]


def open_camera(camera_id: int = 0) -> cv2.VideoCapture:
    system = platform.system()

    backend = {
        "Windows": cv2.CAP_DSHOW,
        "Linux": cv2.CAP_V4L2,
        "Darwin": cv2.CAP_AVFOUNDATION,
    }.get(system, cv2.CAP_ANY)

    cap = cv2.VideoCapture(camera_id, backend)
    if not cap.isOpened():
        cap.release()
        cap = cv2.VideoCapture(camera_id)

    return cap


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
    screen_localtion_in_camera_mm = (-270.0, 10.0)  # (x, y)
    screen_physical_size_mm = (540.0, 290.0)  # (width, height)
    original_screen_pixel_size = (1920, 1080)  # (width, height)
    screen_pixel_ratio = 1.0

    screen_pixel_size = (
        int(original_screen_pixel_size[0] / screen_pixel_ratio),
        int(original_screen_pixel_size[1] / screen_pixel_ratio),
    )

    screen_config = {
        "camera": {
            "screen_x_mm": screen_localtion_in_camera_mm[0],
            "screen_y_mm": screen_localtion_in_camera_mm[1],
        },
        "screen": {
            "h_px": screen_pixel_size[1],
            "w_px": screen_pixel_size[0],
            "h_mm": screen_physical_size_mm[1],
            "w_mm": screen_physical_size_mm[0],
        },
    }

    factory = Factory()
    # Attempt to override default modules with local module files, can still run even if no local files exist
    for root in override_modules:
        factory.load_dir_modules(root)

    pipeline = Pipeline(
        factory,
        packages=[
            "ee7b0a24-c4b3-470d-b095-9c606384d06e",  # AFFNet
            # "17c5187c-beb1-4e7d-a0a7-da2638d679bd", # TdGazeNet
        ],
        attributes=[
            # 'face/gaze_screen'
        ],
    )

    # print(pipeline.problem())
    pipeline.solve()

    # print(pipeline.satisfied())
    pipeline.install_requirements()
    pipeline.cache_models()

    package = pipeline.get_package(provide="face/gaze_screen")
    assert package is not None

    parameter_data = next((p for p in (package.parameters or []) if p.name == "data"), None)
    data_value: dict = parameter_data.value if parameter_data is not None else {}

    data_value = deep_merge(data_value, screen_config)

    pipeline.set_parameters(package.uid, {"data": data_value})

    runner = Runner(pipeline)

    fullscreen = True
    window_name = "Gaze"

    cap = open_camera(0)

    while cap.isOpened():
        ok = cap.grab()
        if not ok:
            break
        ok, frame = cap.retrieve()
        if not ok:
            break
        report = runner.run(data={"default": frame})

        screen_shape = (screen_pixel_size[1], screen_pixel_size[0], 3)
        canvas = numpy.zeros(screen_shape, dtype=numpy.uint8)
        render(canvas, cv2.flip(frame, 1))

        face_gaze_screen = report["face_gaze_screen"]
        if face_gaze_screen:
            first_person_gaze = face_gaze_screen[0]["gaze"]
            success = first_person_gaze["success"]
            gaze_screen_px = first_person_gaze["gaze_screen_px"]
            # gaze_camera_mm = first_person_gaze['gaze_camera_mm']

            if success:
                cv2.circle(canvas, list(map(int, gaze_screen_px["left_eye"])), 20, (0, 0, 255), -1)
                cv2.circle(canvas, list(map(int, gaze_screen_px["right_eye"])), 20, (0, 0, 255), -1)

        if not cv2.getWindowProperty(window_name, cv2.WND_PROP_VISIBLE):
            cv2.namedWindow(window_name, cv2.WINDOW_FULLSCREEN if fullscreen else cv2.WINDOW_NORMAL)

        cv2.putText(canvas, "Press F to toggle fullscreen", (30, 60), cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0, 0, 255), 2)
        cv2.putText(canvas, "Press Esc/Q/X to exit", (30, 120), cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0, 0, 255), 2)

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

    if cv2.getWindowProperty(window_name, cv2.WND_PROP_VISIBLE):
        cv2.destroyWindow(window_name)


if __name__ == "__main__":
    main()
