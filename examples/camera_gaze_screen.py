# -*- coding: utf-8 -*-

import os
import platform
import time
from typing import Any

import cv2
import numpy
from camera_window import CameraWindow, draw_text, fit_canvas, fit_geometry
from seetapsych_lib.runtime.factory import Factory

# from seetapsych_lib.runtime.runner import Runner
from seetapsych_lib.runtime.parallel.future import Future
from seetapsych_lib.runtime.parallel_runner import ParallelRunner as Runner
from seetapsych_lib.runtime.pipeline import Pipeline

override_modules = [
    os.path.join(os.path.dirname(__file__), "../seetapsych_gaze_screen/modules"),
    os.path.join(os.path.dirname(__file__), "../../seetapsych-face-hub/seetapsych_face_hub/modules"),
]


def open_camera(camera_id: int = 0) -> cv2.VideoCapture:
    """Open a camera with a platform backend and default-backend fallback.

    Args:
        camera_id: Camera device index.

    Returns:
        Capture whose open state must be checked by the caller.
    """
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


def deep_merge(base: dict[str, Any], override: dict[str, Any]) -> dict[str, Any]:
    """Merge nested overrides without discarding unrelated model settings.

    Args:
        base: Original parameter mapping.
        override: Values to replace, recursively merging nested mappings.

    Returns:
        New top-level mapping preserving unspecified keys.
    """
    result = base.copy()
    for key, value in override.items():
        if key in result and isinstance(result[key], dict) and isinstance(value, dict):
            result[key] = deep_merge(result[key], value)
        else:
            result[key] = value
    return result


def render(dst: numpy.ndarray, src: numpy.ndarray) -> numpy.ndarray:
    """Fit the entire preview without changing the screen-coordinate canvas.

    Args:
        dst: Screen-sized destination.
        src: Camera preview.

    Returns:
        Destination with the complete preview centered and resized proportionally.
    """
    dst_h, dst_w = dst.shape[:2]
    src_h, src_w = src.shape[:2]
    x, y, width, height = fit_geometry((src_w, src_h), (dst_w, dst_h))
    interpolation = cv2.INTER_AREA if width < src_w else cv2.INTER_CUBIC
    dst[y : y + height, x : x + width] = cv2.resize(src, (width, height), interpolation=interpolation)
    return dst


def run_camera(cap: cv2.VideoCapture) -> None:
    """Run screen-gaze inference independently of camera rendering.

    Args:
        cap: Open camera owned by the caller.
    """
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
    parameter_value = parameter_data.value if parameter_data is not None else {}
    if not isinstance(parameter_value, dict):
        raise ValueError("The gaze package's data parameter must be a mapping.")
    data_value = deep_merge(parameter_value, screen_config)

    pipeline.set_parameters(package.uid, {"data": data_value})

    runner = Runner(pipeline)

    window_name = "Gaze"

    pending: Future[dict[str, Any]] | None = None
    report: dict[str, Any] = {}
    submitted_at = 0.0
    result_at = 0.0
    inference_fps = 0.0
    try:
        window = CameraWindow(window_name, screen_pixel_size, fullscreen=True)
        while cap.isOpened():
            ok, frame = cap.read()
            if not ok:
                print("Camera stopped delivering frames. Check the connection and camera permissions.")
                break
            now = time.monotonic()
            if pending is not None and pending.done():
                report = pending.get()
                result_at = submitted_at
                inference_fps = 1.0 / max(now - submitted_at, 1e-6)
                pending = None
            if pending is None:
                submitted_at = now
                pending = runner.run_async(data={"default": frame.copy()}, timestamp=time.time())

            canvas = numpy.zeros((screen_pixel_size[1], screen_pixel_size[0], 3), dtype=numpy.uint8)
            render(canvas, cv2.flip(frame, 1))
            display_size = window.size()
            offset_x, offset_y, display_width, display_height = fit_geometry(screen_pixel_size, display_size)
            scale_x, scale_y = display_width / screen_pixel_size[0], display_height / screen_pixel_size[1]
            canvas = fit_canvas(canvas, display_size)
            result_ttl = max(1.0, 2.0 / max(inference_fps, 0.01))
            face_gaze_screen = report.get("face_gaze_screen", []) if now - result_at < result_ttl else []
            status = "Waiting for gaze / face not detected"
            if face_gaze_screen:
                gaze = face_gaze_screen[0]["gaze"]
                if gaze["success"]:
                    status = f"Inference: {inference_fps:.1f} FPS"
                    for eye in ("left_eye", "right_eye"):
                        point = gaze["gaze_screen_px"][eye]
                        if numpy.isfinite(point).all():
                            display_point = (round(offset_x + point[0] * scale_x), round(offset_y + point[1] * scale_y))
                            cv2.circle(
                                canvas,
                                display_point,
                                max(3, round(20 * min(scale_x, scale_y))),
                                (0, 0, 255),
                                -1,
                                cv2.LINE_AA,
                            )

            ui_scale = min(display_size[0] / 1920, display_size[1] / 1080)
            draw_text(
                canvas,
                "F = Fullscreen | Esc / Q / X = Exit",
                (round(30 * ui_scale), round(60 * ui_scale)),
                (255, 255, 255),
                scale=0.8 * ui_scale,
                thickness=max(1, round(2 * ui_scale)),
            )
            if not window.fullscreen:
                status += " | Windowed preview; calibration unchanged"
            draw_text(
                canvas,
                status,
                (round(30 * ui_scale), round(100 * ui_scale)),
                (0, 220, 255),
                scale=0.7 * ui_scale,
                thickness=max(1, round(2 * ui_scale)),
            )
            window.show(canvas)
            key = cv2.waitKey(1) & 0xFF
            if key in (27, ord("q"), ord("Q"), ord("x"), ord("X")):
                break
            if cv2.getWindowProperty(window_name, cv2.WND_PROP_VISIBLE) < 1:
                break
            window.handle_key(key)
    finally:
        if pending is not None:
            pending.cancel()
        runner.dispose()


def main() -> None:
    """Open the camera before preparing models and release GUI resources on exit."""
    cap = open_camera(0)
    try:
        if not cap.isOpened():
            print("Could not open camera 0. Check the device index, camera permissions, and other camera apps.")
            return
        run_camera(cap)
    finally:
        cap.release()
        cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
