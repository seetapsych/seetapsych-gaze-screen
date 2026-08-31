# -*- coding: utf-8 -*-

from typing import Any, Callable, cast

import cv2
import numpy as np
import onnxruntime as ort
import torchvision.transforms.functional as TF
from opengaze.runtime.time import TimeViaEMA
from opengaze.utils.gc import FaceAlignment, FaceLandmarks
from PIL import Image
from seetapsych_lib import api
from seetapsych_lib.onnx.session import OnnxSession


# Data Transformations, Model Inference and Result Display
class FrameConsumer:
    def __call__(
        self, src_image: np.ndarray, set_exit_cond: Callable[..., Any], model: ort.InferenceSession
    ) -> tuple[dict[str, Any], bool]:
        result_dict = self.process(src_image, model)
        exit_cond = self.display(result_dict)
        set_exit_cond(exit_cond)
        return result_dict, exit_cond

    def __init__(self, demo_data: dict[str, Any], landmarker: FaceLandmarks | None, alignment: FaceAlignment):
        self.name = "ITrackerPlus Demo"

        self.demo_data = demo_data
        self.landmarker = landmarker
        self.alignment = alignment

        self.time = TimeViaEMA(alpha=0.1)

    def __enter__(self) -> "FrameConsumer":
        # Manage OpenCV resources
        cv2.namedWindow(self.name, cv2.WND_PROP_FULLSCREEN)
        cv2.setWindowProperty(self.name, cv2.WND_PROP_FULLSCREEN, cv2.WINDOW_FULLSCREEN)

        # Manage face landmarker
        if self.landmarker is not None:
            self.landmarker.create()

        return self

    def __exit__(self, exc_type: Any, exc_val: Any, exc_tb: Any):
        cv2.destroyWindow(self.name)
        if self.landmarker is not None:
            self.landmarker.destroy()

    @property
    def source_size(self) -> list[int]:
        return cast(
            list[int],
            [
                self.demo_data["camera"]["frame_h"],
                self.demo_data["camera"]["frame_w"],
            ],
        )

    @property
    def adjust_size(self) -> list[int]:
        return cast(list[int], self.demo_data["pre_process"]["adjust_size"])

    @property
    def norm_params(self) -> dict[str, list[float]]:
        return dict(
            mean=cast(list[float], self.demo_data["pre_process"]["norm_mean"]),
            std=cast(list[float], self.demo_data["pre_process"]["norm_std"]),
        )

    @property
    def screen_origin(self) -> np.ndarray:
        return np.array(
            [
                self.demo_data["camera"]["screen_x_mm"],
                self.demo_data["camera"]["screen_y_mm"],
            ],
            dtype=np.float32,
        )

    @property
    def screen_hw_px(self) -> tuple[int, int]:
        return cast(
            tuple[int, int],
            (self.demo_data["screen"]["h_px"], self.demo_data["screen"]["w_px"]),
        )

    @property
    def screen_hw_mm(self) -> tuple[int, int]:
        return cast(
            tuple[int, int],
            (self.demo_data["screen"]["h_mm"], self.demo_data["screen"]["w_mm"]),
        )

    @property
    def disp_margin(self) -> int:
        return cast(int, self.demo_data["post_process"]["disp_margin"])

    def _adjust_image_size(self, image: np.ndarray) -> np.ndarray:
        src_res, tgt_res = self.source_size, self.adjust_size

        src_asp = src_res[1] / src_res[0]
        tgt_asp = tgt_res[1] / tgt_res[0]

        if tgt_asp > src_asp:
            rescale_h = int(src_res[1] / tgt_asp)
            padding_h = (src_res[0] - rescale_h) // 2
            image = image[padding_h:-padding_h, :]
        if tgt_asp < src_asp:
            rescale_w = int(src_res[0] * tgt_asp)
            padding_w = (src_res[1] - rescale_w) // 2
            image = image[:, padding_w:-padding_w]

        dsize = (tgt_res[1], tgt_res[0])
        image = cv2.resize(image, dsize, interpolation=cv2.INTER_CUBIC)

        return image

    def _image_to_inputs(self, image: np.ndarray) -> np.ndarray:
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        pil_image = Image.fromarray(image, mode="RGB")
        image_tensor = TF.to_tensor(pil_image).unsqueeze(0)
        image_tensor = TF.normalize(image_tensor, **self.norm_params)
        return cast(np.ndarray, image_tensor.numpy().astype(np.float32))

    def _model_data_dict(self, align_dict: dict[str, Any]) -> dict[str, np.ndarray]:
        face = self._image_to_inputs(align_dict["face_crop"])
        reye = self._image_to_inputs(align_dict["reye_crop"])
        leye = self._image_to_inputs(align_dict["leye_crop"])

        reye_c, leye_c = align_dict["ldmks"][468], align_dict["ldmks"][473]
        eyes_ldmk = np.concatenate([reye_c, leye_c], axis=0)
        kpts = np.concatenate([align_dict["face_bbox"], eyes_ldmk], axis=0)
        kpts = np.expand_dims(kpts, axis=0).astype(np.float32)

        return dict(face=face, reye=reye, leye=leye, kpts=kpts)

    def _model_inference(self, model: ort.InferenceSession, data_dict: dict[str, np.ndarray]) -> dict[str, np.ndarray]:
        ort_outputs = model.run(None, data_dict)
        return dict(gaze=ort_outputs[0].squeeze(0))

    def _rotate_gaze(self, gaze: np.ndarray, theta: float) -> np.ndarray:
        radian = np.deg2rad(theta, dtype=np.float32)

        cos, sin = np.cos(radian), np.sin(radian)
        mat = np.array([[cos, -sin], [sin, cos]])

        return cast(np.ndarray, np.dot(mat, gaze))

    def _model_post_proc(self, align_dict: dict[str, Any], output_dict: dict[str, np.ndarray]) -> dict[str, Any]:
        # Note: 1. convert unit length from 1cm (camera) to 1mm (screen)
        #       2. the screen is in parallel to camera's XoY plane
        #
        # Point of Gaze: X-axis points rightward, Y-axis points upward
        #                Origin is at the center of the pinhole camera

        gaze_c = 1e1 * self._rotate_gaze(output_dict["gaze"], align_dict["theta"])

        gaze_s_mm = np.array([1.0, -1.0]) * (gaze_c - self.screen_origin)
        gaze_s_px = np.array(
            [
                gaze_s_mm[0] / self.screen_hw_mm[1] * self.screen_hw_px[1],
                gaze_s_mm[1] / self.screen_hw_mm[0] * self.screen_hw_px[0],
            ],
            dtype=np.float32,
        )

        return dict(gaze_c=gaze_c, gaze_s=gaze_s_px)

    def process(self, frame: np.ndarray, model: ort.InferenceSession) -> dict[str, Any]:
        adjusted_frame = self._adjust_image_size(frame)

        self.time.tick(tag="mediapipe")
        assert self.landmarker is not None, "landmarker required for process() path"
        landmarks = self.landmarker.process(adjusted_frame, bgr2rgb=True)
        self.time.tock(tag="mediapipe")

        if landmarks is None:
            return dict(success=False, frame=frame, message="No face detected.")

        align_dict = self.alignment.align(adjusted_frame, landmarks)

        data_dict = self._model_data_dict(align_dict)

        self.time.tick(tag="inference")
        output_dict = self._model_inference(model, data_dict)
        self.time.tock(tag="inference")

        proc_dict = self._model_post_proc(align_dict, output_dict)

        return dict(success=True, frame=frame, **proc_dict)

    def _draw_frame(self, canvas: np.ndarray, result_dict: dict[str, Any]):
        screen_h, screen_w = self.screen_hw_px

        frame_h, frame_w, _ = result_dict["frame"].shape

        avail_h = screen_h // 1 - 2 * self.disp_margin
        avail_w = screen_w // 2 - 2 * self.disp_margin

        scale_h = avail_h / frame_h
        scale_w = avail_w / frame_w
        scale = min(scale_h, scale_w)

        final_h = int(frame_h * scale)
        final_w = int(frame_w * scale)

        y_offset = (avail_h - final_h) // 2 + self.disp_margin
        x_offset = (avail_w - final_w) // 2 + self.disp_margin

        frame = cv2.resize(result_dict["frame"], (final_w, final_h))
        canvas[y_offset : y_offset + final_h, x_offset : x_offset + final_w] = frame

        if result_dict["success"]:
            center = cast(tuple[int, int], tuple(result_dict["gaze_s"].astype(np.int32)))
            cv2.circle(canvas, center, radius=20, color=(0, 0, 255), thickness=-1)

    def _draw_text(self, canvas: np.ndarray, result_dict: dict[str, Any]):
        screen_h, screen_w = self.screen_hw_px

        if result_dict["success"]:
            # Measured time for different stages
            text = ", ".join(
                [
                    f"MediaPipe: {1e3 * self.time.report(tag='mediapipe'):.2f} ms",
                    f"Inference: {1e3 * self.time.report(tag='inference'):.2f} ms",
                ]
            )
            cv2.putText(
                canvas,
                text,
                (30, screen_h - 30),
                fontFace=cv2.FONT_HERSHEY_SIMPLEX,
                fontScale=0.5,
                color=(255, 255, 255),
                thickness=1,
                lineType=cv2.LINE_AA,
            )

    def display(self, result_dict: dict[str, Any]) -> bool:
        canvas = np.zeros(shape=(*self.screen_hw_px, 3), dtype=np.uint8)
        self._draw_frame(canvas, result_dict)
        self._draw_text(canvas, result_dict)
        cv2.imshow(self.name, canvas)
        return cv2.waitKey(6) & 0xFF == ord("X")

    def _adjust_landmarks(self, image: np.ndarray, landmarks: np.ndarray) -> np.ndarray:
        """
        Map landmarks from original image coordinates to adjusted image coordinates.

        Args:
            image: np.ndarray original image
            landmarks: np.ndarray of shape [478, 2], where each point is [x, y]
                       in pixel coordinates on the original image.

        Returns:
            np.ndarray of shape [478, 2], mapped to the adjusted image coordinates.
        """
        landmarks = landmarks.copy()

        src_res, tgt_res = self.source_size, self.adjust_size
        img_size = image.shape[:2]

        src_asp = src_res[1] / src_res[0]
        tgt_asp = tgt_res[1] / tgt_res[0]

        if tgt_asp > src_asp:
            rescale_h = int(src_res[1] / tgt_asp)
            padding_h = (src_res[0] - rescale_h) // 2
            # image = image[padding_h:-padding_h, :]
            landmarks[:, 1] -= padding_h
            crop_size = (img_size[0] - 2 * padding_h, img_size[1])
        elif tgt_asp < src_asp:
            rescale_w = int(src_res[0] * tgt_asp)
            padding_w = (src_res[1] - rescale_w) // 2
            # image = image[:, padding_w:-padding_w]
            landmarks[:, 0] -= padding_w
            crop_size = (img_size[0], img_size[1] - 2 * padding_w)
        else:
            crop_size = img_size

        ssize = (crop_size[1], crop_size[0])
        dsize = (tgt_res[1], tgt_res[0])

        landmarks *= np.asarray(dsize) / np.asarray(ssize)

        return landmarks

    def process_with_landmarks(
        self, frame: np.ndarray, model: ort.InferenceSession, landmarks: np.ndarray
    ) -> dict[str, Any]:
        adjusted_frame = self._adjust_image_size(frame)
        adjusted_landmarks = self._adjust_landmarks(frame, landmarks)

        align_dict = self.alignment.align(adjusted_frame, adjusted_landmarks)

        data_dict = self._model_data_dict(align_dict)

        self.time.tick(tag="inference")
        output_dict = self._model_inference(model, data_dict)
        self.time.tock(tag="inference")

        proc_dict = self._model_post_proc(align_dict, output_dict)

        return dict(success=True, frame=frame, **proc_dict)


class Instance(api.Instance):
    def __init__(self, model_path: str, demo_data: dict[str, Any], device: api.Device):
        model = OnnxSession(model_path, device)

        # FaceLandmarks(p_detection=0.8, p_presence=0.8)
        consumer_kwargs = dict(
            landmarker=None,
            alignment=FaceAlignment(width_expand=1.6, hw_ratio=1.0),
        )

        consumer = FrameConsumer(demo_data=demo_data, **consumer_kwargs)

        self.__model = model
        self.__consumer = consumer

    def inference(self, *, data: dict[str, Any], report: dict[str, Any], **kwargs: Any) -> dict[str, Any]:
        input_data = data["default"]
        input_data = np.ascontiguousarray(input_data)  # [H, W, C] format

        face_mesh = report.get("face_mesh", [])

        input_height, input_width = input_data.shape[:2]
        self.__consumer.demo_data["camera"]["frame_h"] = input_height
        self.__consumer.demo_data["camera"]["frame_w"] = input_width

        face_gaze_screen = []
        for mesh_data in face_mesh:
            norm_3d_landmarks = mesh_data.get("normalized_3d_landmarks", [])
            if not norm_3d_landmarks:
                face_gaze_screen.append(
                    {
                        "gaze": {
                            "success": False,
                            "gaze_screen_px": {
                                "left_eye": [],
                                "right_eye": [],
                            },
                            "gaze_cm": {
                                "left_eye": [],
                                "right_eye": [],
                            },
                        }
                    }
                )
                continue

            norm_2d_landmarks = np.asarray(norm_3d_landmarks).reshape([-1, 3])[:, :2]
            size = np.asarray([input_width, input_height])
            landmarks = norm_2d_landmarks * size

            result_dict = self.__consumer.process_with_landmarks(input_data, self.__model.session, landmarks)

            success = result_dict["success"]
            gaze_screen_px = result_dict["gaze_s"].tolist() if success else []
            gaze_cm = result_dict["gaze_c"].tolist() if success else []

            face_gaze_screen.append(
                {
                    "gaze": {
                        "success": success,
                        "gaze_screen_px": {
                            "left_eye": gaze_screen_px,
                            "right_eye": gaze_screen_px,
                        },
                        "gaze_cm": {
                            "left_eye": gaze_cm,
                            "right_eye": gaze_cm,
                        },
                    }
                }
            )

        return {"face_gaze_screen": face_gaze_screen}


class Package(api.Package):
    def create(
        self,
        *,
        models: list[api.UsageModel],
        parameters: dict[str, Any],
        device: api.Device | None,
        **kwargs: Any,
    ) -> Instance:
        assert len(models) >= 1, api.MissingModelError("At least one model required")

        demo_data = parameters.get("data", {})

        model_path = models[0].cache()
        return Instance(
            model_path,
            demo_data,
            api.Device("cpu") if device is None else device,
        )


def load() -> api.Package:
    return Package()


def main():
    pass


if __name__ == "__main__":
    main()
