# -*- coding: utf-8 -*-

import collections
import os.path
from typing import Any, Callable, cast

import cv2
import numpy as np
import safetensors.torch
import torch
import torchvision.transforms.functional as TF
from opengaze.model.gaze_3d import XGaze224
from opengaze.runtime.time import TimeViaEMA
from opengaze.utils import FaceBoundingBox, MpiiDataNormalizer
from opengaze.utils.euler import gaze_2d_3d_a
from opengaze.utils.geom import PoseEstimator
from PIL import Image
from seetapsych_lib import api

from ..utils.merge import deep_merge
from .tddfa import SparseFaceLandmarks

ROOT = os.path.dirname(os.path.abspath(__file__))


class HeadPoseEstimator(PoseEstimator):
    def __init__(self, cam_mat: np.ndarray, cam_dist: np.ndarray | None = None):
        super().__init__(cam_mat=cam_mat, cam_dist=cam_dist)

        xgaze_model_path = os.path.join(ROOT, "ethxgaze-generic.txt")
        face_model = np.loadtxt(xgaze_model_path, dtype=np.float32)
        landmarks_indices = [
            20,  # reye, outer
            23,  # reye, inner
            26,  # leye, inner
            29,  # leye, outer
            15,  # mouth, rc
            19,  # mouth, lc
        ]
        self.face_model = face_model[landmarks_indices, :]

    def estimate(self, landmarks_2d: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        return cast(tuple[np.ndarray, np.ndarray], super().estimate(self.face_model, landmarks_2d))


def generate_ellipse_points(bbox: np.ndarray) -> np.ndarray:
    """
    :param bbox: (4, )
    :return: (8, 2)
    """
    x1, y1, x2, y2 = bbox
    cx = (x1 + x2) / 2
    cy = (y1 + y2) / 2

    c = np.asarray([cx, cy], dtype=np.float32)
    size = np.asarray([(x2 - x1) / 2, (y2 - y1) / 2], dtype=np.float32)

    v = 1 / 2**0.5
    tpl = np.asarray(
        [
            0,
            -1,
            v,
            -v,
            1,
            0,
            v,
            v,
            0,
            1,
            -v,
            v,
            -1,
            0,
            -v,
            v,
        ],
        dtype=np.float32,
    ).reshape([-1, 2])

    return tpl * size + c


def keypoints_to_bbox(points: np.ndarray) -> np.ndarray:
    x = points[:, 0]
    y = points[:, 1]
    return np.asarray([x.min(), y.min(), x.max(), y.max()])


# Data Transformations, Model Inference and Result Display
class FrameConsumer:
    def __call__(
        self,
        src_image: np.ndarray,
        set_exit_cond: Callable[..., Any],
        model: XGaze224,
    ) -> tuple[dict[str, Any], bool]:
        result_dict = self.process(src_image, model)
        exit_cond = self.display(result_dict)
        set_exit_cond(exit_cond)
        return result_dict, exit_cond

    def __init__(
        self,
        demo_data: dict[str, Any],
        face_bbox: FaceBoundingBox | None,
        landmarker: SparseFaceLandmarks | None,
        device: torch.device,
    ):
        self.name = "XGaze224 Demo"

        self.demo_data = demo_data
        self.face_bbox = face_bbox
        self.device = device

        if landmarker is None:
            landmarker = SparseFaceLandmarks(width_expand=1.6, image_size=120)

        self.landmarker = landmarker
        self.normalizer = MpiiDataNormalizer(960, (224, 224), distance=300)
        self.pose_estim = HeadPoseEstimator(self.real_intrinsic, self.real_dist_coef)

        self.time = TimeViaEMA(alpha=0.1)

    def __enter__(self) -> "FrameConsumer":
        # Manage OpenCV resources
        cv2.namedWindow(self.name, cv2.WND_PROP_FULLSCREEN)
        cv2.setWindowProperty(self.name, cv2.WND_PROP_FULLSCREEN, cv2.WINDOW_FULLSCREEN)

        # Manage face bounding box
        if self.face_bbox is not None:
            self.face_bbox.create()

        return self

    def __exit__(self, exc_type: Any, exc_val: Any, exc_tb: Any):
        cv2.destroyWindow(self.name)
        if self.face_bbox is not None:
            self.face_bbox.destroy()

    @property
    def real_intrinsic(self) -> np.ndarray:
        return np.array(self.demo_data["camera"]["intrinsic"], dtype=np.float32)

    @property
    def real_extrinsic(self) -> np.ndarray:
        return np.array(self.demo_data["camera"]["extrinsic"], dtype=np.float32)

    @property
    def real_dist_coef(self) -> np.ndarray:
        return np.array(self.demo_data["camera"]["distortion"], dtype=np.float32)

    @property
    def screen_hw_px(self) -> tuple[int, int]:
        return cast(tuple[int, int], (self.demo_data["screen"]["h_px"], self.demo_data["screen"]["w_px"]))

    @property
    def screen_hw_mm(self) -> tuple[int, int]:
        return cast(tuple[int, int], (self.demo_data["screen"]["h_mm"], self.demo_data["screen"]["w_mm"]))

    @property
    def norm_params(self) -> dict[str, list[float]]:
        return dict(
            mean=cast(list[float], self.demo_data["pre_process"]["norm_mean"]),
            std=cast(list[float], self.demo_data["pre_process"]["norm_std"]),
        )

    @property
    def epsilon(self) -> float:
        return cast(float, self.demo_data["post_process"]["epsilon"])

    @property
    def disp_margin(self) -> int:
        return cast(int, self.demo_data["post_process"]["disp_margin"])

    def _calculate_face_center(self, landmarks_3d: np.ndarray) -> np.ndarray:
        # Use convention of ETH-XGaze data normalization

        eyes_center = np.mean(landmarks_3d[:, 0:4], axis=1)
        mouth_center = np.mean(landmarks_3d[:, 4:6], axis=1)
        face_center = 0.5 * (eyes_center + mouth_center)

        return cast(np.ndarray, face_center)

    def _normalize_face(
        self,
        frame: np.ndarray,
        look_at: np.ndarray,
        R1: np.ndarray,
    ) -> tuple[np.ndarray, np.ndarray]:
        Kv, S, R2, W = self.normalizer.normalize_matrices(look_at, R1, self.real_intrinsic)
        face_crop = self.normalizer.warp_image(frame, W)
        return face_crop, R2

    def _denormalize_gaze(self, gaze: np.ndarray) -> np.ndarray:
        gaze_3d = np.array(gaze_2d_3d_a(gaze[0], gaze[1]))
        return cast(np.ndarray, gaze_3d / np.linalg.norm(gaze_3d))

    def _model_data_dict(self, face_crop: np.ndarray) -> dict[str, Any]:
        face = cv2.cvtColor(face_crop, cv2.COLOR_BGR2RGB)
        pil_face = Image.fromarray(face, mode="RGB")
        face = TF.to_tensor(pil_face).unsqueeze(0)
        face = TF.normalize(face, **self.norm_params)
        return dict(face=face)

    def _model_inference(self, model: XGaze224, data_dict: dict[str, Any]) -> dict[str, Any]:
        data_dict = {k: v.to(self.device) for k, v in data_dict.items()}
        with torch.no_grad():
            model_outputs = model(**data_dict)
        return dict(gaze=model_outputs[0].to("cpu"))

    def _project_points_to_image(self, points_3d: np.ndarray) -> np.ndarray:
        # Project 3D points from real camera space to real image space
        image_coords = np.dot(self.real_intrinsic, points_3d.T)
        image_coords = (image_coords[:2, :] / image_coords[2, :]).T
        return cast(np.ndarray, image_coords)

    def _point_of_gaze(self, origin: np.ndarray, vector: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        # Note: 1. the screen is in parallel to camera's XoY plane
        #
        # Point of Gaze: (xc, yc, zc), real camera space, 1mm as unit
        # Gaze Origin: (ox, oy, oz), real camera space, 1mm as unit
        # Gaze Vector: (vx, vy, vz), real camera space, 1mm as unit

        [ox, oy, oz], [vx, vy, vz] = origin, vector

        zc = self.real_extrinsic[2, 3]
        t = (zc - oz) / (vz + self.epsilon)

        xc = ox + t * vx
        yc = oy + t * vy

        gaze_c = np.array([xc, yc, zc], dtype=np.float32)

        R, T = self.real_extrinsic[:, :3], self.real_extrinsic[:, 3]
        gaze_s_mm = np.dot(R.T, gaze_c - T)
        gaze_s_px = np.array(
            [
                gaze_s_mm[0] / self.screen_hw_mm[1] * self.screen_hw_px[1],
                gaze_s_mm[1] / self.screen_hw_mm[0] * self.screen_hw_px[0],
            ],
            dtype=np.float32,
        )

        return gaze_c, gaze_s_px

    def _model_post_proc(self, look_at: np.ndarray, R2: np.ndarray, output_dict: dict[str, Any]) -> dict[str, Any]:
        proc_dict = dict(face_center=look_at)

        gaze_3d = self._denormalize_gaze(output_dict["gaze"])

        origin_2d, target_2d = self._project_points_to_image(np.stack([look_at, look_at + 1e2 * gaze_3d]))
        proc_dict.update(origin_2d=origin_2d, target_2d=target_2d)

        gaze_c, gaze_s = self._point_of_gaze(
            origin=look_at,
            vector=np.dot(R2.T, gaze_3d),
        )
        proc_dict.update(gaze_c=gaze_c, gaze_s=gaze_s)

        return proc_dict

    def process(self, frame: np.ndarray, model: XGaze224) -> dict[str, Any]:
        frame = cv2.undistort(frame, self.real_intrinsic, self.real_dist_coef)

        self.time.tick(tag="face-bbox")
        assert self.face_bbox is not None, "face_bbox required for process() path"
        bboxes = self.face_bbox.process(frame, bgr2rgb=True)
        self.time.tock(tag="face-bbox")

        if bboxes is None:
            return dict(success=False, frame=frame, message="No face detected.")

        ldmks_3d, ldmks_2d = self.landmarker.process(frame, bboxes[0], bgr2rgb=True)
        landmarks_2d = ldmks_2d[
            [
                36,  # reye, outer
                39,  # reye, inner
                42,  # leye, inner
                45,  # leye, outer
                60,  # mouth, rc
                64,  # mouth, lc
            ]
        ]
        hr, ht = self.pose_estim.estimate(landmarks_2d)
        hR = cv2.Rodrigues(hr)[0]
        landmarks_3d = np.dot(hR, self.pose_estim.face_model.T) + ht
        face_center = self._calculate_face_center(landmarks_3d)

        face_crop, R2 = self._normalize_face(frame, face_center, hR)

        data_dict = self._model_data_dict(face_crop)

        self.time.tick(tag="inference")
        output_dict = self._model_inference(model, data_dict)
        self.time.tock(tag="inference")

        proc_dict = self._model_post_proc(face_center, R2, output_dict)

        return dict(success=True, frame=frame, **output_dict, **proc_dict)

    def _draw_frame(self, canvas: np.ndarray, result_dict: dict[str, Any]):
        screen_h, screen_w = self.screen_hw_px

        if result_dict["success"]:
            pt1 = cast(tuple[int, int], tuple(result_dict["origin_2d"].astype(np.int32)))
            pt2 = cast(tuple[int, int], tuple(result_dict["target_2d"].astype(np.int32)))
            cv2.arrowedLine(
                result_dict["frame"],
                pt1,
                pt2,
                color=(0, 0, 255),
                thickness=2,
                line_type=cv2.LINE_AA,
            )

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
            cv2.circle(
                canvas,
                center,
                radius=20,
                color=(0, 0, 255),
                thickness=-1,
            )

    def _draw_text(self, canvas: np.ndarray, result_dict: dict[str, Any]):
        screen_h, screen_w = self.screen_hw_px

        if result_dict["success"]:
            # Measured time for different stages
            text = ", ".join(
                [
                    f"Face-BBox: {1e3 * self.time.report(tag='face-bbox'):.2f} ms",
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

    def process_with_bbox(self, frame: np.ndarray, model: XGaze224, bbox: np.ndarray) -> dict[str, Any]:
        """
        :param frame:
        :param model:
        :param bbox: [4, ]
        :return:
        """
        frame = cv2.undistort(frame, self.real_intrinsic, self.real_dist_coef, None, self.real_intrinsic)
        keypoints = generate_ellipse_points(bbox)
        keypoints = cv2.undistortPoints(
            keypoints.reshape([-1, 1, 2]).astype(np.float32),
            self.real_intrinsic,
            self.real_dist_coef,
            P=self.real_intrinsic,
        )
        bbox = keypoints_to_bbox(keypoints.reshape([-1, 2]))

        # check face out of frame
        frame_height, frame_width = frame.shape[:2]
        x_min, y_min, x_max, y_max = bbox
        if x_min >= frame_width or x_max <= 0 or y_min >= frame_height or y_max <= 0:
            return dict(success=False, frame=frame, message="No face detected.")

        ldmks_3d, ldmks_2d = self.landmarker.process(frame, bbox, bgr2rgb=True)
        landmarks_2d = ldmks_2d[
            [
                36,  # reye, outer
                39,  # reye, inner
                42,  # leye, inner
                45,  # leye, outer
                60,  # mouth, rc
                64,  # mouth, lc
            ]
        ]
        hr, ht = self.pose_estim.estimate(landmarks_2d)
        hR = cv2.Rodrigues(hr)[0]
        landmarks_3d = np.dot(hR, self.pose_estim.face_model.T) + ht
        face_center = self._calculate_face_center(landmarks_3d)

        face_crop, R2 = self._normalize_face(frame, face_center, hR)

        data_dict = self._model_data_dict(face_crop)

        self.time.tick(tag="inference")
        output_dict = self._model_inference(model, data_dict)
        self.time.tock(tag="inference")

        proc_dict = self._model_post_proc(face_center, R2, output_dict)

        return dict(success=True, frame=frame, **output_dict, **proc_dict)


# Entrypoint, Arguments and Top-Level Utilities
def load_wrapped_model(demo_data: dict, state_dict_file: str, device: torch.device) -> XGaze224:
    model = XGaze224().to(device=device)

    state_dict: dict[str, Any] = {}
    match os.path.splitext(state_dict_file)[-1].lower():
        case ".pth":
            state_dict = torch.load(state_dict_file, map_location=device)
            # safetensors.torch.save_file(
            #     state_dict, os.path.splitext(state_dict_file)[0] + ".safetensors"
            # )
        case ".safetensors":
            state_dict = safetensors.torch.load_file(state_dict_file, device=str(device))
        case _:
            raise RuntimeError("Only support file format .pth or .safetensors")

    params = collections.OrderedDict()
    for key in state_dict:
        param_key = key.removeprefix("model.")
        params[param_key] = state_dict[key]
    model.load_state_dict(params, strict=False)

    return model.eval()


class Instance(api.Instance):
    def __init__(
        self,
        model_path: str,
        tddfa_resource_root: str,
        demo_data: dict[str, Any],
        device: api.Device,
    ):
        demo_data = deep_merge(
            {
                "pre_process": {"norm_mean": [0.485, 0.456, 0.406], "norm_std": [0.229, 0.224, 0.225]},
                "post_process": {"epsilon": 1e-9, "disp_margin": 80},
                "camera": {
                    "capture_id": 0,
                    "frame_h": 720,
                    "frame_w": 1280,
                    "intrinsic": [[972.01, 0.0, 652.68], [0.0, 972.35, 373.91], [0.0, 0.0, 1.0]],
                    "extrinsic": [[-1.0, 0.0, 0.0, 155.0], [0.0, 1.0, 0.0, 5.0], [0.0, 0.0, -1.0, 2.5]],
                    "distortion": [0.123508, -0.334222, -0.002206, 0.000207, 0.199979],
                },
                "screen": {"h_px": 1080, "w_px": 1920, "h_mm": 174.0, "w_mm": 310.0},
            },
            demo_data,
        )

        torch_device = torch.device(str(device))
        model = load_wrapped_model(demo_data, model_path, torch_device)

        consumer = FrameConsumer(
            demo_data=demo_data,
            face_bbox=None,
            landmarker=SparseFaceLandmarks(tddfa_resource_root, width_expand=1.6, image_size=120),
            device=torch_device,
        )

        self.__model = model
        self.__consumer = consumer

    def inference(self, *, data: dict[str, Any], report: dict[str, Any], **kwargs: Any) -> dict[str, Any]:
        input_data = data["default"]
        input_data = np.ascontiguousarray(input_data)  # [H, W, C] format

        face_detection = report.get("face_detection", [])

        input_height, input_width = input_data.shape[:2]
        self.__consumer.demo_data["camera"]["frame_h"] = input_height
        self.__consumer.demo_data["camera"]["frame_w"] = input_width

        face_gaze_screen = []
        for detection_data in face_detection:
            xyxy = detection_data.get("xyxy", [])
            bbox = np.asarray(xyxy)

            result_dict = self.__consumer.process_with_bbox(input_data, self.__model, bbox)

            success = result_dict["success"]
            gaze_screen_px = result_dict["gaze_s"].tolist() if success else []
            gaze_camera_mm = (result_dict["gaze_c"] * [-1, -1, 1]).tolist() if success else []

            face_gaze_screen.append(
                {
                    "gaze": {
                        "success": success,
                        "gaze_screen_px": {
                            "left_eye": gaze_screen_px,
                            "right_eye": gaze_screen_px,
                        },
                        "gaze_camera_mm": {
                            "left_eye": gaze_camera_mm,
                            "right_eye": gaze_camera_mm,
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
        using_models = {m.usage: m for m in models}
        assert "" in using_models, api.MissingModelError('Usage "" model required')
        assert "3ddfa" in using_models, api.MissingModelError('Usage "3ddfa" model required')

        model_path = using_models[""].cache()
        tddfa_resource_root = using_models["3ddfa"].cache()

        demo_data = parameters.get("data", {})

        return Instance(
            model_path,
            tddfa_resource_root,
            demo_data,
            api.Device("cpu") if device is None else device,
        )


def load() -> api.Package:
    return Package()


def main():
    pass


if __name__ == "__main__":
    main()
