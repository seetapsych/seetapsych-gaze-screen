# -*- coding: utf-8 -*-

import collections
import math
import os.path
from typing import Any, Callable, cast

import cv2
import numpy as np
import safetensors.torch
import torch
import torchvision.transforms.functional as TF
from opengaze.model.gaze_dp import TdGazeNet
from opengaze.runtime.time import TimeViaEMA
from opengaze.utils.gc import FaceLandmarks
from opengaze.utils.image import scaled_crop
from PIL import Image
from seetapsych_lib import api

from ..utils.merge import deep_merge


# Data Transformations, Model Inference and Result Display
class FrameConsumer:
    def __call__(
        self, src_image: np.ndarray, set_exit_cond: Callable[..., Any], model: TdGazeNet
    ) -> tuple[dict[str, Any], bool]:
        result_dict = self.process(src_image, model)
        exit_cond = self.display(result_dict)
        set_exit_cond(exit_cond)
        return result_dict, exit_cond

    def __init__(self, demo_data: dict[str, Any], landmarker: FaceLandmarks | None, device: torch.device):
        self.name = "TdGazeNet Demo"

        self.demo_data = demo_data
        self.landmarker = landmarker
        self.device = device

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
    def norm_intrinsic(self) -> np.ndarray:
        return np.array(self.demo_data["intrinsic"], dtype=np.float32)

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
    def M_real2norm(self) -> np.ndarray:
        if not hasattr(self, "_M_real2norm"):
            self._M_real2norm = cast(np.ndarray, np.dot(self.norm_intrinsic, np.linalg.inv(self.real_intrinsic)))
        return self._M_real2norm

    @property
    def M_norm2real(self) -> np.ndarray:
        if not hasattr(self, "_M_norm2real"):
            self._M_norm2real = cast(np.ndarray, np.dot(self.real_intrinsic, np.linalg.inv(self.norm_intrinsic)))
        return self._M_norm2real

    @property
    def norm_center(self) -> np.ndarray:
        u_x = self.demo_data["intrinsic"][0][2]
        u_y = self.demo_data["intrinsic"][1][2]
        return np.array([u_x, u_y], dtype=np.float32)

    @property
    def norm_metric(self) -> float:
        u_x = self.demo_data["intrinsic"][0][2]
        u_y = self.demo_data["intrinsic"][1][2]
        return cast(float, (u_x + u_y) / 2.0)

    @property
    def screen_hw_px(self) -> tuple[int, int]:
        return cast(tuple[int, int], (self.demo_data["screen"]["h_px"], self.demo_data["screen"]["w_px"]))

    @property
    def screen_hw_mm(self) -> tuple[int, int]:
        return cast(tuple[int, int], (self.demo_data["screen"]["h_mm"], self.demo_data["screen"]["w_mm"]))

    @property
    def image_size(self) -> tuple[int, int]:
        return cast(tuple[int, int], self.demo_data["pre_process"]["image_size"])

    @property
    def bbox_scale(self) -> float:
        return cast(float, self.demo_data["pre_process"]["bbox_scale"])

    @property
    def norm_params(self) -> dict[str, list[float]]:
        return dict(
            mean=cast(list[float], self.demo_data["pre_process"]["norm_mean"]),
            std=cast(list[float], self.demo_data["pre_process"]["norm_std"]),
        )

    @property
    def gaze_source(self) -> str:
        return cast(str, self.demo_data["post_process"]["gaze_source"])

    @property
    def epsilon(self) -> float:
        return cast(float, self.demo_data["post_process"]["epsilon"])

    @property
    def disp_margin(self) -> int:
        return cast(int, self.demo_data["post_process"]["disp_margin"])

    def _bbox_from_ldmk(self, landmarks: np.ndarray) -> np.ndarray:
        x_min, y_min = np.min(landmarks, axis=0)
        x_max, y_max = np.max(landmarks, axis=0)

        bbox_cx = (x_min + x_max) / 2.0
        bbox_cy = (y_min + y_max) / 2.0
        bbox_ca = math.sqrt((x_max - x_min) * (y_max - y_min))
        bbox_ca = self.bbox_scale * bbox_ca

        x_min = bbox_cx - bbox_ca / 2.0
        y_min = bbox_cy - bbox_ca / 2.0
        x_max = bbox_cx + bbox_ca / 2.0
        y_max = bbox_cy + bbox_ca / 2.0

        bbox = np.array([x_min, y_min, x_max, y_max], dtype=np.float32)

        return bbox

    def _norm_bbox(self, bbox: np.ndarray) -> np.ndarray:
        x_min, y_min, x_max, y_max = bbox

        homo_coords = np.array(
            [
                [x_min, y_min, 1.0],
                [x_max, y_max, 1.0],
            ],
            dtype=np.float32,
        )
        pt_1, pt_2 = np.dot(homo_coords, self.M_real2norm.T)[:, :2]
        [x_min, y_min], [x_max, y_max] = pt_1, pt_2

        norm_bbox = (
            np.concatenate(
                [
                    np.array([(x_min + x_max) / 2, (y_min + y_max) / 2]) - self.norm_center,
                    np.array([x_max - x_min, y_max - y_min]),
                ],
                axis=0,
            )
            / self.norm_metric
        )

        return cast(np.ndarray, norm_bbox)

    def _model_data_dict(self, face_crop: np.ndarray, face_bbox: np.ndarray) -> dict[str, Any]:
        face = cv2.cvtColor(face_crop, cv2.COLOR_BGR2RGB)
        pil_face = Image.fromarray(face, mode="RGB")
        face = TF.to_tensor(pil_face).unsqueeze(0)
        face = TF.normalize(face, **self.norm_params)
        bbox = torch.tensor(face_bbox, dtype=torch.float32).unsqueeze(0)
        return dict(face=face, bbox=bbox)

    def _model_inference(self, model: TdGazeNet, data_dict: dict[str, Any]) -> dict[str, Any]:
        data_dict = {k: v.to(self.device) for k, v in data_dict.items()}
        with torch.no_grad():
            model_outputs = model(**data_dict)
        face_kpts, eyes_kpts, eyes_gaze = [o.to("cpu") for o in model_outputs]

        output_dict = dict(
            face_kpts_3d=face_kpts[0],
            reye_kpts_3d=eyes_kpts[0, 0],
            leye_kpts_3d=eyes_kpts[0, 1],
            reye_origin_3d=eyes_gaze[0, 0, 0],
            reye_vector=eyes_gaze[0, 0, 1],
            leye_origin_3d=eyes_gaze[0, 1, 0],
            leye_vector=eyes_gaze[0, 1, 1],
        )

        return output_dict

    def _project_points_to_image(self, points_3d: np.ndarray) -> np.ndarray:
        # Project 3D points from norm camera space to norm image space
        image_coords = np.dot(self.norm_intrinsic, points_3d.T)
        homo_coords = (image_coords / image_coords[2, :]).T

        # Convert 2D points from norm image space to real image space
        image_coords = np.dot(homo_coords, self.M_norm2real.T)[:, :2]

        return cast(np.ndarray, image_coords)

    def _point_of_gaze(self, origin: np.ndarray, vector: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        # Note: 1. convert unit length from 10cm (norm) to 1mm (real)
        #       2. the screen is in parallel to camera's XoY plane
        #
        # Point of Gaze: (xc, yc, zc), real camera space, 10cm as unit
        # Gaze Origin: (ox, oy, oz), real camera space, 10cm as unit
        # Gaze Vector: (vx, vy, vz), real camera space, 10cm as unit

        [ox, oy, oz], [vx, vy, vz] = origin, vector

        zc = self.real_extrinsic[2, 3] / 1e2
        t = (zc - oz) / (vz + self.epsilon)

        xc = ox + t * vx
        yc = oy + t * vy

        gaze_c = 1e2 * np.array([xc, yc, zc], dtype=np.float32)

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

    def _model_post_proc(self, output_dict: dict[str, Any]) -> dict[str, Any]:
        proc_dict: dict[str, Any] = dict(
            face_kpts_2d=self._project_points_to_image(output_dict["face_kpts_3d"]),
            reye_kpts_2d=self._project_points_to_image(output_dict["reye_kpts_3d"]),
            leye_kpts_2d=self._project_points_to_image(output_dict["leye_kpts_3d"]),
        )

        nasal_distance = 1e2 * np.linalg.norm(output_dict["face_kpts_3d"][2])
        proc_dict.update(nasal_distance=nasal_distance)

        reye_origin_2d, leye_origin_2d = self._project_points_to_image(
            np.stack([output_dict["reye_origin_3d"], output_dict["leye_origin_3d"]])
        )
        proc_dict.update(reye_origin_2d=reye_origin_2d, leye_origin_2d=leye_origin_2d)

        reye_target_2d, leye_target_2d = self._project_points_to_image(
            np.stack(
                [
                    output_dict["reye_origin_3d"] + output_dict["reye_vector"],
                    output_dict["leye_origin_3d"] + output_dict["leye_vector"],
                ]
            )
        )
        proc_dict.update(reye_target_2d=reye_target_2d, leye_target_2d=leye_target_2d)

        if self.gaze_source == "gaze":
            reye_gaze_c, reye_gaze_s = self._point_of_gaze(
                origin=output_dict["reye_origin_3d"],
                vector=output_dict["reye_vector"],
            )
            leye_gaze_c, leye_gaze_s = self._point_of_gaze(
                origin=output_dict["leye_origin_3d"],
                vector=output_dict["leye_vector"],
            )

        if self.gaze_source == "mesh":
            reye_gaze_c, reye_gaze_s = self._point_of_gaze(
                origin=(output_dict["reye_kpts_3d"][35] + output_dict["reye_kpts_3d"][82]) / 2,
                vector=(output_dict["reye_kpts_3d"][35] - output_dict["reye_kpts_3d"][82]) / 2,
            )
            leye_gaze_c, leye_gaze_s = self._point_of_gaze(
                origin=(output_dict["leye_kpts_3d"][33] + output_dict["leye_kpts_3d"][80]) / 2,
                vector=(output_dict["leye_kpts_3d"][33] - output_dict["leye_kpts_3d"][80]) / 2,
            )

        proc_dict.update(reye_gaze_c=reye_gaze_c, reye_gaze_s=reye_gaze_s)
        proc_dict.update(leye_gaze_c=leye_gaze_c, leye_gaze_s=leye_gaze_s)

        return proc_dict

    def process(self, frame: np.ndarray, model: TdGazeNet) -> dict[str, Any]:
        frame = cv2.undistort(frame, self.real_intrinsic, self.real_dist_coef)

        self.time.tick(tag="mediapipe")
        assert self.landmarker is not None, "landmarker required for process() path"
        landmarks = self.landmarker.process(frame, bgr2rgb=True)
        self.time.tock(tag="mediapipe")

        if landmarks is None:
            return dict(success=False, frame=frame, message="No face detected.")

        bbox = self._bbox_from_ldmk(landmarks)

        face_crop = scaled_crop(frame, bbox, self.image_size)
        face_bbox = self._norm_bbox(bbox)

        data_dict = self._model_data_dict(face_crop, face_bbox)

        self.time.tick(tag="inference")
        output_dict = self._model_inference(model, data_dict)
        self.time.tock(tag="inference")

        proc_dict = self._model_post_proc(output_dict)

        return dict(success=True, frame=frame, **output_dict, **proc_dict)

    def _draw_frame(self, canvas: np.ndarray, result_dict: dict[str, Any]):
        screen_h, screen_w = self.screen_hw_px

        if result_dict["success"]:
            kpts_kwargs_list: list[tuple[str, np.ndarray, int, tuple[int, int, int], int]] = [
                ("face_kpts_2d", result_dict["face_kpts_2d"], 2, (0, 255, 0), -1),
                ("reye_kpts_2d", result_dict["reye_kpts_2d"], 2, (255, 255, 0), -1),
                ("leye_kpts_2d", result_dict["leye_kpts_2d"], 2, (255, 255, 0), -1),
            ]
            for _name, pts_arr, radius, color, thickness in kpts_kwargs_list:
                for pt in pts_arr:
                    pt_int = cast(tuple[int, int], tuple(pt.astype(np.int32)))
                    cv2.circle(result_dict["frame"], pt_int, radius=radius, color=color, thickness=thickness)

            color_reye = cast(tuple[int, int, int], (0, 0, 255))
            color_leye = cast(tuple[int, int, int], (255, 0, 0))
            arrow_thickness = 2
            arrow_line_type = cv2.LINE_AA
            reye_orig = cast(tuple[int, int], tuple(result_dict["reye_origin_2d"].astype(np.int32)))
            reye_tgt = cast(tuple[int, int], tuple(result_dict["reye_target_2d"].astype(np.int32)))
            leye_orig = cast(tuple[int, int], tuple(result_dict["leye_origin_2d"].astype(np.int32)))
            leye_tgt = cast(tuple[int, int], tuple(result_dict["leye_target_2d"].astype(np.int32)))
            cv2.arrowedLine(
                result_dict["frame"],
                reye_orig,
                reye_tgt,
                color=color_reye,
                thickness=arrow_thickness,
                line_type=arrow_line_type,
            )
            cv2.arrowedLine(
                result_dict["frame"],
                leye_orig,
                leye_tgt,
                color=color_leye,
                thickness=arrow_thickness,
                line_type=arrow_line_type,
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
            reye_center = cast(tuple[int, int], tuple(result_dict["reye_gaze_s"].astype(np.int32)))
            leye_center = cast(tuple[int, int], tuple(result_dict["leye_gaze_s"].astype(np.int32)))
            cv2.circle(canvas, reye_center, radius=20, color=(0, 0, 255), thickness=-1)
            cv2.circle(canvas, leye_center, radius=20, color=(255, 0, 0), thickness=-1)

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
            # Custom metrics for each demo application
            text = ", ".join(
                [
                    f"Nasal Distance: {result_dict['nasal_distance']:.2f} mm",
                ]
            )
            cv2.putText(
                canvas,
                text,
                (30, screen_h - 60),
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

    def process_with_landmarks(self, frame: np.ndarray, model: TdGazeNet, landmarks: np.ndarray) -> dict[str, Any]:
        frame = cv2.undistort(frame, self.real_intrinsic, self.real_dist_coef, None, self.real_intrinsic)
        landmarks = cv2.undistortPoints(
            landmarks.reshape([-1, 1, 2]),
            self.real_intrinsic,
            self.real_dist_coef,
            P=self.real_intrinsic,
        )
        landmarks = landmarks.reshape([-1, 2])

        bbox = self._bbox_from_ldmk(landmarks)

        # check face out of frame
        frame_height, frame_width = frame.shape[:2]
        x_min, y_min, x_max, y_max = bbox
        if x_min >= frame_width or x_max <= 0 or y_min >= frame_height or y_max <= 0:
            return dict(success=False, frame=frame, message="No face detected.")

        face_crop = scaled_crop(frame, bbox, self.image_size)
        face_bbox = self._norm_bbox(bbox)

        data_dict = self._model_data_dict(face_crop, face_bbox)

        self.time.tick(tag="inference")
        output_dict = self._model_inference(model, data_dict)
        self.time.tock(tag="inference")

        proc_dict = self._model_post_proc(output_dict)

        return dict(success=True, frame=frame, **output_dict, **proc_dict)


# Entrypoint, Arguments and Top-Level Utilities
def load_wrapped_model(
    demo_data: dict,
    state_dict_file: str,
    device: torch.device,
    optimize: str,
) -> TdGazeNet:
    model = TdGazeNet(**demo_data["model"]).to(device=device)

    state_dict: dict[str, Any] = {}
    match os.path.splitext(state_dict_file)[-1].lower():
        case ".pth":
            state_dict = torch.load(state_dict_file, map_location=device)
            # safetensors.torch.save_file(
            #     state_dict,
            #     os.path.splitext(state_dict_file)[0] + ".safetensors",
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

    if optimize == "reparameterize":
        for _name, module in model.named_modules():
            if hasattr(module, "reparameterize"):
                module.reparameterize()

    return model.eval()


class Instance(api.Instance):
    def __init__(self, model_path: str, demo_data: dict[str, Any], optimize: str, device: api.Device):
        demo_data = deep_merge(
            {
                "model": {
                    "backbone": "fastvit-sa12",
                    "fusion": "adaptive",
                    "reg_head": "multi-task",
                    "n_face_kpts": 151,
                    "n_eyes_kpts": 110,
                },
                "pre_process": {
                    "image_size": [224, 224],
                    "bbox_scale": 1.1,
                    "norm_mean": [0.485, 0.456, 0.406],
                    "norm_std": [0.229, 0.224, 0.225],
                },
                "post_process": {"gaze_source": "gaze", "epsilon": 1e-9, "disp_margin": 80},
                "intrinsic": [[1244.44, 0.0, 800.0], [0.0, 1244.44, 800.0], [0.0, 0.0, 1.0]],
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
        model = load_wrapped_model(demo_data, model_path, torch_device, optimize)

        consumer = FrameConsumer(demo_data=demo_data, landmarker=None, device=torch_device)

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
                            "gaze_camera_mm": {
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

            result_dict = self.__consumer.process_with_landmarks(input_data, self.__model, landmarks)

            success = result_dict["success"]
            gaze_screen_px_left = result_dict["leye_gaze_s"].tolist() if success else []
            gaze_screen_px_right = result_dict["reye_gaze_s"].tolist() if success else []
            gaze_camera_mm_left = (result_dict["leye_gaze_c"] * [-1, -1, 1]).tolist() if success else []
            gaze_camera_mm_right = (result_dict["reye_gaze_c"] * [-1, -1, 1]).tolist() if success else []

            face_gaze_screen.append(
                {
                    "gaze": {
                        "success": success,
                        "gaze_screen_px": {
                            "left_eye": gaze_screen_px_left,
                            "right_eye": gaze_screen_px_right,
                        },
                        "gaze_camera_mm": {
                            "left_eye": gaze_camera_mm_left,
                            "right_eye": gaze_camera_mm_right,
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

        optimize = parameters.get("optimize", "none")
        demo_data = parameters.get("data", {})

        model_path = models[0].cache()
        return Instance(
            model_path,
            demo_data,
            optimize,
            api.Device("cpu") if device is None else device,
        )


def load() -> api.Package:
    return Package()


def main():
    pass


if __name__ == "__main__":
    main()
