# SeetaPsych Gaze

> Gaze estimation modules for SeetaPsych

## Usage

This project is already included in the seetapsych-lib default configuration. Download and use it via `seetapsych-manager download`.

For usage, refer to [SeetaPsych](https://github.com/seetapsych/seetapsych-lib).

The gaze estimation algorithms depend on `open-gaze-estimation`. Please install it from GitHub separately:

```bash
uv pip install git+https://github.com/Elorfiniel/open-gaze-estimation-2025-release.git
```

You can additionally add this algorithm module using the following methods.

### WebUI

Run `seetapsych-webui` with the `--files` argument to use it.

```
seetapsych-webui --files \
  seetapsych_gaze_screen/modules/affnet.yml \
  seetapsych_gaze_screen/modules/tdgazenet.yml
```

### Programmatic Usage

Add the following code in your program to use this algorithm module.

```python
from seetapsych_lib.runtime.factory import Factory
from seetapsych_lib.runtime.pipeline import Pipeline

factory = Factory()
factory.load_file_modules("seetapsych_gaze_screen/modules/affnet.yml")

pipeline = Pipeline(factory, ...)

pipeline.add_attributes("face/gaze_screen")
```

### Module Catalog

| Module YAML | Package Name |
|---|---|
| `affnet.yml` | GazeScreen-AFFNet(OpenGaze) |
| `tdgazenet.yml` | GazeScreen-TDGazeNet(OpenGaze) |

### GazeScreen-AFFNet(OpenGaze)

> Open-source gaze estimation toolkit providing screen gaze coordinates from facial landmarks or mesh.

Module config: [affnet.yml](seetapsych_gaze_screen/modules/affnet.yml)

| Package | Provides | Requires |
|---|---|---|
| GazeScreen-AFFNet(OpenGaze) | `face/gaze_screen` | `face/mesh` |

**Description**

Estimate screen gaze coordinates using the AFFNet[^1], which predicts a single gaze location shared by both eyes. Suitable for single-user desktop scenarios with moderate accuracy and computational cost.

**Parameters**

| Name | Type | Default | Description |
|---|---|---|---|
| `data` | object | *(see below)* | Camera/screen calibration and image preprocessing settings. See detailed field breakdown in [data parameter reference](#data-parameter-reference). |

**Models**

| Model | Version | Recommended |
|---|---|---|
| `opengaze-affnet-v2.safetensors` | 2.0 | ✓ |
| `opengaze-affnet.safetensors` | 1.0 |  |

### GazeScreen-TDGazeNet(OpenGaze)

> Open-source gaze estimation toolkit providing screen gaze coordinates from facial landmarks or mesh.

Module config: [tdgazenet.yml](seetapsych_gaze_screen/modules/tdgazenet.yml)

| Package | Provides | Requires |
|---|---|---|
| GazeScreen-TDGazeNet(OpenGaze) | `face/gaze_screen` | `face/mesh` |

**Description**

High-accuracy gaze estimation via TdGazeNet with 3D face prior, camera intrinsics and multi-task head; outputs distinct per-eye gaze vectors for left and right eye. Highest accuracy among OpenGaze variants at the cost of heavier compute.

**Parameters**

| Name | Type | Default | Description |
|---|---|---|---|
| `optimize` | selection | `none` | Post-load structural optimization of the backbone. Set to reparameterize to fold BatchNorm into convolutions for faster inference at load-time cost; use none for training or debug workflows where exact weights must be preserved. Possible values: `none`, `reparameterize`. |
| `data` | object | *(see below)* | Camera/screen calibration and image preprocessing settings. See detailed field breakdown in [data parameter reference](#data-parameter-reference). |

**Models**

| Model | Recommended |
|---|---|
| `opengeze-tdgazenet.safetensors` | ✓ |

## data parameter reference

All gaze-screen modules share the same top-level structure under the `data` parameter. Two layouts are used depending on whether the algorithm relies on a **3D face prior**. Screen dimensions (`w_px`, `h_px`, `w_mm`, `h_mm`) are common to both layouts; camera-section fields differ.

### Common fields (screen section)

Screen physical dimensions and pixel resolution are used to convert between camera-space millimeters and screen-space pixels. Values must match the actual monitor used for the experiment.

| Field | Type | Example | Description |
|---|---|---|---|
| `screen.w_px` | int | `1920` | Screen width in pixels (horizontal resolution). |
| `screen.h_px` | int | `1080` | Screen height in pixels (vertical resolution). |
| `screen.w_mm` | int | `310` | Screen width in physical millimeters (measured on the active display area). |
| `screen.h_mm` | int | `174` | Screen height in physical millimeters. |

### Layout A — Simple camera offset (AFFNet)

Used by algorithms that estimate gaze in the camera coordinate frame directly, then project it onto the screen plane from a known relative position. No per-pixel lens distortion is applied; if your camera has strong distortion, undistort frames before feeding them into the pipeline.

```json
{
  "camera": {
    "screen_x_mm": -155,
    "screen_y_mm": -5
  },
  "screen": {
    "h_px": 1080,
    "w_px": 1920,
    "h_mm": 174,
    "w_mm": 310
  }
}
```

| Field | Type | Default | Description |
|---|---|---|---|
| `camera.screen_x_mm` | float | `-155` | Horizontal offset from the camera optical center to the screen origin (top-left corner of the active display area), **in millimeters along the camera X-axis**. Sign convention: **right = positive, left = negative**. A typical laptop webcam sits above the screen center; the screen therefore lies to the left of the camera, producing a negative value. |
| `camera.screen_y_mm` | float | `-5` | Vertical offset from the camera optical center to the screen origin, **in millimeters along the camera Y-axis**. Sign convention: **up = positive, down = negative**. With the webcam mounted on the top bezel the screen sits slightly below the camera, so this value is usually slightly negative or close to zero. |

### Layout B — Full camera calibration (TDGazeNet)

Used by algorithms that leverage a 3D face prior. They require explicit camera intrinsics, an extrinsic screen-to-camera transform, and OpenCV-format distortion coefficients to lift 2D landmarks back into metric 3D space before projecting the gaze ray onto the screen plane. Run a standard OpenCV/Matlab checkerboard calibration once on your capture setup and paste the matrices here.

```json
{
  "camera": {
    "intrinsic": [
      [972.01, 0.0, 652.68],
      [0.0, 972.35, 373.91],
      [0.0, 0.0, 1.0]
    ],
    "extrinsic": [
      [-1.0, 0.0, 0.0, 155.0],
      [0.0, 1.0, 0.0, 5.0],
      [0.0, 0.0, -1.0, 2.5]
    ],
    "distortion": [0.123508, -0.334222, -0.002206, 0.000207, 0.199979]
  },
  "screen": {
    "h_px": 1080,
    "w_px": 1920,
    "h_mm": 174,
    "w_mm": 310
  }
}
```

| Field | Type | Shape | Description |
|---|---|---|---|
| `camera.intrinsic` | `list[list[float]]` | 3 × 3 | Pinhole camera intrinsic matrix `[[fx, 0, cx], [0, fy, cy], [0, 0, 1]]`. `fx`, `fy` are focal lengths in pixels; `cx`, `cy` is the principal point in pixels. |
| `camera.distortion` | `list[float]` | 5 | OpenCV 5-parameter distortion coefficients `[k1, k2, p1, p2, k3]` in the usual radial + tangential order. Leave as all zeros for an approximately distortion-free lens (e.g. a factory-calibrated industrial camera). |
| `camera.extrinsic` | `list[list[float]]` | 3 × 4 | Screen-to-camera rigid transform `[R \| t]` written in row-major form. The 3×3 block `R` rotates screen-coordinate axes into camera axes; the 3×1 vector `t` is the position of the **screen origin** expressed **in the camera frame**, in millimeters. For a typical webcam sitting above the screen center, `t` follows the same sign convention as Layout A: left / below ⇒ negative X / Y. The default `R = diag(-1, 1, -1)` flips axes so that screen right/down map to camera left/up (matches the webcam-in-front-of-screen mounting). |




## References

[^1]: Yiwei Bao, Yihua Cheng, Yunfei Liu, and Feng Lu. "Adaptive Feature Fusion Network for Gaze Tracking in Mobile Tablets." In *International Conference on Pattern Recognition (ICPR)*, 2020.