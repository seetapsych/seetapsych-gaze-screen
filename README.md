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
  seetapsych_gaze_screen/modules/itracker-plus.yml \
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
| `itracker-plus.yml` | GazeScreen-ITrackerPlus(OpenGaze) |
| `tdgazenet.yml` | GazeScreen-TDGazeNet(OpenGaze) |

### GazeScreen-AFFNet(OpenGaze)

> Open-source gaze estimation toolkit providing screen gaze coordinates from facial landmarks or mesh.

Module config: [affnet.yml](seetapsych_gaze_screen/modules/affnet.yml)

| Package | Provides | Requires |
|---|---|---|
| GazeScreen-AFFNet(OpenGaze) | `face/gaze_screen` | `face/mesh` |

**Description**

Estimate screen gaze coordinates via AFFNet attention fusion network; outputs one shared gaze vector applied to both eyes. Suitable for single-user desktop scenarios with medium accuracy and compute cost.

**Parameters**

| Name | Type | Default | Description |
|---|---|---|---|
| `data` | object | *(see yml)* | Camera/screen calibration and image preprocessing settings. camera and screen physical dimensions (mm/px) and sensor frame size must match the real setup for accurate gaze projection; adjust_size resizes input for throughput, larger increases accuracy but slows inference. |

**Models**

| Model | Version | Recommended |
|---|---|---|
| `opengaze-affnet-v2.safetensors` | 2.0 | ✓ |
| `opengaze-affnet.safetensors` | 1.0 |  |

### GazeScreen-ITrackerPlus(OpenGaze)

> Open-source gaze estimation toolkit providing screen gaze coordinates from facial landmarks or mesh.

Module config: [itracker-plus.yml](seetapsych_gaze_screen/modules/itracker-plus.yml)

| Package | Provides | Requires |
|---|---|---|
| GazeScreen-ITrackerPlus(OpenGaze) | `face/gaze_screen` | `face/mesh` |

**Description**

Lightweight eye-and-face gaze point estimation; outputs one shared gaze vector applied to both eyes. Best for resource-constrained devices or real-time low-latency use cases with acceptable accuracy.

**Parameters**

| Name | Type | Default | Description |
|---|---|---|---|
| `data` | object | *(see yml)* | Camera/screen calibration and image preprocessing settings. camera and screen physical dimensions (mm/px) and sensor frame size must match the real setup for accurate gaze projection; adjust_size resizes input for throughput, larger increases accuracy but slows inference. |

**Models**

| Model | Recommended |
|---|---|
| `opengaze-itrackerplus.onnx` | ✓ |

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
| `data` | object | *(see yml)* | Model topology, camera/screen calibration and image preprocessing settings. camera intrinsic/extrinsic/distortion matrices and screen physical dimensions must match the real setup for the 3D-face-prior projection to be accurate; image_size and bbox_scale tune face crop resolution vs. context margin. |

**Models**

| Model | Recommended |
|---|---|
| `opengeze-tdgazenet.safetensors` | ✓ |
