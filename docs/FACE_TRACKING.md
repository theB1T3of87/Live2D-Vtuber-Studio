# Face tracking — Live2D Vtuber Studio

## Active pipeline

The overlay uses `MnnFaceTrackingController` as its active camera-to-model pipeline:

1. `Camera2Manager` opens the front camera and converts camera frames to rotated RGB.
2. The callback copies each reusable RGB buffer before returning to Camera2, preventing inference from reading a buffer that has already been overwritten by the next frame.
3. `MediaPipeFaceDetector` runs the MNN face detector, 478-point landmark model, head-pose estimator and ARKit-compatible 52 blendshape predictor.
4. The controller calibrates a neutral head pose from 24 valid frames, maps pose/expression/gaze values to Cubism axes, and smooths the result.
5. `Live2DRenderer.setFaceTrackingData()` sends the latest values through JNI.
6. `LAppModel::Update()` applies the face parameters after motion/physics updates so the camera remains the final source of head, eye and mouth values for that frame.

All inference is local to the device. A single inference worker and an in-flight guard avoid stale-frame backlogs.

## Parameter mapping

| Input | Cubism parameter |
|---|---|
| Head yaw | `ParamAngleX` |
| Head pitch | `ParamAngleY` |
| Head roll | `ParamAngleZ` |
| Eye blink left/right | `ParamEyeLOpen`, `ParamEyeROpen` |
| Eye-look blendshapes | `ParamEyeBallX`, `ParamEyeBallY` |
| Jaw open | `ParamMouthOpenY` |
| Smile minus frown | `ParamMouthForm` |

The model must contain the corresponding conventional Cubism parameter IDs for each movement to be visible. Models with custom parameter IDs need a model-specific mapping.

## Render pipeline

`Live2DRenderer` hosts the official Cubism Native Framework on a dedicated EGL/OpenGL ES render thread. The renderer and face pipeline are separate: camera inference publishes immutable tracking values through JNI, while the GL thread owns Cubism rendering. EGL initialization, swap failures, queued command failures and render-thread shutdown are logged rather than silently swallowed.

## Calibration and troubleshooting

- Keep the face centered and look forward briefly while the neutral pose is calibrated.
- The camera permission and Android's foreground camera restrictions must allow the overlay service to use the front camera.
- Check Logcat tags `Live2DFacePipeline` and `Live2DRenderer` for inference and EGL diagnostics.
- If the model does not move, verify that its `model3.json`/MOC3 exposes the standard `ParamAngleX/Y/Z`, `ParamEyeLOpen/ROpen` and `ParamMouthOpenY` IDs.

## Experimental implementation

`FaceTrackingController.kt` contains an older CameraX / MediaPipe Tasks implementation and is not the controller currently wired to `OverlayService`. The top-level `tracking-integration/` folder contains the MNN-based camera, landmark and blendshape implementation used by the active pipeline.
