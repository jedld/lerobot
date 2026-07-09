# MolmoAct2 SO-ARM101 Evaluation UI

This example provides a local browser UI for evaluating the deployed
`allenai/MolmoAct2-SO100_101` inference server on an SO-ARM101 follower.

It is configured for:

- MolmoAct2 endpoint: `http://192.168.0.233:8014/act`
- LeRobot calibration id: `jedld-follower`
- One or two local OpenCV webcams. Use `640x480` for lower latency preview and
  inference requests.
- Camera dropdowns are labeled by OpenCV index because macOS AVFoundation names
  can disagree with the indices OpenCV actually opens.
- SO100/SO101 server schema: `top_cam`, `side_cam`, `instruction`, `state`

The deployed checkpoint expects two camera inputs. Connect **Top Webcam** and
optionally **Side Webcam** to send different frames as `top_cam` and `side_cam`.
If only Top is connected, this UI duplicates the Top frame into both fields.

## Run

From the LeRobot repo root:

```bash
uv run python examples/molmoact_so101_eval/server.py
```

Then open:

```text
http://127.0.0.1:7860
```

## Suggested Flow

1. Press **Auto Detect Robot Port** or **Refresh Ports** and select the
   SO-ARM101 serial port.
2. Press **Check Endpoint** and confirm the endpoint reports
   `allenai/MolmoAct2-SO100_101`.
3. Select and connect **Top Webcam**. Optionally select a different camera and
   press **Connect Side Webcam**. Verify both previews.
4. Press **Connect Robot** with robot id `jedld-follower`.
5. Press **Program Mode** to disable torque and move the joints by hand. The
   live joint table updates while torque is off.
6. Press **Exit Program** before commanding motion again.
7. Enter a task instruction.
8. Press **Dry Run** first. This calls inference but does not move the arm.
9. Press **Start Evaluation** to repeatedly infer and execute action chunks.
10. Press **Stop** before changing the scene or instruction.
11. Press **Reset Home** to stop evaluation and command the follower toward the
   centered home pose.
12. Press **Pack Robot** to stop evaluation and command the follower toward a
    compact packed pose.

## Safety Notes

- Keep an emergency stop or power switch within reach.
- Robot port auto-detection prefers macOS callout ports like
  `/dev/cu.usbmodem*`, then falls back to Linux-style `/dev/ttyACM*` and
  `/dev/ttyUSB*`.
- `Max Step Deg` follows the SO-101 reference implementation's vector-scaled
  per-tick action cap. If any joint would move farther than the cap, the whole
  action delta is scaled down proportionally.
- `Robot Safety Clamp` is also passed into LeRobot's `max_relative_target`,
  giving a second hardware-side clamp.
- Evaluation is synchronous and does not use async temporal ensembling. The
  default loop is observe -> infer -> execute the returned chunk -> observe
  again.
- `Actions Per Chunk` defaults to all actions. Set it to a positive number to
  cap how many actions are executed from each returned chunk before re-querying
  the MolmoAct2 endpoint.
- `Smoothing Alpha` applies EMA filtering to model action targets. Lower values
  smooth more aggressively; `1.0` disables filtering. The default is `0.8`.
- `Interpolation Steps` splits each model action into smaller joint-space
  substeps. This streams commands at `Action FPS * Interpolation Steps` and
  reduces stop-start vibration while preserving the max-step safety clamp. The
  default is `3`.
- Reset, pack, and evaluation commands are serialized so evaluation cannot
  start executing while a pose move is still in progress.
- Inference waits for a fresh camera frame, then immediately reads the robot
  joints for the same observation. The UI reports observation/action IDs,
  image-to-joint skew, camera-to-camera skew, and whether `side_cam` came from
  the Side camera or a Top-frame duplicate under **Obs Sync**.
- **Apply SO-101 v3 to v2.1 joint conversion** is enabled by default for
  MolmoAct2 inference. It sends `signs * arm_state + offsets` to the model and
  converts returned actions back with `(model_action - offsets) * signs`, using
  offsets `[0, 90, 90, 0, 0, 0]` and signs `[1, -1, 1, 1, 1, 1]`.
- **Program Mode** disables all follower motor torque so the arm can be moved
  manually. Hold the arm before entering this mode.
- **Exit Program**, **Start Evaluation**, **Reset Home**, and **Pack Robot**
  re-enable torque before commanding motion.
- **Reset Home** uses the same robot action path and safety clamp. It commands
  arm joints to `0` and gripper to `50`.
- **Pack Robot** also uses the same safety clamp. It commands
  `shoulder_pan=-0.3077`, `shoulder_lift=-103.9121`,
  `elbow_flex=97.3187`, `wrist_flex=72.6593`, `wrist_roll=-0.1319`,
  and `gripper=0.7628`.
- Leave **Run calibration if needed** unchecked when using the existing
  `jedld-follower` calibration.
- **Disconnect** stops the loop, disconnects the robot, and releases the camera.
- UI settings are persisted in browser local storage. When **Auto-connect saved
  cameras and robot on page load** is checked, the page reconnects the saved
  Top/Side cameras and robot after reload or server restart. Auto-connect never
  runs calibration.

## Reference Notes

This UI was checked against
[`irenegracekp/molmoact2-so101`](https://github.com/irenegracekp/molmoact2-so101).
The reference runs MolmoAct2 locally with a RealSense scene camera plus a wrist
webcam, uses `scene-only` as a fallback for out-of-distribution wrist views, and
applies a vector-scaled `max-step-deg` clamp before sending targets. This UI
keeps the single-Brio duplicated-camera setup because the MolmoAct2 inference is
  already hosted at `192.168.0.233:8014`, but supports one or two local OpenCV
  cameras and adopts the reference-style action clamp, synchronous execution
  loop, and SO-101 v3.0 to v2.1 joint-frame conversion.
