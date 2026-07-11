# MolmoAct2 SO-ARM101 Evaluation UI

This example provides a local browser UI for evaluating
`allenai/MolmoAct2-SO100_101` on an SO-ARM101 follower.

Two inference backends are supported:

- **Local (recommended on Halo Strix)** — auto-starts the upstream
  [`molmoact2`](https://github.com/allenai/molmoact2) SO-101 server from
  `~/workspace/molmoact2` using its ROCm-aware launcher. No separate manual
  inference-server step.
- **Remote** — talks to an existing HTTP `/act` endpoint (json-numpy wire
  format).

Defaults live in `default.env`:

- Inference mode: `local`
- MolmoAct2 repo: `~/workspace/molmoact2`
- LeRobot calibration id: `my_awesome_follower_arm`
- One or two local OpenCV webcams at `640x480`

Camera mapping:

- **Local mode:** Top → `front_cam`, Side → `wrist_cam`
- **Remote mode:** Top → `top_cam`, Side → `side_cam`

If only Top is connected, the UI duplicates that frame into both camera inputs.

## Run

From the LeRobot repo root:

```bash
./examples/molmoact_so101_eval/run_eval.sh
```

With `MOLMOACT_INFERENCE=local` in `default.env`, this will:

1. Start `molmoact2/examples/so101/start_server.sh` (ROCm-friendly on AMD Strix)
2. Wait for the model to load on `http://127.0.0.1:8101/act`
3. Launch the browser UI on `http://127.0.0.1:7860`
4. Stop the local inference server when you exit the UI

First-time setup on Halo Strix (in the molmoact2 repo):

```bash
cd ~/workspace/molmoact2
sudo ./examples/so101/install_rocm_system.sh   # once
./examples/so101/setup_amd.sh                  # once
```

Override settings in `default.env` or create an untracked `.env.local`. CLI
flags are forwarded to the UI server, e.g. `--port 8080`.

Force a mode from the CLI:

```bash
./examples/molmoact_so101_eval/run_eval.sh --local
./examples/molmoact_so101_eval/run_eval.sh --remote
```

You can also run the launcher directly:

```bash
uv run --extra feetech python examples/molmoact_so101_eval/run_eval.py --local
```

Or start only the UI (remote endpoint or an already-running local server):

```bash
uv run --extra feetech python examples/molmoact_so101_eval/server.py \
  --endpoint http://127.0.0.1:8101/act \
  --inference-mode local \
  --inference-schema front_wrist
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
- `Robot Safety Clamp` is passed into LeRobot's `max_relative_target` for arm
  joints only. The gripper keeps the full 0-100 travel so close commands are
  not clipped to 10 units per tick.
- Gripper values use the same LeRobot 0-100 scale as the follower; only the
  five arm joints get the v3.0 → v2.1 frame conversion.
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
