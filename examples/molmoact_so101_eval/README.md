# SO-ARM101 Policy Evaluation UI

This example provides a local browser UI for evaluating vision-language-action
policies on an SO-ARM101 follower. Two policy backends are supported:

- **MolmoAct2** — `allenai/MolmoAct2-SO100_101` via the upstream
  [`molmoact2`](https://github.com/allenai/molmoact2) SO-101 server or a remote
  `/act` endpoint.
- **Pi (pi05 / pi0)** — open-source LeRobot Pi policies via
  `host_server_pi.py`. **Pi 0.7 weights are not publicly released yet**; use
  `lerobot/pi05_base`, `lerobot/pi0_base`, or a community SO-101 fine-tune such
  as [`L7-Robotics/pi05_so101_v6.1`](https://huggingface.co/L7-Robotics/pi05_so101_v6.1).

Both backends expose the same json-numpy `/act` wire format, so the UI loop is
unchanged. Each backend supports **local managed inference** (auto-started
subprocess) and **remote inference** (existing HTTP endpoint).

Defaults live in `default.env`:

- Policy: `molmoact2`
- Inference mode: `local`
- MolmoAct2 repo: `~/workspace/molmoact2`
- Pi checkpoint: `lerobot/pi05_base`
- LeRobot calibration id: `my_awesome_follower_arm`
- One or two local OpenCV webcams at `640x480`

The browser previews use persistent, latest-frame MJPEG streams instead of polling
individual JPEGs. The default 12 FPS, JPEG quality 65, and 640-pixel maximum width
keeps latency and bandwidth substantially lower over the internet. For constrained
links, use the UI's preview controls to select 8 FPS, quality 55, and width 480.

Camera mapping:

| Policy | Local mode | Remote mode |
| --- | --- | --- |
| MolmoAct2 | Top → `front_cam`, Side → `wrist_cam` | Top → `top_cam`, Side → `side_cam` |
| Pi | Top → `top_cam`, Side → `side_cam` (renamed server-side to checkpoint keys) | same |

If only Top is connected, the UI duplicates that frame into both camera inputs.

## Run

From the LeRobot repo root:

```bash
# MolmoAct2 (default)
./examples/molmoact_so101_eval/run_eval.sh

# Pi pi05 local inference (Pi 0.7 is not available — uses open pi05/pi0)
./examples/molmoact_so101_eval/run_eval.sh --policy pi --local

# Pi remote inference on another GPU machine
./examples/molmoact_so101_eval/run_eval.sh --policy pi --remote
```

With `EVAL_INFERENCE=local`, the launcher will:

1. Start the selected policy server (`molmoact2` or `host_server_pi.py`)
2. Wait for the model to load on the local `/act` port (8101 MolmoAct2, 8102 Pi)
3. Launch the browser UI on `http://127.0.0.1:7860`
4. Stop the local inference server when you exit the UI

First-time MolmoAct2 setup on Halo Strix (in the molmoact2 repo):

```bash
cd ~/workspace/molmoact2
sudo ./examples/so101/install_rocm_system.sh   # once
./examples/so101/setup_amd.sh                  # once
```

Pi local inference requires the LeRobot `pi` extra. On **AMD Strix Halo** you
also need a ROCm PyTorch venv (the default LeRobot venv is CUDA-only and will
not see the Radeon 890M):

```bash
# once — creates examples/molmoact_so101_eval/.venv-rocm
./examples/molmoact_so101_eval/setup_amd_pi.sh

# Hugging Face gated tokenizer (required for pi05/pi0)
# 1. Accept terms: https://huggingface.co/google/paligemma-3b-pt-224
# 2. huggingface-cli login

# then
./examples/molmoact_so101_eval/run_eval.sh --policy pi --local
```

First Pi launch downloads ~14 GB (`lerobot/pi05_base`). Health check:

```bash
curl http://127.0.0.1:8102/act
# should report backend like lerobot-pi05-rocm, dtype bfloat16, gpu "AMD Radeon 890M"
```

Speed knobs (Strix defaults are already set in `default.env`):

- `PI_DTYPE=bfloat16` — much faster than float32 on the 890M
- `PI_NUM_INFERENCE_STEPS=5` — fewer flow-matching denoise steps (try `3` if still slow; quality drops)

Restart the launcher after changing these. On NVIDIA, `run_eval.sh --policy pi --local` uses the normal LeRobot CUDA venv.

Override settings in `default.env` or create an untracked `.env.local`. CLI flags
are forwarded to the UI server, e.g. `--port 8080`.

Force a mode from the CLI:

```bash
./examples/molmoact_so101_eval/run_eval.sh --local
./examples/molmoact_so101_eval/run_eval.sh --remote
./examples/molmoact_so101_eval/run_eval.sh --policy pi --pi-checkpoint L7-Robotics/pi05_so101_v6.1
```

You can also run the launcher directly:

```bash
uv run --extra feetech --extra pi python examples/molmoact_so101_eval/run_eval.py --policy pi --local
```

### Remote Pi server only

For a full agent-oriented setup packet targeting an **RTX 3090** (or any NVIDIA
CUDA box), see [`AGENT_SETUP_PI_3090.md`](./AGENT_SETUP_PI_3090.md).

On a GPU machine:

```bash
uv run --extra pi python examples/molmoact_so101_eval/host_server_pi.py \
  --host 0.0.0.0 \
  --port 8102 \
  --checkpoint lerobot/pi05_base \
  --policy-type pi05
```

Then on the robot machine, set `PI_ENDPOINT=http://jedld-lab:8101/act` (default in
`default.env`) and run with `--policy pi --remote`.

Or start only the UI (remote endpoint or an already-running local server):

```bash
uv run --extra feetech --extra pi python examples/molmoact_so101_eval/server.py \
  --policy pi \
  --endpoint http://127.0.0.1:8102/act \
  --inference-mode local \
  --inference-schema top_side \
  --default-apply-joint-conversion false
```

Then open:

```text
http://127.0.0.1:7860
```

## Suggested Flow

1. Press **Auto Detect Robot Port** or **Refresh Ports** and select the
   SO-ARM101 serial port.
2. Press **Check Endpoint** and confirm the endpoint reports the expected
   checkpoint (`allenai/MolmoAct2-SO100_101` or your Pi repo id).
3. Select and connect **Top Webcam**. Optionally select a different camera and
   press **Connect Side Webcam**. Verify both previews.
4. Press **Connect Robot** with your calibration id.
5. Press **Program Mode** to disable torque and move the joints by hand. The
   live joint table updates while torque is off.
6. Press **Exit Program** before commanding motion again.
7. Enter a task instruction.
8. Press **Dry Run** first. This calls inference but does not move the arm.
9. Press **Start Evaluation** to repeatedly infer and execute action chunks.
10. Press **Stop** before changing the scene or instruction.
11. Press **Reset Start** to stop evaluation and command the follower toward the
    MolmoAct2 research start pose (training-state median).
12. Press **Pack Robot** to stop evaluation and command the follower toward a
    compact packed pose.

## Pi vs MolmoAct2 Notes

- **Joint conversion** (SO-101 v3.0 → v2.1 frame) applies to MolmoAct2 only.
  Leave it **off** for Pi policies — they expect native LeRobot joint degrees.
- Pi checkpoints may expect different camera key names (`base_0_rgb`,
  `left_wrist_0_rgb`, etc.). `host_server_pi.py` infers a mapping from the
  checkpoint config; override with `PI_TOP_IMAGE_KEY` / `PI_SIDE_IMAGE_KEY` or
  `--top-image-key` / `--side-image-key` if needed.
- Community SO-101 Pi fine-tunes often work better than `lerobot/pi05_base`
  zero-shot, but always dry-run and verify gripper behavior first.

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
- Gripper values use the same LeRobot 0-100 scale as the follower; only MolmoAct2
  arm joints get the v3.0 → v2.1 frame conversion when enabled.
- Evaluation is synchronous. MolmoAct2 is forced fully open-loop: observe →
  infer → execute the **entire** returned chunk → observe again. There is no
  real-time chunking (RTC), no leftover-chunk mixing, and no EMA action
  blending (`smoothing_alpha` is locked to `1.0`).
- For Pi, `Actions Per Chunk` defaults to all actions. Set it to a positive
  number to cap how many actions are executed from each returned chunk before
  re-querying the policy endpoint.
- For Pi, `Smoothing Alpha` applies EMA filtering to model action targets.
  Lower values smooth more aggressively; `1.0` disables filtering. The default
  is `0.8`. MolmoAct2 ignores this control.
- `Interpolation Steps` splits each model action into smaller joint-space
  substeps. This streams commands at `Action FPS * Interpolation Steps` and
  reduces stop-start vibration while preserving the max-step safety clamp. The
  default is `3`. This is control-rate densification only, not prediction
  mixing.
- Reset, pack, and evaluation commands are serialized so evaluation cannot
  start executing while a pose move is still in progress.
- Inference waits for a fresh camera frame, then immediately reads the robot
  joints for the same observation. The UI reports observation/action IDs,
  image-to-joint skew, camera-to-camera skew, and whether `side_cam` came from
  the Side camera or a Top-frame duplicate under **Obs Sync**.
- **Apply SO-101 v3 to v2.1 joint conversion** is enabled by default for
  MolmoAct2 inference only.
- **Program Mode** disables all follower motor torque so the arm can be moved
  manually. Hold the arm before entering this mode.
- **Exit Program**, **Start Evaluation**, **Reset Start**, and **Pack Robot**
  re-enable torque before commanding motion.
- **Reset Start** uses the same robot action path and safety clamp. It commands
  the MolmoAct2-SO100_101 research start pose: the checkpoint
  `norm_stats.json` state q50 for `so100_so101_molmoact2` (model / v2.1 frame),
  converted into the LeRobot v3 arm frame
  (`shoulder_pan≈3.07`, `shoulder_lift≈-33.16`, `elbow_flex≈34.40`,
  `wrist_flex≈57.89`, `wrist_roll≈-11.04`, `gripper≈9.24`).
- **Pack Robot** also uses the same safety clamp. It commands
  `shoulder_pan=-0.3077`, `shoulder_lift=-103.9121`,
  `elbow_flex=97.3187`, `wrist_flex=72.6593`, `wrist_roll=-0.1319`,
  and `gripper=0.7628`.
- Leave **Run calibration if needed** unchecked when using an existing
  calibration.
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
applies a vector-scaled `max-step-deg` clamp before sending targets.

Pi support follows the LeRobot
[`using_pi0_example.py`](../../examples/tutorial/pi0/using_pi0_example.py)
pattern, wrapped in an `/act` HTTP server compatible with this UI.
