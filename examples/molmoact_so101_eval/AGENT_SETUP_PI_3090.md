# Agent setup packet: Pi policy server on RTX 3090

**Audience:** an AI coding agent (or human) setting up **remote Pi inference** on a
machine with an **NVIDIA RTX 3090**, so a separate SO-101 robot host (e.g. AMD
Strix Halo) can call it over the LAN.

**Goal:** run an HTTP `/act` server on the 3090 that serves LeRobot **pi05** (or
**pi0**) with the same json-numpy wire format used by
`examples/molmoact_so101_eval/`.

**Out of scope on the 3090:** robot serial, cameras, and the browser UI. Those
stay on the robot machine.

---

## Architecture

```text
┌─────────────────────────────┐         LAN HTTP          ┌──────────────────────────────┐
│ Robot host (Strix / etc.)   │  POST /act (images+state) │ GPU host (RTX 3090)          │
│ run_eval.sh --policy pi     │ ─────────────────────────►│ host_server_pi.py :8102      │
│ --remote                    │ ◄─────────────────────────│ LeRobot pi05, CUDA, bf16     │
│ cameras + SO-101 + UI :7860 │      actions + dt_ms      │ bind 0.0.0.0                  │
└─────────────────────────────┘                           └──────────────────────────────┘
```

**Important model note:** Physical Intelligence **Pi 0.7 weights are not public**.
Use open-source LeRobot **`pi05`** (default) or **`pi0`**. Do not block on pi0.7.

---

## Success criteria (verify before handing off)

1. On the 3090: `curl -s http://127.0.0.1:8102/act` returns JSON with
   `"status":"ok"`, `"backend"` containing `lerobot-pi05-cuda`, `"gpu"` mentioning
   the 3090 / NVIDIA, `"dtype":"bfloat16"`, and **`"state_dim": 6`** (SO-101
   joints — not `32`).
2. From the **robot machine**: `curl -s http://<3090-LAN-IP>:8102/act` succeeds
   (same JSON).
3. A dry-run POST with dummy cameras returns `"actions"` shaped `(N, 6)` in under
   ~1–2 s on a warm 3090 (first call may be slower due to kernel warmup).

---

## Prerequisites (3090 machine)

| Requirement | Notes |
| --- | --- |
| Ubuntu (or similar) + NVIDIA driver | `nvidia-smi` must show the 3090 |
| CUDA-capable PyTorch | LeRobot’s default Linux torch is CUDA (`cu128`) |
| Git + `uv` | https://astral.sh/uv |
| Hugging Face account | Logged in; **accept gated PaliGemma terms** |
| Disk | ~20 GB free (pi05 weights ~14 GB + deps) |
| Network | LAN reachability to the robot host; open TCP **8102** |

### Hugging Face gated tokenizer (required)

1. While logged into Hugging Face, open and accept:
   https://huggingface.co/google/paligemma-3b-pt-224
2. On the 3090:

```bash
huggingface-cli login
# or: export HF_TOKEN=hf_...
```

Without this, the server dies after loading weights with a 403 on the tokenizer.

---

## Setup steps (3090 machine)

Work from a clone of this LeRobot repo (same revision the robot host uses if
possible).

### 1) Confirm GPU

```bash
nvidia-smi
# Expect RTX 3090 listed, driver OK
```

### 2) Install LeRobot with Pi extra

```bash
cd /path/to/lerobot
uv sync --locked --extra pi
```

Optional sanity check:

```bash
uv run --extra pi python -c "import torch; print(torch.__version__, torch.cuda.is_available(), torch.cuda.get_device_name(0))"
# Expect: True and a name containing 3090 / NVIDIA
```

### 3) Pre-download weights (optional but recommended)

```bash
uv run --extra pi huggingface-cli download lerobot/pi05_base
# Tokenizer (after accepting gated terms):
uv run --extra pi huggingface-cli download google/paligemma-3b-pt-224
```

### 4) Start the Pi `/act` server (listen on LAN)

Bind **`0.0.0.0`** so the robot host can connect. Use **bfloat16** and **5**
denoise steps (good 3090 defaults; raise steps to 10 only if quality needs it).

```bash
cd /path/to/lerobot

export PI_LOCAL_HOST=0.0.0.0
export PI_LOCAL_PORT=8102
export PI_CHECKPOINT=lerobot/pi05_base
export PI_POLICY_TYPE=pi05
export PI_DEVICE=cuda
export PI_DTYPE=bfloat16
export PI_NUM_INFERENCE_STEPS=5

# Preferred launcher (sets env + python):
./examples/molmoact_so101_eval/start_server_pi.sh

# Equivalent direct command:
# uv run --extra pi python examples/molmoact_so101_eval/host_server_pi.py \
#   --host 0.0.0.0 \
#   --port 8102 \
#   --checkpoint lerobot/pi05_base \
#   --policy-type pi05 \
#   --device cuda \
#   --dtype bfloat16 \
#   --num-inference-steps 5
```

First start: model load + warmup can take several minutes. Leave the process
running (tmux/systemd recommended).

### 5) Local health check

```bash
curl -s http://127.0.0.1:8102/act | python3 -m json.tool
```

Expect fields like:

- `"status": "ok"`
- `"backend": "lerobot-pi05-cuda"`
- `"device": "cuda"`
- `"dtype": "bfloat16"`
- `"num_inference_steps": 5`
- `"gpu": "...3090..."` (or similar NVIDIA name)
- `"repo_id": "lerobot/pi05_base"`

### 6) Firewall / LAN

```bash
# Example (ufw); adjust to site policy
sudo ufw allow 8102/tcp
sudo ufw status

hostname -I   # note the LAN IP, e.g. 192.168.0.50
```

From the **robot machine**:

```bash
curl -s http://<3090-LAN-IP>:8102/act | python3 -m json.tool
```

### 7) Optional: dummy inference smoke test (on 3090)

```bash
uv run --extra pi python - <<'PY'
import base64, json, time, urllib.request
import numpy as np

def enc(a):
    a = np.ascontiguousarray(a)
    return {"__numpy__": base64.b64encode(a.data).decode(), "dtype": a.dtype.str, "shape": list(a.shape)}

payload = {
    "top_cam": enc(np.zeros((480, 640, 3), np.uint8)),
    "side_cam": enc(np.zeros((480, 640, 3), np.uint8)),
    "instruction": "pick up the object",
    "state": enc(np.zeros(6, np.float32)),
}
data = json.dumps(payload).encode()
req = urllib.request.Request(
    "http://127.0.0.1:8102/act",
    data=data,
    headers={"Content-Type": "application/json"},
)
t0 = time.time()
with urllib.request.urlopen(req, timeout=180) as resp:
    body = json.loads(resp.read().decode())
print("wall_s", round(time.time() - t0, 2), "dt_ms", body.get("dt_ms"))
acts = body["actions"]
if isinstance(acts, dict) and "__numpy__" in acts:
    arr = np.frombuffer(base64.b64decode(acts["__numpy__"]), dtype=np.dtype(acts["dtype"])).reshape(acts["shape"])
else:
    arr = np.asarray(acts)
print("actions", arr.shape)
assert arr.ndim == 2 and arr.shape[1] == 6
PY
```

---

## Robot-host configuration (Strix / SO-101 machine)

Do **not** start local Pi on the robot host when using the 3090.

Create or edit `examples/molmoact_so101_eval/.env.local`:

```bash
EVAL_POLICY=pi
EVAL_INFERENCE=remote
PI_ENDPOINT=http://jedld-lab:8102/act
# Joint conversion stays off for Pi (launcher sets this).
```

Then:

```bash
cd /path/to/lerobot
./examples/molmoact_so101_eval/run_eval.sh --policy pi --remote
```

Open `http://127.0.0.1:7860`, connect cameras + robot, **Check Endpoint**, then
**Dry Run** before **Start Evaluation**.

UI expectations:

- Backend: `remote endpoint`
- Policy: Pi (pi05/pi0)
- Joint conversion: **off**
- Endpoint status: `lerobot/pi05_base` (or your checkpoint id)

---

## Optional: SO-101 fine-tuned checkpoint

If using a community SO-101 finetune instead of the base model:

```bash
export PI_CHECKPOINT=L7-Robotics/pi05_so101_v6.1   # example
# or a local path to a LeRobot checkpoint directory
./examples/molmoact_so101_eval/start_server_pi.sh
```

Camera key mapping is inferred from the checkpoint; override only if needed:

```bash
export PI_TOP_IMAGE_KEY=base_0_rgb
export PI_SIDE_IMAGE_KEY=left_wrist_0_rgb
```

---

## Suggested systemd unit (3090, optional)

`/etc/systemd/system/pi-act.service`:

```ini
[Unit]
Description=LeRobot Pi /act server (SO-101)
After=network.target

[Service]
Type=simple
User=YOUR_USER
WorkingDirectory=/path/to/lerobot
Environment=PI_LOCAL_HOST=0.0.0.0
Environment=PI_LOCAL_PORT=8102
Environment=PI_CHECKPOINT=lerobot/pi05_base
Environment=PI_POLICY_TYPE=pi05
Environment=PI_DEVICE=cuda
Environment=PI_DTYPE=bfloat16
Environment=PI_NUM_INFERENCE_STEPS=5
Environment=HF_TOKEN=hf_...
ExecStart=/path/to/lerobot/examples/molmoact_so101_eval/start_server_pi.sh
Restart=on-failure
RestartSec=5

[Install]
WantedBy=multi-user.target
```

```bash
sudo systemctl daemon-reload
sudo systemctl enable --now pi-act.service
journalctl -u pi-act.service -f
```

---

## Troubleshooting

| Symptom | Fix |
| --- | --- |
| `403` / gated repo on `google/paligemma-3b-pt-224` | Accept model terms + `huggingface-cli login` |
| `cuda False` / CPU fallback | Install CUDA torch via `uv sync --extra pi`; check `nvidia-smi` |
| Robot cannot curl `:8102` | Bind `0.0.0.0`, open firewall, confirm LAN IP, no VPN isolation |
| Slow first request | Normal (warmup); server does one warmup at start — wait until “listening” |
| Still ~seconds per call on 3090 | Confirm `"dtype":"bfloat16"`; try `PI_NUM_INFERENCE_STEPS=3`; ensure no other GPU jobs |
| `POST /act` returns **400** `state must be shape (32,)` | Wrong server build on the GPU host. Redeploy **`host_server_pi.py` from this repo** (expects **6** SO-101 joints, returns `(N, 6)` actions). Health JSON should show `"state_dim": 6`, not `32`. |
| `POST /act` returns **400** (other) | Check server logs; response body includes `"error"`. Run the dummy POST in step 7 with **6-dim** state. |
| Robot uses joint conversion | Pi mode must leave conversion **off** (MolmoAct2-only) |

---

## Files involved (do not reinvent)

| Path | Role |
| --- | --- |
| `examples/molmoact_so101_eval/host_server_pi.py` | `/act` HTTP server |
| `examples/molmoact_so101_eval/start_server_pi.sh` | Launcher (env → CLI) |
| `examples/molmoact_so101_eval/run_eval.py` | Robot-side UI launcher (`--policy pi --remote`) |
| `examples/molmoact_so101_eval/default.env` | Defaults; override with `.env.local` |
| `examples/molmoact_so101_eval/README.md` | Broader eval UI docs |

**Do not** run `setup_amd_pi.sh` on the 3090 — that is ROCm/Strix-only.

---

## Agent checklist (copy/paste)

```text
[ ] nvidia-smi shows RTX 3090
[ ] uv sync --locked --extra pi
[ ] Accepted https://huggingface.co/google/paligemma-3b-pt-224
[ ] huggingface-cli login (or HF_TOKEN set)
[ ] start_server_pi.sh with PI_LOCAL_HOST=0.0.0.0 PI_DTYPE=bfloat16
[ ] curl localhost:8102/act → status ok, cuda, bfloat16
[ ] curl from robot host → same
[ ] Robot .env.local: EVAL_POLICY=pi EVAL_INFERENCE=remote PI_ENDPOINT=http://jedld-lab:8102/act
[ ] Robot: run_eval.sh --policy pi --remote → Check Endpoint + Dry Run OK
```
