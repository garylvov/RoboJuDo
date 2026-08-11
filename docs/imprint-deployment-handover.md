# Imprint × RoboJuDo H1-2 deployment — handover

**Branch:** `dev/imprint-h1_2-deployment`
**Base:** `release` @ `377b0c5bbcfa94c10690fcb5b3e8ee2f97414e92`
**Written:** 2026-08-11
**Audience:** someone with zero prior context on this integration.

> **Read this first.** The single most important finding of this handover is in
> §2: **the RoboJuDo-side deployment patch that Imprint has been carrying is
> already merged into `release`.** There is nothing to port. If you were handed
> `patches/third_party/RoboJuDo/0001-native-imprint-deployment.patch` and told to
> apply it, do not — it will fail, and it *should* fail, because its content is
> already here.

---

## 1. What RoboJuDo is to us

RoboJuDo is the **deployment / sim2real control stack** for the Unitree H1-2.
It is the thing that takes a trained policy and actually *runs* it — on a
simulated robot or on hardware — with a real controller, a real observation
pipeline, and a real safety gate.

It is **not** a scoring harness. This distinction has burned people repeatedly,
so it is stated bluntly:

| | **RoboJuDo (this repo)** | **`wt-evalharness` scoring harness** |
|---|---|---|
| Question it answers | "Can this policy *run* the robot?" | "How *good* is this policy?" |
| Output | Robot motion, receipts, logs | SR / body-error / wrist-error numbers |
| Location | `github.com/garylvov/RoboJuDo` | `/oscar/data/stellex/glvov/wt-evalharness/scripts/wbc/eval/` |
| Metric authority | none — it does not score | `mj_metrics.py`, single source of truth |

**The confusion trap:** a directory named `robojudo_eval/` once existed. It was
**deleted entirely** and rebuilt as the scoring harness now living at
`wt-evalharness/scripts/wbc/eval/`. Verified today: there is no `robojudo_eval`
directory in `wt-evalharness`, and no commit in its history ever touched that
path. So if you find a document referencing `robojudo_eval/`, that document is
stale. "RoboJuDo" (deployment) and "the eval harness" (scoring) are two
different things that happen to both drive MuJoCo.

### The deployment path, component by component

All Imprint-side code lives in `src/imprint/robojudo/` in the Imprint repo
(`github.com/garylvov-chewy/imprint`). Eleven modules:

| Module | Role |
|---|---|
| `native.py` | Registers Imprint configs (`imprint_newton`, `imprint_mujoco`) into **RoboJuDo's own** launcher. This is the "native deployment" entrypoint. |
| `run_policy.py` | ONNX policy runner — loads an exported policy and drives the control loop. |
| `recurrent_onnx.py` | Recurrent-policy ONNX runner (carries hidden state across ticks). Separate from `run_policy.py` because the state plumbing is genuinely different. |
| `sage_export.py` | SAGE export path. |
| `recorder.py` | Records rollouts to disk for replay/inspection. |
| `h1_2_ability_scene.py` | The H1-2 + bilateral PSYONIC Ability-hand MuJoCo scene (47 joints, 39 actuators). |
| `excite.py` | Excitation-signal generation (system-ID trajectories). |
| `sysid_windows.py` | System-ID windowing over recorded data. |
| `shared_runtime.py` | The shared deployment runtime — the `run(steps)` lifecycle object the RoboJuDo adapter consumes. Largest module (33 KB). |
| `newton_adapter.py` | Imprint's Newton/IsaacLab backend shim. |
| `__init__.py` | Gated inference entrypoints (Enter-to-damp / `g`-to-resume safety gate). |

### The architectural idea

RoboJuDo owns the **finite rollout lifecycle**; Imprint owns the **backend**.
The seam is a registered adapter, so a mislabeled rollout fails before its first
control tick rather than silently driving the wrong simulator:

```
run_pipeline.py --config-module imprint.robojudo.native --config imprint_newton
      │
      ▼
DeploymentPipelineCfg  (robojudo/pipeline/pipeline_cfgs.py)
      │  adapter_type: ImprintNewtonAdapter | ImprintMujocoAdapter
      ▼
deployment_adapter_registry  (robojudo/deployment/__init__.py)
      │
      ▼
_ImprintAdapter.build(cfg)   (robojudo/deployment/imprint_adapter.py)
      │  resolves cfg.bootstrap "module:callable"
      │  ASSERTS session.backend.capabilities.kind == adapter's backend_kind
      ▼
Imprint SharedDeploymentRuntime  →  .run(steps)
```

The `bootstrap` is resolved **only at pipeline construction time**. That is
deliberate and load-bearing: Isaac/Newton must establish `AppLauncher` before any
simulator module is imported, so the bootstrap cannot be eagerly resolved at
config-parse time.

---

## 2. Exact state — what is committed vs what is patch-only

### The headline: the patch is already merged

Imprint carries two patch files:

- `patches/third_party/RoboJuDo/0001-native-imprint-deployment.patch` (+ `series`, `target`)
- `patches/third_party/robojudo-deployment-wip.patch`

**Both encode the same change to the same 7 files, in different file order.**
They are not two different pieces of work.

Verified today, in a fresh clone:

```
$ diff -q <(git diff 2c7dacc^ 2c7dacc) patches/third_party/robojudo-deployment-wip.patch
IDENTICAL
```

`robojudo-deployment-wip.patch` is **byte-identical** to the diff of commit
`2c7dacc`. And `2c7dacc` is `refs/heads/deployment-pipeline-wip`, which is an
**ancestor of `origin/release`**:

```
$ git merge-base --is-ancestor 2c7dacc origin/release && echo YES
YES: 2c7dacc IS an ancestor of origin/release
```

`0001-native-imprint-deployment.patch` is the same content with the three
modified files hoisted above the four new files — a hand-reordered regeneration
of the same commit.

### Why `git apply` fails (and why that is correct)

```
$ git apply --check 0001-native-imprint-deployment.patch
error: robojudo/pipeline/__init__.py: patch does not apply
error: robojudo/pipeline/pipeline_cfgs.py: patch does not apply
error: scripts/run_pipeline.py: patch does not apply
```

It fails because the pre-image context it searches for no longer exists —
the change is already in the tree. Reverse-apply also fails
(`git apply --check -R` → `scripts/run_pipeline.py:28`), because two of the
seven files have received **further** edits since `2c7dacc`.

**This is the trap the campaign was warned about, in its inverted form.** The
danger here was never a patch that silently mis-applies; it is that someone
reads "patch does not apply" as "our work is missing" and force-resolves it back
in — re-introducing an *older* version of two files that have since moved on.
Do not do that.

### Per-file ledger

Blob hashes, patch post-image vs `origin/release`:

| File | at `2c7dacc` (= patch output) | at `origin/release` | status |
|---|---|---|---|
| `robojudo/deployment/__init__.py` | `3c71622` | `3c71622` | identical |
| `robojudo/deployment/base_adapter.py` | `998a47e` | `998a47e` | identical |
| `robojudo/deployment/imprint_adapter.py` | `f9279fd` | `f9279fd` | identical |
| `robojudo/pipeline/deployment_pipeline.py` | `275e1de` | `275e1de` | identical |
| `robojudo/pipeline/__init__.py` | `617dc3a` | `617dc3a` | identical |
| `robojudo/pipeline/pipeline_cfgs.py` | `d9c5948` | **`dfd94c8`** | drifted forward |
| `scripts/run_pipeline.py` | `7b0a8cb` | **`bb614bb`** | drifted forward |

The two drifted files were checked to confirm the deployment additions
**survived** the drift — `DeploymentPipelineCfg` is present at
`pipeline_cfgs.py:32`, and `--config-module` / `--steps` / `--bootstrap` /
`--bootstrap-kwargs-json` / the `run_to_completion` dispatch are all present in
`scripts/run_pipeline.py` on `release`.

**Conclusion: nothing in either Imprint patch is missing from `release`.
Both patch files are obsolete and should be deleted from Imprint.**

### Branch choice

Created `dev/imprint-h1_2-deployment` off **`release`** (`377b0c5`), per owner
directive not to use `dev/h1_2-teleop`.

`release` exists under exactly that name on the remote and is the repo's default
HEAD, so no substitution was necessary. For the record, the full remote branch
list at time of writing: `release`, `deployment-pipeline-wip`, `dev/agent-init`,
`dev/h1_2-teleop`, `dev/sim-test-harness`, `feat/torso-imu`,
`glvov/deploy-controls-july19`, `glvov/orireach-deploy-r14`,
`glvov/release-anchor-py310`, `glvov/requires-python-310`,
`glvov/teleop-odom-ctrl-fix`, `h1_2-protomotions-support`.

Name rationale: `dev/` matches the repo's existing convention for integration
branches; `imprint-h1_2-deployment` states the two systems and the robot, and
distinguishes it from the teleop line.

### ⚠️ `release` is missing masked-mimic — this affects the MM student

`dev/h1_2-teleop` is **1 commit ahead** of `release` (and `release` is 31 ahead
of it). That one commit is `f76c14d`, and it is not cosmetic:

```
robojudo/config/h1_2/h1_2_masked_mimic_pipeline_cfg.py       |  53 ++
robojudo/config/h1_2/policy/h1_2_masked_mimic_cfg.py         |  55 ++
robojudo/policy/protomotions_masked_mimic_policy.py          | 703 +++++++
robojudo/controller/teleop_ctrl.py                           |  33 +-
robojudo/environment/mujoco_env.py                           |  36 +-
robojudo/policy/protomotions_tracker_policy.py               |  15 +-
robojudo/utils/motion_utils.py                               |  42 +-
robojudo/config/h1_2/__init__.py                             |   4 +
robojudo/policy/__init__.py                                  |   1 +
9 files changed, 932 insertions(+), 10 deletions(-)
```

Confirmed by tree listing: `origin/release` contains **no** file matching
`masked_mimic`; `origin/dev/h1_2-teleop` contains three.

**Consequence:** the teacher champion (`v62_ep6050`) can be deployed from this
branch. The **MM student (`ep7900`) cannot** — `release` has no masked-mimic
policy class to load it with. Deploying the MM student requires first
cherry-picking or merging `f76c14d` onto this branch. That is named and sized in
§5.

### Submodule pointer

Imprint's `.gitmodules` currently reads:

```
[submodule "third_party/RoboJuDo"]
	path = third_party/RoboJuDo
	url = https://github.com/garylvov/RoboJuDo.git
	branch = dev/h1_2-teleop
```

pinned at `ed7601fec766070bcd8de0b302b305a5aa73b06c`.

`ed7601fe` ("docs: clarify protomotion docstring in g1_cfg") **is an ancestor of
`origin/release`** — so the current pin is simply *behind*. Moving to this
branch is a fast-forward in content terms, not a rewrite.

**Recommended bump:** point `branch` at `dev/imprint-h1_2-deployment` and the
pin at this branch's tip. **I did not perform this bump** — see §8.

---

## 3. The environment

**Prefix:** `/oscar/data/stellex/glvov/glvov-envs/robojudo-eval`

This section is filled in from an actual scrubbed-process execution, not from
reading a manifest. See §3.1 for the verbatim command and output.

Context for why this matters: a sibling investigation in this campaign found a
*different* env that was reported "broken" was in fact broken for a completely
different reason than the received story claimed — three undeclared imports, not
a purged cache. Received stories about environments in this campaign have a poor
track record. Verify before you trust.

### 3.0 It is not a conda env

The brief called this "the conda env". It is not. It is a **venv layered on the
`protomotions-newton` conda env with `--system-site-packages`**:

```
$ cat /oscar/data/stellex/glvov/glvov-envs/robojudo-eval/pyvenv.cfg
home = /oscar/data/stellex/glvov/glvov-envs/protomotions-newton/bin
include-system-site-packages = true
version = 3.11.15
executable = /oscar/data/stellex/glvov/glvov-envs/protomotions-newton/bin/python3.11
command = .../protomotions-newton/bin/python3.11 -m venv --system-site-packages .../robojudo-eval
```

`bin/python3.11` is a symlink into `protomotions-newton`. The venv's own
site-packages holds only ~15 light packages (`onnxruntime`, `pygame`, `pyzmq`,
`pynput`, `msgpack`, `joblib`, …). **Everything heavy — torch, mujoco, numpy,
scipy — comes from `protomotions-newton`.** There is no `pytest` in this env's
`bin/` either; it is inherited.

**Consequence:** `robojudo-eval` is not self-contained. Anything that damages or
moves `protomotions-newton` breaks it. Treat the two as one unit.

- Python: **3.11.15** (conda-forge, GCC 14.3.0)
- site-packages (both, in order):
  `/oscar/data/stellex/glvov/glvov-envs/robojudo-eval/lib/python3.11/site-packages`
  `/oscar/data/stellex/glvov/glvov-envs/protomotions-newton/lib/python3.11/site-packages`

### 3.1 ⚠️ Out of the box, `import robojudo` FAILS

Verified in a scrubbed process from a neutral cwd:

```
$ env -i HOME="$HOME" /oscar/data/stellex/glvov/glvov-envs/robojudo-eval/bin/python \
    -c "import robojudo, mujoco, torch, onnxruntime, numpy"
Traceback (most recent call last):
  File "<string>", line 2, in <module>
ModuleNotFoundError: No module named 'robojudo'
```

**Root cause — a dangling editable install, verified not inferred.** `robojudo`
was `pip install -e`'d from a worktree that no longer exists:

- `site-packages/__editable__.robojudo-1.5.0.pth` → installs a finder
- `__editable___robojudo_1_5_0_finder.py`:
  `MAPPING = {'robojudo': '/oscar/data/stellex/glvov/wt_visual_s2r/third_party/RoboJuDo/robojudo'}`
- `robojudo-1.5.0.dist-info/direct_url.json`: `"editable": true`, url
  `file:///oscar/data/stellex/glvov/wt_visual_s2r/third_party/RoboJuDo`

```
$ ls -ld /oscar/data/stellex/glvov/wt_visual_s2r
ls: cannot access '/oscar/data/stellex/glvov/wt_visual_s2r': No such file or directory
```

The whole `wt_visual_s2r` worktree is gone. The `.pth` still loads cleanly — the
finder simply returns `None` — so you get a **silent `ModuleNotFoundError`
rather than a loud broken-path error**. This is a broken editable install. It is
**not** a missing package and **not** a purged cache.

*(This is precisely the "check rather than trust" case the brief warned about.
The received framing — "conda env, works" — was wrong on both counts.)*

### 3.2 ⚠️ `robojudo` also needs a second repo, undeclared

Putting the clone on `PYTHONPATH` alone is still not enough:

```
$ env -i HOME="$HOME" PYTHONPATH=/oscar/data/stellex/glvov/robojudo-handover \
    .../robojudo-eval/bin/python -c "import robojudo"
  .../robojudo/policy/h1_2_lift_teacher_onnx_policy.py:78, in <module>
    from imprint_isaaclab_ext.wbc.runner import WbcOnnxRunner
ModuleNotFoundError: No module named 'imprint_isaaclab_ext'
```

This is an **eager, top-level** import reached from `robojudo/__init__.py` →
`robojudo.config` → `config/h1_2/`. `imprint_isaaclab_ext` is **not in
`requirements.txt` or `pyproject.toml`**; it lives in the grove repo at
`/oscar/data/stellex/glvov/grove/repo/imprint_isaaclab_ext`.

Cross-repo imports inside this clone:

| Location | Import | Severity |
|---|---|---|
| `robojudo/policy/h1_2_lift_teacher_onnx_policy.py:78` | `imprint_isaaclab_ext.wbc.runner` | **eager — blocks `import robojudo`** |
| `robojudo/policy/h1_2_reach_teacher_onnx_policy.py:109` | `imprint_isaaclab_ext.wbc.runner` | **eager — blocks `import robojudo`** |
| `robojudo/environment/utils/newton_camera.py:52` | `imprint.integrations.visual_sim2sim.camera` | module-level, lazily reached |
| `robojudo/controller/teleop_ctrl.py:31` | `imprint.robojudo.teleop.provider` | function-local, fine |

### 3.3 The command that actually works

**This is the verification command. Record it; it is the one to re-run.**

```bash
env -i HOME="$HOME" \
  PYTHONPATH=/oscar/data/stellex/glvov/robojudo-handover:/oscar/data/stellex/glvov/grove/repo \
  /oscar/data/stellex/glvov/glvov-envs/robojudo-eval/bin/python -c "<import probe>"
```

Output:

```
python 3.11.15 /oscar/data/stellex/glvov/glvov-envs/robojudo-eval/bin/python
[DEBUG] [robojudo] ========== robojudo-1.5.0 init done ==========
OK   robojudo 1.5.0        /oscar/data/stellex/glvov/robojudo-handover/robojudo/__init__.py
OK   mujoco 3.5.0          .../protomotions-newton/lib/python3.11/site-packages/mujoco/__init__.py
OK   torch 2.7.0+cu128     .../protomotions-newton/lib/python3.11/site-packages/torch/__init__.py
FAIL onnx                  ModuleNotFoundError: No module named 'onnx'
OK   onnxruntime 1.27.0    .../robojudo-eval/lib/python3.11/site-packages/onnxruntime/__init__.py
OK   numpy 2.4.6           .../protomotions-newton/lib/python3.11/site-packages/numpy/__init__.py
OK   robojudo.deployment   /oscar/data/stellex/glvov/robojudo-handover/robojudo/deployment/__init__.py
OK   DeploymentPipelineCfg <class 'robojudo.pipeline.pipeline_cfgs.DeploymentPipelineCfg'>
```

`robojudo.__file__` resolves to the **clone**, never to an installed copy —
because the installed copy's target does not exist. **In practice the only
working mechanism is `PYTHONPATH`.**

### 3.4 Other env gaps

| Gap | Detail | Impact |
|---|---|---|
| **`onnx` absent** | Not installed in either env. Note `requirements.txt` declares only `onnxruntime`. | Skips `test_robojudo_recurrent_onnx.py:81`. Inference works (onnxruntime present); **export/inspection does not**. |
| **`sage` absent** | Not installed. | Skips `test_robojudo_sage_export.py:347`. |
| **`pynput` fails headless** | `ImportError: this platform is not supported: failed to acquire X connection` | Not fatal — `robojudo` does not import it eagerly — but **the teleop / keyboard safety-gate path dies on a headless node**. |

All other declared deps import cleanly: `scipy, yaml, easydict, box, tqdm,
pygame, zmq, pydantic, msgpack, msgpack_numpy, colorlog, joblib`.

---

## 4. How to run it end to end

Shortest path from nothing to a policy actually running.

### 4.1 Get the code

```bash
# Never clone into /oscar/scratch — it runs ~91% full and silently deleted
# three corpus packs on 2026-08-08. readlink -f any path there before trusting it.
cd /oscar/data/stellex/glvov
git clone https://github.com/garylvov/RoboJuDo.git robojudo
cd robojudo
git checkout dev/imprint-h1_2-deployment
```

You also need Imprint, which supplies the backend:

```bash
# already present on this machine:
#   /oscar/data/stellex/glvov/imprint-consolidate-20260810
```

### 4.2 Fix OSMesa *before* starting the process

Headless MuJoCo rendering needs `libOSMesa.so`, which lives in Oscar's Lmod mesa
module and is **not** on `LD_LIBRARY_PATH` by default. It must be prepended
**before the process starts** — you cannot fix this from inside Python, because
by the time Python runs, the dynamic loader has already been configured.

```bash
module load mesa/25.0.5-2kqs
# equivalently, without Lmod:
export LD_LIBRARY_PATH=/oscar/rt/9.6/25/spack/x86_64_v3/mesa-25.0.5-2kqswd2l3fkmb62exxaxvj7ykx63pnft/lib:$LD_LIBRARY_PATH
```

Verified present today:
`/oscar/rt/9.6/25/spack/x86_64_v3/mesa-25.0.5-2kqswd2l3fkmb62exxaxvj7ykx63pnft/lib/libOSMesa.so.8.0.0` (18.5 MB).

### 4.3 Run a deployment

The native path, driven by RoboJuDo's own launcher:

```bash
python scripts/run_pipeline.py \
  --config-module imprint.robojudo.native \
  --config imprint_mujoco \
  --bootstrap imprint.robojudo.native:build_newton_replay_session \
  --bootstrap-kwargs-json '{"python": "...", "episode": "...", "output": "...", "metrics": "..."}' \
  --steps 100
```

- `--config-module` imports a module that registers external configs into
  RoboJuDo's `cfg_registry`. This is how Imprint's configs become visible.
- `--config` selects `imprint_newton` (device `cuda`) or `imprint_mujoco`
  (device `cpu`).
- `--bootstrap` is `module:callable` returning an object with `.run(steps)` and
  a `.backend.capabilities.kind` matching the adapter.
- `--steps` overrides the finite step count; must be positive, and is rejected
  unless the config is a `DeploymentPipeline`.

The adapter verifies `backend.capabilities.kind` against the adapter's declared
`backend_kind` and raises `TypeError` on mismatch — so pointing a Newton adapter
at a MuJoCo session fails immediately rather than at tick 1.

### 4.4 Generate the H1-2 + Ability scene

```bash
cd /oscar/data/stellex/glvov/imprint-consolidate-20260810
PYTHONPATH=src python -m imprint.integrations.unitree_lab.h1_2_usd \
  ability-mjcf --output artifacts/h1_2_ability.xml

# with the sprint table + resettable free box, for manipulation replay:
PYTHONPATH=src python -m imprint.integrations.unitree_lab.h1_2_usd \
  ability-mjcf --include-task-scene \
  --output artifacts/h1_2_ability_table_box.xml
```

Each writes a sibling `*.validation.json` receipt after a headless MuJoCo
compile/settle check.

**Do not** point RoboJuDo's positional 27-DOF `MujocoEnv` config at this scene.
It is a 47-joint robot plus a free object; the `MujocoEnv` state slices and
control width are not valid for it. Use `MujocoReplayPlant --xml` instead, which
uses named qpos/qvel/actuator indices.

---

## 5. What is unfinished — named and sized

| # | Item | Size | Notes |
|---|---|---|---|
| 1 | **Delete both obsolete patch files from Imprint** | XS (~10 min) | `patches/third_party/RoboJuDo/{0001-native-imprint-deployment.patch,series,target}` and `patches/third_party/robojudo-deployment-wip.patch`. Proven redundant in §2. Leaving them is an active hazard — they invite a destructive re-apply. |
| 2 | **Bump the Imprint submodule pointer** | S (~30 min incl. verification) | Point `branch` at `dev/imprint-h1_2-deployment`, pin at this branch tip. Deliberately not done here; see §8. |
| 3 | **Land masked-mimic on this branch** | M (~half day) | Cherry-pick/merge `f76c14d` from `dev/h1_2-teleop` (932 insertions, 9 files). **Blocks MM-student deployment entirely.** Touches `mujoco_env.py` and `teleop_ctrl.py`, so it is not a clean isolated add — expect to re-verify the teleop path after. |
| 4 | **`bootstrap: str = ""` defaults** | S | Both `imprint_newton` and `imprint_mujoco` in `src/imprint/robojudo/native.py` ship with an empty-string bootstrap and `steps: int = 1`. Unusable without CLI override. Either give real defaults or make the field required so the failure is a clear validation error rather than an obscure resolve failure. |
| 5 | **`DeploymentPipeline.step()` is a hard error** | XS | It raises `RuntimeError` by design (the pipeline is finite, `run_to_completion()` is the entry). Fine, but it means `DeploymentPipeline` is not substitutable for other pipelines — worth a note in the class docstring for anyone writing generic pipeline-driving code. |
| 6 | **`NewtonReplaySession` shells out to a subprocess** | M | It runs `python -m imprint.sim2sim.newton_manipulation_runner` via `subprocess.run`. Necessary today (Isaac must start in a restored env before simulator imports) but it means no in-process error propagation — you get a `CalledProcessError` and a metrics-file existence check, nothing richer. |
| 7 | **Two Imprint test modules are undocumented in the handover brief** | XS | See §7 — there are **9** `test_robojudo_*` files, not 7. |
| 8 | **Initialize `third_party/RoboJuDo` in `grove/repo`, then fix the 4 tests that fail** | M (~half day) | **Highest priority.** The submodule is empty, so the deployment path's own integration tests skip; wire the env in and 4 of them *fail* (§7.4 D2). Until this is green, no deployment claim is backed by a passing test. |
| 9 | **Restore `src/imprint/robojudo/gated_inference.py` on `curric-dawn`** | S (~1 h) | Lost in a merge; present on 15 other branches. Blocks collection of the entire 672-line `test_robojudo_excite.py` (§7.4 D1). Also makes `__init__.py`'s docstring true again. |
| 10 | **Repair the broken editable install** | S | Points at deleted `wt_visual_s2r` (§3.1). Fails *silently*. Either re-install `-e` against a real path or drop the dist-info and standardise on `PYTHONPATH`. |
| 11 | **Make the two `imprint_isaaclab_ext` imports lazy, or declare the dep** | S | Two eager imports in `h1_2_{lift,reach}_teacher_onnx_policy.py` make `import robojudo` require a second repo that is in no manifest (§3.2). |
| 12 | **Install `onnx` (and decide about `sage`)** | XS | `onnx` absent; inference works via `onnxruntime` but export/inspection does not. Unblocks 2 skips. |
| 13 | **Generate `resolved_configs_inference.pt` for the MM student** | S | Neither MM staging dir has it, so an export today silently falls back to the DR-on training config (§6.1). |

The filename `robojudo-deployment-wip.patch` says "WIP" — but as established in
§2 it is not unfinished work, it is a **finished** commit already on `release`,
carrying a WIP name from when it was authored on the
`deployment-pipeline-wip` branch. The name is misleading. Item 1 disposes of it.

---

## 6. Known traps

### 6.1 The ONNX export silently falls back to a domain-randomised config ⚠️

**This is the most dangerous trap for deployment.** In
`ProtoMotions/deployment/export_bm_tracker_onnx.py`:

```python
resolved_path = checkpoint_path.parent / "resolved_configs_inference.pt"
if not resolved_path.exists():
    log.warning(
        "resolved_configs_inference.pt not found, falling back to "
        "resolved_configs.pt.  Domain randomization may still be active!"
    )
    resolved_path = checkpoint_path.parent / "resolved_configs.pt"
```

If `resolved_configs_inference.pt` is absent, the exporter falls back to the
**training** config — with **domain randomisation ON** — and continues. The only
signal is a `log.warning`, which is trivially lost in export noise.

**Why this matters enormously for deployment:** the exported policy is then *not
the policy that was scored*. You can score a checkpoint at 94.4% SR, export it,
deploy it, and be running something measurably different — with randomised
dynamics baked into the inference path.

**Mitigation — do this every time:**
1. Before exporting, assert `resolved_configs_inference.pt` exists next to the
   checkpoint. Fail loudly if not.
2. Grep the export log for `Domain randomization may still be active` and treat
   any hit as an export failure, not a warning.
3. Confirm the `Loading configs from <path>` line names the `_inference` file.

Note the related comment at line 153 of that file: *"IN-MEMORY ONLY:
`resolved_configs*.pt` on disk keeps the ORIGINAL launch gains."* Disk configs
are not the whole story for gains either.

#### This trap is LIVE right now for the MM student

Checked on disk today:

| Checkpoint dir | `resolved_configs.pt` | `resolved_configs_inference.pt` | Verdict |
|---|---|---|---|
| `pm_run_v60/third_party/ProtoMotions/results/canonical_teacher_20260809_v62/` (teacher `v62_ep6050`) | ✅ | ✅ | **safe to export** |
| `mm_eval/stage/` (MM `epoch_7900.ckpt`) | ✅ | ❌ | **would silently fall back** |
| `mm_eval/stage_fk/base7900/` (MM `epoch_7900.ckpt`) | ✅ | ❌ | **would silently fall back** |

So exporting the **teacher champion** from its results dir is safe — the
inference config is present. Exporting the **MM student** from either staging
directory as they stand today would hit the fallback and produce a policy built
from the training config with domain randomisation potentially active. That is a
second, independent reason (alongside the missing masked-mimic policy in §2) not
to reach for the MM student casually.

### 6.2 OSMesa must be on `LD_LIBRARY_PATH` before process start

Covered in §4.2. Restating the failure mode: `libOSMesa.so` is in the Lmod mesa
module, off the default `LD_LIBRARY_PATH`. Setting it from inside Python does
not work — the loader is already configured by then. Prepend it in the shell, or
`module load mesa/25.0.5-2kqs`, before launching.

### 6.3 `MUJOCO_GL` import-time trap — real mechanism, but NOT active in this suite

`motion_aug/pack_io.py:42` does `os.environ.setdefault("MUJOCO_GL", "osmesa")`
**at import time**. Where that module is imported, the GL backend is decided as a
side effect of collection order, and a standalone pass would not imply an
in-suite pass.

**Measured for the RoboJuDo tests: it does not fire.** No test file imports
`motion_aug`; `motion_aug/__init__.py` does not import `pack_io`; `MUJOCO_GL`
stays `None` through collection and session end; `mujoco` never enters
`sys.modules`. B1 and B2 are identical. Full evidence in §7.3.

Keep running both ways anyway — the mechanism is real and one added import would
re-arm it — but do not attribute unrelated failures to it. It was checked, and
it is clean.

### 6.4 Do not point `MujocoEnv` at the 47-joint Ability scene

Covered in §4.4. The state slices and control width are wrong for a 47-joint
robot plus a free object. Use `MujocoReplayPlant --xml`.

### 6.5 The Ability-hand mount transform is an approximation

From Imprint's `docs/integrations/robojudo-h1_2-ability-mujoco.md`: the MuJoCo
mount uses the vendor hand frame rotated onto the H1-2 wrist-yaw x-axis
(`0.10 m`, +90° about y), because the composed USD provides no portable MJCF
mount-transform receipt. It is the transform already exercised by the replay
evidence, **but it remains an approximation rather than an exact cross-format
proof.** Relevant if you care about absolute hand placement.

### 6.6 `/oscar/scratch` is not safe storage

~91% full; silently deleted three corpus packs on 2026-08-08. `readlink -f`
before trusting any path that might resolve there. Clone and write under
`/oscar/data/stellex/glvov/`.

---

## 7. Tests

There are **9** `test_robojudo_*` files, not 7. The handover brief listed seven;
these two were omitted and are substantial:

- `test_robojudo_shared_runtime.py` (618 lines) — covers `shared_runtime.py`,
  the largest Imprint module and the actual `run(steps)` lifecycle object the
  RoboJuDo adapter consumes. **This is arguably the most load-bearing test of
  the whole integration.**
- `test_robojudo_sysid_windows.py` (473 lines)

Full inventory (in `test/` of the Imprint repo; present in both
`imprint-consolidate-20260810/test/` and `grove/repo/test/`):

| File | Lines |
|---|---|
| `test_robojudo_excite.py` | 672 |
| `test_robojudo_shared_runtime.py` | 618 |
| `test_robojudo_sysid_windows.py` | 473 |
| `test_robojudo_recorder.py` | 418 |
| `test_robojudo_sage_export.py` | 358 |
| `test_robojudo_run_policy_onnx.py` | 243 |
| `test_robojudo_native_deployment.py` | 133 |
| `test_robojudo_recurrent_onnx.py` | 95 |
| `test_robojudo_h1_2_ability_scene.py` | 84 |

### 7.1 How the suite is actually run

There is no `pytest.ini`/`setup.cfg`/`tox.ini`. Config is
`[tool.pytest.ini_options]` in `/oscar/data/stellex/glvov/grove/repo/pyproject.toml`
(`testpaths=["test"]`, markers `network`/`gpu`/`live`, `norecursedirs` includes
`third_party`). Two conftests apply: `grove/repo/conftest.py` (autouse environ
save/restore) and `grove/repo/test/conftest.py` (arms
`IMPRINT_FORBID_REAL_SPAWN=1`, hermetic roots).

Docstrings say `pixi run -e test python -m pytest …`, **but `.pixi/envs/` does
not exist on this machine** — the pixi env is not materialized. CI
(`.github/workflows/dawn2-tests.yaml`) uses bare python3.11 with
`PYTHONPATH: src`. What was used here (pytest 9.1.1, numpy 2.4.6 — both matching
the CI pins):

```bash
cd /oscar/data/stellex/glvov/grove/repo
PYTHONPATH=/oscar/data/stellex/glvov/grove/repo/src \
  /oscar/data/stellex/glvov/glvov-envs/protomotions-newton/bin/python -m pytest ...
```

Running the same under `robojudo-eval/bin/python` gives **byte-identical**
results, as expected — same inherited site-packages.

### 7.2 Both numbers: standalone vs in-suite

| Run | Passed | Failed | Skipped | Collection errors |
|---|---|---|---|---|
| **B1 — the 7 alone** | 33 | 0 | 9 | 1 |
| **B2 — the 7 with the whole `test/` dir collected** | 33 | 0 | 9 | 1 |

**They do not differ.** Same 33 passes, same 9 skips with identical reasons at
identical line numbers, same single collection error. **No test changed status.**

B2 method: passed `test/` (296 files, **5741 tests collected**) so every module
is imported, then `-k` to select only the 7 — confirmed `5699 deselected`,
`33 + 9 = 42` selected, exactly matching B1's collection. B2's raw line reads
`33 passed, 18 skipped, 5699 deselected, 8 errors`; the extra 9 skips and 7
errors are **other people's modules** (`langgraph` ×4, `fcl` ×2, `onnx`,
`rsl_rl_g` ×4, …) firing at collection regardless of `-k`, not our tests.

⚠️ Without `--continue-on-collection-errors` the run **aborts entirely**:
`!!!! Interrupted: 1 error during collection !!!!`. Anyone reporting "the
robojudo tests pass" without that flag is reporting on a run that never happened.

### 7.3 ⚠️ The `MUJOCO_GL` collection-order trap is FALSIFIED for this suite

The brief warned that `motion_aug/pack_io.py` sets `MUJOCO_GL=osmesa` at import
time, so results depend on collection order. **Measured, this does not happen
here.** Instrumented with a pytest plugin hooking `pytest_collection_finish` and
`pytest_sessionfinish`:

```
[PROBE-COLLECT] MUJOCO_GL = None | pack_io loaded: [] | mujoco loaded: False
[PROBE-FINISH]  MUJOCO_GL = None | pack_io loaded: [] | mujoco loaded: False
```

Same `None` on a full `--co -q` of the entire `test/` dir. Why:

- `grep -rln motion_aug test/*.py` → **zero hits**. No test file imports
  `motion_aug`.
- `src/imprint/integrations/wbc/training/motion_aug/__init__.py` does **not**
  import `pack_io` — it only mentions it in a comment. So
  `pack_io.py:42 os.environ.setdefault("MUJOCO_GL", "osmesa")` never executes.
- `mujoco` is never even loaded into `sys.modules` by this suite.

Belt-and-braces, the trap condition was forced anyway:

```
$ MUJOCO_GL=osmesa ... pytest <the 7> -q
33 passed, 9 skipped, 1 error in 9.03s     # identical
```

**These 7 tests need neither OSMesa nor the mesa module.** The
`LD_LIBRARY_PATH` prefix in §4.2 remains correct and necessary for *headless
MuJoCo rendering* generally — it is simply not exercised by this test set. The
run-both-ways discipline stays worthwhile; it just came back clean this time.

### 7.4 ⚠️ The "9 skipped" and "1 error" are hiding two real defects

**D1 — `src/imprint/robojudo/gated_inference.py` does not exist on branch
`curric-dawn`.** `test_robojudo_excite.py:20` does
`from imprint.robojudo import excite, gated_inference`, and
`src/imprint/robojudo/__init__.py`'s own docstring promises *"See
`imprint.robojudo.gated_inference`"* — but `git ls-files src/imprint/robojudo/`
lists 11 files and `gated_inference.py` is not among them, and `git log` on that
path is empty (it was **never** on `curric-dawn`). It exists on **15 other
remote branches** including `origin/golden/env-locked` and
`origin/stepback/retread/integration`. It was lost in a merge/branch selection,
not deleted. This kills the whole 23 KB `test_robojudo_excite.py` file at
collection. **Fix: restore it from one of the branches that has it.**

**D2 — the 4 `ROBOJUDO_PYTHON` skips are hiding 4 hard failures.** Wiring the env
in as the tests intend converts skips into failures:

```
$ ROBOJUDO_PYTHON=/oscar/data/stellex/glvov/glvov-envs/robojudo-eval/bin/python \
    pytest test/test_robojudo_native_deployment.py -q
FAILED test_imprint_adapters_and_pipeline_are_robojudo_registered
FAILED test_robojudo_entrypoint_runs_finite_imprint_rollout[imprint_newton-newton]
FAILED test_robojudo_entrypoint_runs_finite_imprint_rollout[imprint_mujoco-mujoco]
FAILED test_registered_adapter_rejects_mislabeled_backend
4 failed, 1 passed
```

with

```
can't open file '/oscar/data/stellex/glvov/grove/repo/third_party/RoboJuDo/scripts/run_pipeline.py':
[Errno 2] No such file or directory
```

Cause: the tests resolve RoboJuDo as `ROOT/"third_party"/"RoboJuDo"`, and that
submodule is **empty and uninitialized** in `grove/repo`. The same emptiness
produces the 3 `"pinned RoboJuDo/Ability submodules are not initialized"` skips
in `test_robojudo_h1_2_ability_scene.py`.

**So the honest reading of "33 passed / 9 skipped" is: the deployment path's own
integration tests have never actually run on this machine.** They skip because
the submodule is empty, and when you fill that gap in they fail. This is the
single most important thing to fix before trusting any deployment claim.

### 7.5 Gap table

| # | Gap | Fix |
|---|---|---|
| A1 | `robojudo` editable install points at deleted `wt_visual_s2r` | `pip install -e /oscar/data/stellex/glvov/robojudo-handover`, or repoint the finder `MAPPING`, or use `PYTHONPATH` |
| A2 | `robojudo/__init__` eagerly needs undeclared `imprint_isaaclab_ext` | put `grove/repo` on `PYTHONPATH`, install it, or make the two `h1_2_*_teacher_onnx_policy.py` imports lazy |
| A3 | `onnx`, `sage` not installed | `pip install onnx` unblocks 2 skips |
| A4 | `pynput` ImportError headless | only affects teleop / keyboard-gate |
| D1 | `gated_inference.py` missing on `curric-dawn` | restore from one of 15 branches that have it |
| D2 | `third_party/RoboJuDo` submodule empty in `grove/repo` | `git submodule update --init`; then fix the 4 tests that will fail |

---

## 8. What we would deploy today

### Candidates

| Candidate | SR | body err | wrist err | Runs on this branch? |
|---|---|---|---|---|
| **Teacher champion `v62_ep6050`** | 94.4 % | 3.40 cm | 2.21 cm | ✅ yes |
| **MM student `ep7900`** | 97.5 % | 10.35 cm | 16.49 cm | ❌ **no** — needs `f76c14d` (§2, §5 item 3) |

Both scored on `canonical_eval_v1` (284 clips). Sources, verified today:
`wt-evalharness/scripts/wbc/eval/TREND.md:160` for the teacher
(`| **ALL** | 284 | 94.4 / 3.40 / 2.21 |`) and `MM_TREND.md:69` for the student
(`| ep7900 | 97.5 | 10.35 | 16.49 | 277/284 |`).

### ⚠️ The global-frame caveat — read before deploying open-loop

**The headline metric is anchor-relative.** From the harness README:

> `local_frame(pos, rot)` maps world body states into the **pelvis-anchored,
> yaw-aligned** frame: subtract the anchor body's position, then rotate by the
> inverse *heading* (yaw-only) quaternion of the anchor.

So the reported body/wrist errors have **root position and yaw subtracted out**.
This is the correct frame for judging *tracking* — it is the frame the teacher is
optimised in, and a world-frame error would be dominated by root drift (the
harness's `env.py` says so explicitly about the wrist stat). The harness does
carry a world-frame channel, `pos_err_global`, but it is explicitly labelled
**"diagnostic only"** and is not what the scoreboard reports.

**The practical consequence:** the champion carries roughly **18–25 cm of global
position error**, and on `forward_locomotion` it drifts about **64 cm** — none of
which appears in the 3.40 cm headline, because the metric subtracts it.

- **Deploying open-loop in world coordinates?** You need to know this. A policy
  that tracks beautifully in the pelvis frame can still walk your robot most of a
  metre away from where you expected.
- **Teleop operator closing the loop?** Probably does not care — the human is the
  outer loop correcting exactly this drift.

*Provenance note, stated honestly:* the **mechanism** is confirmed in the source
(the anchor-relative frame definition and the diagnostic-only `pos_err_global`
channel are both verified in `wt-evalharness/scripts/wbc/eval/README.md`, lines
~68–86). The two specific figures — 18–25 cm global and 64 cm on
`forward_locomotion` — are carried over from the handover brief and were **not**
independently reproduced during this handover. Re-derive them from
`pos_err_global` before quoting them in anything load-bearing.

### Recommendation

Deploy the **teacher champion `v62_ep6050`**. Three independent reasons, each
sufficient on its own:

1. **It runs on this branch.** The MM student needs `f76c14d` merged before
   `release` even has a class that can load it (§2).
2. **Its export is safe.** The teacher's results dir has
   `resolved_configs_inference.pt`; **neither MM staging dir does**, so an MM
   export today silently falls back to the DR-on training config (§6.1).
3. **Its tracking is far tighter** — 3.40 cm body / 2.21 cm wrist versus the
   student's 10.35 / 16.49. The student's higher SR (97.5 % vs 94.4 %) buys
   robustness at roughly 3× the body error and 7× the wrist error.

Note from `TREND.md:137`: *"v62 has plateaued, and past ep6050 it is going
backwards"* — `ep6050` is the right stopping point, not a later epoch. `ep7200`
scores **worse** (91.5 % SR).

---

## 9. Verification record for this handover

Everything asserted about branch state was verified against the **remote**, not
local `git log`:

- Remote branch list: `git ls-remote --heads https://github.com/garylvov/RoboJuDo.git`
- Ancestry: `git merge-base --is-ancestor 2c7dacc origin/release` after `git fetch origin`
- File presence: `git ls-tree -r --name-only origin/release`
- Blob identity: `git rev-parse origin/release:<path>`

**Submodule pointer: NOT bumped.** Imprint's `third_party/RoboJuDo` pin is
untouched at `ed7601fe` and `.gitmodules` still names `dev/h1_2-teleop`.

The bump was withheld deliberately, and §7.4 D2 is why. The deployment path's
own integration tests have **never actually run** on this machine — they skip
because the submodule is empty, and when the env is wired in as they intend,
**4 of them fail**. Bumping the pointer would advertise a verified integration
that is not verified. The correct order is:

1. `git submodule update --init third_party/RoboJuDo` in `grove/repo`
2. Fix the 4 failures in `test_robojudo_native_deployment.py`
3. Restore `gated_inference.py` so `test_robojudo_excite.py` collects
4. *Then* bump the pin, with a green suite behind it

§5 items 8, 9 and 2 size that sequence.

**One more caveat on the bump:** whoever does it must decide the masked-mimic
question first (§2). Pointing Imprint at a `release`-derived branch silently
drops the masked-mimic policy that `dev/h1_2-teleop` carries. If anything in
Imprint depends on it, the bump is a regression, not an upgrade.

**No force-push was used at any point.**
