# Image to Three.js miner (tamarisk-weir)

Generates a Three.js module for a prompt image through the competition witness: drafts, selection, rewrites against
renders and a final pick, within the per-prompt budget. Every module is centered on the origin and scaled uniformly to
fit the unit cube.

Entry point: `main.py` -> `async generate(image, seed, witness) -> str | None`. The plan is `ACTIVE_PLAN` in
`miner404/versions.py`.

## Live run

```sh
uv venv --python 3.13 .venv-live
uv pip install --python .venv-live/bin/python -r requirements.txt
git clone --branch miner-runtime-v0.1.5 --depth 1 https://github.com/404-Repo/404-gen-subnet.git ../404-sdk
```

Create `witness.json` next to `main.py`, with `witness_url` set to `https://witness.subnet404.xyz` and
`proof_parameters` set to the absolute path of `404-sdk/witness-server/verifier/params`. Then:

```sh
.venv-live/bin/python main.py --prompts prompts.txt --seed PUBLISHED_SEED --wallet-name NAME --wallet-hotkey HOTKEY
```

## Credits

The pipeline is a fork of [TiltGilt/VertexLoom](https://github.com/TiltGilt/VertexLoom) at 2385ec1, itself built on
taohunter251/lark-ridge, SmoothTight/newvictor, Rabbit692/sedge-hollow and research-somthing-t/lumen-kit.

A budget reserve setting is adapted from the public round-5 submissions taohunter251/knot-stack and
bottleneck-earbud/plover-gate.

`miner404/patch.py` and the SEARCH/REPLACE edit format come from kaito0311/3d-gen-sub (MIT License, see `NOTICE.md`).

## Changes in this build

Based on bujhmml/forge3d-4 @ 646cfce6 (itself a fork of TiltGilt/VertexLoom). Adds the ga gate (front-left + from-above
views with a free photo-match lift, `miner404/match.py`), a vehicle-facing prompt rule, and on the finished module a snap of
partial whole-object rotations to the nearest quarter turn plus an exact 90-degree stand-up of flat-lying objects.
