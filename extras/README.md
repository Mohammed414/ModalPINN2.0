# extras

Standalone visualizations, not part of the R1-R6 training pipeline.

## Flow animation

`flow_animation_vorticity.gif` animates the ground-truth CFD flow
(`data/fixed_cylinder_atRe100`) as vorticity over the cylinder wake,
cropped to the same region ModalPINN trains/evaluates on
(x in [-4,8], y in [-4,4]). Shows the classic von Karman vortex street
at Re=100 across t=[400,420] (~3 shedding periods).

To regenerate:
```bash
cd extras
python3 make_flow_animation.py --field vorticity
```

The first run parses the raw 1.1GB flow file (~15s) and caches the result
to `flow_cache.npz` (gitignored, ~170MB, regenerate anytime from
`data/fixed_cylinder_atRe100`). Subsequent runs reuse the cache.

Other options:
```bash
python3 make_flow_animation.py --test               # single PNG frame, for tuning
python3 make_flow_animation.py --field speed         # speed instead of vorticity
python3 make_flow_animation.py --field pressure      # pressure instead of vorticity
python3 make_flow_animation.py --stride 2 --fps 20   # shorter/faster animation
```

`parse_flow.py` is a fast replacement for `src/text_flow.py`'s `read_flow()`
for this specific use case: it reads each timestep block with a single
`np.loadtxt(..., max_rows=N_nodes)` call instead of parsing one float per
line in pure Python, which matters at this file's scale (Nt=201,
N_nodes=82872, ~16.6M data lines).
