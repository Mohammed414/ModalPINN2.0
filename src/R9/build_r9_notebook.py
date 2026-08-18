"""Builds R9/notebooks/R9_trust_street_32taps.ipynb. Run locally, not on Colab.

Same skeleton as build_r8_notebook.py (proven Colab setup: miniconda py3.7 +
TF1.14-GPU + Zenodo dataset + smoke-test gate), with R8's loss-term stack
REPLACED by the R9 trust-street ansatz:

  DROPPED: --BVF, --K0Loss, --PhaseLoss (R9_wake_rescue's controlled arms
           showed loss-term fixes leave the wake dead; the ansatz fix makes
           them redundant - and PhaseLoss actively diverged in R8)
  ADDED:   --TrustStreet --StreetPrior street_prior_Ntap32.npz
           (+ the street_prior.py derivation step, taps-only, runs in ~1 min)

No warm start: the trust ansatz replaces the basin-escape role a warm start
would play; k=0 trains from scratch. A composable follow-up (R9b) could add
--WarmStartFrom R3 for the near-cylinder mean if R9's k=0 lags R7's.
"""
import json
import os
import uuid

ROOT = os.path.dirname(os.path.abspath(__file__))          # .../src/R9
REPO = os.path.dirname(os.path.dirname(ROOT))
SRC = os.path.join(ROOT, 'src')
OUT = os.path.join(REPO, 'notebooks', 'R9_trust_street_32taps.ipynb')


def md(src):
    return {"cell_type": "markdown", "id": uuid.uuid4().hex[:8],
            "metadata": {}, "source": src.splitlines(keepends=True)}


def code(src):
    return {"cell_type": "code", "id": uuid.uuid4().hex[:8], "metadata": {},
            "execution_count": None, "outputs": [],
            "source": src.splitlines(keepends=True)}


def writefile_cell(fname, path):
    with open(path) as f:
        content = f.read()
    return code('%%writefile ' + fname + '\n' + content)


cells = []

cells.append(md("""# R9 - trust-region ansatz around a taps-only vortex-street prior, 32 taps

The mechanism change (not another loss term): the k>=1 mode networks are wrapped as a BOUNDED
correction around a closed-form von Karman street derived from the 32 taps alone,

    q_k = S_k + (rho |S_k| + cap) * tanh-bounded network correction,   k = 1..3

so the dead-wake solution q_k = 0 that every run R1-R8 converged to is EXCLUDED from the search
space wherever the street prior is alive. The k=0 (mean) modes stay free networks.

Why this and not another loss fix: R9_wake_rescue/'s controlled testbed arms showed that
relative-residual, RZIF, lift-anchored CV budgets and hard symmetry all leave the wake dead
(near-wake E_v ~ 1.0) even though they re-order the loss landscape correctly - the optimizer
still walks into the dead basin. Removing the basin worked: testbed far-wake E_v 1.0 -> 0.40,
near-wake 0.96 -> 0.64, robust to 1-5% tap noise, 16 taps, and seed restarts. Full account:
R9_wake_rescue/REPORT.md.

The prior is derived by `street_prior.py` (numpy-only, ~1 min) from tap-measured quantities:
shedding frequency (lift sinusoid fit), circulation (Karman drag relation == measured pressure
drag / 0.75), formation point / core size / temporal phase (matching the image-vortex street's
induced surface-pressure pattern to the measured tap k=1 harmonics). The reference CFD file is
touched only to extract the same 32 tap signals the training script itself trains on.

Loss stack is deliberately MINIMAL: physics + taps + FSBC/FIBC. BVF/K0Loss/PhaseLoss are dropped
(see builder docstring). No warm start.

Same discipline as R7/R8: smoke test first (full production point counts, only Tmax/maxit
reduced), hard assertion gate, then the real ~9h run."""))

cells.append(md("## 1. Confirm GPU (and that this is a high-RAM runtime)"))
cells.append(code("!nvidia-smi\n!free -h\n"))

cells.append(md("## 2. Mount Google Drive"))
cells.append(code("""from google.colab import drive
drive.mount('/content/drive')
import os
os.makedirs('/content/drive/MyDrive/ModalPINN_results', exist_ok=True)
print('Drive mounted and target folder ready')
"""))

cells.append(md("## 3. Install Miniconda + Python 3.7 + CUDA 10.0 + cuDNN 7.6.5"))
cells.append(code("""!curl -sL -o /tmp/miniconda.sh https://repo.anaconda.com/miniconda/Miniconda3-latest-Linux-x86_64.sh
!bash /tmp/miniconda.sh -b -p /content/miniconda
!/content/miniconda/bin/conda create -y -n modalpinn -c conda-forge --override-channels python=3.7 cudatoolkit=10.0 cudnn=7.6.5
"""))

cells.append(md("## 4. Install TensorFlow-GPU 1.14 and other pinned dependencies"))
cells.append(code("""!/content/miniconda/envs/modalpinn/bin/pip install \\
    tensorflow-gpu==1.14.0 numpy==1.17.4 scipy==1.3.2 \\
    matplotlib==3.1.1 gputil protobuf==3.11.3
"""))

cells.append(md("## 5. Verify TensorFlow actually sees the GPU"))
cells.append(code("""import subprocess, os
env = os.environ.copy()
env['LD_LIBRARY_PATH'] = '/content/miniconda/envs/modalpinn/lib:' + env.get('LD_LIBRARY_PATH', '')
r = subprocess.run(
    ['/content/miniconda/envs/modalpinn/bin/python', '-c',
     "import tensorflow as tf; sess = tf.compat.v1.Session(config=tf.compat.v1.ConfigProto(log_device_placement=True))"],
    capture_output=True, text=True, env=env)
print(r.stdout[-500:])
print(r.stderr[-1500:])
"""))

cells.append(md("## 6. Write the source files\n\n"
                "R9 sources are based on `src/pressure_only` (the R7-era shared tree), NOT on `R8/src` - "
                "R8's fixed-c PhaseLoss and its DRIFT_CHECK_EVERY logging never enter R9 at all (they were "
                "R8-only changes, and R9 drops PhaseLoss deliberately). R9 changes on top of that base: "
                "`--TrustStreet/--StreetPrior/--TrustRho/--TrustCap` flags + `street_params` threading in "
                "`ModalPINN_VortexShedding.py`; `street_modes_k` + trust wrap in `NN_functions.py`; "
                "trust-aware restore in `evaluate_regions.py`; new `street_prior.py`."))
cells.append(writefile_cell('Load_train_data_desync.py', os.path.join(SRC, 'Load_train_data_desync.py')))
cells.append(writefile_cell('ModalPINN_VortexShedding.py', os.path.join(SRC, 'ModalPINN_VortexShedding.py')))
cells.append(writefile_cell('NN_functions.py', os.path.join(SRC, 'NN_functions.py')))
cells.append(writefile_cell('evaluate_regions.py', os.path.join(SRC, 'evaluate_regions.py')))
cells.append(writefile_cell('street_prior.py', os.path.join(SRC, 'street_prior.py')))
cells.append(writefile_cell('text_flow.py', os.path.join(REPO, 'src', 'text_flow.py')))
cells.append(writefile_cell('reactions_process.py', os.path.join(REPO, 'src', 'reactions_process.py')))

cells.append(md("## 7. Get the real dataset (Boudina et al., Zenodo, ~1.17 GB)"))
cells.append(code("""import os, shutil
os.makedirs('Data', exist_ok=True)
drive_cache = '/content/drive/MyDrive/ModalPINN_data/fixed_cylinder_atRe100'
local_path = 'Data/fixed_cylinder_atRe100'
if os.path.exists(drive_cache):
    print('Found cached dataset in Drive, copying locally...')
    shutil.copyfile(drive_cache, local_path)
else:
    print('Not cached yet, downloading from Zenodo...')
    exit_code = os.system(
        'wget -q -O ' + local_path +
        ' https://zenodo.org/record/5039610/files/fixed_cylinder_atRe100?download=1')
    assert exit_code == 0, 'Download failed'
    os.makedirs(os.path.dirname(drive_cache), exist_ok=True)
    shutil.copyfile(local_path, drive_cache)
    print('Saved a copy to Drive for future runs:', drive_cache)
print('Dataset ready:', local_path, os.path.getsize(local_path), 'bytes')
"""))

cells.append(md("""## 8. Derive the street prior from the 32 taps (numpy-only, ~1 min)

This is the R9-specific step. Everything in `street_prior_Ntap32.npz` derives from the tap
pressures + classical physics; the printed `tap-p1 corr` (expect ~0.74) and
`corr vs numeric` (expect >0.99, asserted in-script) are the sanity numbers. Reference values
from the local run: omega0_hat=1.03575, Gamma=2.527, Uc=0.821, xf=1.2, r0=0.4, phase=-0.709,
amp_scale=0.743, scale_p=1.239."""))
cells.append(code("""!/content/miniconda/envs/modalpinn/bin/python street_prior.py \\
    --DataFile Data/fixed_cylinder_atRe100 --NTaps 32
import numpy as np
sp = np.load('street_prior_Ntap32.npz')
print({k: round(float(sp[k]), 4) for k in sp.files})
assert float(sp['cf_corr_vs_numeric']) > 0.95
assert abs(float(sp['omega']) - 1.036) < 0.02, 'tap omega far from expected - investigate before training'
print('Street prior OK')
"""))

cells.append(md("""## 9. Smoke test FIRST - TrustStreet has never touched real TF1.14 before this run

Same proven gate as R6/R7/R8: full production point counts (`--Nmes 5000 --Nint 50000
--Ngrid 5 --NgridTurn 200`), only `--Tmax` reduced to 0.05 and L-BFGS maxit patched to 300 -
memory problems in this project have consistently been graph-topology costs, not data-volume
ones, so shrinking point counts would test the wrong configuration. `TrustStreet` is the one
new ingredient. Runs in an isolated `smoke_test/` subfolder. Checks: clean exit, no traceback,
the `R9 TrustStreet prior:` line is printed (proof the ansatz is actually active), loss is
finite, L-BFGS actually iterated."""))
cells.append(code(r"""import os, re, shutil, subprocess, time, threading, queue

os.makedirs('smoke_test', exist_ok=True)

# ModalPINN_VortexShedding.py hardcodes filename_data = 'Data/fixed_cylinder_atRe100'
# resolved relative to cwd (see R8's builder for the history of this gotcha).
os.makedirs('smoke_test/Data', exist_ok=True)
smoke_data_link = 'smoke_test/Data/fixed_cylinder_atRe100'
if not os.path.exists(smoke_data_link):
    os.symlink(os.path.abspath('Data/fixed_cylinder_atRe100'), smoke_data_link)
    print('Symlinked dataset into smoke_test/Data/')

def patch(src_path, dst_path, replacements):
    with open(src_path) as f:
        content = f.read()
    for old, new in replacements:
        assert old in content, 'patch target not found: %r' % old
        content = content.replace(old, new)
    with open(dst_path, 'w') as f:
        f.write(content)

# NOTE: R8's builder also patched DRIFT_CHECK_EVERY here - that constant is
# an R8-only addition (R8/src); the R9 sources are based on src/pressure_only
# and don't have it, so ModalPINN_VortexShedding.py is copied unpatched.
shutil.copyfile('ModalPINN_VortexShedding.py', 'smoke_test/ModalPINN_VortexShedding.py')
patch('NN_functions.py', 'smoke_test/NN_functions.py',
      [("def declare_LBFGS(loss,maxit=50000,maxfun=50000,ftol=1.0 * np.finfo(float).eps):",
        "def declare_LBFGS(loss,maxit=300,maxfun=300,ftol=1.0 * np.finfo(float).eps):")])
for fname in ['Load_train_data_desync.py', 'evaluate_regions.py', 'street_prior.py',
              'text_flow.py', 'reactions_process.py', 'street_prior_Ntap32.npz']:
    shutil.copyfile(fname, os.path.join('smoke_test', fname))

env = os.environ.copy()
env['LD_LIBRARY_PATH'] = '/content/miniconda/envs/modalpinn/lib:' + env.get('LD_LIBRARY_PATH', '')

cmd = [
    '/content/miniconda/envs/modalpinn/bin/python', 'ModalPINN_VortexShedding.py',
    '--SparseData', '--PressureOnly', '--NTaps', '32', '--FreestreamBC', '--FluctuationInletBC', '--Seed', '0',
    '--TrustStreet', '--StreetPrior', 'street_prior_Ntap32.npz',
    '--LBFGSFtol', '1e-12',
    # --SkipDiagnostics: the first smoke run PROVED training+save work, then
    # hung >10 min inside the post-training 'Error details' block (TF 1.14
    # builds the never-before-evaluated modal-equation residual graph on
    # first sess.run - >600 s of silence tripped the stall guard). The gate's
    # job is training + save; diagnostics are exercised by the real run,
    # whose weights are protected by the pre-diagnostics safety save.
    '--SkipDiagnostics',
    '--Tmax', '0.05', '--Nmes', '5000', '--Nint', '50000',
    '--multigrid', '--Ngrid', '5', '--NgridTurn', '200',
    '--WidthLayer', '25', '--Nmodes', '3',
]
print('Running smoke test (up to 1h at full Nint=50000). Streaming output live below;')
print('a stall (no output for STALL_LIMIT s) is flagged explicitly, not silent.')
print('=' * 70)

STALL_LIMIT = 600
HARD_LIMIT = 3600

# Background reader thread + Queue polling - a bare readline() loop would hang
# with the process (see R8's builder for the verified failure mode).
proc = subprocess.Popen(cmd, cwd='smoke_test', env=env, stdout=subprocess.PIPE,
                         stderr=subprocess.STDOUT, text=True, bufsize=1)
line_queue = queue.Queue()

def _reader():
    for line in iter(proc.stdout.readline, ''):
        line_queue.put(line)
    line_queue.put(None)

threading.Thread(target=_reader, daemon=True).start()

lines = []
start = time.time()
last_line_time = start
stalled = False
stream_closed = False
while True:
    try:
        line = line_queue.get(timeout=1.0)
        if line is None:
            stream_closed = True
        else:
            print(line, end='', flush=True)
            lines.append(line)
            last_line_time = time.time()
    except queue.Empty:
        pass
    now = time.time()
    if stream_closed and proc.poll() is not None:
        break
    if now - last_line_time > STALL_LIMIT:
        print('\n[[[ NO NEW OUTPUT FOR %ds - likely hung, killing the process ]]]' % STALL_LIMIT)
        proc.kill()
        stalled = True
        break
    if now - start > HARD_LIMIT:
        print('\n[[[ HARD TIME LIMIT (%ds) REACHED - killing the process ]]]' % HARD_LIMIT)
        proc.kill()
        stalled = True
        break

proc.wait()
smoke_stdout = ''.join(lines)
smoke_returncode = -1 if stalled else proc.returncode

print('=' * 70)
print('process finished: returncode=%s, stalled=%s, wall_time=%.0fs' % (smoke_returncode, stalled, time.time() - start))
"""))

cells.append(md("### 9.1 Smoke test verdict - hard stop if this fails"))
cells.append(code(r"""import re

checks = {}
checks['exit_code_zero'] = (smoke_returncode == 0)
checks['no_traceback'] = ('Traceback' not in smoke_stdout)
checks['trust_prior_active'] = ('R9 TrustStreet prior:' in smoke_stdout)
# 'Loss mesures training' is printed by every configuration (see
# ModalPINN_VortexShedding.py's tf_print block) - the tap loss, the one
# term that must be present and finite regardless of flags.
m = re.search(r'Loss mesures training\s*[:=]?\s*([-\d.eE+]+)', smoke_stdout)
checks['loss_printed'] = m is not None
if m is not None:
    val = float(m.group(1))
    checks['loss_finite'] = (val == val) and abs(val) != float('inf')
    print('Smoke-test loss value: %.6e' % val)
else:
    checks['loss_finite'] = False
checks['lbfgs_ran'] = ('CONVERGENCE' in smoke_stdout or 'ITERATIONS REACHED LIMIT' in smoke_stdout
                        or 'At iterate' in smoke_stdout)
# The pre-diagnostics safety save is the thing that protects the real 9h
# run's weights - the gate must prove it actually executes.
checks['model_saved'] = ('safety save done' in smoke_stdout)

print()
print('=' * 70)
for k, v in checks.items():
    print('  %-22s %s' % (k, v))
smoke_passed = all(checks.values())
print('-' * 70)
print('SMOKE TEST: %s' % ('PASS' if smoke_passed else 'FAIL'))
print('=' * 70)

assert smoke_passed, ('SMOKE TEST FAILED - stopping before the real ~9h run. '
                       'Read the printed output above for the actual error before debugging; '
                       'do not just re-run this notebook as-is.')
print()
print('Smoke test passed - proceeding to the real run below.')
"""))

cells.append(md("## 10. Run the real experiment (only reached if the smoke test above passed)\n\n"
                "Note: after 'End of training' the script now saves the model IMMEDIATELY ('safety save "
                "done'), then runs the full 'Error details' diagnostics - expect a stretch of total "
                "silence there (>10 min on the smoke config; TF 1.14 builds the modal-equation residual "
                "graph on its first evaluation). That silence is normal, and even if the runtime died "
                "during it the weights would already be on disk."))
cells.append(code("""import os
os.environ['LD_LIBRARY_PATH'] = '/content/miniconda/envs/modalpinn/lib:' + os.environ.get('LD_LIBRARY_PATH', '')
!/content/miniconda/envs/modalpinn/bin/python ModalPINN_VortexShedding.py \\
    --SparseData --PressureOnly --NTaps 32 --FreestreamBC --FluctuationInletBC --Seed 0 \\
    --TrustStreet --StreetPrior street_prior_Ntap32.npz \\
    --LBFGSFtol 1e-12 \\
    --Tmax 9 --Nmes 5000 --Nint 50000 \\
    --multigrid --Ngrid 5 --NgridTurn 200 \\
    --WidthLayer 25 --Nmodes 3
"""))

cells.append(md("## 11. Copy results to Drive and verify"))
cells.append(code("""import glob, shutil, os, datetime
runs = sorted(glob.glob('OutputPythonScript/ModalPINN_*'))
print('Found run folders:', runs)
assert runs, 'No output folder found - training may not have finished'
latest = runs[-1]
# Keep the street prior WITH the weights - evaluate_regions.py needs it to
# rebuild the trust ansatz (same bug class as --HardSym restore, see there).
shutil.copyfile('street_prior_Ntap32.npz', os.path.join(latest, 'street_prior_used.npz'))
run_name = ('R9_TRUST_street_noBVF_noK0_noPhase_pressure_only_Re100_Nm3_Nint50000_Nmes5000_WL25_Ntap32_'
            'FSBC_FIBC_rho0p6_cap0p12_seed0_' + datetime.date.today().strftime('%Y%m%d'))
dest = os.path.join('/content/drive/MyDrive/ModalPINN_results', run_name)
shutil.copytree(latest, dest, dirs_exist_ok=True)
print('Original folder:', latest)
print('Copied to Drive as:', dest)
"""))

cells.append(md("## 12. Confirm by reading a file back from Drive"))
cells.append(code("""with open(os.path.join(dest, 'out.txt')) as f:
    content = f.read()
print('Read', len(content), 'bytes from Drive-backed file')
print(content[-3000:])
"""))

cells.append(md("## 13. Regional accuracy evaluation - the headline comparison\n\n"
                "Expect (from the local testbed, which should be a floor, not a ceiling, at this "
                "scale): near-wake E_v <~ 0.64, far-wake E_v <~ 0.40. Every prior run sits at ~1.0."))
cells.append(code("""import subprocess
r = subprocess.run(
    ['/content/miniconda/envs/modalpinn/bin/python', 'evaluate_regions.py',
     '--RunDir', latest, '--WidthLayer', '25', '--Nmodes', '3',
     '--StreetPrior', 'street_prior_Ntap32.npz'],
    capture_output=True, text=True)
print(r.stdout)
print(r.stderr[-3000:])
output_text = r.stdout + ('\\n--- stderr ---\\n' + r.stderr if r.returncode != 0 else '')
with open(os.path.join(dest, 'regional_evaluation.txt'), 'w') as f:
    f.write(output_text)
print('Saved evaluation output to', os.path.join(dest, 'regional_evaluation.txt'))
"""))

nb = {
    "cells": cells,
    "metadata": {
        "kernelspec": {"display_name": "Python 3", "name": "python3"},
        "language_info": {"name": "python"},
    },
    "nbformat": 4,
    "nbformat_minor": 5,
}

os.makedirs(os.path.dirname(OUT), exist_ok=True)
with open(OUT, 'w') as f:
    json.dump(nb, f, indent=1)
print('Wrote %s (%d cells)' % (OUT, len(cells)))
