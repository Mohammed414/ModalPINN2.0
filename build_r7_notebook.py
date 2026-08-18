"""Builds R7/notebooks/R7_phase_loss_32taps.ipynb. Run locally, not on Colab."""
import json
import os
import re
import uuid

ROOT = os.path.dirname(os.path.abspath(__file__))
SRC = os.path.join(ROOT, 'src', 'R7')
OUT = os.path.join(ROOT, 'notebooks', 'R7_phase_loss_32taps.ipynb')


def md(src):
    return {"cell_type": "markdown", "id": uuid.uuid4().hex[:8], "metadata": {}, "source": src.splitlines(keepends=True)}


def code(src):
    return {"cell_type": "code", "id": uuid.uuid4().hex[:8], "metadata": {}, "execution_count": None,
            "outputs": [], "source": src.splitlines(keepends=True)}


def writefile_cell(fname, path):
    with open(path) as f:
        content = f.read()
    return code('%%writefile ' + fname + '\n' + content)


cells = []

cells.append(md("""# R7 - dispersion-consistency (wavenumber) loss, 32 taps

Adds `--PhaseLoss` on top of R6's exact recipe (warm start from R3, `--K0Loss --K0WakeOnly --LambdaK0 0.5`,
`--LBFGSFtol 1e-12`) - a single new ingredient, not a rebuild.

**Why**: a phase-0 audit (`src/pressure_only/audit_r7_mistimed_vs_deleted.py`,
`audit_r7_phase_term.py`) found that K0Loss/CV1Loss are BOTH exactly, provably invariant to the
mode-1 field's spatial phase - proven analytically (their formulas are built from conjugate-paired
products that cancel any common phase rotation exactly) and confirmed numerically to 5+ significant
figures. CV1's residual is additionally EXACTLY zero at zero amplitude by construction - a trivial,
perfect optimum at total wake death. Neither term can ever push the network toward the correct
spatial wavenumber; only the (comparatively weak, sparse) tap measurement loss can, and R7's earlier
observability diagnostic found the real R5 checkpoint has an amplitude deficit (5.7% of true) but an
even worse wavenumber deficit (13% of true, i.e. an essentially flat, non-propagating phase) -
consistent with exactly this asymmetry.

`Loss_phase` (see `ModalPINN_VortexShedding.py::loss_phase`) is normalized by `|mode1|^2 + eps` (a
FIXED floor, not derived from the current field's own scale - an earlier version had that bug and it
silently reproduced the same amplitude-blindness this term exists to fix), so its residual does NOT
vanish as amplitude collapses. Validated on the actual R5 checkpoint's own forward pass (numpy,
`audit_r7_phase_term.py`): 631x elevated over the true-field floor, with the checkpoint's own
estimated wavenumber essentially flat (0.0024 vs a target of 1.13 implied by its own mean flow) -
i.e. it would have flagged exactly the failure R5 produced.

**This term has never been run through actual TF1.14 before this notebook.** Every prior loss
addition in this project (K0Loss, CV1Loss) hit real problems (OOM, memory blowups) the first time it
touched the real graph - caught only by actually running it. So this notebook smoke-tests itself
first (few minutes, isolated throwaway run) and only proceeds to the real ~9h run if that passes
cleanly - if you wake up and the notebook stopped partway with an assertion error, read the smoke
test's printed output above it, don't just re-run blindly."""))

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

cells.append(md("## 6. Write the source files (real, un-reduced - maxit=50000, DRIFT_CHECK_EVERY=2000)\n\n"
                 "Includes the new `--PhaseLoss` term (see `loss_phase` in `ModalPINN_VortexShedding.py`) - "
                 "everything else is byte-for-byte R6."))
cells.append(writefile_cell('Load_train_data_desync.py', os.path.join(SRC, 'Load_train_data_desync.py')))
cells.append(writefile_cell('ModalPINN_VortexShedding.py', os.path.join(SRC, 'ModalPINN_VortexShedding.py')))
cells.append(writefile_cell('NN_functions.py', os.path.join(SRC, 'NN_functions.py')))
cells.append(writefile_cell('evaluate_regions.py', os.path.join(SRC, 'evaluate_regions.py')))
cells.append(writefile_cell('bvf_targets.py', os.path.join(SRC, 'bvf_targets.py')))
cells.append(writefile_cell('text_flow.py', os.path.join(ROOT, 'src', 'text_flow.py')))
cells.append(writefile_cell('reactions_process.py', os.path.join(ROOT, 'src', 'reactions_process.py')))

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
        'curl -L -o ' + local_path +
        ' "https://zenodo.org/records/5039610/files/fixed_cylinder_atRe100?download=1"')
    assert exit_code == 0, 'Download failed'
    os.makedirs(os.path.dirname(drive_cache), exist_ok=True)
    shutil.copyfile(local_path, drive_cache)
    print('Saved a copy to Drive for future runs:', drive_cache)
print('Dataset ready:', local_path, os.path.getsize(local_path), 'bytes')
"""))

cells.append(md("## 8. Build the BVF target from the 32 pressure taps"))
cells.append(code("""!/content/miniconda/envs/modalpinn/bin/python bvf_targets.py \\
    --DataFile Data/fixed_cylinder_atRe100 --NTaps 32 --Seed 0 --NoPlot
"""))

cells.append(md("## 9. Get R3's checkpoint to warm-start from (same choice R6 made)"))
cells.append(code("""import os
os.makedirs('warm_start', exist_ok=True)
r3_pickle = 'warm_start/R3_DNN2_75_75_3_tanh.pickle'
exit_code = os.system(
    'curl -sL -o ' + r3_pickle +
    ' "https://raw.githubusercontent.com/Mohammed414/ModalPINN2.0/main/runs/R3_fibc_bvf_pressure_only_Re100_Nm3_Nint50000_Nmes5000_WL25_Ntap32_FSBC_FIBC_BVF_lam0p1_seed0_20260804/DNN2_75_75_3_tanh.pickle"')
assert exit_code == 0 and os.path.getsize(r3_pickle) > 1000, 'R3 checkpoint download failed'
print('R3 checkpoint ready:', r3_pickle, os.path.getsize(r3_pickle), 'bytes')
"""))

# ---- smoke test gate ----
cells.append(md("""## 10. Smoke test FIRST - PhaseLoss has never touched real TF1.14 before this run

Deliberately kept as close as possible to R6's own PROVEN Gate B smoke test
(`R6/notebooks/R6_smoketest_gateB.ipynb`, which used exactly `--Tmax 0.05` with the FULL production
`--Nmes 5000 --Nint 50000 --Ngrid 5 --NgridTurn 200` - not smaller placeholder values): the K0Loss/
CV1Loss OOM problems found earlier in this project were confirmed to be a fixed GRAPH-TOPOLOGY cost,
not a data-volume one, so shrinking point counts for a smoke test would exercise a different,
untested configuration and could hide exactly the kind of memory issue this gate exists to catch.
Only `--Tmax` (wall-clock budget, hours) is reduced, plus maxit is patched down separately (300, not
Tmax-dependent - Tmax only bounds the later Adam phase, not L-BFGS's own iteration count) so this
still finishes in a bounded time. `PhaseLoss` is the one new ingredient relative to that already-
passing run. Runs in an isolated `smoke_test/` subfolder so it can never be confused with the real
run's output. Checks: process exits cleanly, no Python traceback, `Loss phase (component)` is
printed and is a finite number (not NaN/Inf), L-BFGS produced its `disp=True` termination message
(i.e. it actually ran, not just crashed before the first iteration)."""))
cells.append(code(r"""import os, re, shutil, subprocess, time, threading, queue

os.makedirs('smoke_test', exist_ok=True)

# ModalPINN_VortexShedding.py hardcodes filename_data = 'Data/fixed_cylinder_atRe100'
# (no --DataFile override exists), resolved relative to the process's cwd - since the
# smoke test runs with cwd='smoke_test', it would otherwise look for
# smoke_test/Data/fixed_cylinder_atRe100 and fail with FileNotFoundError (this is
# exactly what happened on the first real run of this notebook). Symlink rather than
# copy - the dataset is ~1.17GB and duplicating it would be slow and wasteful.
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

patch('ModalPINN_VortexShedding.py', 'smoke_test/ModalPINN_VortexShedding.py',
      [('DRIFT_CHECK_EVERY = 2000', 'DRIFT_CHECK_EVERY = 20')])
patch('NN_functions.py', 'smoke_test/NN_functions.py',
      [("def declare_LBFGS(loss,maxit=50000,maxfun=50000,ftol=1.0 * np.finfo(float).eps):",
        "def declare_LBFGS(loss,maxit=300,maxfun=300,ftol=1.0 * np.finfo(float).eps):")])
for fname in ['Load_train_data_desync.py', 'evaluate_regions.py', 'bvf_targets.py',
              'text_flow.py', 'reactions_process.py']:
    shutil.copyfile(fname, os.path.join('smoke_test', fname))

env = os.environ.copy()
env['LD_LIBRARY_PATH'] = '/content/miniconda/envs/modalpinn/lib:' + env.get('LD_LIBRARY_PATH', '')

cmd = [
    '/content/miniconda/envs/modalpinn/bin/python', 'ModalPINN_VortexShedding.py',
    '--SparseData', '--PressureOnly', '--NTaps', '32', '--FreestreamBC', '--FluctuationInletBC', '--Seed', '0',
    '--BVF', '--LambdaBVF', '0.1', '--BVFTargets', '../bvf_targets_Ntap32_seed0.npz',
    '--WarmStartFrom', '../warm_start/R3_DNN2_75_75_3_tanh.pickle',
    '--K0Loss', '--K0WakeOnly', '--LambdaK0', '0.5', '--LBFGSFtol', '1e-12',
    '--PhaseLoss', '--LambdaPhase', '0.1',
    '--Tmax', '0.05', '--Nmes', '5000', '--Nint', '50000',
    '--multigrid', '--Ngrid', '5', '--NgridTurn', '200',
    '--WidthLayer', '25', '--Nmodes', '3',
]
print('Running smoke test (R6''s equivalent gate took a while at full Nint=50000 - allowing up to 1h).')
print('Streaming output live below (unlike the first version of this cell, which buffered')
print('everything silently until the process exited - impossible to tell "slow" from "stuck").')
print('If no new line appears for STALL_LIMIT seconds, this is flagged explicitly, not just silent.')
print('=' * 70)

STALL_LIMIT = 600  # seconds with zero new output before flagging a likely hang
HARD_LIMIT = 3600  # total wall-clock budget

# NOTE: proc.stdout.readline() BLOCKS until a line is available or the stream
# closes - polling it directly in the same loop as the stall/hard-limit checks
# would mean those checks never run while nothing is being printed, i.e. a
# genuine hang would hang THIS cell too, silently, exactly the failure mode
# this is meant to catch. Verified locally with a mock hanging subprocess
# before trusting this: a naive single-threaded readline() loop hung
# indefinitely; a background reader thread + Queue.get(timeout=...) polling
# loop (below) correctly detected the stall in ~2s in that test.
proc = subprocess.Popen(cmd, cwd='smoke_test', env=env, stdout=subprocess.PIPE,
                         stderr=subprocess.STDOUT, text=True, bufsize=1)
line_queue = queue.Queue()

def _reader():
    for line in iter(proc.stdout.readline, ''):
        line_queue.put(line)
    line_queue.put(None)  # sentinel: stream closed

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
smoke_stderr = ''  # merged into stdout above (stderr=subprocess.STDOUT)
smoke_returncode = -1 if stalled else proc.returncode

print('=' * 70)
print('process finished: returncode=%s, stalled=%s, wall_time=%.0fs' % (smoke_returncode, stalled, time.time() - start))
"""))

cells.append(md("### 10.1 Smoke test verdict - hard stop if this fails"))
cells.append(code(r"""import re

checks = {}
checks['exit_code_zero'] = (smoke_returncode == 0)
checks['no_traceback'] = ('Traceback' not in smoke_stdout)
m = re.search(r'Loss phase \(component\)\s*[:=]?\s*([-\d.eE+]+)', smoke_stdout)
checks['phase_loss_printed'] = m is not None
if m is not None:
    val = float(m.group(1))
    checks['phase_loss_finite'] = (val == val) and abs(val) != float('inf')  # NaN check without numpy
    print('Loss phase (component) smoke-test value: %.6e' % val)
else:
    checks['phase_loss_finite'] = False
    print('Could not find "Loss phase (component)" in smoke test output at all - see stdout above.')
checks['lbfgs_ran'] = ('CONVERGENCE' in smoke_stdout or 'ITERATIONS REACHED LIMIT' in smoke_stdout
                        or 'At iterate' in smoke_stdout)

print()
print('=' * 70)
for k, v in checks.items():
    print('  %-22s %s' % (k, v))
smoke_passed = all(checks.values())
print('-' * 70)
print('SMOKE TEST: %s' % ('PASS' if smoke_passed else 'FAIL'))
print('=' * 70)

assert smoke_passed, ('SMOKE TEST FAILED - stopping before the real ~9h run. '
                       'Read the printed stdout/stderr above (cell 10) for the actual error before '
                       'debugging further; do not just re-run this notebook as-is.')
print('\\nSmoke test passed - proceeding to the real run below.')
"""))

cells.append(md("## 11. Run the real experiment (only reached if the smoke test above passed)"))
cells.append(code("""import os
os.environ['LD_LIBRARY_PATH'] = '/content/miniconda/envs/modalpinn/lib:' + os.environ.get('LD_LIBRARY_PATH', '')
!/content/miniconda/envs/modalpinn/bin/python ModalPINN_VortexShedding.py \\
    --SparseData --PressureOnly --NTaps 32 --FreestreamBC --FluctuationInletBC --Seed 0 \\
    --BVF --LambdaBVF 0.1 --BVFTargets bvf_targets_Ntap32_seed0.npz \\
    --WarmStartFrom warm_start/R3_DNN2_75_75_3_tanh.pickle \\
    --K0Loss --K0WakeOnly --LambdaK0 0.5 --LBFGSFtol 1e-12 \\
    --PhaseLoss --LambdaPhase 0.1 \\
    --Tmax 9 --Nmes 5000 --Nint 50000 \\
    --multigrid --Ngrid 5 --NgridTurn 200 \\
    --WidthLayer 25 --Nmodes 3
"""))

cells.append(md("## 12. Copy results to Drive and verify"))
cells.append(code("""import glob, shutil, os, datetime
runs = sorted(glob.glob('OutputPythonScript/ModalPINN_*'))
print('Found run folders:', runs)
assert runs, 'No output folder found - training may not have finished'
latest = runs[-1]
run_name = ('R7_phase_pressure_only_Re100_Nm3_Nint50000_Nmes5000_WL25_Ntap32_FSBC_FIBC_BVF_lam0p1_'
            'K0_0p5_wake_PH_0p1_WS_seed0_' + datetime.date.today().strftime('%Y%m%d'))
dest = os.path.join('/content/drive/MyDrive/ModalPINN_results', run_name)
shutil.copytree(latest, dest, dirs_exist_ok=True)
print('Original folder:', latest)
print('Copied to Drive as:', dest)
"""))

cells.append(md("## 13. Confirm by reading a file back from Drive"))
cells.append(code("""with open(os.path.join(dest, 'out.txt')) as f:
    content = f.read()
print('Read', len(content), 'bytes from Drive-backed file')
print(content[-3000:])
"""))

cells.append(md("## 14. Regional accuracy evaluation - the headline comparison"))
cells.append(code("""import subprocess
r = subprocess.run(
    ['/content/miniconda/envs/modalpinn/bin/python', 'evaluate_regions.py',
     '--RunDir', latest, '--WidthLayer', '25', '--Nmodes', '3'],
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
