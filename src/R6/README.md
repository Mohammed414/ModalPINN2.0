# R6 Isolated Workspace

This folder contains the isolated R6 pressure-only worktree copy.

Contents:
- `pressure_only/ModalPINN_VortexShedding.py` - R6-copied training script with warm start, wake-restricted k=0 loss, k0 ramp, LBFGS ftol override, and drift monitoring.
- `pressure_only/NN_functions.py` - copied helper library used by the training script.
- `pressure_only/Load_train_data_desync.py` - copied data loader used by the training script.
- `pressure_only/DNN2_75_75_3_tanh.pickle` - warm-start checkpoint copied from R3.
- `pressure_only/bvf_targets_Ntap32_seed0.npz` - BVF targets copied locally for the notebook runs.
- `notebooks/R6_smoketest.ipynb` - notebook entrypoint for the warm-start and full R6 smoke tests.

The original `src/pressure_only/` files are untouched.