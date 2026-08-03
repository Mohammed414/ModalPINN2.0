# Local-only dev shim. Not used on Colab (real TensorFlow is installed there).
#
# Load_train_data_desync.py imports `tensorflow` unconditionally at module
# level, even though the tap-extraction functions bvf_targets.py needs
# (read_cut_simulation_data_exp_point_and_cylinder, cut_simu_cylinder_only,
# etc.) never touch it. This stub exists only so those functions can be
# reused as-is (no duplicated logic) on a local machine that has no
# TensorFlow install. It is added to sys.path only as a fallback, and only
# by bvf_targets.py, when `import tensorflow` genuinely fails.


class _V1:
    def disable_eager_execution(self):
        pass


class _Compat:
    def __init__(self):
        self.v1 = _V1()


compat = _Compat()
