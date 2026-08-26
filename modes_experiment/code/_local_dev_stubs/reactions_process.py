# Local-only dev shim. Not used on Colab (the pinned conda env there has
# scipy 1.3.2, which still has scipy.integrate.trapz).
#
# The real src/reactions_process.py imports scipy.integrate.trapz, which
# newer scipy releases removed in favor of trapezoid. Load_train_data_desync.py
# imports this module unconditionally just for extract_reactions(), which
# bvf_targets.py never calls. This stub lets the real tap-extraction code be
# reused as-is locally without needing an old scipy pinned just for this.


def extract_reactions(filename):
    raise NotImplementedError('local dev stub only - not implemented, not used by bvf_targets.py')
