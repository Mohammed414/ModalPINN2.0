# -*- coding: utf-8 -*-
"""
ModalPINN Python Code
This is the main Python file for performing flow reconstruction using ModalPINN
as described in the paper

    ModalPINN : an extension of Physics-Informed Neural Networks with enforced 
    truncated Fourier decomposition  for periodic flow reconstruction using a 
    limited number of imperfect sensors. 
    G. Raynaud, S. Houde, F. P. Gosselin (2021)

This file contains the main losses functions of the ModalPINN as well as the 
main steps of the training. Nonetheless, it calls functions from 
-Load_train_data_desync.py:
    Python file containing functions that extract and prepare data for training 
    and validation.
-NN_functions.py:
    Python file containing functions specific to
        o neural networks (construction, initialisation),
        o optimisers (calling from scipy or tf interfaces, initialisation, training steps),
        o plots.

This file is designed to be launched on a computationel cluster (initially for 
Compute Canada - Graham server) using the following batch commands:
    #!/bin/bash
    #SBATCH --gres=gpu:t4:1
    #SBATCH --nodelist=gra1337
    #SBATCH --cpus-per-task=2
    #SBATCH --mem=50G
    #SBATCH --job-name=ModalPINN
    #SBATCH --time=0-10:00

    module load python/3.7.4
    source ~/ENV/bin/activate
    python ./ModalPINN_VortexShedding.py --Tmax 9 --Nmes 5000 --Nint 50000 --multigrid --Ngrid 5 --NgridTurn 200 --WidthLayer 25 --Nmodes 3 
    deactivate

For each job launched, a folder is created in ./OutputPythonScript and is 
identified by date-time information. In this folder, the content of consol prints 
is saved in a out.txt file alongside other files (mode shapes, various plots...)
including the model itself in a pickle archive.

Please refer to the help for the arguments sent to the parser and to the readme 
for librairies requirements.


@author: Gaétan Raynaud. 
ORCID : orcid.org/0000-0002-2802-7366
email : gaetan.raynaud (at) polymtl.ca
"""

# =============================================================================
# Librairies Import
# =============================================================================

import numpy as np
import tensorflow as tf
tf.compat.v1.disable_eager_execution()  # required: this codebase uses tf.compat.v1 placeholders
import matplotlib as mpl
mpl.use('Agg')
import matplotlib.pyplot as plt
import datetime
import os
import pickle
from shutil import copyfile
import sys
import GPUtil
import time
import argparse
import resource
from tensorflow.python.client import device_lib

# Code parts
import NN_functions as nnf
import Load_train_data_desync as ltd

def print_mem(tag):
    '''Peak resident memory so far (KB on Linux), printed on its own line at
    a few key checkpoints - added while diagnosing an OOM traced to R5's
    --K0Loss/--CV1Loss graph construction (see PROJECT_LOG.md). Cheap
    (a single getrusage syscall), left in permanently since colab-cli
    sessions have a hard 12GB memcg limit and future loss terms may hit it
    again - this makes the next one fast to diagnose instead of another
    multi-hour trial-and-error round.'''
    print('MEM %s : %d MB' % (tag, resource.getrusage(resource.RUSAGE_SELF).ru_maxrss // 1024))

# Link to simulations data 
# In the paper, we used those from Boudina et al. (2020) that can be downloaded 
# at https://zenodo.org/record/5039610
filename_data = 'Data/fixed_cylinder_atRe100'

t0 = time.time()
# =============================================================================
# matplotlib parameters
# =============================================================================

plt.rc('text', usetex=False)
plt.rc('font', family='serif')
plt.rc('font', size=18)
plt.rc('axes',titlesize=20)
plt.rc('legend',fontsize=18)
plt.rc('figure',titlesize=24)


# =============================================================================
# Preparing the writing of console prints in out.txt
# =============================================================================

class Tee(object):
    def __init__(self, *files):
        self.files = files
    def write(self, obj):
        for f in self.files:
            f.write(obj)
            f.flush() # If you want the output to be visible immediately
    def flush(self) :
        for f in self.files:
            f.flush()

# =============================================================================
# File copy and folder creation
# Here we create a folder containing all the data of this job
# And we copy current python files to keep track of how the job was launched
# =============================================================================
r = int(np.ceil(1000*np.random.rand(1)[0])) # This random number is used in case 2 jobs are launched at the exact same time so that the newly created folders does not merge the one into the other
d = datetime.datetime.now()
pythonfile = os.path.basename(__file__)
repertoire = 'OutputPythonScript/ModalPINN_'+ d.strftime("%Y_%m_%d-%H_%M_%S") + '__' +str(r)
os.makedirs(repertoire, exist_ok=True)
copyfile(pythonfile,repertoire+'/Copy_python_script.py')
copyfile('NN_functions.py',repertoire+'/NN_functions.py')
copyfile('Load_train_data_desync.py',repertoire+'/Load_train_data_desync.py')

f = open(repertoire+'/out.txt', 'w')
original = sys.stdout
sys.stdout = Tee(sys.stdout, f)
print('File copy and stdout ok')


# Print devices available


list_devices = device_lib.list_local_devices()
print('Devices available')
print(list_devices)

# =============================================================================
# Set arguments passed through bash
# =============================================================================

parser = argparse.ArgumentParser()

parser.add_argument('--Tmax',type=float,default=None,help="Define the max time allowed for optimisation (in hours)")
parser.add_argument('--Nmodes',type=int,default=2,help="Number of modes, including zero frequency")
parser.add_argument('--Nmes',type=int,default=5000,help="Number of measurement points to provide for optimisation")
parser.add_argument('--Nint',type=int,default=50000,help="Number of computing points to provide for equation evaluation during optimisation")
parser.add_argument('--LossModes',action="store_true",default=False,help="Use of modal equations during optimisation")
parser.add_argument('--multigrid',action="store_true",default=False,help="Use of multi grid")
parser.add_argument('--Ngrid',type=int,default=1,help="Number of batch for Adam optimization")
parser.add_argument('--NgridTurn',type=int,default=1000,help="Number of iterations between each batch changement")
parser.add_argument('--Noise',type=float,default=0.,help="Define standard deviation of gaussian noise added to measurements")
parser.add_argument('--WidthLayer',type=int,default=20,help="Number of neurons per layer and per mode")
parser.add_argument('--SparseData',action="store_true",default=False,help="if activated, use simulated  measurements data for training. Else use dense data")
parser.add_argument('--DesyncSparseData',action="store_true",default=False,help="if activated (and --SparseData == True), then simulated measurements are randomly made out of synchronisation")
parser.add_argument('--TwoZonesSampling',action="store_true",default=False,help="if activated, the sampling of equation penalisation points is carried out using 2 zones (with more points near the cylinder). Else use a uniform sampling")
parser.add_argument('--WakeBiasedSampling',action="store_true",default=False,help="R12: use 30% whole-domain uniform + 35% formation wake + 25% far wake + 10% near-cylinder annulus for NS collocation.")
parser.add_argument('--PressureOnly',action="store_true",default=False,help="if activated (requires --SparseData), drop pitot (u,v) velocity measurements and train only on cylinder-surface pressure taps.")
parser.add_argument('--NTaps',type=int,default=30,help="Number of pressure taps sampled uniformly around the cylinder border when --SparseData is used.")
parser.add_argument('--Seed',type=int,default=0,help="Seed for numpy and TensorFlow RNGs, for reproducible comparisons across tap counts.")
parser.add_argument('--FreestreamBC',action="store_true",default=False,help="Blend the network's mean velocity mode toward the known free-stream value (u=u_in, v=0) near the inlet, upstream of the cylinder. A second, independent prior alongside the existing cylinder no-slip encoding - not used downstream/in the wake, where the real flow is not free-stream.")
parser.add_argument('--FluctuationInletBC',action="store_true",default=False,help="Damp the fluctuating velocity modes (k>=1) toward zero at the inlet, using the same ramp as --FreestreamBC. Shedding is a wake phenomenon; nothing previously stopped a spurious oscillation from leaking upstream. Velocity only - pressure fluctuations do reach the inlet.")
parser.add_argument('--BVF',action="store_true",default=False,help="Add the Lighthill boundary-vorticity-flux loss term: enforces (1/Re)*d(omega)/dn = (1/R)*dp/dtheta at the cylinder wall, using a target derived from the same pressure taps (see bvf_targets.py). Requires --BVFTargets.")
parser.add_argument('--LambdaBVF',type=float,default=1.0,help="Weight of the BVF loss term when --BVF is set.")
parser.add_argument('--BVFTargets',type=str,default=None,help="Path to the .npz file produced by bvf_targets.py, required when --BVF is set.")
parser.add_argument('--CausalWeighting',action="store_true",default=False,help="Reweight the physics-residual loss so points near x_front (starting near the cylinder) count for more early in training, expanding downstream over --CausalWarmupIters iterations. Addresses the PINN 'causality violation' pathology (Wang et al. 2022) where distant collocation points can score near-zero residual for a near-zero (wrong) field just as easily as for a correct one, removing any training-time incentive to propagate the wake downstream. No new loss term, no new derivative order - same NS residual, just reweighted.")
parser.add_argument('--CausalSteepness',type=float,default=2.0,help="Sigmoid steepness of the causal weighting ramp (see --CausalWeighting).")
parser.add_argument('--CausalStartX',type=float,default=-1.0,help="Starting x position of the causal weighting frontier (see --CausalWeighting).")
parser.add_argument('--CausalEndX',type=float,default=8.0,help="Ending x position of the causal weighting frontier - the domain's downstream edge (see --CausalWeighting).")
parser.add_argument('--CausalWarmupIters',type=int,default=3000,help="Number of L-BFGS/Adam iterations over which the causal weighting frontier sweeps from --CausalStartX to --CausalEndX (see --CausalWeighting). Deliberately an absolute iteration count, not a fraction of some assumed total iteration budget: L-BFGS's actual convergence point isn't known in advance (R3 converged at ~15452 iterations, far short of the 50000 cap), so anchoring to a fraction of maxit risks the frontier never finishing its sweep before ftol is satisfied. 3000 is comfortably smaller than that precedent, leaving most of training at full-domain weighting to consolidate.")
parser.add_argument('--K0Loss',action="store_true",default=False,help="Add the k=0 harmonic momentum residual (mean-flow / Reynolds-stress balance, from the existing modal-equation machinery) as a separate weighted loss term. A dead (zero-amplitude) wake cannot satisfy this equation, since it needs the quadratic Reynolds-stress divergence that only a live oscillating wake supplies (Noack 2003 / Mantic-Lugo 2014 mean-field coupling). Also hard-projects the k=0 (mean) mode onto its real part in out_nn_modes_uv/out_nn_modes_p (see NN_functions), removing an otherwise-unconstrained imaginary gauge direction that would poison this loss (see 'R5 measured best candidates plan.md').")
parser.add_argument('--LambdaK0',type=float,default=0.025,help="Weight of the k=0 harmonic momentum residual loss term when --K0Loss is set. Calibrated via audit_r5_losses.py --Mode checkpoint against R3's early/mid-training loss magnitude (~0.1-0.3, the iteration range where R3's wake-collapse decision actually got locked in), not its tiny converged value (1.4e-4) - a lambda calibrated against the converged loss would be ~1000x smaller and make this term negligible during the exact training window where it needs to matter.")
parser.add_argument('--CV1Loss',action="store_true",default=False,help="Add the k=1 control-volume integral momentum balance (six fixed wake boxes) as a separate weighted loss term - a first-derivative-only, integral form of the k=1 momentum equation that measured strong (and adversarially robust) sensitivity to wake collapse in the diagnostic loss study. See 'R5 measured best candidates plan.md'.")
parser.add_argument('--LambdaCV1',type=float,default=0.0075,help="Weight of the k=1 control-volume integral loss term when --CV1Loss is set. Calibrated the same way as --LambdaK0 (see there) - against R3's early/mid-training loss magnitude, not its converged value.")
parser.add_argument('--HardSym',action="store_true",default=False,help="Hard-enforce the Karman-street mode parity (u_k,p_k even/odd, v_k odd/even in y per (-1)^k) by reflection-symmetrizing each mode's output in out_nn_modes_uv/out_nn_modes_p. Roughly doubles mode-network forward evaluations (graph size/step time). Optional, go/no-go at the smoke test (see plan).")
parser.add_argument('--SkipDiagnostics',action="store_true",default=False,help="R9: skip the post-training 'Error details' tf_print block. Those diagnostics evaluate graphs never built during training (notably the modal-equation residual, Loss_int_mode_wrap, when training used the time-collocation loss); TF 1.14 builds+optimizes each on first sess.run, which measured >10 min of total silence with the street ops threaded in - the R9 smoke test's stall guard killed exactly this. Used by the smoke test (training+save is what the gate proves); the real run keeps full diagnostics, protected by the pre-diagnostics safety save.")
parser.add_argument('--LBFGSFtol',type=float,default=1e-12,help="ftol passed to declare_LBFGS (ported from R6/R8, same rationale): the scipy default (~2.22e-16, machine epsilon) made R5's L-BFGS terminate after only 4574 iterations via a line-search failure at that unusually tight tolerance, not genuine convergence. 1e-12 is the value R6-R8 ran with.")
parser.add_argument('--TrustStreet',action="store_true",default=False,help="R9: wrap the k>=1 mode networks as a bounded trust-region correction around the closed-form von Karman street prior derived from the taps alone (see street_prior.py + NN_functions.street_modes_k). q_k = S_k + (rho|S_k| + cap)*tanh-bounded network correction, so the dead-wake solution q_k=0 of R1-R8 is excluded from the search space wherever the street is alive. The k=0 (mean) modes stay free networks. Validated in R9_wake_rescue/ (testbed: far-wake E_v 1.0 -> 0.40).")
parser.add_argument('--StreetPrior',type=str,default=None,help="Path to the street_prior_Ntap<N>.npz file produced by street_prior.py (required when --TrustStreet is set).")
parser.add_argument('--TrustRho',type=float,default=0.6,help="Trust-region radius as a fraction of the local street mode amplitude |S_k| (see --TrustStreet).")
parser.add_argument('--TrustCap',type=float,default=0.12,help="Additive floor of the trust-region radius - keeps a minimal correction capacity where the street prior is small (formation region, upstream). NOTE: in this codebase's one-sided mode convention (no factor 2 in NN_time_*) this value corresponds to 0.06 in the R9_wake_rescue testbed's two-sided convention (see street_modes_k's docstring).")

# R10: targeted radial trust candidate selected by the CFD diagnostic notebook.
parser.add_argument('--V1RadialTrust',action="store_true",default=False,help="Apply a proper complex radial trust region ONLY to v mode k=1 in the downstream wake. u, p, k=0 and k>=2 remain ordinary ModalPINN outputs.")
parser.add_argument('--V1TrustRho',type=float,default=0.70,help="Radial correction radius fraction rho for downstream v1. Candidate diagnostic selected rho=0.70 at x>=3D.")
parser.add_argument('--V1TrustXStart',type=float,default=3.0,help="Streamwise start of the v1 radial trust gate in D.")
parser.add_argument('--V1TrustXWidth',type=float,default=0.30,help="Smooth tanh transition width of the v1 radial trust gate.")
parser.add_argument('--V1TrustYMax',type=float,default=2.0,help="Half-width of the central wake band for v1 radial trust.")
parser.add_argument('--V1TrustYWidth',type=float,default=0.20,help="Smooth tanh transverse transition width.")
parser.add_argument('--LBFGSMaxit',type=int,default=50000,help="Maximum L-BFGS iterations. R10 smoke uses a deliberately short cap.")
parser.add_argument('--LBFGSMaxfun',type=int,default=50000,help="Maximum L-BFGS function evaluations.")
parser.add_argument('--SkipAdam',action="store_true",default=False,help="Skip the Adam polish after L-BFGS. Used for matched short smoke comparisons.")
parser.add_argument('--ExitAfterSafetySave',action="store_true",default=False,help="Exit cleanly immediately after the pre-diagnostics safety model save. Intended for smoke/preflight runs that only need trained weights for external evaluation; avoids legacy post-processing that assumes a non-empty Adam history.")
parser.add_argument('--RestoreModel',type=str,default=None,help='Warm-start from an existing DNN...pickle checkpoint. Restored tensors remain trainable.')


args = parser.parse_args()

if args.PressureOnly and not args.SparseData:
    raise ValueError('--PressureOnly requires --SparseData to also be set.')

if args.TrustStreet and args.StreetPrior is None:
    raise ValueError('--TrustStreet requires --StreetPrior <path to street_prior_Ntap<N>.npz produced by street_prior.py>.')

if args.V1RadialTrust and args.StreetPrior is None:
    raise ValueError('--V1RadialTrust requires --StreetPrior <street_prior_Ntap<N>.npz>.')
if args.V1RadialTrust and args.TrustStreet:
    raise ValueError('Use either --V1RadialTrust or legacy --TrustStreet, not both in the same run.')

if args.V1RadialTrust:
    if not (0.0 < args.V1TrustRho < 1.0):
        raise ValueError('--V1TrustRho must satisfy 0 < rho < 1 for the anti-collapse guarantee.')
    if args.V1TrustXWidth <= 0.0 or args.V1TrustYWidth <= 0.0:
        raise ValueError('--V1TrustXWidth and --V1TrustYWidth must be strictly positive.')

if args.BVF and args.BVFTargets is None:
    raise ValueError('--BVF requires --BVFTargets <path to the .npz file produced by bvf_targets.py>.')

if args.CV1Loss and not args.SparseData:
    raise ValueError('--CV1Loss requires --SparseData to also be set (the Vin-append anti-escape-hatch mechanism is only wired into the sparse-data collocation path).')

print('Args passed to python script')
print('Tmax '+str(args.Tmax)+' (h)')
print('Nmodes %d' % (args.Nmodes))
print('Nmes %d' % (args.Nmes))
print('Nint %d' % (args.Nint))
print('Use Loss Modes : ' + str(args.LossModes))
print('Multigrid : '+str(args.multigrid))
print('Ngrid : '+str(args.Ngrid))
print('Ngrid Turn : '+str(args.NgridTurn))
print('STD Noise : %.2e' % (args.Noise))
print('Neurons per layer and per mode : %d' % (args.WidthLayer))
print('Sparse Data : ' + str(args.SparseData))
print('Desync Sparse Data : ' + str(args.DesyncSparseData))
print('Pressure Only : ' + str(args.PressureOnly))
print('NTaps : %d' % (args.NTaps))
print('Seed : %d' % (args.Seed))
print('Freestream BC : ' + str(args.FreestreamBC))
print('Fluctuation Inlet BC : ' + str(args.FluctuationInletBC))
print('BVF : ' + str(args.BVF))
print('Lambda BVF : %.4g' % (args.LambdaBVF))
print('BVF Targets : ' + str(args.BVFTargets))
print('Causal Weighting : ' + str(args.CausalWeighting))
print('Causal Steepness : %.4g' % (args.CausalSteepness))
print('Causal Start X : %.4g' % (args.CausalStartX))
print('Causal End X : %.4g' % (args.CausalEndX))
print('Causal Warmup Iters : %d' % (args.CausalWarmupIters))
print('K0 Loss : ' + str(args.K0Loss))
print('Lambda K0 : %.4g' % (args.LambdaK0))
print('CV1 Loss : ' + str(args.CV1Loss))
print('Lambda CV1 : %.4g' % (args.LambdaCV1))
print('Hard Sym : ' + str(args.HardSym))
print('V1 Radial Trust : ' + str(args.V1RadialTrust))
print('V1 Trust rho/xstart/xwidth/ymax/ywidth : %.3g / %.3g / %.3g / %.3g / %.3g' % (args.V1TrustRho,args.V1TrustXStart,args.V1TrustXWidth,args.V1TrustYMax,args.V1TrustYWidth))
print('L-BFGS maxit/maxfun : %d / %d' % (args.LBFGSMaxit,args.LBFGSMaxfun))
print('Skip Adam : ' + str(args.SkipAdam))

if args.TwoZonesSampling and args.WakeBiasedSampling:
    raise ValueError('Choose only one of --TwoZonesSampling or --WakeBiasedSampling.')

if args.WakeBiasedSampling:
    IntSampling = 'wake_biased'
elif args.TwoZonesSampling:
    IntSampling = '2zones'
else:
    IntSampling = 'uniform'

print('Sampling of V_in : '+IntSampling)
if args.WakeBiasedSampling:
    print('R12 wake-biased source mixture: 30% whole / 35% formation / 25% far / 10% annulus')

# =============================================================================
# Reproducibility: seed numpy and TF graph-level RNG
# =============================================================================
np.random.seed(args.Seed)
tf.compat.v1.set_random_seed(args.Seed)
print('Random seed set to %d' % (args.Seed))
print_mem('startup (before graph construction)')

repertoire_new = repertoire
if args.PressureOnly:
    repertoire_new = repertoire_new + '_Ponly_Ntap%d' % (args.NTaps)
if args.FreestreamBC:
    repertoire_new = repertoire_new + '_FSBC'
if args.FluctuationInletBC:
    repertoire_new = repertoire_new + '_FIBC'
if args.TwoZonesSampling:
    repertoire_new = repertoire_new + '_2zones'
if args.WakeBiasedSampling:
    repertoire_new = repertoire_new + '_WAKEBIAS'
if args.BVF:
    repertoire_new = repertoire_new + '_BVF_lam%s' % (str(args.LambdaBVF).replace('.', 'p'))
if args.CausalWeighting:
    repertoire_new = repertoire_new + '_Causal_s%s' % (str(args.CausalSteepness).replace('.', 'p'))
if args.K0Loss:
    repertoire_new = repertoire_new + '_K0RS_lam%s' % (str(args.LambdaK0).replace('.', 'p'))
if args.CV1Loss:
    repertoire_new = repertoire_new + '_CV1_lam%s' % (str(args.LambdaCV1).replace('.', 'p'))
if args.HardSym:
    repertoire_new = repertoire_new + '_SYM'
if args.V1RadialTrust:
    _r = str(args.V1TrustRho).replace('.', 'p')
    _x = str(args.V1TrustXStart).replace('.', 'p')
    repertoire_new = repertoire_new + '_V1RAD_rho%s_x%s' % (_r, _x)
if args.RestoreModel is not None:
    repertoire_new = repertoire_new + '_WARM'
if repertoire_new != repertoire:
    os.rename(repertoire, repertoire_new)
    repertoire = repertoire_new
    print('Repertoire renamed to '+repertoire)

if args.RestoreModel is not None:
    copyfile(args.RestoreModel, repertoire+'/initial_model_used.pickle')
    print('Copied warm-start checkpoint into run folder.')

# =============================================================================
# Physical and geometrical parameters 
# =============================================================================

Re = 100.
Lxmin = -4. 
Lxmax = 8. 
Lx = Lxmax-Lxmin
Lymin = -4. 
Lymax = 4. 
Ly = Lymax-Lymin
x_c = 0. # x-Position of the centre of the cylindre
y_c = 0. # y-Position of the centre of the cylindre
r_c = 0.5 # radius
d = 2.*r_c
u_in = 1. 
rho_0 = 1.

omega_0 = 1.036 #Dimensionless frequency

geom = [Lxmin,Lxmax,Lymin,Lymax,x_c,y_c,r_c]


def xbc5(s):
    '''
    Compute cylinders border x coordinate as a function of curvilinear abscissa s \in [0,1]
    input : s (tf tensor, usually of shape [Nbc,1])
    return a tf tensor of the same shape as s
    '''
    return x_c + r_c*tf.cos(2*np.pi*s)
def ybc5(s):
    '''
    Compute cylinders border y coordinate as a function of curvilinear abscissa s \in [0,1]
    input : s (tf tensor, usually of shape [Nbc,1])
    return a tf tensor of the same shape as s
    '''
    return y_c + r_c*tf.sin(2*np.pi*s)

# =============================================================================
# Choix de discretisation
# =============================================================================

Nmodes = args.Nmodes

Nmes = args.Nmes # Number of measurement points in the domain in case of dense data
Nint = args.Nint # Number of points to penalize NS equations in \Omega_f
Nbc = 1000 # Number of points to sample on cylinders norder


multigrid = args.multigrid # If true, Adam optimiser will change of V_in sampling 
# every NgridTurn iterations between the Ngrid generated
Ngrid = args.Ngrid
NgridTurn = args.NgridTurn 

stdNoise = args.Noise # In case of artificially noised data, it defines the 
# standard deviation inputted in the Gaussian distribution

# List of frequencies associated with each mode shapes
# Note that it could be replaced with an arbitrary list of frequencies
# or even tf.Variables() that could be optimized during training
list_omega = np.asarray([k*omega_0 for k in range(Nmodes)]) 

# Structure of each Neural Network that approximate a mode shape
layers = [2,args.WidthLayer*Nmodes,args.WidthLayer*Nmodes,Nmodes]

# =============================================================================
# Training tracking variables
# =============================================================================

global it
global listeErrTimeSerie
global listeErrValidTimeSerie

it=0
listeErrTimeSerie = []
listeErrValidTimeSerie = []

plot_config = False

if args.Tmax==None:
    Tmax = None  #0.5*3600 #8h
else:
    Tmax = 3600*args.Tmax


# =============================================================================
# Placeholders declaration
# In TF<2, one can define placeholders and build an operation graph based on these.
# Values are provided only at the computation in a dictionary tf_dict when running
# session.run(TF quantity that depends on placeholders,feed_dict=TF dictionary containing placeholders values)
# =============================================================================

Nxpitot = 40 # Number of simulated pitot probe locations in the flow (4 sections of 10 points)
Ncyl = 30 # Number of points around the cylinder to simulated pressure probes
Ntimes = 201 # Number of timesteps in simulations data

# Placeholders for V_in (penalization of equations)
x_tf_int = tf.compat.v1.placeholder(dtype=tf.float32,shape=[None,1])
y_tf_int = tf.compat.v1.placeholder(dtype=tf.float32,shape=[None,1])
t_tf_int = tf.compat.v1.placeholder(dtype=tf.float32,shape=[None,1])

# Placeholders for general fitting data (especially dense data)
x_tf_mes = tf.compat.v1.placeholder(dtype=tf.float32,shape=[None,1])
y_tf_mes = tf.compat.v1.placeholder(dtype=tf.float32,shape=[None,1])
t_tf_mes = tf.compat.v1.placeholder(dtype=tf.float32,shape=[None,1])
u_tf_mes = tf.compat.v1.placeholder(dtype=tf.float32,shape=[None,1])
v_tf_mes = tf.compat.v1.placeholder(dtype=tf.float32,shape=[None,1])
p_tf_mes = tf.compat.v1.placeholder(dtype=tf.float32,shape=[None,1])

# Placeholder for simulated pitot probe
x_tf_mes_pitot = tf.compat.v1.placeholder(dtype=tf.float32,shape=[Ntimes*Nxpitot,1])
y_tf_mes_pitot = tf.compat.v1.placeholder(dtype=tf.float32,shape=[Ntimes*Nxpitot,1])
t_tf_mes_pitot = tf.compat.v1.placeholder(dtype=tf.float32,shape=[Ntimes*Nxpitot,1])
u_tf_mes_pitot = tf.compat.v1.placeholder(dtype=tf.float32,shape=[Ntimes*Nxpitot,1])
v_tf_mes_pitot = tf.compat.v1.placeholder(dtype=tf.float32,shape=[Ntimes*Nxpitot,1])
p_tf_mes_pitot = tf.compat.v1.placeholder(dtype=tf.float32,shape=[Ntimes*Nxpitot,1]) #  Not really used since only u and v are used at these locations for training


# Preparing desynchronisation of pitot probe. Especially Used if args.DesyncSparseData == True
Delta_phi_np_pitot = 0.*np.random.uniform(low=0.0,high=2*np.pi/omega_0, size=Nxpitot)

if args.DesyncSparseData:
    Delta_t_tf_pitot = tf.Variable(Delta_t_np_pitot,dtype=tf.float32,shape=[Nxpitot])
else:
    Delta_phi_tf_pitot = tf.constant(Delta_phi_np_pitot,dtype=tf.float32,shape=[Nxpitot])


t_tf_mes_pitot_unflatten = tf.reshape(t_tf_mes_pitot,[Ntimes,Nxpitot])
t_tf_mes_pitot_resync_unflatten = tf.convert_to_tensor([[ t_tf_mes_pitot_unflatten[t,k] - Delta_phi_tf_pitot[k] for k in range(Nxpitot)] for t in range(Ntimes)])
t_tf_mes_pitot_resync = tf.reshape(t_tf_mes_pitot_resync_unflatten,[Ntimes*Nxpitot,1]) 

# Cylindre data for simulated pressure probe
x_tf_mes_cyl = tf.compat.v1.placeholder(dtype=tf.float32,shape=[None,1])
y_tf_mes_cyl = tf.compat.v1.placeholder(dtype=tf.float32,shape=[None,1])
t_tf_mes_cyl = tf.compat.v1.placeholder(dtype=tf.float32,shape=[None,1])
p_tf_mes_cyl = tf.compat.v1.placeholder(dtype=tf.float32,shape=[None,1])

# Lighthill boundary-vorticity-flux enforcement grid (only fed when --BVF is
# set; declaring these unconditionally is harmless since an unused
# placeholder never needs feeding and doesn't affect any other tensor)
x_tf_bvf = tf.compat.v1.placeholder(dtype=tf.float32,shape=[None,1])
y_tf_bvf = tf.compat.v1.placeholder(dtype=tf.float32,shape=[None,1])
t_tf_bvf = tf.compat.v1.placeholder(dtype=tf.float32,shape=[None,1])
g_tf_bvf = tf.compat.v1.placeholder(dtype=tf.float32,shape=[None,1])


# Border
s_tf = tf.compat.v1.placeholder(dtype=tf.float32,shape=[None,1])
one_s_tf = tf.compat.v1.placeholder(dtype=tf.float32,shape=[None,1])

# Frequencies
w_tf = tf.constant(list_omega,dtype=tf.float32,shape=[Nmodes])


# =============================================================================
# Model construction
# =============================================================================

# Initialisation / trainable warm-start of weights and biases.
if args.RestoreModel is None:
    print('Initialising u/v/p networks from Xavier random weights.')
    w_u,b_u = nnf.initialize_NN(layers)
    w_v,b_v = nnf.initialize_NN(layers)
    w_p,b_p = nnf.initialize_NN(layers)
else:
    if not os.path.exists(args.RestoreModel):
        raise IOError('Restore checkpoint not found: '+args.RestoreModel)
    print('Warm-starting TRAINABLE u/v/p networks from:', args.RestoreModel)
    w_u,b_u,w_v,b_v,w_p,b_p = nnf.restore_NN(
        layers,args.RestoreModel,tf_as_constant=False)


# Known free-stream velocity, used as a prior near the inlet when
# --FreestreamBC is set (see NN_functions.f_freestream_weight/out_nn_modes_uv).
# None when the flag is off, so behaviour is unchanged by default.
freestream_target_u = u_in if args.FreestreamBC else None
freestream_target_v = 0. if args.FreestreamBC else None
# Damps the fluctuating velocity modes (k>=1) toward zero at the inlet when set
# (see NN_functions.out_nn_modes_uv). None of BVF/measurement/interior losses
# call out_nn_modes_uv/NN_time_uv directly - they all go through these four
# wrappers - so this one flag propagates everywhere automatically, including
# the end-of-run mode plots.
damp_fluct = bool(args.FluctuationInletBC)
# Hard-kills the k=0 (mean) mode's imaginary part in every u/v/p wrapper below
# (see NN_functions.out_nn_modes_uv/out_nn_modes_p) - only matters once --K0Loss
# actually reads the complex k=0 mode; a no-op change in behaviour otherwise.
kill_k0 = bool(args.K0Loss)
# Hard-enforces Karman-street mode parity in every u/v/p wrapper below (see
# NN_functions.out_nn_modes_uv/out_nn_modes_p) when --HardSym is set.
hard_sym = bool(args.HardSym)
# R9: closed-form street prior parameters for the trust ansatz (--TrustStreet).
# Plain python-float dict (baked into the TF graph as constants - the prior is
# frozen; only the correction networks train). None when the flag is off, so
# behaviour is unchanged by default. Threaded through every u/v/p wrapper
# below - like damp_fluct/hard_sym, no loss or plot code path bypasses these.
street_params = None
v1_radial_params = None
if args.TrustStreet or args.V1RadialTrust:
    _sp = np.load(args.StreetPrior)
    _loaded_street = {k: float(_sp[k]) for k in
                      ('Gamma', 'Uc', 'xf', 'r0', 'omega', 'phase',
                       'amp_scale', 'scale_p', 'ramp', 'delta')}
    if args.TrustStreet:
        street_params = _loaded_street
        print('R9 legacy TrustStreet prior:', street_params)
    if args.V1RadialTrust:
        v1_radial_params = _loaded_street
        print('R10 v1-only radial prior:', v1_radial_params)
    if abs(_loaded_street['omega'] - omega_0) > 0.02:
        print('WARNING: tap-fitted omega %.4f differs from omega_0=%.4f used '
              'by the modal ansatz' % (_loaded_street['omega'], omega_0))
trust_rho = float(args.TrustRho)
trust_cap = float(args.TrustCap)
v1_trust_rho = float(args.V1TrustRho)
v1_xstart = float(args.V1TrustXStart)
v1_xwidth = float(args.V1TrustXWidth)
v1_ymax = float(args.V1TrustYMax)
v1_ywidth = float(args.V1TrustYWidth)

def fluid_u(x,y):
    '''
    Compute mode shapes of u
    Input : x,y TF tensors of shape [Nint,1]
    Return TF tensor of shape [1,Nint,Nmodes] with complex values
    '''
    return nnf.out_nn_modes_uv(x,y,w_u,b_u,geom,freestream_target=freestream_target_u,damp_fluctuations=damp_fluct,kill_k0_imag=kill_k0,hard_sym=hard_sym,is_v=False,street_params=street_params,trust_rho=trust_rho,trust_cap=trust_cap)

def fluid_u_t(x,y,t):
    '''
    Compute u at instant t and position x,y
    Input: x,y,t TF tensors of shape [Nint,1]
    Return TF tensor of shape [Nint,1] with real values
    '''
    return nnf.NN_time_uv(x,y,t,w_u,b_u,geom,omega_0,freestream_target=freestream_target_u,damp_fluctuations=damp_fluct,kill_k0_imag=kill_k0,hard_sym=hard_sym,is_v=False,street_params=street_params,trust_rho=trust_rho,trust_cap=trust_cap)

def fluid_v(x,y):
    '''
    Compute mode shapes of v
    Input : x,y TF tensors of shape [Nint,1]
    Return TF tensor of shape [1,Nint,Nmodes] with complex values
    '''
    return nnf.out_nn_modes_uv(x,y,w_v,b_v,geom,freestream_target=freestream_target_v,damp_fluctuations=damp_fluct,kill_k0_imag=kill_k0,hard_sym=hard_sym,is_v=True,street_params=street_params,trust_rho=trust_rho,trust_cap=trust_cap,v1_radial_params=v1_radial_params,v1_trust_rho=v1_trust_rho,v1_xstart=v1_xstart,v1_xwidth=v1_xwidth,v1_ymax=v1_ymax,v1_ywidth=v1_ywidth)

def fluid_v_t(x,y,t):
    '''
    Compute v at instant t and position x,y
    Input: x,y,t TF tensors of shape [Nint,1]
    Return TF tensor of shape [Nint,1] with real values
    '''
    return nnf.NN_time_uv(x,y,t,w_v,b_v,geom,omega_0,freestream_target=freestream_target_v,damp_fluctuations=damp_fluct,kill_k0_imag=kill_k0,hard_sym=hard_sym,is_v=True,street_params=street_params,trust_rho=trust_rho,trust_cap=trust_cap,v1_radial_params=v1_radial_params,v1_trust_rho=v1_trust_rho,v1_xstart=v1_xstart,v1_xwidth=v1_xwidth,v1_ymax=v1_ymax,v1_ywidth=v1_ywidth)

def fluid_p(x,y):
    '''
    Compute mode shapes of p
    Input : x,y TF tensors of shape [Nint,1]
    Return TF tensor of shape [1,Nint,Nmodes] with complex values
    '''
    return nnf.out_nn_modes_p(x,y,w_p,b_p,kill_k0_imag=kill_k0,hard_sym=hard_sym,street_params=street_params,trust_rho=trust_rho,trust_cap=trust_cap)

def fluid_p_t(x,y,t):
    '''
    Compute p at instant t and position x,y
    Input: x,y,t TF tensors of shape [Nint,1]
    Return TF tensor of shape [Nint,1] with real values
    '''
    return nnf.NN_time_p(x,y,t,w_p,b_p,omega_0,kill_k0_imag=kill_k0,hard_sym=hard_sym,street_params=street_params,trust_rho=trust_rho,trust_cap=trust_cap)

# =============================================================================
# Forces on cylinder
# =============================================================================

def force_cylinder_flatten(t):
    '''
    t : tf.float32 tensor shape [Nt,1]  
    ----
    return
    fx_tf,fy_tf :  tf.float32 tensor of shape [Nt,] containing averaged horizontal force on cylinder at time t
    '''
    Nt = int(t.shape[0])
    Ns = 1000 # Number of points to perform the integration over the border
    s_cyl = tf.constant(np.linspace(0.,1.,Ns), dtype = tf.float32, shape = [Ns,1])*tf.transpose(1+0.*t)
    # s_cyl = tf.random.uniform([Ns,1], minval=0., maxval = 1., dtype = tf.float32)*tf.transpose(1+0.*t)
    # Reshaping Space x Times on a same dimension
    s_cyl_r = tf.reshape(s_cyl,[Nt*Ns,1])
    x_cyl_r = tf.reshape(xbc5(s_cyl_r),[Nt*Ns,1]) 
    y_cyl_r = tf.reshape(ybc5(s_cyl_r),[Nt*Ns,1])
    t_cyl = (1.+0*s_cyl)*tf.transpose(t)
    t_cyl_r = tf.reshape(t_cyl,[Nt*Ns,1])

    # Computing fluid values along the border
    u = fluid_u_t(x_cyl_r,y_cyl_r,t_cyl_r)
    v = fluid_v_t(x_cyl_r,y_cyl_r,t_cyl_r)
    p = fluid_p_t(x_cyl_r,y_cyl_r,t_cyl_r)

    # Computing differentiated quantities
    u_x = tf.gradients(u, x_cyl_r)[0]
    u_y = tf.gradients(u, y_cyl_r)[0]
    u_xx = tf.gradients(u_x, x_cyl_r)[0]
    u_yy = tf.gradients(u_y, y_cyl_r)[0]

    v_x = tf.gradients(v, x_cyl_r)[0]
    v_y = tf.gradients(v, y_cyl_r)[0]
    v_xx = tf.gradients(v_x, x_cyl_r)[0]
    v_yy = tf.gradients(v_y, y_cyl_r)[0]

    # Computing normal and tangent vectors
    nx_base = - tf.gradients(y_cyl_r, s_cyl_r)[0]
    ny_base = tf.gradients(x_cyl_r, s_cyl_r)[0]
    normalisation = tf.sqrt(tf.square(nx_base) + tf.square(ny_base))
    nx = nx_base/normalisation
    ny = ny_base/normalisation

    # Computing local forces elements
    fx_tf_local = -p*nx + 2.*(1./Re)*u_x*nx + (1./Re)*(u_y+v_x)*ny
    fy_tf_local = -p*ny + 2.*(1./Re)*v_y*ny + (1./Re)*(u_y+v_x)*nx

    # Reshape to [Ns,Nt]
    fx_tf_local_r2 = tf.reshape(fx_tf_local,[Ns,Nt])
    fy_tf_local_r2 = tf.reshape(fy_tf_local,[Ns,Nt])

    # Integrating along the border for every time step
    fx_tf = -2.*np.pi*r_c*tf.reduce_mean(fx_tf_local_r2,axis=0)
    fy_tf = -2.*np.pi*r_c*tf.reduce_mean(fy_tf_local_r2,axis=0)

    return fx_tf,fy_tf


# =============================================================================
# Definition of functions for loss
# =============================================================================

def loss_int_mode_per_k(x,y):
    '''
    Parameters
    ----------
    x,y : float 32 tensor [Nint,1]

    Returns
    -------
    Return a tf.float32 tensor of shape [1,Nint,Nmodes]: the per-mode combined
    squared residual (x-momentum + y-momentum + continuity), NOT reduced over
    the mode axis - unlike loss_int_mode (below), which sums this over k and
    is otherwise unchanged. Split out so a single mode's contribution (e.g.
    k=0, the mean-flow/Reynolds-stress balance - see --K0Loss) can be used as
    its own loss term without re-deriving or duplicating this computation.
    '''
    all_u = fluid_u(x,y)
    all_v = fluid_v(x,y)
    all_p = fluid_p(x,y)


    one = tf.transpose(0.*x + 1.)

    def customgrad(fgrad,xgrad):
        '''
        Input frgad,xgrad : tf.complex64 tensor of shape [1,Nint,N+1] and [1,Nint] resp.
        Return a tf.complex64 tensor df/dx of shape [1,Nint,N+1]
        (tf.gradients does not seem to work with complex values and with f being of order 3... But it is mainly the same thing here)
        '''
        fgrad_xgrad =  [tf.complex(tf.gradients(tf.real(fgrad[:,:,k]), xgrad, grad_ys = one)[0],tf.gradients(tf.imag(fgrad[:,:,k]), xgrad, grad_ys = one)[0]) for k in range(Nmodes)]
        return tf.transpose(tf.convert_to_tensor(fgrad_xgrad), perm=[2,1,0])

    all_u_x = customgrad(all_u,x)
    all_u_y = customgrad(all_u,y)

    all_v_x = customgrad(all_v,x)
    all_v_y = customgrad(all_v,y)

    all_p_x = customgrad(all_p,x)
    all_p_y = customgrad(all_p,y)

    all_u_xx = customgrad(all_u_x,x)
    all_u_yy = customgrad(all_u_y,y)

    all_v_xx = customgrad(all_v_x,x)
    all_v_yy = customgrad(all_v_y,y)


    # x axis momentum equation
    f_u = tf.transpose(tf.convert_to_tensor([tf.complex(0.,k*omega_0)*all_u[:,:,k] for k in range(Nmodes)]), perm=[1,2,0])
    f_u += all_p_x
    f_u += (-1./Re)*(all_u_xx + all_u_yy)

    f_u_4a = [tf.reduce_sum(tf.convert_to_tensor([all_u[:,:,l]*all_u_x[:,:,k-l] for l in range(k+1)]), axis = 0) for k in range(Nmodes)]
    f_u += tf.transpose(tf.convert_to_tensor(f_u_4a), perm = [1,2,0])

    f_u_4b = [tf.reduce_sum(tf.convert_to_tensor([all_v[:,:,l]*all_u_y[:,:,k-l] for l in range(k+1)]), axis = 0) for k in range(Nmodes)]
    f_u += tf.transpose(tf.convert_to_tensor(f_u_4b), perm = [1,2,0])

    f_u_5a = [tf.reduce_sum(tf.convert_to_tensor([all_u[:,:,l]*tf.conj(all_u_x[:,:,l-k]) for l in range(k+1,Nmodes)]),axis=0) for k in range(Nmodes)]
    f_u_5a[-1] = f_u_5a[-2]*0.
    f_u += tf.transpose(tf.convert_to_tensor(f_u_5a), perm=[1,2,0])

    f_u_5b = [tf.reduce_sum(tf.convert_to_tensor([tf.conj(all_u[:,:,l-k])*all_u_x[:,:,l] for l in range(k+1,Nmodes)]),axis=0) for k in range(Nmodes)]
    f_u_5b[-1] = f_u_5b[-2]*0.
    f_u += tf.transpose(tf.convert_to_tensor(f_u_5b), perm=[1,2,0])

    f_u_5c = [tf.reduce_sum(tf.convert_to_tensor([all_v[:,:,l]*tf.conj(all_u_y[:,:,l-k]) for l in range(k+1,Nmodes)]),axis=0) for k in range(Nmodes)]
    f_u_5c[-1] = f_u_5c[-2]*0.
    f_u += tf.transpose(tf.convert_to_tensor(f_u_5c), perm=[1,2,0])

    f_u_5d = [tf.reduce_sum(tf.convert_to_tensor([tf.conj(all_v[:,:,l-k])*all_u_y[:,:,l] for l in range(k+1,Nmodes)]),axis=0) for k in range(Nmodes)]
    f_u_5d[-1] = f_u_5d[-2]*0.
    f_u += tf.transpose(tf.convert_to_tensor(f_u_5d), perm=[1,2,0])


    f_u_sq = nnf.square_norm(f_u)

    # y axis Momentum equation
    f_v = tf.transpose(tf.convert_to_tensor([tf.complex(0.,k*omega_0)*all_v[:,:,k] for k in range(Nmodes)]), perm=[1,2,0])
    f_v += all_p_y
    f_v += (-1./Re)*(all_v_xx + all_v_yy)

    f_v_4a = [tf.reduce_sum(tf.convert_to_tensor([all_u[:,:,l]*all_v_x[:,:,k-l] for l in range(k+1)]), axis = 0) for k in range(Nmodes)]
    f_v += tf.transpose(tf.convert_to_tensor(f_v_4a), perm = [1,2,0])

    f_v_4b = [tf.reduce_sum(tf.convert_to_tensor([all_v[:,:,l]*all_v_y[:,:,k-l] for l in range(k+1)]), axis = 0) for k in range(Nmodes)]
    f_v += tf.transpose(tf.convert_to_tensor(f_v_4b), perm = [1,2,0])

    f_v_5a = [tf.reduce_sum(tf.convert_to_tensor([all_u[:,:,l]*tf.conj(all_v_x[:,:,l-k]) for l in range(k+1,Nmodes)]),axis=0) for k in range(Nmodes)]
    f_v_5a[-1] = f_v_5a[-2]*0.
    f_v += tf.transpose(tf.convert_to_tensor(f_v_5a), perm=[1,2,0])

    f_v_5b = [tf.reduce_sum(tf.convert_to_tensor([tf.conj(all_u[:,:,l-k])*all_v_x[:,:,l] for l in range(k+1,Nmodes)]),axis=0) for k in range(Nmodes)]
    f_v_5b[-1] = f_v_5b[-2]*0.  #quand k=N, k+1 > N
    f_v += tf.transpose(tf.convert_to_tensor(f_v_5b), perm=[1,2,0])

    f_v_5c = [tf.reduce_sum(tf.convert_to_tensor([all_v[:,:,l]*tf.conj(all_v_y[:,:,l-k]) for l in range(k+1,Nmodes)]),axis=0) for k in range(Nmodes)]
    f_v_5c[-1] = f_v_5c[-2]*0.
    f_v += tf.transpose(tf.convert_to_tensor(f_v_5c), perm=[1,2,0])

    f_v_5d = [tf.reduce_sum(tf.convert_to_tensor([tf.conj(all_v[:,:,l-k])*all_v_y[:,:,l] for l in range(k+1,Nmodes)]),axis=0) for k in range(Nmodes)]
    f_v_5d[-1] = f_v_5d[-2]*0.
    f_v += tf.transpose(tf.convert_to_tensor(f_v_5d), perm=[1,2,0])


    f_v_sq = nnf.square_norm(f_v)


    # Mass conservation equation
    div_u = all_u_x + all_v_y
    div_u_sq = nnf.square_norm(div_u)

    return div_u_sq + f_u_sq + f_v_sq


def loss_int_mode(x,y):
    '''
    Parameters
    ----------
    x,y : float 32 tensor [Nint,1]

    Returns
    -------
    Return a tf.float32 tensor of shape [Nint,1] computing squared errors on
    modal equations - bit-identical to the pre-refactor implementation, now
    just the mode-axis sum of loss_int_mode_per_k (see there).
    '''
    return tf.reduce_sum(loss_int_mode_per_k(x,y), axis=2)


# =============================================================================
# k=1 control-volume integral momentum balance (R5, --CV1Loss) - see
# "R5 measured best candidates plan.md". Integral (not pointwise) form of the
# k=1 momentum equation over six fixed wake boxes: for the true field,
# momentum in = momentum out, i.e. R_j ~ 0. A dead wake was measured (in the
# diagnostic loss study this plan is based on) to satisfy the *pointwise*
# k>=1 equations almost as well as the true field (they are nearly
# homogeneous in amplitude), but not this integral form - it stayed
# adversarially robust. Explicitly NOT implementing a k=2 version: the same
# study measured the k=2 CV balance at 0.47x sensitivity - the dead wake
# satisfies it *better* than the truth, i.e. it is actively harmful as a
# training signal. Do not add one.
# =============================================================================

def conv_mode_k(a, b, k, Nmodes_local):
    '''
    Complex convolution at harmonic k of two per-mode complex tensors a,b
    (each [1,Npts,Nmodes], the same truncated-Fourier convention used by
    loss_int_mode_per_k's f_u_4a/4b + 5a-5d terms - the direct sum over
    l in [0,k] plus the two conjugate sums over l in (k,Nmodes) that account
    for the negative-frequency image of a real signal's higher harmonics).
    No derivative and no extra factor: this is the raw (a*b)-product's k-th
    Fourier coefficient (the momentum-flux tensor for --CV1Loss), not the
    pointwise NS residual (which multiplies one factor by its gradient).
    Written generically in Nmodes_local rather than as a fixed-length formula:
    a literal k=1 flux written out for 4 harmonics (k=0..3) silently drops two
    of its terms whenever, as in every run so far including R5's own planned
    command, --Nmodes 3 means only k=0,1,2 exist (Nmodes is the network's
    actual output width, confirmed against ModalPINN_VortexShedding.py's own
    layer construction - not assumed).
    Returns a tf.complex64 tensor of shape [1,Npts].
    '''
    direct = tf.reduce_sum(tf.convert_to_tensor([a[:,:,l]*b[:,:,k-l] for l in range(k+1)]), axis=0)
    if k+1 < Nmodes_local:
        conj_a = tf.reduce_sum(tf.convert_to_tensor([a[:,:,l]*tf.conj(b[:,:,l-k]) for l in range(k+1,Nmodes_local)]), axis=0)
        conj_b = tf.reduce_sum(tf.convert_to_tensor([tf.conj(a[:,:,l-k])*b[:,:,l] for l in range(k+1,Nmodes_local)]), axis=0)
        return direct + conj_a + conj_b
    else:
        return direct

def conv_deriv_k(a, ad, k, Nmodes_local):
    '''
    TF twin of audit_r5_losses.py's conv_deriv_k_np - the derivative-
    convolution pattern used by loss_int_mode_per_k's f_u_4a/4b + 5a/5b
    terms (one raw factor a, one derivative factor ad), factored out here
    for reuse by k0_residual's targeted mode-0 computation (see there for
    why this needs to exist separately from loss_int_mode_per_k).
    Returns a tf.complex64 tensor of shape [1,Npts].
    '''
    direct = tf.reduce_sum(tf.convert_to_tensor([a[:,:,l]*ad[:,:,k-l] for l in range(k+1)]), axis=0)
    if k+1 < Nmodes_local:
        conj_a = tf.reduce_sum(tf.convert_to_tensor([a[:,:,l]*tf.conj(ad[:,:,l-k]) for l in range(k+1,Nmodes_local)]), axis=0)
        conj_b = tf.reduce_sum(tf.convert_to_tensor([tf.conj(a[:,:,l-k])*ad[:,:,l] for l in range(k+1,Nmodes_local)]), axis=0)
        return direct + conj_a + conj_b
    return direct

def k0_residual(x, y):
    '''
    Minimal, mode-0-targeted computation of the k=0 harmonic momentum
    residual - mathematically equivalent to loss_int_mode_per_k(x,y)[:,:,0],
    but deliberately NOT built by calling that function.

    loss_int_mode_per_k computes 2nd derivatives (customgrad applied twice)
    and p_x/p_y for EVERY mode (0,1,2), because the full all-modes residual
    needs that. The k=0 slice specifically only ever reads mode 0's 2nd
    derivatives and mode 0's pressure gradient (modes 1,2 only enter k=0's
    formula through their FIRST derivatives, via the Reynolds-stress
    convolution sums) - the other ~24 gradient ops loss_int_mode_per_k would
    build are pure waste for this purpose on a forward evaluation, and
    actively harmful once backpropagated: --K0Loss puts this inside the
    optimized Loss, so declare_LBFGS's tf.gradients(Loss, weights) call has
    to differentiate whatever graph this function builds a SECOND time.
    Sharing loss_int_mode_per_k's full ~60-gradient-op graph for that OOM-
    killed a gate smoke test at ~11.7GB RSS on a 12GB colab-cli session -
    confirmed (by testing with as few as 10 collocation points, no change in
    peak memory) to be a fixed graph-topology cost, not a data-volume one,
    so trimming Nmodes-per-array here (not point count) is what actually
    matters. See PROJECT_LOG.md's R5 entry for the full diagnosis.

    Returns a tf.float32 tensor of shape [1,Npts]: |f_u|^2+|f_v|^2+|div_u|^2 at k=0.
    '''
    all_u = fluid_u(x, y)
    all_v = fluid_v(x, y)
    all_p = fluid_p(x, y)

    one = tf.transpose(0.*x + 1.)

    def cgrad_all(fgrad, xgrad):
        parts = [tf.complex(tf.gradients(tf.real(fgrad[:,:,k]), xgrad, grad_ys=one)[0],tf.gradients(tf.imag(fgrad[:,:,k]), xgrad, grad_ys=one)[0]) for k in range(Nmodes)]
        return tf.transpose(tf.convert_to_tensor(parts), perm=[2,1,0])

    def cgrad_mode0(fgrad_mode0, xgrad):
        return tf.complex(tf.gradients(tf.real(fgrad_mode0), xgrad, grad_ys=one)[0], tf.gradients(tf.imag(fgrad_mode0), xgrad, grad_ys=one)[0])

    # 1st derivatives - genuinely needed for all Nmodes (the convolution
    # sums read u_x/u_y/v_x/v_y at every mode, not just mode 0).
    all_u_x = cgrad_all(all_u, x)
    all_u_y = cgrad_all(all_u, y)
    all_v_x = cgrad_all(all_v, x)
    all_v_y = cgrad_all(all_v, y)

    # Mode-0-only: pressure gradient and 2nd derivatives.
    p0_x = cgrad_mode0(all_p[:,:,0], x)
    p0_y = cgrad_mode0(all_p[:,:,0], y)
    u0_xx = cgrad_mode0(all_u_x[:,:,0], x)
    u0_yy = cgrad_mode0(all_u_y[:,:,0], y)
    v0_xx = cgrad_mode0(all_v_x[:,:,0], x)
    v0_yy = cgrad_mode0(all_v_y[:,:,0], y)

    k = 0
    f_u = tf.complex(0., k*omega_0)*all_u[:,:,k] + p0_x - (1./Re)*(u0_xx + u0_yy)
    f_u = f_u + conv_deriv_k(all_u, all_u_x, k, Nmodes)
    f_u = f_u + conv_deriv_k(all_v, all_u_y, k, Nmodes)

    f_v = tf.complex(0., k*omega_0)*all_v[:,:,k] + p0_y - (1./Re)*(v0_xx + v0_yy)
    f_v = f_v + conv_deriv_k(all_u, all_v_x, k, Nmodes)
    f_v = f_v + conv_deriv_k(all_v, all_v_y, k, Nmodes)

    div_u = all_u_x[:,:,k] + all_v_y[:,:,k]

    return nnf.square_norm(f_u) + nnf.square_norm(f_v) + nnf.square_norm(div_u)

def grad_mode1(fgrad,xgrad):
    '''
    1st derivative of JUST the k=1 mode slice w.r.t. xgrad - used by the k=1
    control-volume loss (--CV1Loss), which only ever needs mode 1's
    derivative on box faces. Deliberately NOT a loop over all Nmodes like
    loss_int_mode_per_k's internal customgrad (which needs every mode):
    an earlier version reused that all-modes pattern here, computing and
    discarding gradients for modes 0 and 2 at every one of the 4
    derivatives (u1_x,u1_y,v1_x,v1_y) x 4 faces x 4 boxes - 3x more
    tf.gradients calls than necessary, which OOM-killed a gate smoke test
    (see PROJECT_LOG.md, R5 entry) on a 12GB colab-cli CPU session before
    training even started. Fixed by only ever differentiating mode 1.
    Input fgrad,xgrad : tf.complex64 [1,Npts,Nmodes] and tf.float32 [Npts,1] resp.
    Return tf.complex64 tensor d(mode 1)/dx of shape [1,Npts]
    '''
    one = tf.transpose(0.*xgrad + 1.)
    f1 = fgrad[:,:,1]
    return tf.complex(tf.gradients(tf.real(f1), xgrad, grad_ys=one)[0], tf.gradients(tf.imag(f1), xgrad, grad_ys=one)[0])

# Four fixed boxes, |y|<=2, clear of the cylinder (r_c=0.5) and strictly
# inside the domain (Lxmin=-4,Lxmax=8,Lymin=-4,Lymax=4): three paired
# (upstream x, downstream x) boxes sweeping into the near/mid wake, plus one
# full-wake box spanning all of them. The plan originally specified five
# paired boxes (x_up in {0.5,1,2,3,4}, x_down in {2,3,4,5,6}) plus the
# full-wake box (six total); Phase 0's R3-checkpoint audit
# (audit_r5_losses.py --Mode checkpoint) found the two furthest-downstream
# boxes (x in [3,5] and [4,6]) had INVERTED sensitivity - the dead R3 wake
# scored *better* on them than the true field (ratios 0.13x and 0.08x) - the
# same pathology the plan explicitly forbade for the k=2 harmonic (there
# measured at 0.47x). Dropped for the same reason: a box scoring the dead
# wake as more correct than the truth would train against wake revival in
# exactly the region R5 is trying to fix.
#
# Down to a single box (from the four survivors above) for an unrelated,
# later reason: --CV1Loss's memory cost turned out to be a FIXED graph-
# topology cost per box (confirmed by testing quadrature resolution from
# 64pts/face, 32x16 area down to 16pts/face, 8x8 area with ZERO change in
# peak memory - 10459MB vs 10406MB), not a data-volume cost, so cutting
# point density doesn't help but cutting box count does. Two boxes still
# OOM-killed a combined --K0Loss --CV1Loss smoke test at ~11.8GB on the
# 12GB colab-cli session (reached the multigrid/first-training-iteration
# step, which needs its own headroom on top of the ~10.9GB already used
# just building the graph and loading data) - down to one box for a real
# safety margin. Kept the full-wake box [0.5,6] (7.7x correct-sign
# discrimination in the R3-checkpoint audit) over the numerically stronger
# but narrower [0.5,2] box (16.5x): R5's primary acceptance criterion is
# far-wake E_v revival, and the full-wake box spans both near and far wake
# while [0.5,2] sits entirely in the near-cylinder region - see
# PROJECT_LOG.md for the full audit numbers and memory-debugging trace.
CV1_X_UP   = [0.5]
CV1_X_DOWN = [6. ]
CV1_YMIN, CV1_YMAX = -2., 2.
CV1_N_FACE_PTS = 64   # quadrature points per face (surface integrals) - resolution isn't the memory driver (see above), kept high for integration accuracy
CV1_N_AREA_X, CV1_N_AREA_Y = 32, 16  # quadrature grid (area/volume integral)

# Per-box normalizer for Loss_cv1 (see loss_cv1 below) - the full-wake
# box's R3-checkpoint |R_j|^2 from audit_r5_losses.py --Mode checkpoint, so
# it contributes ~O(1) to Loss_cv1 at the R3 checkpoint.
CV1_NORMALIZERS = [1.70640340664987e-2]

def _cv1_trapz_nodes_weights(a, b, n):
    '''Uniform trapezoid quadrature nodes+weights on [a,b], n points.'''
    nodes = np.linspace(a, b, n)
    w = np.full(n, (b - a) / (n - 1))
    w[0] *= 0.5
    w[-1] *= 0.5
    return nodes, w

def _build_cv1_boxes():
    '''Precompute, per box, the fixed numpy quadrature node coordinates +
    trapezoid weights for the 4 faces (surface integral) and the area grid
    (volume/storage integral). Pure numpy constants - the boxes do not move
    over training, so this only needs to run once at import time.'''
    boxes = []
    for x_up, x_down in zip(CV1_X_UP, CV1_X_DOWN):
        y_face, wy_face = _cv1_trapz_nodes_weights(CV1_YMIN, CV1_YMAX, CV1_N_FACE_PTS)
        x_face, wx_face = _cv1_trapz_nodes_weights(x_up, x_down, CV1_N_FACE_PTS)
        xa, wxa = _cv1_trapz_nodes_weights(x_up, x_down, CV1_N_AREA_X)
        ya, wya = _cv1_trapz_nodes_weights(CV1_YMIN, CV1_YMAX, CV1_N_AREA_Y)
        Xa, Ya = np.meshgrid(xa, ya, indexing='ij')
        Wa = np.outer(wxa, wya)
        boxes.append(dict(
            left_x=np.full(CV1_N_FACE_PTS, x_up, dtype=np.float32), left_y=y_face.astype(np.float32),
            left_w=wy_face.astype(np.float32), left_n=(-1., 0.),
            right_x=np.full(CV1_N_FACE_PTS, x_down, dtype=np.float32), right_y=y_face.astype(np.float32),
            right_w=wy_face.astype(np.float32), right_n=(1., 0.),
            bottom_x=x_face.astype(np.float32), bottom_y=np.full(CV1_N_FACE_PTS, CV1_YMIN, dtype=np.float32),
            bottom_w=wx_face.astype(np.float32), bottom_n=(0., -1.),
            top_x=x_face.astype(np.float32), top_y=np.full(CV1_N_FACE_PTS, CV1_YMAX, dtype=np.float32),
            top_w=wx_face.astype(np.float32), top_n=(0., 1.),
            area_x=Xa.flatten().astype(np.float32), area_y=Ya.flatten().astype(np.float32),
            area_w=Wa.flatten().astype(np.float32),
        ))
    return boxes

CV1_BOXES = _build_cv1_boxes()

def _cv1_all_quadrature_xy():
    '''All quadrature node (x,y) coordinates across all six boxes and all
    five sub-integrals (4 faces + area), concatenated and deduplicated by
    (x,y) pair. Used to append these exact points to the interior
    collocation set (Vin) so the ordinary pointwise physics loss also polices
    them every iteration - the anti-escape-hatch rule: every point the CV
    integral reads must also be physics-policed elsewhere, so the network
    cannot satisfy the box balance via a field that is only locally
    pathological exactly at the quadrature nodes.'''
    xs, ys = [], []
    for box in CV1_BOXES:
        for face in ['left', 'right', 'bottom', 'top']:
            xs.append(box[face + '_x']); ys.append(box[face + '_y'])
        xs.append(box['area_x']); ys.append(box['area_y'])
    x_all = np.concatenate(xs).astype(np.float32)
    y_all = np.concatenate(ys).astype(np.float32)
    xy = np.unique(np.stack([x_all, y_all], axis=1), axis=0)
    return xy[:, 0], xy[:, 1]

CV1_VIN_X, CV1_VIN_Y = _cv1_all_quadrature_xy()

def _cv1_box_residual(box):
    '''
    Complex 2-vector (Rx,Ry) k=1 control-volume momentum residual for one
    box (see module docstring above): the standard control-volume momentum
    balance, i*omega_0*integral(u1)dA (storage) + surface flux + pressure -
    viscous traction, matching sign conventions exactly against loss_int_time
    (pressure enters as +grad(p), viscous as -(1/Re)*Laplacian) and against
    force_cylinder_flatten's already-existing stress-tensor construction
    (fx_local = -p*nx + 2*(1/Re)*u_x*nx + (1/Re)*(u_y+v_x)*ny) for the
    viscous traction term, just evaluated on the k=1 complex mode instead of
    the real time-domain field.
    Returns (Rx, Ry) : two tf.complex64 scalar tensors.
    '''
    def col(a):
        return tf.constant(a.reshape(-1, 1), dtype=tf.float32)

    xa, ya = col(box['area_x']), col(box['area_y'])
    wa_c = tf.complex(tf.constant(box['area_w'], dtype=tf.float32), 0.)
    u1_a = fluid_u(xa, ya)[0, :, 1]
    v1_a = fluid_v(xa, ya)[0, :, 1]
    Rx = tf.complex(0., omega_0) * tf.reduce_sum(wa_c * u1_a)
    Ry = tf.complex(0., omega_0) * tf.reduce_sum(wa_c * v1_a)

    Re_c = tf.complex(Re, 0.)
    for face in ['left', 'right', 'bottom', 'top']:
        xf, yf = col(box[face + '_x']), col(box[face + '_y'])
        wf_c = tf.complex(tf.constant(box[face + '_w'], dtype=tf.float32), 0.)
        nx, ny = box[face + '_n']
        nx_c, ny_c = tf.complex(nx, 0.), tf.complex(ny, 0.)

        all_u = fluid_u(xf, yf)
        all_v = fluid_v(xf, yf)
        all_p = fluid_p(xf, yf)
        p1 = all_p[0, :, 1]
        Qxx = conv_mode_k(all_u, all_u, 1, Nmodes)[0, :]
        Qxy = conv_mode_k(all_u, all_v, 1, Nmodes)[0, :]
        Qyy = conv_mode_k(all_v, all_v, 1, Nmodes)[0, :]
        u1_x = grad_mode1(all_u, xf)[0, :]
        u1_y = grad_mode1(all_u, yf)[0, :]
        v1_x = grad_mode1(all_v, xf)[0, :]
        v1_y = grad_mode1(all_v, yf)[0, :]

        flux_x = Qxx*nx_c + Qxy*ny_c
        flux_y = Qxy*nx_c + Qyy*ny_c
        visc_x = (1./Re_c)*(2.*u1_x*nx_c + (u1_y+v1_x)*ny_c)
        visc_y = (1./Re_c)*((u1_y+v1_x)*nx_c + 2.*v1_y*ny_c)

        Rx = Rx + tf.reduce_sum(wf_c * (flux_x + p1*nx_c - visc_x))
        Ry = Ry + tf.reduce_sum(wf_c * (flux_y + p1*ny_c - visc_y))

    return Rx, Ry

def loss_cv1():
    '''
    Sum over all six k=1 control-volume boxes of |Rx|^2+|Ry|^2, each
    normalized by a fixed per-box constant (CV1_NORMALIZERS - see there;
    must be calibrated by audit_r5_losses.py before a real run) so no single
    box dominates purely from its size.
    Returns a tf.float32 scalar tensor.
    '''
    total = 0.
    for j, box in enumerate(CV1_BOXES):
        Rx, Ry = _cv1_box_residual(box)
        total = total + (nnf.square_norm(Rx) + nnf.square_norm(Ry)) / CV1_NORMALIZERS[j]
    return total


def loss_int_time(x,y,t):
    '''
    Parameters
    ----------
    x,y,t : tf.float 32 tensor [Nint,1]

    Returns
    -------
    Return [Nint,1] tensor containing squared error on NS equations
    '''
    u = fluid_u_t(x,y,t)
    v = fluid_v_t(x,y,t)
    p = fluid_p_t(x,y,t)

    u_t = tf.gradients(u,t)[0]
    v_t = tf.gradients(v,t)[0]

    u_x = tf.gradients(u, x)[0]
    u_y = tf.gradients(u, y)[0]
    u_xx = tf.gradients(u_x, x)[0]
    u_yy = tf.gradients(u_y, y)[0]

    v_x = tf.gradients(v, x)[0]
    v_y = tf.gradients(v, y)[0]
    v_xx = tf.gradients(v_x, x)[0]
    v_yy = tf.gradients(v_y, y)[0]

    p_x = tf.gradients(p, x)[0]
    p_y = tf.gradients(p, y)[0]

    f_u = u_t + (u*u_x + v*u_y) + p_x - (1./Re)*(u_xx + u_yy) 
    f_v = v_t + (u*v_x + v*v_y) + p_y - (1./Re)*(v_xx + v_yy)
    div_u = u_x + v_y

    return tf.square(f_u)+tf.square(f_v)+tf.square(div_u)


def loss_mes(xmes,ymes,tmes,umes,vmes,pmes):
    '''
    xmes,ymes,tmes,umes,vmes,pmes : [Nmes,1] tf.float32 tensor
    Return [Nmes,1] tf.float32 tensor containing square difference to measurements 
    '''
    u_DNN = fluid_u_t(xmes,ymes,tmes)
    v_DNN = fluid_v_t(xmes,ymes,tmes)
    p_DNN = fluid_p_t(xmes,ymes,tmes)

    return tf.square(u_DNN-umes) + tf.square(v_DNN-vmes) + tf.square(p_DNN-pmes)

def loss_mes_uv(xmes,ymes,tmes,umes,vmes):
    '''
    xmes,ymes,tmes,umes,vmes : [Nmes,1] tf.float32 tensor
    Return [Nmes,1] tf.float32 tensor containing square difference to measurements of velocity
    '''
    u_DNN = fluid_u_t(xmes,ymes,tmes)
    v_DNN = fluid_v_t(xmes,ymes,tmes)

    return tf.square(u_DNN-umes) + tf.square(v_DNN-vmes)

def loss_mes_p(xmes,ymes,tmes,pmes):
    '''
    xmes,ymes,tmes,pmes : [Nmes,1] tf.float32 tensor
    Return [Nmes,1] tf.float32 tensor containing square difference to measurements of pressure
    '''
    p_DNN = fluid_p_t(xmes,ymes,tmes)

    return tf.square(p_DNN-pmes)


def loss_bvf(x,y,t,g,residual_clip=50.):
    '''
    Lighthill wall relation (see bvf.md): at a stationary no-slip wall, all
    nonlinear/unsteady terms in the momentum equation vanish exactly, leaving
    (1/Re)*d(omega)/dn = (1/R)*dp/dtheta, omega = v_x - u_y, n = (x,y)/R
    outward. x,y must lie exactly on r = R = r_c (bvf_targets.py's analytic
    wall grid, not the ~0.4999 mesh nodes).

    Needs a third derivative through the network (omega is already a first
    derivative of u,v; w_x,w_y are second), which can occasionally produce
    very large - not NaN, just numerically extreme - values at a handful of
    points once weights move away from their small random init (observed on
    a real full-scale GPU run: a single L-BFGS step landed on a residual
    large enough to spike the loss to ~1e6 and abnormally terminate the line
    search - not reproduced in smaller-scale CPU checks, consistent with it
    being a rare event that needs many collocation points to hit). The
    residual is clipped before squaring so one such point can't dominate the
    mean or blow up the gradient - it still contributes its full unclipped
    gradient anywhere within a generous +-50 band (Phase 0 validation showed
    the true LHS/RHS are both O(1) in magnitude), only saturating for
    genuine outliers.
    Input x,y,t,g : [Nbvf,1] tf.float32 tensor
    Return [Nbvf,1] tf.float32 tensor of squared residuals
    '''
    u = fluid_u_t(x,y,t)
    v = fluid_v_t(x,y,t)
    w = tf.gradients(v,x)[0] - tf.gradients(u,y)[0]
    w_x = tf.gradients(w,x)[0]
    w_y = tf.gradients(w,y)[0]
    dwdn = (x*w_x + y*w_y) / r_c
    residual = (1./Re)*dwdn - g
    residual = tf.clip_by_value(residual, -residual_clip, residual_clip)
    return tf.square(residual)


def loss_BC(s):
    '''
    Return error on u=v=0 on cylinder border for each mode
    Input s : [Nbc,1] tf.float32 tensor of coordinates \in [0,1]
    Output : [] tf.float32 real positive number
    '''    
    x = xbc5(s)
    y = ybc5(s)
    u_k = fluid_u(x,y)
    v_k = fluid_v(x,y)

    err = tf.convert_to_tensor([nnf.square_norm(u_k[0,:,k]) + nnf.square_norm(v_k[0,:,k]) for k in range(Nmodes)])

    return tf.reduce_sum(tf.reduce_mean(err,axis=1))


# =============================================================================
# Causal weighting of the physics residual (R4, see "R4 fluctuation..." plan
# note: this addresses the PINN "causality violation" pathology (Wang et al.
# 2022) - distant collocation points can score near-zero residual for a
# near-zero (wrong) field just as easily as for a correct one, so nothing
# forces the network to solve the harder far-field/wake problem before it's
# "cheap" not to. x_front sweeps downstream over training so credit expands
# outward - same NS residual (loss_int_time, unchanged), only reweighted.
# =============================================================================

# x_front is scheduled externally (see causal_step_hook below), not learned -
# a plain tf.Variable(trainable=False) rather than a placeholder, so it never
# needs to be threaded through the multigrid tf_dict list: TF reads a
# variable's current graph-stored value on every sess.run automatically,
# and trainable=False already excludes it from tf.trainable_variables(), so
# neither declare_LBFGS's ScipyOptimizerInterface nor declare_Adam (both of
# which default to optimizing tf.trainable_variables()) will try to learn it.
x_front_var = tf.Variable(args.CausalStartX, dtype=tf.float32, trainable=False, name='x_front')
x_front_new_ph = tf.compat.v1.placeholder(dtype=tf.float32, shape=[])
x_front_assign_op = tf.compat.v1.assign(x_front_var, x_front_new_ph)

def causal_weight_fn(x, x_front, steepness):
    '''
    1 near/upstream of the frontier, smoothly -> 0 well past it.
    Input x : [Nint,1] tf.float32 tensor. x_front : scalar tf.Variable.
    Output : [Nint,1] tf.float32 tensor in (0,1)
    '''
    return tf.sigmoid(-(x - x_front) * steepness)

def causal_step_hook(it):
    '''Advance x_front linearly in iteration count from CausalStartX to
    CausalEndX over CausalWarmupIters, then hold at CausalEndX (full-domain
    weighting) for the remainder of training. No-op unless --CausalWeighting.
    Prints x_front's progress on its own line every 100 iterations (not
    appended to the "Loss: %.3e" line, which several scripts in this project
    parse via regex on that exact format) - for a loss-vs-iteration figure
    annotated with frontier position.'''
    if not args.CausalWeighting:
        return
    frac = min(1.0, it / max(1, args.CausalWarmupIters))
    new_x_front = args.CausalStartX + frac * (args.CausalEndX - args.CausalStartX)
    sess.run(x_front_assign_op, feed_dict={x_front_new_ph: new_x_front})
    if it % 100 == 0:
        print('Causal x_front @ it %d : %.4f' % (it, new_x_front))

# =============================================================================
# Training loss creation
# =============================================================================

# Wrap error on modal equations - the existing all-modes diagnostic, unchanged
# (Nint=50000 points, forward-evaluated once at the end for the "Loss eqs.
# modes" print; never part of the optimized Loss unless --LossModes, which no
# run uses, so this never gets differentiated a second time).
Loss_int_mode_wrap = tf.reduce_mean(loss_int_mode(x_tf_int, y_tf_int))

# k=0 harmonic residual (R5's --K0Loss - see the CLI arg docstring): the
# mean-flow/Reynolds-stress balance. A dead wake cannot satisfy this - it
# needs the quadratic Reynolds-stress divergence that only a live oscillating
# (k>=1) wake supplies. Uses k0_residual (see there), NOT
# loss_int_mode_per_k(x,y)[:,:,0] - mathematically the same k=0 formula, but
# built without the ~24 wasted 2nd-derivative/pressure-gradient ops
# loss_int_mode_per_k computes for modes 1,2 (never read by k=0's formula).
# That waste is harmless for a forward-only evaluation (e.g. the diagnostic
# above), but --K0Loss puts this INSIDE the optimized Loss, so
# declare_LBFGS's tf.gradients(Loss, weights) call has to differentiate
# whatever graph this builds a SECOND time - confirmed (by testing with as
# few as 10 collocation points, no change in peak memory) to be a fixed
# graph-topology cost, not a data-volume one, so K0_N_POINTS below is kept
# small mainly for per-iteration runtime, not as the OOM fix - see
# PROJECT_LOG.md's R5 entry for the full diagnosis (this OOM-killed a gate
# smoke test at ~11.7GB RSS on a 12GB colab-cli session before this fix).
K0_N_POINTS = 2000
_k0_rng = np.random.RandomState(42)
def _build_k0_points():
    x = _k0_rng.uniform(Lxmin + 0.5, Lxmax - 0.5, K0_N_POINTS * 2).astype(np.float32)
    y = _k0_rng.uniform(Lymin + 0.5, Lymax - 0.5, K0_N_POINTS * 2).astype(np.float32)
    r = np.sqrt((x - x_c) ** 2 + (y - y_c) ** 2)
    keep = r > 1.5 * r_c
    return x[keep][:K0_N_POINTS].reshape(-1, 1), y[keep][:K0_N_POINTS].reshape(-1, 1)
K0_X, K0_Y = _build_k0_points()
x_tf_k0 = tf.constant(K0_X, dtype=tf.float32)
y_tf_k0 = tf.constant(K0_Y, dtype=tf.float32)
Loss_k0_wrap = tf.reduce_mean(k0_residual(x_tf_k0, y_tf_k0))

# Wrap error on physical equations - unweighted mean by default (unchanged
# behavior); when --CausalWeighting is set, a weighted mean that upweights
# points near/upstream of x_front and downweights points still ahead of it.
Loss_int_time_raw = loss_int_time(x_tf_int, y_tf_int, t_tf_int)
if args.CausalWeighting:
    causal_w = causal_weight_fn(x_tf_int, x_front_var, args.CausalSteepness)
    Loss_int_time_wrap = tf.reduce_sum(causal_w * Loss_int_time_raw) / (tf.reduce_sum(causal_w) + 1e-8)
else:
    Loss_int_time_wrap = tf.reduce_mean(Loss_int_time_raw)

# Wrap error on (u,v,p) measurements
Loss_dense_mes = tf.reduce_mean(loss_mes(x_tf_mes,y_tf_mes,t_tf_mes,u_tf_mes,v_tf_mes,p_tf_mes))

# Wrap error on (u,v) measurements at simulated pitot probes locations
Loss_mes_pitot = tf.reduce_mean(loss_mes_uv(x_tf_mes_pitot,y_tf_mes_pitot,t_tf_mes_pitot_resync,u_tf_mes_pitot,v_tf_mes_pitot))
Loss_mes_pitot_desync = tf.reduce_mean(loss_mes_uv(x_tf_mes_pitot,y_tf_mes_pitot,t_tf_mes_pitot,u_tf_mes_pitot,v_tf_mes_pitot))

# Wrap error on pressure measurement around cylindre border
Loss_mes_cyl = tf.reduce_mean(loss_mes_p(x_tf_mes_cyl,y_tf_mes_cyl,t_tf_mes_cyl,p_tf_mes_cyl))

# Wrap error on the Lighthill boundary-vorticity-flux identity (only built when --BVF is set)
if args.BVF:
    Loss_bvf_wrap = tf.reduce_mean(loss_bvf(x_tf_bvf,y_tf_bvf,t_tf_bvf,g_tf_bvf))

# Simulated experimental losses
if args.PressureOnly:
    # Pressure-only mode: cylinder-surface pressure taps only, pitot velocity dropped entirely
    Loss_mes_exp = Loss_mes_cyl
else:
    Loss_mes_exp = Loss_mes_pitot + Loss_mes_cyl

if args.SparseData:
    Loss_mes = Loss_mes_exp
else: # Dense measurements are used for training
    Loss_mes = Loss_dense_mes

if args.LossModes:
    Loss = Loss_int_mode_wrap + Loss_mes
else: #Physical equations are used instead of modal equations
    Loss = Loss_int_time_wrap + Loss_mes

if args.BVF:
    Loss = Loss + args.LambdaBVF * Loss_bvf_wrap

if args.K0Loss:
    Loss = Loss + args.LambdaK0 * Loss_k0_wrap

if args.CV1Loss:
    Loss_cv1_wrap = loss_cv1()
    Loss = Loss + args.LambdaCV1 * Loss_cv1_wrap

print_mem('after Loss assembly (forward graph built)')

# =============================================================================
# Optimizer configuration
# =============================================================================

opt_LBFGS = nnf.declare_LBFGS(Loss, maxit=args.LBFGSMaxit, maxfun=args.LBFGSMaxfun, ftol=args.LBFGSFtol)
print_mem('after declare_LBFGS (d(Loss)/d(weights) graph built)')

opt_Adam = nnf.declare_Adam(Loss, lr=1e-5)
print_mem('after declare_Adam')

sess = nnf.declare_init_session()
print_mem('after session init')


# =============================================================================
# GPU use before loading data
# =============================================================================
print('GPU use before loading data')
GPUtil.showUtilization()

# =============================================================================
# Data set preparation
# =============================================================================

if args.SparseData:
    # Let's load data only at locations defined for simulated measurements
    print('Loading Sparse Data')

    x_int,y_int,t_int,s_train,xmes_pitot,ymes_pitot,tmes_pitot,umes_pitot,vmes_pitot,pmes_pitot,xmes_cyl,ymes_cyl,tmes_cyl,umes_cyl,vmes_cyl,pmes_cyl,Delta_phi_np_pitot_applied = ltd.training_dict(Nmes,Nint,Nbc,filename_data,geom,Tintmax=1e2,data_selection = 'cylinder_pitot',desync=args.DesyncSparseData, multigrid=multigrid,Ngrid=Ngrid,stdNoise=stdNoise,method_int = IntSampling, n_taps=args.NTaps)
    Ncyl = len(xmes_cyl)
    Npitot = len(xmes_pitot)
    Tmin = 400.

    if args.CV1Loss:
        # Anti-escape-hatch (see "R5 measured best candidates plan.md"):
        # append the k=1 CV integral's own fixed quadrature nodes to the
        # interior collocation set, so the ordinary pointwise physics loss
        # also polices these exact points every iteration, not just the CV
        # loss's own dedicated evaluation of them. Random times, same scale
        # as the rest of t_int (Tintmax=1e2 above) - a physically valid
        # periodic solution's NS residual should hold at any time.
        n_cv1 = len(CV1_VIN_X)
        if multigrid:
            for k in range(Ngrid):
                x_int[k] = np.concatenate([np.asarray(x_int[k]).flatten(), CV1_VIN_X])
                y_int[k] = np.concatenate([np.asarray(y_int[k]).flatten(), CV1_VIN_Y])
                t_int[k] = np.concatenate([np.asarray(t_int[k]).flatten(), np.random.uniform(0., 1e2, size=n_cv1).astype(np.float32)])
        else:
            x_int = np.concatenate([np.asarray(x_int).flatten(), CV1_VIN_X])
            y_int = np.concatenate([np.asarray(y_int).flatten(), CV1_VIN_Y])
            t_int = np.concatenate([np.asarray(t_int).flatten(), np.random.uniform(0., 1e2, size=n_cv1).astype(np.float32)])
        print('CV1: appended %d quadrature nodes to the interior collocation set (Vin)' % n_cv1)

    if multigrid:
        tf_dict = []
        for k in range(Ngrid):
            tf_dict_temp = {x_tf_int : np.reshape(x_int[k],(-1,1)),
             y_tf_int : np.reshape(y_int[k],(-1,1)),
             t_tf_int : np.reshape(t_int[k],(-1,1)),
             s_tf : np.reshape(s_train,(Nbc,1)),
             x_tf_mes_cyl : np.reshape(xmes_cyl,(Ncyl,1)),
             y_tf_mes_cyl : np.reshape(ymes_cyl,(Ncyl,1)),
             p_tf_mes_cyl : np.reshape(pmes_cyl,(Ncyl,1)),
             t_tf_mes_cyl : np.reshape(tmes_cyl,(Ncyl,1)),
             x_tf_mes_pitot : np.reshape(xmes_pitot,(Npitot,1)),
             y_tf_mes_pitot : np.reshape(ymes_pitot,(Npitot,1)),
             u_tf_mes_pitot : np.reshape(umes_pitot,(Npitot,1)),
             v_tf_mes_pitot : np.reshape(vmes_pitot,(Npitot,1)),
             p_tf_mes_pitot : np.reshape(pmes_pitot,(Npitot,1)),
             t_tf_mes_pitot : np.reshape(tmes_pitot,(Npitot,1)),
             }
            tf_dict.append(tf_dict_temp)

    else:
        tf_dict = {x_tf_int : np.reshape(x_int,(-1,1)),
             y_tf_int : np.reshape(y_int,(-1,1)),
             t_tf_int : np.reshape(t_int,(-1,1)),
             s_tf : np.reshape(s_train,(Nbc,1)),
             x_tf_mes_cyl : np.reshape(xmes_cyl,(Ncyl,1)),
             y_tf_mes_cyl : np.reshape(ymes_cyl,(Ncyl,1)),
             p_tf_mes_cyl : np.reshape(pmes_cyl,(Ncyl,1)),
             t_tf_mes_cyl : np.reshape(tmes_cyl,(Ncyl,1)),
             x_tf_mes_pitot : np.reshape(xmes_pitot,(Npitot,1)),
             y_tf_mes_pitot : np.reshape(ymes_pitot,(Npitot,1)),
             u_tf_mes_pitot : np.reshape(umes_pitot,(Npitot,1)),
             v_tf_mes_pitot : np.reshape(vmes_pitot,(Npitot,1)),
             p_tf_mes_pitot : np.reshape(pmes_pitot,(Npitot,1)),
             t_tf_mes_pitot : np.reshape(tmes_pitot,(Npitot,1))
             }

else:
    print('Loading Dense Data')
    x_int,y_int,t_int,s_train,xmes,ymes,tmes,umes,vmes,pmes = ltd.training_dict(Nmes,Nint,Nbc,filename_data,geom,Tintmax=1e2,data_selection = 'all',desync=False, multigrid=multigrid,Ngrid=Ngrid,stdNoise=stdNoise,cut=True,method_int=IntSampling)
    Nmes = len(xmes)
    Tmin = 400.

    if multigrid:
        tf_dict = []
        for k in range(Ngrid):
            tf_dict_temp = {x_tf_int : np.reshape(x_int[k],(Nint,1)),
              y_tf_int : np.reshape(y_int[k],(Nint,1)),
              t_tf_int : np.reshape(t_int[k],(Nint,1)),
              s_tf : np.reshape(s_train,(Nbc,1)),
              x_tf_mes : np.reshape(xmes,(Nmes,1)),
              y_tf_mes : np.reshape(ymes,(Nmes,1)),
              p_tf_mes : np.reshape(pmes,(Nmes,1)),
              t_tf_mes : np.reshape(tmes,(Nmes,1)),
              u_tf_mes : np.reshape(umes,(Nmes,1)),
              v_tf_mes : np.reshape(vmes,(Nmes,1))
              }
            tf_dict.append(tf_dict_temp)

    else:      
        tf_dict = {x_tf_int : np.reshape(x_int,(Nint,1)),
              y_tf_int : np.reshape(y_int,(Nint,1)),
              t_tf_int : np.reshape(t_int,(Nint,1)),
              s_tf : np.reshape(s_train,(Nbc,1)),
              x_tf_mes : np.reshape(xmes,(Nmes,1)),
              y_tf_mes : np.reshape(ymes,(Nmes,1)),
              p_tf_mes : np.reshape(pmes,(Nmes,1)),
              t_tf_mes : np.reshape(tmes,(Nmes,1)),
              u_tf_mes : np.reshape(umes,(Nmes,1)),
              v_tf_mes : np.reshape(vmes,(Nmes,1))
              }

if args.BVF:
    # Feed the same fixed wall grid + target into every entry of tf_dict
    # (a plain dict in the non-multigrid case, a list of dicts otherwise),
    # since Loss now depends on these placeholders whenever --BVF is set.
    bvf_npz = np.load(args.BVFTargets)
    x_bvf_wall = bvf_npz['x_wall'].astype(np.float32)
    y_bvf_wall = bvf_npz['y_wall'].astype(np.float32)
    t_bvf_grid = bvf_npz['t_grid'].astype(np.float32)
    G_bvf = bvf_npz['G'].astype(np.float32)  # [Ntheta, Ntime]
    Ntheta_bvf = len(x_bvf_wall)
    Ntime_bvf = len(t_bvf_grid)
    X_bvf = np.tile(x_bvf_wall.reshape(-1,1), (1,Ntime_bvf)).reshape(-1,1)
    Y_bvf = np.tile(y_bvf_wall.reshape(-1,1), (1,Ntime_bvf)).reshape(-1,1)
    T_bvf = np.tile(t_bvf_grid.reshape(1,-1), (Ntheta_bvf,1)).reshape(-1,1)
    Gflat_bvf = G_bvf.reshape(-1,1)
    bvf_feed = {x_tf_bvf: X_bvf, y_tf_bvf: Y_bvf, t_tf_bvf: T_bvf, g_tf_bvf: Gflat_bvf}

    if isinstance(tf_dict, list):
        for tf_dict_k in tf_dict:
            tf_dict_k.update(bvf_feed)
    else:
        tf_dict.update(bvf_feed)
    print('BVF targets loaded from %s: %d wall points x %d times = %d enforcement points' %
          (args.BVFTargets, Ntheta_bvf, Ntime_bvf, Ntheta_bvf*Ntime_bvf))


# Validation data set loading
# We extract 10 times more points for both dense measurements and equation penalisation
print('Loading validation data set')

x_int_valid,y_int_valid,t_int_valid,s_train,xmes_valid,ymes_valid,tmes_valid,umes_valid,vmes_valid,pmes_valid = ltd.training_dict(10*Nmes,10*Nint,Nbc,filename_data,geom,Tintmax=1e2,cut=True,method_int='uniform')
Nmesvalid = len(xmes_valid)

tf_dict_valid = {x_tf_int : np.reshape(x_int_valid,(10*Nint,1)),
     y_tf_int : np.reshape(y_int_valid,(10*Nint,1)),
     t_tf_int : np.reshape(t_int_valid,(10*Nint,1)),
     s_tf : np.reshape(s_train,(Nbc,1)),
     x_tf_mes : np.reshape(xmes_valid,(Nmesvalid,1)),
     y_tf_mes : np.reshape(ymes_valid,(Nmesvalid,1)),
     u_tf_mes : np.reshape(umes_valid,(Nmesvalid,1)),
     v_tf_mes : np.reshape(vmes_valid,(Nmesvalid,1)),
     p_tf_mes : np.reshape(pmes_valid,(Nmesvalid,1)),
     t_tf_mes : np.reshape(tmes_valid,(Nmesvalid,1))}


# =============================================================================
# GPU use after loading data
# =============================================================================
print('GPU use after loading data')
GPUtil.showUtilization()
print_mem('after data loading')


# =============================================================================
# Training
# =============================================================================

nnf.print_bar()
t1 = time.time()
print('Start training after %d s'%(t1-t0))

print('Start L-BFGS-B training')
List_it_loss_LBFGS,List_it_loss_valid_LBFGS = nnf.model_train_scipy(opt_LBFGS,sess,Loss,tf_dict[0],List_loss = True,tf_dict_valid=tf_dict_valid,loss_valid = Loss_dense_mes,step_hook=causal_step_hook)

t2 = time.time()
print('L-BFGS-B training ended after %d s'%(t2-t1))

if args.SkipAdam:
    print('Skipping Adam training (--SkipAdam)')
    List_it_loss_Adam = []
    List_it_loss_valid_Adam = []
    t3 = t2
else:
    print('Start Adam training')
    # Here Adam training is stopped if it reaches a time limit AdamTmax, or number of iterations Nit or if training loss goes under tolAdam
    AdamTmax = Tmax-(t2-t0)
    List_it_loss_Adam,List_it_loss_valid_Adam = nnf.model_train_Adam(opt_Adam,sess,Loss,liste_tf_dict=tf_dict,Nit=1e5,tolAdam=1e-5,it=it,itdisp=100,maxTime=AdamTmax,multigrid=multigrid,NgridTurn=NgridTurn,List_loss = True,tf_dict_valid=tf_dict_valid,loss_valid = Loss_dense_mes,step_hook=causal_step_hook)
    t3 = time.time()
    print('Adam training ended after %d s'%(t3-t2))

# =============================================================================
# GPU use after training
# =============================================================================
print('GPU use after training')
GPUtil.showUtilization()
print('End of training')

# =============================================================================
# R9: Save NN Model coefficients FIRST, before any post-training diagnostics.
# The R9 smoke test caught the modal-equation diagnostic below hanging for
# >10 min on its first evaluation (TF 1.14 builds/optimizes that large
# residual graph - now larger still with the street ops threaded in - on
# first sess.run). Saving used to happen ~70 lines further down, i.e. a hang
# or kill in the diagnostics would lose a full 9h run's weights. The save at
# the original location below is kept (it appends to the same file; loaders
# read the first pickle record) so downstream tooling is unchanged.
# =============================================================================

# R11: save a compact final loss breakdown BEFORE the clean exit.
# With pressure-only and no auxiliary loss terms:
#   total loss = physics loss + pressure-tap loss.
_eval_dict = tf_dict[0] if isinstance(tf_dict, list) else tf_dict
_physics_tensor = Loss_int_mode_wrap if args.LossModes else Loss_int_time_wrap
_r11_total, _r11_phys, _r11_taps = sess.run(
    [Loss, _physics_tensor, Loss_mes], feed_dict=_eval_dict)
_r11_summary = {
    'total_loss': float(_r11_total),
    'physics_loss': float(_r11_phys),
    'pressure_tap_loss': float(_r11_taps),
    'warm_started': bool(args.RestoreModel is not None),
    'restore_model': args.RestoreModel,
    'Nint': int(Nint),
    'Nmes': int(Nmes),
    'LBFGS_maxit': int(args.LBFGSMaxit),
    'LBFGS_maxfun': int(args.LBFGSMaxfun),
    'adam_skipped': bool(args.SkipAdam),
    'int_sampling': IntSampling,
}
with open(repertoire+'/training_loss_summary.json','w') as _fj:
    import json as _json
    _json.dump(_r11_summary,_fj,indent=2)
print('R11 FINAL LOSS SUMMARY:', _r11_summary)

print('Saving NN Model (pre-diagnostics safety save)...')
str_layers_fluid = [str(j) for j in layers]
filename_fluid = repertoire + '/DNN' + '_'.join(str_layers_fluid) + '_tanh.pickle'
Data_fluid = sess.run([w_u,b_u,w_v,b_v,w_p,b_p])
pcklfile_fluide = open(filename_fluid,'ab+')
pickle.dump(Data_fluid,pcklfile_fluide)
pcklfile_fluide.close()
print('Model exported in '+repertoire+' (safety save done)')

if args.ExitAfterSafetySave:
    print('Clean smoke exit requested (--ExitAfterSafetySave).')
    print('Weights are safely saved; skipping all legacy post-training plotting/history code.')
    sys.stdout.flush()
    sys.stderr.flush()
    sys.exit(0)


# =============================================================================
# Print residuals errors and losses
# =============================================================================

nnf.print_bar()
print('Error details')
nnf.print_bar()

if not(multigrid):
    tf_dict = [tf_dict]

print('')
if args.SkipDiagnostics:
    print('Skipping post-training diagnostics (--SkipDiagnostics); only the tap loss:')
    nnf.tf_print('Loss mesures training',Loss_mes,sess,tf_dict[0])
else:
    nnf.tf_print('Border',loss_BC(s_tf),sess,tf_dict[0])
    nnf.tf_print('Loss eqs. modes',Loss_int_mode_wrap,sess,tf_dict[0])
    nnf.tf_print('Loss eqs. int time',Loss_int_time_wrap,sess,tf_dict[0])
    nnf.tf_print('Loss mesures training',Loss_mes,sess,tf_dict[0])
    nnf.tf_print('Loss mesures validation',Loss_dense_mes,sess,tf_dict_valid)
    if args.SparseData:
        nnf.tf_print('Loss mes pitot (component)',Loss_mes_pitot,sess,tf_dict[0])
        nnf.tf_print('Loss mes cyl (component)',Loss_mes_cyl,sess,tf_dict[0])

    if args.BVF:
        nnf.tf_print('Loss BVF (component)',Loss_bvf_wrap,sess,tf_dict[0])

    if args.K0Loss:
        nnf.tf_print('Loss k0 harmonic (component)',Loss_k0_wrap,sess,tf_dict[0])

    if args.CV1Loss:
        nnf.tf_print('Loss cv1 (component)',Loss_cv1_wrap,sess,tf_dict[0])

    if args.CausalWeighting:
        print('Causal x_front final value : %.4f (target was %.4f)' % (sess.run(x_front_var), args.CausalEndX))

if args.DesyncSparseData:

    def r_div_eucli(a,b):
        '''
        a,b real numbers
        return r with a = n*b + r, n (int) and -b/2 <= r < b/2
        '''
        rtemp = a%b
        return np.where(rtemp>0.5*b,rtemp-b,rtemp)


    print('Validation Resync')
    Delta_phi_tf_pitot_found_o = sess.run(Delta_phi_tf_pitot)
    err_rms_resync = np.sqrt(np.mean(np.square((r_div_eucli(Delta_phi_tf_pitot_found_o-Delta_phi_np_pitot_applied,2*np.pi/omega_0)))))
    err_rms_resync_normalized = err_rms_resync/np.sqrt(np.mean(np.square(Delta_phi_np_pitot_applied)))
    print('Err RMS Resynchro : %.3e'%(err_rms_resync))
    print('Err RMS Resynchro normalized : %.3e'%(err_rms_resync_normalized))

    # Plot répartition des  erreurs de resyncro
    xpitot = np.reshape(xmes_pitot,[Ntimes,Nxpitot])[0,:]
    ypitot = np.reshape(ymes_pitot,[Ntimes,Nxpitot])[0,:]
    err_resync_pitot = r_div_eucli(Delta_phi_tf_pitot_found_o-Delta_phi_np_pitot_applied,2*np.pi/omega_0)

    size_resync = np.log10(err_resync_pitot)

    plt.figure()
    plt.scatter(xpitot,ypitot,c=np.log10(err_resync_pitot),marker='o',s=1.+size_resync)
    plt.colorbar()
    plt.scatter(xmes_cyl,ymes_cyl,c='black',marker='.',s=1.)
    plt.xlabel('$x$')
    plt.ylabel('$y$')
    plt.axis('equal')
    plt.xlim((Lxmin,Lxmax))
    plt.ylim((Lymin,Lymax))
    plt.title('Synchronisation error - log')
    plt.tight_layout()
    plt.savefig(repertoire+'/resync_err.png')
    plt.close()



# =============================================================================
# Save NN Model coefficients in a pickle archive
# =============================================================================

print('Saving NN Model...')

str_layers_fluid = [str(j) for j in layers]
filename_fluid = repertoire + '/DNN' + '_'.join(str_layers_fluid) + '_tanh.pickle'

Data_fluid = sess.run([w_u,b_u,w_v,b_v,w_p,b_p])
pcklfile_fluide = open(filename_fluid,'ab+')
pickle.dump(Data_fluid,pcklfile_fluide)
pcklfile_fluide.close()
print('Model exported in '+repertoire)

# =============================================================================
# Save convergence history
# =============================================================================

print('Saving convergence history...')

filename_hist = repertoire + '/Convergence_history.pickle'

Data_loss_history = [List_it_loss_LBFGS,List_it_loss_valid_LBFGS,List_it_loss_Adam,List_it_loss_valid_Adam]
pckl_hist = open(filename_hist,'ab+')
pickle.dump(Data_loss_history,pckl_hist)
pckl_hist.close()
print('History exported in '+repertoire)

plt.figure()
plt.scatter(np.array(List_it_loss_LBFGS)[:,0],np.array(List_it_loss_LBFGS)[:,1],label='LBFGS train',marker='.',s=1.,c='red')
# plt.scatter(np.array(List_it_loss_valid_LBFGS)[:,0],np.array(List_it_loss_valid_LBFGS)[:,1],label='LBFGS valid',marker='.',s=1.,c='pink')
# Validation loss does not seem to be accessible during L-BFGS-B training. It returns constant values
plt.scatter(np.array(List_it_loss_Adam)[:,0]+np.max(np.array(List_it_loss_LBFGS)[:,0]),np.array(List_it_loss_Adam)[:,1],label='Adam train',marker='.',s=1.,c='blue')
plt.scatter(np.array(List_it_loss_valid_Adam)[:,0]+np.max(np.array(List_it_loss_LBFGS)[:,0]),np.array(List_it_loss_valid_Adam)[:,1],label='Adam valid',marker='.',s=1.,c='green')
plt.xlabel('Iterations')
plt.ylabel('Error')
plt.yscale('log')
plt.legend()
plt.tight_layout()
plt.savefig(repertoire+'/Convergence_history.png')
plt.close()



# =============================================================================
# Plot of modal shapes
# =============================================================================


for k in range(Nmodes):
    nnf.tf_plot_scatter_complex(x_tf_int[:,0],y_tf_int[:,0],fluid_u(x_tf_int,y_tf_int)[0,:,k],
                        sess,
                        title='u Mode '+str(k),
                        xlabel='$x$',ylabel='$y$',
                        tf_dict=tf_dict_valid)
    plt.savefig(repertoire+'/u_mode_'+str(k)+'.png')
    plt.close()



for k in range(Nmodes):
    nnf.tf_plot_scatter_complex(x_tf_int[:,0],y_tf_int[:,0],fluid_v(x_tf_int,y_tf_int)[0,:,k],
                        sess,
                        title='v Mode '+str(k),
                        xlabel='$x$',ylabel='$y$',
                        tf_dict=tf_dict_valid)
    plt.savefig(repertoire+'/v_mode_'+str(k)+'.png')
    plt.close()

for k in range(Nmodes):
    nnf.tf_plot_scatter_complex(x_tf_int[:,0],y_tf_int[:,0],fluid_p(x_tf_int,y_tf_int)[0,:,k],
                        sess,
                        title='p Mode '+str(k),
                        xlabel='$x$',ylabel='$y$',
                        tf_dict=tf_dict_valid)
    plt.savefig(repertoire+'/p_mode_'+str(k)+'.png')
    plt.close()


# =============================================================================
# Comparison at a given timestep between modalPINN and simulations data
# =============================================================================
inst = 16

Re, Ur, times, nodes_X, nodes_Y, Us, Vs, Ps = ltd.read_cut_simulation_data(filename_data,geom)

tf_dict_compare = {
    x_tf_mes : np.reshape(nodes_X[0,:],(len(nodes_X[0,:]),1)),
    y_tf_mes : np.reshape(nodes_Y[0,:],(len(nodes_Y[0,:]),1)),
    t_tf_mes : np.reshape(times[inst]*np.ones(len(nodes_X[0,:])),(len(nodes_Y[0,:]),1)),
    u_tf_mes : np.reshape(Us[inst,:],(len(nodes_X[0,:]),1))
    }

suptitle='u difference at t = '+'{0:.2f}'.format(times[inst])

nnf.tf_plot_compare_3plot(x_tf_mes,y_tf_mes,u_tf_mes,fluid_u_t(x_tf_mes,y_tf_mes,t_tf_mes),sess,xlabel='$x$',ylabel='$y$',title1='Exact',title2='ModalPINN',suptitle='',tf_dict=tf_dict_compare)
plt.savefig(repertoire+'/diff_u_t_'+'{0:.2f}'.format(times[inst])+'.png')
