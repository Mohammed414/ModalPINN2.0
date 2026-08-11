"""
Flatten/unflatten a CylinderFlowSolver's velocity field to/from the
assimilated state vector x = [u, v] at INTERIOR ACTIVE fluid DOFs only
(see docs/DESIGN.md Sec 1 & 9): excludes Dirichlet-boundary faces/rows
(inflow, top/bottom far-field; outflow is a copy, not an independent DOF)
and excludes points inside the immersed cylinder.

Excluding these from x is what makes "the analysis stays physically
valid" structural rather than a post-hoc patch: boundary DOFs are never
touched by the EnKF at all (they're re-imposed by the solver's own BC
application every step regardless), and divergence-free-ness is exact
because div is linear and every forecast member is already divergence-free
on exactly this same active set (see Sec 9's proof sketch).
"""
import numpy as np


def active_masks(solver):
    Nx, Ny = solver.Nx, solver.Ny
    active_u = np.zeros_like(solver.solid_u, dtype=bool)
    active_u[:, 1:Nx] = True          # interior u-faces only (exclude inflow i=0, outflow i=Nx)
    active_u &= ~solver.solid_u

    active_v = np.zeros_like(solver.solid_v, dtype=bool)
    active_v[1:Ny, :] = True          # interior v-faces only (exclude top/bottom j=0,Ny)
    active_v &= ~solver.solid_v
    return active_u, active_v


class StateVectorizer:
    """Built once per grid configuration; reused for every member/step."""

    def __init__(self, solver):
        self.active_u, self.active_v = active_masks(solver)
        self.n_u = int(self.active_u.sum())
        self.n_v = int(self.active_v.sum())
        self.n_state = self.n_u + self.n_v

    def flatten(self, solver):
        return np.concatenate([solver.u[self.active_u], solver.v[self.active_v]])

    def unflatten_into(self, solver, x):
        solver.u[self.active_u] = x[:self.n_u]
        solver.v[self.active_v] = x[self.n_u:]
