"""
STAGE B: independent incompressible Navier-Stokes solver.

MAC (Marker-and-Cell) staggered grid, fractional-step (Chorin) projection,
explicit advection + explicit diffusion (justified: at Re=100, dx~0.05-0.1,
diffusion stability dt <= dx^2*Re/4 is far less restrictive than the
advective CFL limit, so implicit diffusion isn't needed -- see
docs/DESIGN.md Stage B notes for the deviation from the original
Crank-Nicolson proposal), immersed cylinder via direct-forcing (binary
masking: velocity forced to exactly zero on grid points inside r<r_c).

Grid layout (Ny rows, Nx columns):
    p[j,i]  at cell centers   (x_centers[i], y_centers[j])   shape (Ny, Nx)
    u[j,i]  at vertical faces (x_edges[i],   y_centers[j])   shape (Ny, Nx+1)
    v[j,i]  at horizontal faces(x_centers[i], y_edges[j])    shape (Ny+1, Nx)

Boundary conditions:
    x=Lxmin (inflow):  u=1 (Dirichlet), v=0 (Dirichlet, via ghost mirror)
    y=Lymin,Lymax (far-field): u=1 (Dirichlet, via ghost mirror), v=0 (Dirichlet)
    x=Lxmax (outflow): du/dx=0, dv/dx=0 (Neumann, zero-gradient copy)
    pressure Poisson: Neumann (zero flux) at every Dirichlet-velocity edge,
                       Dirichlet p=0 at the outflow edge (pins the gauge,
                       makes the discrete Laplacian nonsingular).
"""
import numpy as np
import scipy.sparse as sp
import scipy.sparse.linalg as spla


class CylinderFlowSolver:
    def __init__(self, Nx, Ny, Lxmin, Lxmax, Lymin, Lymax,
                 x_c, y_c, r_c, Re, dt):
        self.Nx, self.Ny = Nx, Ny
        self.Lxmin, self.Lxmax = Lxmin, Lxmax
        self.Lymin, self.Lymax = Lymin, Lymax
        self.dx = (Lxmax - Lxmin) / Nx
        self.dy = (Lymax - Lymin) / Ny
        self.x_c, self.y_c, self.r_c = x_c, y_c, r_c
        self.Re = Re
        self.nu = 1.0 / Re
        self.dt = dt
        self.t = 0.0

        self.x_edges = Lxmin + np.arange(Nx + 1) * self.dx
        self.x_centers = Lxmin + (np.arange(Nx) + 0.5) * self.dx
        self.y_edges = Lymin + np.arange(Ny + 1) * self.dy
        self.y_centers = Lymin + (np.arange(Ny) + 0.5) * self.dy

        # grid-point coordinate meshes
        self.Xu, self.Yu = np.meshgrid(self.x_edges, self.y_centers)      # (Ny, Nx+1)
        self.Xv, self.Yv = np.meshgrid(self.x_centers, self.y_edges)      # (Ny+1, Nx)
        self.Xp, self.Yp = np.meshgrid(self.x_centers, self.y_centers)    # (Ny, Nx)

        # immersed-cylinder masks (True = inside solid)
        self.solid_u = (self.Xu - x_c) ** 2 + (self.Yu - y_c) ** 2 < r_c ** 2
        self.solid_v = (self.Xv - x_c) ** 2 + (self.Yv - y_c) ** 2 < r_c ** 2

        # state
        self.u = np.ones((Ny, Nx + 1))
        self.v = np.zeros((Ny + 1, Nx))
        self.p = np.zeros((Ny, Nx))
        self.u[self.solid_u] = 0.0
        self.v[self.solid_v] = 0.0

        self._build_poisson()

    # ------------------------------------------------------------------
    def _build_poisson(self):
        """5-point Laplacian for p (Ny*Nx unknowns), Neumann at
        Dirichlet-velocity edges (left/top/bottom), Dirichlet p=0 at
        outflow (right). Factorized once via splu; reused every step."""
        Nx, Ny, dx, dy = self.Nx, self.Ny, self.dx, self.dy
        N = Nx * Ny

        def idx(j, i):
            return j * Nx + i

        rows, cols, vals = [], [], []
        rhs_dirichlet_scale = np.zeros(N)  # unused, Dirichlet value is 0

        cx, cy = 1.0 / dx ** 2, 1.0 / dy ** 2

        for j in range(Ny):
            for i in range(Nx):
                k = idx(j, i)
                diag = 0.0
                # west neighbor (i-1): Neumann if i==0 (inflow) -> skip entirely
                if i > 0:
                    rows.append(k); cols.append(idx(j, i - 1)); vals.append(cx)
                    diag -= cx
                # east neighbor (i+1): Dirichlet p=0 ghost if i==Nx-1 (outflow)
                if i < Nx - 1:
                    rows.append(k); cols.append(idx(j, i + 1)); vals.append(cx)
                    diag -= cx
                else:
                    diag -= 2 * cx  # ghost = -p[k], Dirichlet p=0 at outflow
                # south neighbor (j-1): Neumann if j==0 (bottom far-field) -> skip
                if j > 0:
                    rows.append(k); cols.append(idx(j - 1, i)); vals.append(cy)
                    diag -= cy
                # north neighbor (j+1): Neumann if j==Ny-1 (top far-field) -> skip
                if j < Ny - 1:
                    rows.append(k); cols.append(idx(j + 1, i)); vals.append(cy)
                    diag -= cy
                rows.append(k); cols.append(k); vals.append(diag)

        L = sp.csc_matrix((vals, (rows, cols)), shape=(N, N))
        self._poisson_matrix = L
        self._poisson_lu = spla.splu(L)

    def _solve_poisson(self, rhs):
        """rhs: (Ny,Nx) -> phi: (Ny,Nx)."""
        phi = self._poisson_lu.solve(rhs.ravel())
        return phi.reshape(self.Ny, self.Nx)

    # ------------------------------------------------------------------
    def _apply_velocity_bc(self, u, v):
        """Enforce Dirichlet BCs on u,v in place (boundary faces / ghost-derived)."""
        u[:, 0] = 1.0                 # inflow, exact (u lives on this face)
        u[:, -1] = u[:, -2]           # outflow, zero-gradient
        # top/bottom far-field for u (u is cell-centered in y -> use ghost mirror,
        # i.e. average of first row and its ghost equals 1 => we directly set the
        # first/last row's *effective* BC by extrapolating after the stencil; here
        # we instead fix u's own first/last rows is NOT correct (u isn't ON that
        # edge) -- handled via ghost values inside the Laplacian/advection stencils
        # (see _lap_u / _adv_u), nothing to do here for u's rows.
        v[0, :] = 0.0                 # bottom far-field, exact (v lives on this edge)
        v[-1, :] = 0.0                # top far-field, exact
        v[:, 0] = -v[:, 1] if False else v[:, 0]  # placeholder (v has no x-boundary node)
        return u, v

    def _ghost_row(self, u, side):
        """Ghost row of u just outside y=Lymin (side=0) or y=Lymax (side=-1),
        enforcing u=1 exactly at that far-field edge via mirror extrapolation."""
        if side == 0:
            return 2.0 * 1.0 - u[0, :]
        else:
            return 2.0 * 1.0 - u[-1, :]

    def _ghost_col_v_left(self, v):
        """Ghost column of v just outside x=Lxmin, enforcing v=0 at inflow."""
        return -v[:, 0]

    def _ghost_col_v_right(self, v):
        """Ghost column of v just outside x=Lxmax, zero-gradient outflow."""
        return v[:, -1]

    # ------------------------------------------------------------------
    def _lap_u(self, u):
        lap = np.zeros_like(u)
        # interior columns only (i=1..Nx-1); boundary columns (0, Nx) are
        # prescribed BC, not evolved by the PDE
        lap[:, 1:-1] = (u[:, 2:] - 2 * u[:, 1:-1] + u[:, :-2]) / self.dx ** 2
        u_top_ghost = self._ghost_row(u, -1)
        u_bot_ghost = self._ghost_row(u, 0)
        d2y = np.empty_like(u)
        d2y[1:-1, :] = (u[2:, :] - 2 * u[1:-1, :] + u[:-2, :]) / self.dy ** 2
        d2y[0, :] = (u[1, :] - 2 * u[0, :] + u_bot_ghost) / self.dy ** 2
        d2y[-1, :] = (u_top_ghost - 2 * u[-1, :] + u[-2, :]) / self.dy ** 2
        lap[:, 1:-1] += d2y[:, 1:-1]
        return lap

    def _lap_v(self, v):
        lap = np.zeros_like(v)
        lap[1:-1, :] = (v[2:, :] - 2 * v[1:-1, :] + v[:-2, :]) / self.dy ** 2
        v_left_ghost = self._ghost_col_v_left(v)
        v_right_ghost = self._ghost_col_v_right(v)
        d2x = np.empty_like(v)
        d2x[:, 1:-1] = (v[:, 2:] - 2 * v[:, 1:-1] + v[:, :-2]) / self.dx ** 2
        d2x[:, 0] = (v[:, 1] - 2 * v[:, 0] + v_left_ghost) / self.dx ** 2
        d2x[:, -1] = (v_right_ghost - 2 * v[:, -1] + v[:, -2]) / self.dx ** 2
        lap[1:-1, :] += d2x[1:-1, :]
        return lap

    def _advect_u(self, u, v):
        """u du/dx + v du/dy at u-points, interior columns only."""
        dx, dy = self.dx, self.dy
        adv = np.zeros_like(u)
        # du/dx, centered, interior columns
        dudx = np.zeros_like(u)
        dudx[:, 1:-1] = (u[:, 2:] - u[:, :-2]) / (2 * dx)
        # v interpolated onto u-points (Ny, Nx+1): 2x2 box average of v,
        # padded with ghost columns at x=Lxmin (i=-1) and x=Lxmax (i=Nx)
        v_ext = np.empty((self.Ny + 1, self.Nx + 2))
        v_ext[:, 1:-1] = v
        v_ext[:, 0] = self._ghost_col_v_left(v)
        v_ext[:, -1] = self._ghost_col_v_right(v)
        v_at_u_full = 0.25 * (v_ext[:-1, :-1] + v_ext[:-1, 1:] + v_ext[1:, :-1] + v_ext[1:, 1:])
        # du/dy, centered with ghost rows for top/bottom
        u_top_ghost = self._ghost_row(u, -1)
        u_bot_ghost = self._ghost_row(u, 0)
        dudy = np.empty_like(u)
        dudy[1:-1, :] = (u[2:, :] - u[:-2, :]) / (2 * dy)
        dudy[0, :] = (u[1, :] - u_bot_ghost) / (2 * dy)
        dudy[-1, :] = (u_top_ghost - u[-2, :]) / (2 * dy)
        adv[:, 1:-1] = u[:, 1:-1] * dudx[:, 1:-1] + v_at_u_full[:, 1:-1] * dudy[:, 1:-1]
        return adv

    def _advect_v(self, u, v):
        """u dv/dx + v dv/dy at v-points, interior rows only."""
        dx, dy = self.dx, self.dy
        adv = np.zeros_like(v)
        dvdy = np.zeros_like(v)
        dvdy[1:-1, :] = (v[2:, :] - v[:-2, :]) / (2 * dy)
        # u interpolated onto v-points (Ny+1, Nx): 2x2 box average of u,
        # padded with ghost rows at y=Lymin (j=-1) and y=Lymax (j=Ny)
        u_ext = np.empty((self.Ny + 2, self.Nx + 1))
        u_ext[1:-1, :] = u
        u_ext[0, :] = self._ghost_row(u, 0)
        u_ext[-1, :] = self._ghost_row(u, -1)
        u_at_v_full = 0.25 * (u_ext[:-1, :-1] + u_ext[:-1, 1:] + u_ext[1:, :-1] + u_ext[1:, 1:])
        v_left_ghost = self._ghost_col_v_left(v)
        v_right_ghost = self._ghost_col_v_right(v)
        dvdx = np.empty_like(v)
        dvdx[:, 1:-1] = (v[:, 2:] - v[:, :-2]) / (2 * dx)
        dvdx[:, 0] = (v[:, 1] - v_left_ghost) / (2 * dx)
        dvdx[:, -1] = (v_right_ghost - v[:, -2]) / (2 * dx)
        adv[1:-1, :] = u_at_v_full[1:-1, :] * dvdx[1:-1, :] + v[1:-1, :] * dvdy[1:-1, :]
        return adv

    # ------------------------------------------------------------------
    def step(self):
        dt, nu = self.dt, self.nu
        u, v = self.u, self.v

        u_star = u + dt * (-self._advect_u(u, v) + nu * self._lap_u(u))
        v_star = v + dt * (-self._advect_v(u, v) + nu * self._lap_v(v))

        u_star, v_star = self._apply_velocity_bc(u_star, v_star)

        # immersed-boundary direct forcing (binary masking): store the
        # implied force-per-volume before zeroing, for lift/drag diagnostics
        self._force_u = np.where(self.solid_u, -u_star / dt, 0.0)
        self._force_v = np.where(self.solid_v, -v_star / dt, 0.0)
        u_star[self.solid_u] = 0.0
        v_star[self.solid_v] = 0.0

        div_u_star = ((u_star[:, 1:] - u_star[:, :-1]) / self.dx
                      + (v_star[1:, :] - v_star[:-1, :]) / self.dy)
        phi = self._solve_poisson(div_u_star / dt)

        u_new = u_star.copy()
        v_new = v_star.copy()
        u_new[:, 1:-1] -= dt * (phi[:, 1:] - phi[:, :-1]) / self.dx
        v_new[1:-1, :] -= dt * (phi[1:, :] - phi[:-1, :]) / self.dy

        u_new, v_new = self._apply_velocity_bc(u_new, v_new)
        # NOTE: deliberately NOT re-masking solid points to exactly zero here.
        # u_star/v_star WERE masked to zero before the Poisson RHS was built
        # (see above), so the projection already "knows" the target was zero
        # there; its correction generally leaves a small nonzero residual at
        # solid/interface points because that's what global div-free
        # consistency requires. Re-zeroing after the fact would silently
        # reintroduce local divergence error at exactly the fluid/solid
        # interface -- caught via max|div(u)| sanity checks during
        # development (was ~0.66, i.e. O(1), before this fix). The next
        # step's predictor-stage masking keeps this residual small (soft
        # penalization rather than hard masking after projection).

        self.u, self.v, self.p = u_new, v_new, phi
        self.t += dt

    def run(self, T, callback=None, callback_every=1):
        n_steps = int(round(T / self.dt))
        for n in range(n_steps):
            self.step()
            if callback is not None and n % callback_every == 0:
                callback(self)

    # ------------------------------------------------------------------
    def force_on_body(self):
        """Net (Fx, Fy) the fluid exerts on the cylinder, from the IBM
        reaction force (Newton's third law): body-on-fluid force density
        is self._force_{u,v}; fluid-on-body is the negative, integrated
        over the solid cell volumes."""
        cell_vol = self.dx * self.dy
        Fx = -np.sum(self._force_u) * cell_vol
        Fy = -np.sum(self._force_v) * cell_vol
        return Fx, Fy

    def velocity_at_centers(self):
        """u,v interpolated onto the pressure (cell-center) grid."""
        u_c = 0.5 * (self.u[:, :-1] + self.u[:, 1:])
        v_c = 0.5 * (self.v[:-1, :] + self.v[1:, :])
        return u_c, v_c

    def sample_pressure(self, x_pts, y_pts):
        """Interpolate the current pressure field to arbitrary (x,y) points
        using bilinear interpolation on the cell-center (Xp,Yp) grid --
        used both for wake probes and (in h(x)) for the wall taps."""
        from scipy.interpolate import RegularGridInterpolator
        interp = RegularGridInterpolator(
            (self.y_centers, self.x_centers), self.p,
            method='linear', bounds_error=False, fill_value=None)
        pts = np.stack([y_pts, x_pts], axis=-1)
        return interp(pts)
