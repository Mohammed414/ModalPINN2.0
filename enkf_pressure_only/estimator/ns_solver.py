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

    # ------------------------------------------------------------------
    # Pressure sampling / observation operator h(x)
    # ------------------------------------------------------------------
    def _bilinear_plan(self, x_pts, y_pts):
        """Bilinear-interpolation plan for a cell-centred field at points
        (x_pts, y_pts).

        Returns (jj, ii, ww), each shape (n_pts, 4): the row/column indices
        of the four stencil cells and their weights, so that

            f(x_k, y_k) = sum_m ww[k,m] * F[jj[k,m], ii[k,m]]

        Base cell indices are clamped to [0, N-2] but the local coordinates
        are NOT clamped, so points just outside the cell-centre grid are
        linearly extrapolated -- identical behaviour to the previous
        RegularGridInterpolator(..., bounds_error=False, fill_value=None).
        """
        x_pts = np.atleast_1d(np.asarray(x_pts, dtype=float))
        y_pts = np.atleast_1d(np.asarray(y_pts, dtype=float))
        fx = (x_pts - self.x_centers[0]) / self.dx
        fy = (y_pts - self.y_centers[0]) / self.dy
        i0 = np.clip(np.floor(fx).astype(int), 0, self.Nx - 2)
        j0 = np.clip(np.floor(fy).astype(int), 0, self.Ny - 2)
        tx = fx - i0
        ty = fy - j0
        ii = np.stack([i0, i0 + 1, i0, i0 + 1], axis=1)
        jj = np.stack([j0, j0, j0 + 1, j0 + 1], axis=1)
        ww = np.stack([(1 - tx) * (1 - ty), tx * (1 - ty),
                       (1 - tx) * ty, tx * ty], axis=1)
        return jj, ii, ww

    def _stencil_solid_count(self, jj, ii):
        """How many of each point's 4 stencil cell-centres lie inside the
        cylinder (r < r_c). Shape (n_pts,)."""
        rr = np.hypot(self.x_centers[ii] - self.x_c, self.y_centers[jj] - self.y_c)
        return np.sum(rr < self.r_c, axis=1)

    def wall_probe_plan(self, x_pts, y_pts, probe_offsets=(1.5, 2.5, 3.5),
                        extrap_order=2, max_offset=6.0, offset_increment=0.25):
        """Build (and validate) the normal-direction wall-probe geometry for
        surface taps -- docs/DESIGN.md Sec 3.

        For each tap, the local outward unit normal is n = (x-x_c, y-y_c)/r.
        Pressure is sampled at ``len(probe_offsets)`` points stepping OUTWARD
        along that normal, at wall-normal distances d_m = probe_offsets[m]*h
        (h = min(dx,dy)), and then extrapolated back to the wall (d=0) by a
        least-squares polynomial of degree ``extrap_order`` in d; the wall
        value is that polynomial's constant term.

        extrap_order=2 (quadratic, the default, needs >= 3 offsets) was
        selected on three controlled tests in which the true wall pressure is
        known, NOT on the innovation against the tap data (which is dominated
        by a separate model bias and so cannot discriminate sensor models):
          (i)  analytic potential flow past a cylinder, solid cells zeroed to
               emulate the IBM artefact -- wall err rms 0.153 (quadratic) vs
               0.246 (linear) vs 0.561 (plain bilinear);
          (ii) a synthetic field with the same surface Cp but zero wall-normal
               pressure gradient at r=r_c (viscous-like) -- 0.072 vs 0.111 vs
               0.493;
          (iii) self-consistency on 11 real solver snapshots across the limit
               cycle: predict the (fully fluid) d=1.5h station from stations
               further out -- 0.0105 vs 0.0158 vs 0.0605 (constant).
        extrap_order=1 with probe_offsets=(1.5, 2.5) reproduces the plain
        linear two-point extrapolation and remains available.

        Why an offset of 1.5*h and not 1.0*h: a query point sits somewhere
        inside a dual cell of size dx-by-dy, so its farthest bilinear stencil
        corner can be up to sqrt(dx^2+dy^2) = 1.414*h away and therefore up to
        1.414*h closer to the cylinder centre in the worst case. 1.5*h is the
        smallest round multiple that clears that bound for every tap angle;
        it is nevertheless CHECKED per point, not assumed -- any probe point
        whose 4-cell bilinear stencil still touches a cell with r < r_c is
        pushed further out in steps of ``offset_increment``*h until its whole
        stencil is fluid (or ``max_offset`` is exceeded, which raises).

        Returns a dict with the per-point plans and diagnostics:
            jj, ii, ww      (n_pts, n_probe, 4) stencil indices/weights
            offsets_used    (n_pts, n_probe) offsets in units of h
            dists           (n_pts, n_probe) wall-normal distances
            extrap_w        (n_pts, n_probe) weights s.t. p_wall = sum_m w_m p_m
            solid_touch     (n_pts, n_probe) solid cells per probe stencil
                            (must be all-zero for a valid plan)
            n_escalated     how many probe points needed pushing outward
        """
        x_pts = np.atleast_1d(np.asarray(x_pts, dtype=float))
        y_pts = np.atleast_1d(np.asarray(y_pts, dtype=float))
        n_pts = x_pts.size
        n_probe = len(probe_offsets)
        h = min(self.dx, self.dy)

        rad = np.hypot(x_pts - self.x_c, y_pts - self.y_c)
        if np.any(rad < 1e-12):
            raise ValueError('wall probe undefined for a point at the cylinder centre')
        nx = (x_pts - self.x_c) / rad
        ny = (y_pts - self.y_c) / rad

        offsets_used = np.tile(np.asarray(probe_offsets, dtype=float), (n_pts, 1))
        jj = np.empty((n_pts, n_probe, 4), dtype=int)
        ii = np.empty((n_pts, n_probe, 4), dtype=int)
        ww = np.empty((n_pts, n_probe, 4), dtype=float)
        solid = np.empty((n_pts, n_probe), dtype=int)
        n_escalated = 0

        for m in range(n_probe):
            for _ in range(int(np.ceil((max_offset - min(probe_offsets)) / offset_increment)) + 2):
                # sample point: on the ray from the centre, at radius r_c + d
                d = offsets_used[:, m] * h
                px = self.x_c + nx * (self.r_c + d)
                py = self.y_c + ny * (self.r_c + d)
                Jm, Im, Wm = self._bilinear_plan(px, py)
                Sm = self._stencil_solid_count(Jm, Im)
                bad = Sm > 0
                if not np.any(bad):
                    break
                offsets_used[bad, m] += offset_increment
                n_escalated += int(np.sum(bad))
                if np.any(offsets_used[:, m] > max_offset):
                    raise RuntimeError(
                        'wall probe could not find a fully-fluid stencil within '
                        '%.2f*h of the wall -- grid too coarse for r_c=%.3f'
                        % (max_offset, self.r_c))
            jj[:, m], ii[:, m], ww[:, m] = Jm, Im, Wm
            solid[:, m] = Sm

        dists = offsets_used * h

        # Least-squares polynomial fit p(d) = a_0 + a_1 d + ... + a_k d^k over
        # the probe points; the wall value is a_0. Expressed as a fixed linear
        # functional of the probe samples (extrap_w) so that h(x) stays a cheap
        # matrix-vector product and its linearity in the state is explicit --
        # which matters because the EnKF's Y anomalies assume exactly that.
        if n_probe < extrap_order + 1:
            raise ValueError('extrap_order=%d needs >= %d probe offsets (got %d)'
                             % (extrap_order, extrap_order + 1, n_probe))
        extrap_w = np.empty((n_pts, n_probe))
        for k in range(n_pts):
            V = np.vander(dists[k], extrap_order + 1, increasing=True)  # (n_probe, k+1)
            # row 0 of the pseudo-inverse maps samples -> constant term a_0
            extrap_w[k] = np.linalg.pinv(V)[0]

        return dict(jj=jj, ii=ii, ww=ww, offsets_used=offsets_used, dists=dists,
                    extrap_w=extrap_w, solid_touch=solid, n_escalated=n_escalated,
                    extrap_order=extrap_order, normal_x=nx, normal_y=ny, radius=rad)

    def _cached_wall_plan(self, x_pts, y_pts, probe_offsets, extrap_order):
        key = (np.asarray(x_pts, dtype=float).tobytes(),
               np.asarray(y_pts, dtype=float).tobytes(),
               tuple(float(o) for o in probe_offsets), int(extrap_order))
        cache = getattr(self, '_wall_plan_cache', None)
        if cache is None:
            cache = self._wall_plan_cache = {}
        if key not in cache:
            cache[key] = self.wall_probe_plan(x_pts, y_pts, probe_offsets=probe_offsets,
                                              extrap_order=extrap_order)
        return cache[key]

    def sample_pressure(self, x_pts, y_pts, method='wall_probe',
                        probe_offsets=(1.5, 2.5, 3.5), extrap_order=2,
                        return_plan=False):
        """Sample the current pressure field at points (x,y). This is h(x).

        method='wall_probe' (default) -- docs/DESIGN.md Sec 3. For points on
            (or very near) the cylinder surface, pressure is extrapolated to
            r = r_c along the local outward normal from probe points whose
            interpolation stencils are entirely in fluid cells. Points that
            are already comfortably in the fluid (r > r_c + 1.5*h) are
            bilinearly interpolated as before -- their stencils are checked
            and are solid-free by construction at that distance.

        method='bilinear' -- the original behaviour: plain bilinear
            interpolation of p at (x,y) on the cell-centre grid. For wall
            taps this reads cells INSIDE the cylinder, where the pressure of
            a binary-masking direct-forcing IBM has no physical meaning.
            Retained only so the two can be compared directly.

        Returns an array of shape (n_pts,).
        """
        x_pts = np.atleast_1d(np.asarray(x_pts, dtype=float))
        y_pts = np.atleast_1d(np.asarray(y_pts, dtype=float))

        if method == 'bilinear':
            jj, ii, ww = self._bilinear_plan(x_pts, y_pts)
            out = np.sum(ww * self.p[jj, ii], axis=1)
            return (out, dict(jj=jj, ii=ii, ww=ww)) if return_plan else out

        if method != 'wall_probe':
            raise ValueError("method must be 'wall_probe' or 'bilinear' (got %r)" % method)

        h = min(self.dx, self.dy)
        rad = np.hypot(x_pts - self.x_c, y_pts - self.y_c)
        near_wall = rad < self.r_c + max(probe_offsets) * h

        out = np.empty(x_pts.size)
        plan = {}
        if np.any(near_wall):
            wp = self._cached_wall_plan(x_pts[near_wall], y_pts[near_wall],
                                        probe_offsets, extrap_order)
            p_probe = np.sum(wp['ww'] * self.p[wp['jj'], wp['ii']], axis=2)  # (n_near, n_probe)
            out[near_wall] = np.sum(wp['extrap_w'] * p_probe, axis=1)
            plan['wall'] = wp
        if np.any(~near_wall):
            jj, ii, ww = self._bilinear_plan(x_pts[~near_wall], y_pts[~near_wall])
            out[~near_wall] = np.sum(ww * self.p[jj, ii], axis=1)
            plan['far'] = dict(jj=jj, ii=ii, ww=ww)
        return (out, plan) if return_plan else out
