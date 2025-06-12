import numpy as np
import osqp
from math import sin, cos, tan, pi  # Added tan, pi
from scipy import sparse
from scipy.linalg import block_diag
from base_controller import BaseController


class MPCController(BaseController):  # Inherit from BaseController

    def __init__(
        self,
        model,  # Added model argument
        path_x=None,
        path_y=None,
        path_theta=None,
        ref_v=None,  # For full path
        dt=0.1,
        Q=None,
        R=None,
        horizon=10,
        control_horizon_m=None,  # Explicit control horizon
        use_differential=False,
        q_diag=None,  # Simpler way to pass Q diagonals
        r_diag=None,  # Simpler way to pass R diagonals
        **kwargs):  # Catch-all for other base kwargs or future params

        # Call BaseController's __init__
        # It will store model parameters like self.L, self.b, self.r (if model has them)
        # and the initial full path if provided.
        super().__init__(model,
                         path_x=path_x,
                         path_y=path_y,
                         path_theta=path_theta,
                         **kwargs)

        self.control_name = "mpc"
        self.ts = dt  # MPC sample time
        self.N = horizon  # Prediction horizon

        # Control horizon M: usually M <= N. If None, set M = N.
        self.M = control_horizon_m if control_horizon_m is not None and control_horizon_m <= self.N else self.N
        self.ref_v = ref_v
        self.use_differential = use_differential
        # Vehicle constants L and b are now inherited from BaseController (self.L, self.b)
        # Ensure your 'model' object passed to BaseController has 'L' and 'b' attributes.
        # self.L = model.L # Already done by super()
        # self.b = model.b # Already done by super()
        self.epsilon = 0.000001
        # Dimensions for slip constraint if M constraints (one per step in control horizon)
        # self.slip_const is 1x3, so A_cons for M steps will be M x (M*nu)
        self.l_cons_slip = -self.epsilon * np.ones(
            (self.M, 1)) if self.use_differential else None
        self.u_cons_slip = self.epsilon * np.ones(
            (self.M, 1)) if self.use_differential else None

        # States: x, y, theta,delta, v_vehicle_avg (vehicle speed)
        self.nx = 5
        # Control inputs: delta_vl, delta_vr, delta_steer (changes in wheel speeds and steering)
        # OR absolute: vl, vr, steer. Current MPC is formulated for delta_u.
        # Inputs u = [v_left_wheel, v_right_wheel, steering_angle_delta]
        self.nu = 3
        self.ny = 5  #[x, y, theta. v] Output dimension (typically same as state for MPC tracking)

        # self.x_aug = np.zeros((self.nx + self.nu, 1)) # Old augmented state, not directly used in this structure

        # Linearized model matrices (updated at each step)
        # A is simplified to identity in mdl_update, implying x_k+1 = x_k + B*du_k + f(x_k, u_k-1)*dt
        # For a pure delta_u formulation like x_k+1 = A x_k + B delta_u_k, A should be I here.
        self.A_mpc = np.eye(
            self.nx)  # Renamed to avoid conflict if base had self.A
        self.B_mpc = np.zeros((self.nx, self.nu))
        self.B_mpc[3, 2] = 1
        self.B_mpc[4, 0] = 0.5
        self.B_mpc[4, 1] = 0.5

        # self.C_mpc = np.zeros((self.ny, self.nx))
        self.C_mpc = np.eye((self.ny))
        # self.C_mpc[0:3, 0:3] = np.eye(self.ny - 1)
        # self.C_mpc[3, 4] = 1

        #self.slip_const_matrix_row = np.zeros((1, 3))    # Stores the actual slip constraint coefficients for one step

        # Reference trajectory for the horizon (N steps)
        self.ref_horizon = np.zeros((self.N * self.ny, 1))

        # Cost matrices
        # Q for states [x, y, theta, delta, v]
        q_diag_default = [1.0, 1.0, 1.0, 1.0, 1.0]  # Default Q diagonals
        actual_q_diag = q_diag if q_diag is not None and len(
            q_diag) == self.ny else q_diag_default

        # R for control inputs [v_left, v_right, steering_angle_delta]
        r_diag_default = [1.0, 1.0, 1.0]  # Default R diagonals
        actual_r_diag = r_diag if r_diag is not None and len(
            r_diag) == self.nu else r_diag_default

        self.Q_mpc = np.zeros((self.ny * self.N, self.ny * self.N))
        for indx in range(self.N):
            self.Q_mpc[indx * self.ny:indx * self.ny + self.ny,
                       indx * self.ny:indx * self.ny +
                       self.ny] = np.diag(actual_q_diag) / (1.0**indx)

        self.R_mpc = np.zeros((self.nu * self.M, self.nu * self.M))
        for indx in range(self.M):
            self.R_mpc[indx * self.nu:indx * self.nu + self.nu,
                       indx * self.nu:indx * self.nu +
                       self.nu] = np.diag(actual_r_diag) / (1.0**indx)

        # self.Q_mpc = sparse.block_diag([q_single] * self.N).tocsc()
        # self.R_mpc = sparse.block_diag(
        #     [r_single] * self.M).tocsc()  # Use self.M for control horizon

        self.du = np.zeros((self.nu, 1))  # Change in control input
        self.u_prev = np.zeros(
            (self.nu, 1))  # Previous control input [vl, vr, delta_steer]

        # Absolute constraints on control inputs (u = [vl, vr, delta_steer])
        # Note: These are not directly used in the OSQP formulation for du yet.
        # To use them, bounds on du would be: u_min - u_prev <= du <= u_max - u_prev
        self.u_min_abs = np.array([0.0, 0.0, -np.deg2rad(30)
                                   ])  # Example: vl, vr >=0, delta +/-30 deg
        self.u_max_abs = np.array([1.5, 1.5, np.deg2rad(30)
                                   ])  # Example: vl, vr <=1.5 m/s

        self.osqp_prob = osqp.OSQP()
        print(self.osqp_prob.version())
        self.osqp_solver_initialized = False

    def _update_horizon_reference(self):
        """
        Updates self.ref_horizon based on the current vehicle state (self.current_x, self.current_y)
        and the stored full path (self._path_x, self._path_y, self._path_theta, self._path_v).
        This replicates the logic from the old simulator's update_mpc_reference.
        """
        if self._path_x.size == 0:
            # print("[MPC] Warning: No reference path set. Using current state as reference.")
            current_state_ref = np.array([
                self.current_x, self.current_y, self.current_theta,
                self.current_v
            ])
            self.ref_horizon = np.tile(current_state_ref,
                                       self.N).reshape(-1, 1)
            return

        # Find the closest point on the path to the current vehicle position
        dists = np.hypot(self._path_x - self.current_x,
                         self._path_y - self.current_y)
        closest_idx = np.argmin(dists)

        # Extract N points for the horizon starting from the closest point
        ref_list = []
        path_len = len(self._path_x)

        for i in range(self.N):
            idx = min(closest_idx + i,
                      path_len - 1)  # Ensure we don't go out of bounds

            current_ref_x = self._path_x[idx]
            current_ref_y = self._path_y[idx]
            current_ref_theta = self._path_theta[
                idx] if self._path_theta.size > 0 else self.current_theta  # Use path theta if available
            current_ref_v = self._path_v[
                idx] if self._path_v.size > 0 else self.current_v  # Use path velocity if available

            ref_list.extend([
                current_ref_x, current_ref_y, current_ref_theta, 0, self.ref_v
            ])

        self.ref_horizon = np.array(ref_list).reshape(-1, 1)

    def _mdl_update_linearized(self, current_v_vehicle: float, prev_vl: float,
                               prev_vr: float, prev_delta: float):
        """
        Updates the linearized state-space model (A_mpc, B_mpc, C_mpc)
        around the current state (self.current_theta, current_v_vehicle) and previous inputs.
        Note: self.A_mpc is kept as Identity for a delta_u formulation type: x_k+1 = x_k + B_k * du_k + f(xk,uk-1)*dt
        The f(xk,uk-1)*dt term is implicit            current_ref_v = self._path_v[idx] if self._path_v.size > 0 else self.current_v    # Use path velocity if available
        ly handled by predicting from current state and adding G*dU.
        """
        current_theta = self.current_theta  # From update_states

        # A_mpc remains Identity matrix for this common delta_u MPC formulation

        self.A_mpc[(0, 4)] = (self.ts) * cos(current_theta)
        self.A_mpc[(1, 4)] = (self.ts) * sin(current_theta)

        # --- Compute B_base (continuous time d(state_derivs)/du) ---
        # Inputs u = [v_left_wheel, v_right_wheel, steering_angle_delta]
        # States x = [x, y, theta, v_vehicle_avg]
        # Vehicle speed v is (vl+vr)/2 for the kinematic part of B_base
        # The vl0, vr0 passed are u_k-1 and are used for linearization point of derivative wrt delta.

        # Corrected B_base computation:
        tan_prev_delta = tan(prev_delta)  # Use math.tan
        # tan_prev_delta = np.clip(tan_prev_delta, -1.0, 1.0) # Clipping might be aggressive here. Max steer is usually less.

        # Derivative of theta_dot = ((vl+vr)/(2*L)) * tan(delta)
        # d(theta_dot)/dvl = tan(delta)/(2L)
        # d(theta_dot)/dvr = tan(delta)/(2L)
        # d(theta_dot)/ddelta = (vl+vr)/(2L*cos^2(delta))
        # Note: vl0, vr0 are from u_prev, representing the linearization point for speed.

        # Coeff for d(theta_dot)/ddelta
        a_coeff = (prev_vl + prev_vr) / (2 * self.L *
                                         (cos(prev_delta)**2 + 1e-6)
                                         )  # Added epsilon for cos^2
        # a_coeff = ((prev_vl + prev_vr) *
        #            (1 + tan_prev_delta**2)) / (2 * self.L)
        # Coeff for d(theta_dot)/dvl and d(theta_dot)/dvr
        b_coeff = tan_prev_delta / (2 * self.L)
        self.B_mpc[2, 0] = b_coeff * self.ts
        self.B_mpc[2, 1] = b_coeff * self.ts
        self.B_mpc[2, 2] = a_coeff * self.ts

    def update_pred_mdl(self):

        G = np.zeros((self.ny * self.N, self.nu * self.M))
        Phi = np.zeros((self.ny * self.N, self.nx))

        aux = self.C_mpc @ self.B_mpc
        Phi[0:self.ny, :] = self.C_mpc @ self.A_mpc

        for i in range(self.N):
            j = 0
            if i != 0:
                Phi[i * self.ny:(i + 1) *
                    self.ny, :] = Phi[(i - 1) * self.ny:i *
                                      self.ny, :] @ self.A_mpc
                aux = self.C_mpc @ (self.A_mpc @ self.B_mpc)

            while (j < self.M) and (i + j < self.N):
                G[(i + j) * self.ny:(i + j + 1) * self.ny,
                  j * self.nu:(j + 1) * self.nu] = aux
                j += 1

        return Phi, G

    def _solve_qp(self, current_physical_state_vec: np.ndarray, prev_vl: float,
                  prev_vr: float, prev_delta: float) -> np.ndarray:
        """
        Core MPC QP solver.
        Args:
            current_physical_state_vec: [x, y, theta, v_vehicle_avg]
            prev_vl, prev_vr, prev_delta: Elements of u_{k-1}
        Returns:
            u_k: The new absolute control command [vl, vr, delta_steer]
        """
        # Update the linearized model (A_mpc, B_mpc) around current state and u_prev
        self._mdl_update_linearized(current_physical_state_vec[3], prev_vl,
                                    prev_vr, prev_delta)

        # Get prediction matrices Phi (maps x_current to future x) and G (maps DU to future x)
        Phi_pred, G_pred = self.update_pred_mdl()

        # Cost function: J = (G*DU + Phi*x_current - X_ref)^T Q (G*DU + Phi*x_current - X_ref) + DU^T R DU
        # J = DU^T (G^T Q G + R) DU + 2 * (Phi*x_current - X_ref)^T Q G * DU + const
        # H = 2 * (G^T Q G + R)
        # F = 2 * (Phi*x_current - X_ref)^T Q G

        H = 2 * (G_pred.T @ self.Q_mpc @ G_pred + self.R_mpc)
        # current_state_for_pred needs to be (nx,1)
        x_curr_reshaped = current_physical_state_vec.reshape(self.nx, 1)
        error_term_for_F = (Phi_pred @ x_curr_reshaped) - self.ref_horizon
        F_transposed = 2 * (error_term_for_F.T @ self.Q_mpc @ G_pred)

        # Constraints on DU
        # 1. Control input rate limits (optional, not explicitly in original)
        #    du_min <= du <= du_max

        # 2. Absolute input limits: u_min_abs <= u_prev + du <= u_max_abs
        #    u_min_abs - u_prev <= du <= u_max_abs - u_prev
        # These bounds need to be formulated for the entire horizon M for DU vector.
        u_prev_vec = np.array([prev_vl, prev_vr,
                               prev_delta]).reshape(self.nu, 1)

        lower_bounds_du_abs = np.tile(
            self.u_min_abs.reshape(self.nu, 1) - u_prev_vec, (self.M, 1))
        upper_bounds_du_abs = np.tile(
            self.u_max_abs.reshape(self.nu, 1) - u_prev_vec, (self.M, 1))

        # 3. Slip constraint (if active): A_slip_horizon * DU approx -A_slip_single_step * U_prev_horizon
        # User's formulation was l_cons_slip <= A_slip_block_diag * (U_prev_horizon + DU) <= u_cons_slip
        # Which means l_cons_slip - A_slip * U_prev <= A_slip * DU <= u_cons_slip - A_slip * U_prev
        # For simplicity here, using the user's original constraint A_cons_slip_block_diag on DU with fixed bounds
        # (This is an area for refinement if slip constraint is critical)
        # A_constraints_list = [sparse.csc_matrix(lower_bounds_du_abs.shape[0], G_pred.shape[1])]    # Placeholder for actual constraint matrix build
        # l_bounds_list = [lower_bounds_du_abs]
        # u_bounds_list = [upper_bounds_du_abs]

        # Build the actual constraint matrix for absolute limits
        # Constraint is I * DU >= lower_bounds_du_abs and I * DU <= upper_bounds_du_abs
        # So, A_for_abs_limits is block_diag([I_nu]*M)
        # # l_for_abs_limits is lower_bounds_du_abs.flatten()
        # # u_for_abs_limits is upper_bounds_du_abs.flatten()
        # A_abs_lim = sparse.block_diag([sparse.eye(self.nu)] * self.M).tocsc()
        # l_abs_lim = lower_bounds_du_abs.flatten()
        # u_abs_lim = upper_bounds_du_abs.flatten()

        # A_final_cons = A_abs_lim
        # l_final_cons = l_abs_lim
        # u_final_cons = u_abs_lim

        # if self.use_differential and self.b > 1e-6:
        #     # A_slip applies to each u_k = u_{k-1} + du_k.
        #     # slip_const_matrix_row @ (u_prev_step_j + du_j) is between -eps and eps
        #     # => -eps - slip_const @ u_prev_j <= slip_const @ du_j <= eps - slip_const @ u_prev_j
        #     # This means bounds change at each step 'j' of control horizon M if u_prev is updated.
        #     # The original user code built A_cons from self.slip_const_matrix_row, which is updated once per call.
        #     # This A_cons was M x (M*nu). Its l/u bounds were fixed at -eps, eps.
        #     # This implied slip_const_matrix_row @ du_j ~ 0.

        #     # Replicating original logic for A_cons for slip (applied to DU directly):
        #     A_cons_slip_block_diag = sparse.block_diag([self.slip_const_matrix_row] * self.M).tocsc()
        #     l_cons_slip_active = self.l_cons_slip.flatten()    # Should be M x 1
        #     u_cons_slip_active = self.u_cons_slip.flatten()    # Should be M x 1

        #     A_final_cons = sparse.vstack([A_abs_lim, A_cons_slip_block_diag], format="csc")
        #     l_final_cons = np.hstack([l_abs_lim, l_cons_slip_active])
        #     u_final_cons = np.hstack([u_abs_lim, u_cons_slip_active])

        osqp_prob = osqp.OSQP()
        H = sparse.csc_matrix(H)
        osqp_prob.setup(P=H,
                        q=F_transposed.flatten(),
                        A=None,
                        l=None,
                        u=None,
                        verbose=False,
                        warm_start=True)
        # if not self.osqp_solver_initialized:
        #     self.osqp_prob.setup(P=H,
        #                          q=F_transposed.flatten(),
        #                          A=None,
        #                          l=None,
        #                          u=None,
        #                          verbose=False,
        #                          warm_start=True)
        #     self.osqp_solver_initialized = True
        # else:
        #     self.osqp_prob.update(Px=sparse.triu(H).data,
        #                           Ax=None,
        #                           q=F_transposed.flatten(),
        #                           l=None,
        #                           u=None)

        res = osqp_prob.solve()

        if res.info.status != "solved":
            # print(f"[MPC] QP not solved. Status: {res.info.status}. Using zero du.")
            optimal_du_sequence = np.zeros((self.nu * self.M, 1))
        else:
            optimal_du_sequence = res.x

        # Extract the first control action from the sequence
        self.du = optimal_du_sequence[0:self.nu].reshape(self.nu, 1)

        if (self.du[2] > 1):
            pass

        # Update u_prev for the next iteration
        # u_k = u_{k-1} + du_0
        u_current_applied = u_prev_vec + self.du

        self.u_prev = u_current_applied  # Store u_k as u_prev for next step

        return self.u_prev.flatten()  # Return [vl_k, vr_k, delta_k]

    # Inherited from BaseController, called by Simulator
    def compute_control(self) -> dict:
        """
        Main control computation method for MPC.
        Updates horizon reference, solves QP, returns commands.
        """
        # 1. Update the N-step reference trajectory for the QP
        self._update_horizon_reference()

        # 2. Get current physical state and previous control input u_{k-1}
        current_physical_state_vec = np.array([
            self.current_x, self.current_y, self.current_theta,
            self.current_delta, self.current_v
        ])
        prev_vl, prev_vr, prev_delta = self.u_prev.flatten(
        )  # u_prev is u_{k-1}

        # 3. Solve the MPC QP to get new absolute command u_k = [vl, vr, delta]
        # The _solve_qp method also updates self.u_prev to u_k internally.
        new_u_abs = self._solve_qp(current_physical_state_vec, prev_vl,
                                   prev_vr, prev_delta)

        vl_cmd_linear, vr_cmd_linear, delta_cmd_rad = new_u_abs[0], new_u_abs[
            1], new_u_abs[2]

        # 4. Return commands in the expected dictionary format
        # Simulator expects linear wheel velocities (v_left, v_right) and delta in rad.
        return {
            'delta': delta_cmd_rad,
            'v_left': vl_cmd_linear,  # Linear speed of left wheel (m/s)
            'v_right': vr_cmd_linear  # Linear speed of right wheel (m/s)
        }

    # Kept original set_reference for now, but _update_horizon_reference is preferred for dynamic use
    # This might be useful if a fixed N-step reference is to be set externally once.
    def set_fixed_horizon_reference(self, x_refs_horizon: np.ndarray):
        """
        Sets a fixed N-step reference trajectory for the MPC.
        Args:
            x_refs_horizon (np.array): Array of shape (N, ny) or (N*ny, 1).
        """
        if x_refs_horizon.shape[0] != self.N and x_refs_horizon.shape[
                0] != self.N * self.ny:
            raise ValueError(
                f"[MPC] Fixed reference trajectory must have {self.N} steps for {self.ny} states each."
            )
        if x_refs_horizon.ndim == 2 and x_refs_horizon.shape[
                0] == self.N:  # Shape (N, ny)
            self.ref_horizon = x_refs_horizon.flatten('C').reshape(-1, 1)
        elif x_refs_horizon.shape == (self.N * self.ny, 1):  # Shape (N*ny, 1)
            self.ref_horizon = x_refs_horizon
        else:
            raise ValueError(
                f"[MPC] Incorrect shape for fixed reference trajectory. Expected ({self.N},{self.ny}) or ({self.N*self.ny},1)."
            )
