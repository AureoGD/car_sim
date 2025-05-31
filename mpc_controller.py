import numpy as np
import osqp
from math import sin, cos
from scipy import sparse
from scipy.linalg import block_diag


class MPCController:

    def __init__(self, dt=0.1, Q=None, R=None, horizon=10, use_differential=False):
        self.prob = osqp.OSQP()
        self.use_differential = use_differential
        self.control_name = "mpc"
        self.ts = dt
        self.N = horizon
        self.M = 30
        self.L = 0.21
        self.b = 0.137
        self.epsilon = 0.000001
        self.l_cons = -self.epsilon * np.ones((self.M, 1))
        self.u_cons = self.epsilon * np.ones((self.M, 1))

        # Estados: x, y, theta, v
        self.nx = 4
        self.nu = 3
        self.ny = 4

        self.x = np.zeros((self.nx + self.nu, 1))
        self.A = np.eye(self.nx)
        self.B = np.zeros((self.nx, self.nu))
        self.C = np.eye(self.ny)
        self.slip_const = np.zeros((1, 3))
        self.ref = np.zeros((self.N * self.ny, 1))

        q_single = Q if Q is not None else np.diag([5, 5, 6, 5])
        r_single = R if R is not None else np.diag([1.0, 1.0, 0.001])

        self.Q = sparse.block_diag([q_single] * self.N).tocsc()
        self.R = sparse.block_diag([r_single] * self.M).tocsc()

        self.init_flag = False
        self.du = np.zeros((self.nu, 1))
        self.u_prev = np.zeros((self.nu, 1))

        self.u_min = np.array([0.0, 0.0, -1.0])
        self.u_max = np.array([1.5, 1.5, 1.0])

    def control(self, x, delta0, vl0, vr0):
        self.mdl_update(x, delta0, vl0, vr0)
        Phi, G = self.update_pred_mdl()

        H = 2 * sparse.csc_matrix((G.T @ self.Q @ G + self.R))
        F = 2 * (((Phi @ x.reshape(self.nx, 1)) - self.ref).T) @ self.Q @ G

        if not self.use_differential:
            A_cons = None
            self.l_cons = None
            self.u_cons = None
        else:
            A_cons = self.slip_const
            for index in range(self.M - 1):
                A_cons = block_diag(A_cons, self.slip_const)

        solver = osqp.OSQP()
        solver.setup(P=H, q=F.flatten(), A=sparse.csc_matrix(A_cons), l=self.l_cons, u=self.u_cons, verbose=False)
        res = solver.solve()

        if res.info.status != "solved":
            self.du = np.zeros((self.nu, 1))
            print("[MPC] QP not solved.")
        else:
            self.du = res.x[0:self.nu].reshape(-1, 1)

        u = np.array([vl0, vr0, delta0])
        self.u_prev = u + self.du
        return self.u_prev

    def mdl_update(self, x0, delta0, vl0, vr0):
        theta = x0[2]
        if isinstance(theta, np.ndarray):
            theta = theta[0]

        self.A = np.eye(self.nx)
        B_base = self.compute_B(theta, delta0, vl0, vr0)
        B_discrete = self.ts * B_base
        B_v = np.array([[0.5, 0.5, 0.0]])
        self.B = np.vstack((B_discrete, B_v))
        self.C = np.eye(self.ny)

    def update_pred_mdl(self):
        G = np.zeros((self.ny * self.N, self.nu * self.M))
        Phi = np.zeros((self.ny * self.N, self.nx))

        aux = self.C @ self.B
        Phi[0:self.ny, :] = self.C @ self.A

        for i in range(self.N):
            j = 0
            if i != 0:
                Phi[i * self.ny:(i + 1) * self.ny, :] = Phi[(i - 1) * self.ny:i * self.ny, :] @ self.A
                aux = self.C @ (self.A @ self.B)

            while (j < self.M) and (i + j < self.N):
                G[(i + j) * self.ny:(i + j + 1) * self.ny, j * self.nu:(j + 1) * self.nu] = aux
                j += 1

        return Phi, G

    def compute_B(self, theta, delta0, vl0, vr0):
        tan_delta = np.tan(delta0)
        tan_delta = np.clip(tan_delta, -1.0, 1.0)
        a = (vl0 + vr0) * (1 + tan_delta**2)
        a = a[0] / (2 * self.L)
        b = tan_delta[0] / (2 * self.L)

        B = np.array([[np.cos(theta), np.cos(theta), 0], [np.sin(theta), np.sin(theta), 0], [b, b, a]])
        self.slip_const[0, 0] = -(1 / (2 * self.b) + b)
        self.slip_const[0, 1] = (1 / (2 * self.b) - b)
        self.slip_const[0, 2] = -a
        return B

    def set_reference(self, x_refs):
        if len(x_refs) != self.N:
            raise ValueError(f"[MPC] Reference trajectory must have {self.N} steps.")
        self.ref = x_refs.flatten('C').reshape(-1, 1)

    def compute_steering_angle(self, x, y, theta, v_meas, L):
        state = np.array([x, y, theta, v_meas])
        vl0, vr0, delta0 = self.u_prev
        u = self.control(state, delta0, vl0, vr0)
        vl, vr, delta = u.flatten()
        return float(delta), (float(vl), float(vr))
