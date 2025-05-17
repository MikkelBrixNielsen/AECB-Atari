import torch
import time
import numpy as np
from utils import VALUE_CONTAINER, extract_and_batch, log

class VectorizedMDP:
    def __init__(self, model, num_actions, log_dir, batch=256):
        self.model = model
        self.num_actions = num_actions
        self.batch_size = batch
        self.log_dir = log_dir

        self.state2idx = {}
        self.idx_counter = 0

        self.N_sa = np.zeros((1, num_actions), dtype=np.int32)
        self.R_sums = np.zeros((1, num_actions), dtype=np.float32)
        self.D_sums = np.zeros((1, num_actions), dtype=np.float32)
        self.N_sas = np.zeros((1, num_actions, 1), dtype=np.int32)
        self.V = np.zeros(1, dtype=np.float32)

    def _flatten_indices(self, z):
        return [tuple(row.tolist()) for row in z]

    def _ensure_state(self, z_flat):
        if z_flat not in self.state2idx:
            self.state2idx[z_flat] = self.idx_counter
            self._ensure_capacity(self.idx_counter)
            self.idx_counter += 1
        return self.state2idx[z_flat]

    def _ensure_capacity(self, idx):
        cur = self.N_sa.shape[0]
        if idx >= cur:
            new_size = idx + 1

            def grow(arr, shape):
                pad = [(0, max(0, s - arr.shape[i])) for i, s in enumerate(shape)]
                return np.pad(arr, pad, mode='constant')

            self.N_sa = grow(self.N_sa, (new_size, self.num_actions))
            self.R_sums = grow(self.R_sums, (new_size, self.num_actions))
            self.D_sums = grow(self.D_sums, (new_size, self.num_actions))
            self.V = grow(self.V, (new_size,))
            self.N_sas = grow(self.N_sas, (new_size, self.num_actions, new_size))

    def _discretize_multiple(self, s_batch):
        with torch.no_grad():
            z_e = self.model.encoder(s_batch)
            _, _, z_q_indices = self.model.quantizer(z_e)
            z_q_indices = z_q_indices.view(z_q_indices.shape[0], -1)
        return z_q_indices.cpu().numpy()

    def _discretized_extract_and_batch(self, transitions):
        z_list, a_list, zp_list, r_list, d_list = [], [], [], [], []

        for i in range(0, len(transitions), self.batch_size):
            mini_batch = transitions[i:i+self.batch_size]
            s_batch, a_batch, sp_batch, r_batch, d_batch = extract_and_batch(mini_batch)

            z_batch = self._discretize_multiple(s_batch)
            zp_batch = self._discretize_multiple(sp_batch)

            z_list.append(z_batch)
            zp_list.append(zp_batch)
            a_list.append(a_batch.cpu().numpy())
            r_list.append(r_batch.cpu().numpy())
            d_list.append(d_batch.cpu().numpy())

        z = np.concatenate(z_list, axis=0)
        zp = np.concatenate(zp_list, axis=0)
        a = np.concatenate(a_list, axis=0)
        r = np.concatenate(r_list, axis=0)
        d = np.concatenate(d_list, axis=0)

        return z, a, zp, r, d

    def _compute_known_transition_percentage(self, M=1):
        N_sa_trimmed = self.N_sa[:self.idx_counter, :]
        known = np.sum(N_sa_trimmed >= M)
        total_possible = self.idx_counter * self.num_actions
        return 100.0 * known / total_possible if total_possible > 0 else 0.0

    def update(self, transitions):
        st = time.time()
        log(self.log_dir, "\tUpdating MDP...", console_log=True, no_log=True)

        self.model.eval()
        z, a, zp, r, d = self._discretized_extract_and_batch(transitions)

        z_flat = self._flatten_indices(z)
        zp_flat = self._flatten_indices(zp)

        for zf, act, zpf, rew, done in zip(z_flat, a, zp_flat, r, d):
            si = self._ensure_state(zf)
            sip = self._ensure_state(zpf)

            self.N_sa[si, act] += 1
            self.R_sums[si, act] += rew
            self.D_sums[si, act] += done
            self.N_sas[si, act, sip] += 1
        
        VALUE_CONTAINER.transition_percentage = f"Percentage of Transitions Known: {self._compute_known_transition_percentage():.4f}%"
        log(self.log_dir, f"\tMDP updated in {time.time() - st:.4f}, {VALUE_CONTAINER.transition_percentage}", console_log=True, no_log=True)

    def _VI(self, P, R, gamma, eps, max_iters):
        st = time.time()
        for i in range(max_iters):
            Q = R + gamma * (P @ self.V)
            V_new = np.max(Q, axis=1)
            delta = np.max(np.abs(V_new - self.V))
            log(self.log_dir, f"\t\tVI - Round: {i}, Delta: {delta}, Target: {eps}, Duration: {time.time() - st:.4f}", console_log=VALUE_CONTAINER.debug_mode, no_log=True)
            if delta < eps:
                break
            self.V = V_new
        return np.argmax(Q, axis=1)

    def solve(self, gamma=0.99, eps=1e-5, max_iters=2500):
        st = time.time()
        log(self.log_dir, "\tDoing Value Iteration...", console_log=True, no_log=True)
        n = self.idx_counter
        N_sa = self.N_sa[:n]
        R = np.divide(self.R_sums[:n], N_sa, where=N_sa > 0)
        P = self.N_sas[:n, :, :n].astype(np.float32)
        denom = P.sum(axis=2, keepdims=True)
        P = np.divide(P, denom, where=denom > 0)
        pi = self._VI(P, R, gamma, eps, max_iters)
        log(self.log_dir, f"\tMDP solved: {time.time() - st:.4f}", console_log=True, no_log=True)
        return pi
    
    def _discretize(self, s):
        self.model.eval()
        with torch.no_grad():
            z_e = self.model.encoder(s)
            _, _, z_q_indices = self.model.quantizer(z_e)
            z_q_indices = z_q_indices.view(-1).cpu().numpy()
        return tuple(z_q_indices.tolist())

    def get_index_if_known(self, s):
        return self.state2idx.get(self._discretize(s), None)

    def discretize_and_index(self, s):
        return self._ensure_state(self._discretize(s))