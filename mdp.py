import torch
import time
from collections import defaultdict, Counter
from utils import VC, compute_known_transition_percentage, log
import numpy as np

class MDP:
    def __init__(self, model, device, log_dir, gamma=0.99, min_visits=1, R_max=1.0, use_hash=True):
        self.gamma = gamma
        self.R_max = R_max
        self.M = min_visits
        self.model = model
        self.device = device
        self.log_dir = log_dir
        self.use_hash = use_hash

        self.N_sa = defaultdict(int)
        self.N_sas = defaultdict(Counter)
        self.rewards_sum = defaultdict(float)
        self.terminal_counts = defaultdict(int)

        self.unique_states = set()
        self.dirty = set()
        self.num_actions = float('-inf')

        self.state2idx = {}
        self.idx2state = []
        self.num_states = 0

        self.P = None
        self.R = None
        self.D = None
        self.V = None
        self.pi = {}

    def _build(self):
        st = time.time()
        log(self.log_dir, "\t\tCalculating P, R, D...", console_log=VC.debug_mode, no_log=True)
        
        for (s, a) in self.dirty:
            self.unique_states.add(s)
            self.unique_states.update(self.N_sas[(s, a)].keys())

        self.idx2state = list(self.unique_states)
        self.state2idx = {s: i for i, s in enumerate(self.idx2state)}
        self.num_states = len(self.idx2state)
        if len(self.dirty) > 0:
            self.num_actions = max(max([a for (_, a) in self.dirty]) + 1, self.num_actions)

        self.P = np.zeros((self.num_states, self.num_actions, self.num_states), dtype=np.float32)
        self.R = np.full((self.num_states, self.num_actions), self.R_max, dtype=np.float32)
        self.D = np.zeros((self.num_states, self.num_actions), dtype=np.float32)

        for (s, a) in self.dirty:
            s_idx = self.state2idx[s]
            total = self.N_sa[(s, a)]
            if total < self.M:
                self.P[s_idx, a, :] = 1.0 / self.num_states # uniform probability to all states
                self.R[s_idx, a] = self.R_max
                self.D[s_idx, a] = 0 # Assume non-terminal state
                continue
            for sp, count in self.N_sas[(s, a)].items():
                sp_idx = self.state2idx[sp]
                self.P[s_idx, a, sp_idx] = count / total
            self.R[s_idx, a] = self.rewards_sum[(s, a)] / total
            self.D[s_idx, a] = (self.terminal_counts[(s, a)] / total) > 0.5

        self.dirty.clear()
        log(self.log_dir, f"\t\tCompleted calculating P, R, D in {time.time() - st:.4f}", console_log=VC.debug_mode, no_log=True)
    
    def _encode_and_quantize(self, s_batch):
        with torch.no_grad():
            _, _, z_q_indices = self.model.quantizer(self.model.encoder(s_batch))
            return z_q_indices

    # def _encode_state(self, z):
    #     if isinstance(z, torch.Tensor):
    #         z = z.cpu().numpy()
    #     return z.flatten().tobytes() if self.use_hash else tuple(z.flatten().tolist())
    
    def _encode_state(self, z_indices):
        hist = torch.bincount(z_indices.view(-1), minlength=self.model.quantizer.num_embeddings)
        # return tuple(hist.tolist()) # preserves frequency
        return tuple((hist > 0).int().tolist()) # simpler, less states

    def _add_transition(self, z, a, z_next, r, done):
        s = self._encode_state(z)
        sp = self._encode_state(z_next)
        self.N_sas[(s, a)][sp] += 1
        self.rewards_sum[(s, a)] += r
        self.N_sa[(s, a)] += 1
        if done:
            self.terminal_counts[(s, a)] += 1
        self.dirty.add((s, a))

    def _add_transitions_aux(self, transitions, current, goal):
        self.model.eval()
        s_batch, a_batch, sp_batch, r_batch, d_batch = zip(*transitions)
        z_q = self._encode_and_quantize(torch.stack(s_batch).to(self.device))
        zp_q = self._encode_and_quantize(torch.stack(sp_batch).to(self.device))

        self.unique_codes_used.add(torch.unique(z_q))
        self.unique_codes_used.add(torch.unique(zp_q))

        for z_q, a, zp_q, r, d in zip(z_q, a_batch, zp_q, r_batch, d_batch):
            log(self.log_dir, f"\t\t\tAdding transition: {current + 1} / {goal}...", console_log=VC.debug_mode, no_log=True)
            self._add_transition(z_q, a.item(), zp_q, r.item(), d.item())
            current += 1
    
    def _add_transitions(self, transitions, mini_batch_size):
        st = time.time()
        log(self.log_dir, f"\t\t Adding transitions...", console_log=VC.debug_mode, no_log=True)
        num_transitions = len(transitions)
        for i in range(0, num_transitions, mini_batch_size):
            mini_batch = transitions[i : i + mini_batch_size]
            self._add_transitions_aux(mini_batch, i, num_transitions)
        log(self.log_dir, f"\t\tCompleted adding transitions in {time.time() - st:.4f}", console_log=VC.debug_mode, no_log=True)

    def update(self, transitions, mini_batch_size=256):
        self.unique_codes_used = set()
        ast = time.time()
        log(self.log_dir, "\tUpdating MDP...", console_log=True, no_log=True)
        self._add_transitions(transitions, mini_batch_size)
        self._build()
        VC.transition_percentage = f"Transitions known: {compute_known_transition_percentage(self.N_sa, self.num_states, self.num_actions, self.M):.2f}%"
        s_states = len(set(s for (s, _) in self.N_sa))
        sp_states = len(set(sp for counter in self.N_sas.values() for sp in counter))
        log(self.log_dir, f"\tMDP updated completed in {time.time() - ast:.4f}, " + 
            f"Unique codes used: {len(self.unique_codes_used)}, " +
            f"{VC.transition_percentage}, " +
            f"#(s, a)-pairs: {len(self.N_sa.keys())}, " +
            f"#states: {self.num_states} - {s_states, sp_states}, " +
            f"#actions: {self.num_actions}", 
            console_log=True, no_log=True)

    def get_action(self, s, action_space): # expects a single frame on form (4, 84, 84) 
        s_idx = self.state2idx.get(self._encode_state(self._encode_and_quantize(s.unsqueeze(0))), None)
        return self.pi[s_idx] if s_idx is not None and s_idx < len(self.pi) else action_space.sample()

    def _check_VV_size(self):
        if self.V is None:
            self.V = np.zeros(self.num_states, dtype=np.float32)
        elif self.V.shape[0] < self.num_states:
            V_old = self.V
            self.V = np.zeros(self.num_states, dtype=np.float32)
            self.V[:V_old.shape[0]] = V_old

    def solve(self, tol=1e-6, max_iterations=10000): # Currently does value iteration
        if self.P is None or self.P.shape[0] == 0:
            log(self.log_dir, "\tSkipping value iteration: no transitions known.", console_log=True)
            return

        ast = time.time()
        log(self.log_dir, "\tDoing value iteration...", console_log=True, no_log=True)
        self._check_VV_size()

        for i in range(max_iterations):
            st = time.time()
            new_V = np.max(self.R + self.gamma * np.einsum('sak,k->sa', self.P, self.V) * (1 - self.D), axis=1)
            delta = np.max(np.abs(new_V - self.V))
            self.V[:] = new_V # in-place update
            log(self.log_dir, f"\t\tVI round {i}: Delta: {delta:.6f}, Target: {tol:.6f}, Duration: {time.time() - st:.4f}",  console_log=VC.debug_mode, no_log=True)
            if delta < tol:
                break

        self.pi = np.argmax(self.R + self.gamma * np.einsum('sak,k->sa', self.P, self.V), axis=1)
        log(self.log_dir, f"\tValue iteration completed in {time.time() - ast:.4f}", console_log=True, no_log=True)
