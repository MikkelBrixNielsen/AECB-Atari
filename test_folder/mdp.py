import torch
import time
import math
from collections import defaultdict, Counter
from utils import VC, compute_known_transition_percentage, log, extract_and_batch

def validate_transition_probabilities(P, tolerance=1e-6, log_dir=None, console_log=False):
    invalid_pairs = []

    for (s, a), transitions in P.items():
        total_prob = sum(transitions.values())
        if abs(total_prob - 1) > tolerance:
            invalid_pairs.append(((s, a), total_prob))

    if invalid_pairs:
        msg = f"\t\t[WARNING] {len(invalid_pairs)} (s, a) pairs have invalid transition probability sums:"
        if log_dir:
            log(log_dir, msg, console_log=console_log, no_log=True)
            for (s, a), total in invalid_pairs:
                log(log_dir, f"\t(s, a) = ({s}, {a}) → total probability = {total:.6f}", console_log=False, no_log=True)
        else:
            print(msg)
            for i, ((s, a), total) in enumerate(invalid_pairs):
                print(f"\t\t\tPair {i} total probability = {total:.6f}")
    else:
        msg = "\t\t[OK] All transition probability distributions sum to ~1.0"
        if log_dir:
            log(log_dir, msg, console_log=console_log, no_log=True)
        else:
            print(msg)

    return len(invalid_pairs) == 0

def discretize(model, s):
    model.eval()
    with torch.no_grad():
        z_e = model.encoder(s)
        _, _, z_q_indices = model.quantizer(z_e)
        z_q_indices = z_q_indices.cpu().numpy()

    return z_q_indices.flatten().tobytes()

def discretize_multiple(model, s_batch):
    model.eval()
    with torch.no_grad():
        z_e = model.encoder(s_batch)
        _, _, z_q_indices = model.quantizer(z_e)
        z_q_indices = z_q_indices.cpu().numpy()

    return [z_q_indices[i].flatten().tobytes() for i in range(z_q_indices.shape[0])]


def discretized_extract_and_batch(model, transitions, batch_size=128): # Larger batch increases speed but also memory usage
    model.eval()
    ds_list, a_list, dsp_list, r_list, d_list = [], [], [], [], []

    for i in range(0, len(transitions), batch_size):
        mini_batch = transitions[i:i+batch_size]
        s_batch, a_batch, sp_batch, r_batch, d_batch = extract_and_batch(mini_batch)

        with torch.no_grad():
            ds_batch = discretize_multiple(model, s_batch)
            dsp_batch = discretize_multiple(model, sp_batch)

        ds_list.extend(ds_batch)
        a_list.extend(a_batch)
        dsp_list.extend(dsp_batch)
        r_list.extend(r_batch)
        d_list.extend(d_batch)

    return zip(ds_list, a_list, dsp_list, r_list, d_list)

def is_known(N_sa_val, M):
    return N_sa_val >= M

# P(s'|s, a) = { N(s, a, s) / N(s, a)    if N(s, a) >= M  |  R(s, a) = { R_sum / N(s, a)     if N(s, a) >= M  |  D(s, a) = { 1 if D_sum / N(s, a) > 0.5 else 0     if N(s, a) >= M
#              { I[s' = s]}              otherwise        |            { R_max               otherwise        |            { 0                                     otherwise
# Note: The otherwise part of R and D is provided by the default behaviour of defaultdicts, so can be omitted 
def update_P_R_D(items, N_sa, N_sas, R_sum, D_sum, states, actions, s_max, R_max, P, R, D, M=1):
    for (s, a), total in items:
        if is_known(total, M):
            P[(s, a)] = {
                sp: N_sas[(s, a, sp)] / total
                for sp in states
                if N_sas[(s, a, sp)] > 0 # keep P as sparse as possible
            }
            R[(s, a)] = R_sum[(s, a)] / total
            D[(s, a)] = 1 if D_sum[(s, a)] / total > 0.5 else 0

    # add self loop to unknown (s, a)-pairs
    for s in states: # observed discritized states
        for a in actions: # full action space from env
            if (s, a) not in P.keys() or not is_known(N_sa[(s, a)], M):
                P[(s, a)] = {s_max: 1.0}                            # Optimistic transition to absorbing state
                R[(s, a)] = R_max / math.sqrt(N_sa[(s, a)] + 1)     # R_max value
                D[(s, a)] = 0                                       # Assume non-terminal

    return P, R, D

def compute_P_R_D(N_sa, N_sas, R_sum, D_sum, states, actions, s_max, M=1, R_max=1.0):
    P = defaultdict(dict)
    R = defaultdict(lambda: R_max)    # R-MAX fallback
    D = defaultdict(int)              # defaults to 0
    return update_P_R_D(N_sa.items(), N_sa, N_sas, R_sum, D_sum, states, actions, s_max, R_max, P, R, D, M)

def create_mdp(model, actions, transitions, log_dir, M=1, R_max=1.0):
    st = time.time()
    log(log_dir, "\tCreating MDP...", console_log=True, no_log=True)
    processed_transitions = discretized_extract_and_batch(model, transitions)

    N_sa = Counter()
    N_sas = Counter()
    R_sum = Counter()
    D_sum = Counter()
    states = set()

    for s, a, sp, r, d in processed_transitions:
        a, r, d = a.item(), r.item(), d.item() # tensor -> value
        N_sa[(s, a)] += 1
        N_sas[(s, a, sp)] += 1
        R_sum[(s, a)] += r
        if d:
            D_sum[(s, a)] += 1
        states.update([s, sp])

    s_max = b"\xff" * model.quantizer.num_embeddings

    P, R, D = compute_P_R_D(N_sa, N_sas, R_sum, D_sum, states, actions, s_max, M, R_max)

    if VC.debug_mode:
        validate_transition_probabilities(P, tolerance=1e-6, log_dir=None, console_log=VC.debug_mode)
    VC.transition_percentage = f"Percentage of Transitions Known: {compute_known_transition_percentage(N_sa, states, actions, M):.4f}%"
    log(log_dir, f"\tMDP created in {time.time() - st:.4f}, {VC.transition_percentage}", console_log=True, no_log=True)

    return {
        'N_sa': N_sa,           # Count of observed (s, a)-pairs
        'N_sas': N_sas,         # Count of observed (s, a, s')-pairs
        'R_sum': R_sum,         # Total reward for all observed (s, a)-pairs
        'D_sum': D_sum,         # Total number of observed s, a)-pairs leading to a terminal state
        'P': P,                 # Estimated P(s'|s, a)
        'R': R,                 # Estimated R(s, a)
        'D': D,                 # Estimation of whether (s, a) -> terminal state
        'states': states,       # Observed discretized states
        'actions': actions,     # Iterable containing all possible actions in env
        's_max': s_max,         # Optimistic absorbing state
        'R_max': R_max          # R_max used when creating MDP 
    }

def update_mdp(mdp, model, transitions, log_dir, M=1):
    st = time.time()
    log(log_dir, "\tUpdating MDP...", console_log=True, no_log=True)
    updated_sa = set()
    processed_transitions = discretized_extract_and_batch(model, transitions)

    for s, a, sp, r, d in processed_transitions:
        a, r, d = a.item(), r.item(), d.item()
        mdp['N_sa'][(s, a)] += 1
        mdp['N_sas'][(s, a, sp)] += 1
        mdp['R_sum'][(s, a)] += r
        if d:
            mdp['D_sum'][(s, a)] += 1
        mdp['states'].update([s, sp])
        updated_sa.add((s, a))

    items = [((s, a), mdp['N_sa'][(s, a)]) for (s, a) in updated_sa]
    mdp['P'], mdp['R'], mdp['D'] = update_P_R_D(items, mdp['N_sa'], mdp['N_sas'], mdp['R_sum'], mdp['D_sum'], mdp['states'], mdp['actions'], mdp['s_max'], mdp['R_max'], mdp['P'], mdp['R'], mdp['D'], M)

    if VC.debug_mode:
        validate_transition_probabilities(mdp['P'], tolerance=1e-6, log_dir=None, console_log=VC.debug_mode)
    VC.transition_percentage = f"Transition Percentage: {compute_known_transition_percentage(mdp['N_sa'], mdp['states'], mdp['actions'], M):.4f}%"
    log(log_dir, f"\tMDP update completed in: {time.time() - st:.4f}, {VC.transition_percentage}", console_log=True, no_log=True)

def VI(P, R, states, actions, log_dir, V, gamma=0.99, max_iterations=10000, tol=1e-6, max_patience=10, s_max=None, R_max=1.0):
    ast = time.time()
    log(log_dir, "\tDoing Value Iteration...", console_log=True, no_log=True)
    Q = defaultdict(float)
    pi = {}
    patience = 0
    prev_delta = float('inf')

    if s_max is not None:
        V[s_max] = R_max / (1-gamma) # Value of optimistic absorbing state set to max discounted reward e.g. R_max=1, gamma=0.99 -> V[s_max]=100

    for i in range(max_iterations):
        st = time.time()
        delta = 0
        for s in states:
            q_max = float('-inf')
            best_a = None
            for a in actions:
                q = sum([p * (R[(s, a)] + (gamma * V[sp])) for sp, p in P[(s, a)].items()])
                if q > q_max:
                    q_max = q
                    best_a = a
                Q[(s, a)] = q
            delta = max(delta, abs(q_max - V[s]))
            V[s] = q_max
            pi[s] = best_a

        log(log_dir, f"\t\tVI - Round: {i}, Delta: {delta}, Target: {tol}, Duration: {time.time() - st:.4f}", console_log=VC.debug_mode, no_log=True)
        if delta < tol or patience >= max_patience: # convergence check
            break
        if (delta == prev_delta):
            patience += 1
        prev_delta = delta

    log(log_dir, f"\tVI completed in: {time.time() - ast:.4f}", console_log=True, no_log=True)
    return pi, V