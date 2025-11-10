from collections import deque
from pomdpy.pomdp import AtomicPropPOMDP
from typing import Any
import spot


def get_next_aut_state_by_observation(
    env: AtomicPropPOMDP, aut: Any, obs_idx: int, state_aut_idx: int
):
    """
    Get the next automaton state based on observation.
    
    Args:
        env: POMDP with atomic propositions on observations
        aut: spot automaton
        obs_idx: observation index
        state_aut_idx: current automaton state index
        
    Returns:
        Next automaton state index
    """
    bdict = aut.get_dict()
    formula = env.generate_observation_formula(obs_idx)
    transitions = aut.out(state_aut_idx)
    for t in transitions:
        if formula == spot.bdd_format_formula(bdict, t.cond):
            return t.dst
    raise ValueError(
        f"No matching transition found for formula: {formula} "
        f"in automaton state {state_aut_idx} for observation {obs_idx}"
    )


class BeliefSuppMDP:
    """
    Belief-Support MDP constructed from a POMDP and parity automaton.
    
    This class builds the belief-support MDP where:
    - States are belief supports (sets of (pomdp_state, automaton_state) pairs)
    - Actions are the same as the POMDP actions
    - Transitions follow observations and update both POMDP and automaton states
    - Priorities are inherited from the automaton states
    """

    def __init__(self, env: AtomicPropPOMDP, aut: Any):
        """
        Constructs the Belief-Support MDP from a POMDP and parity automaton.
        
        Args:
            env: AtomicPropPOMDP with atomic propositions on observations
            aut: spot parity automaton
        """
        self.pomdp = env
        self.aut = aut
        self.actions = env.actions
        self.actionsinv = env.actionsinv
        self.trans = {}  # state -> action -> list of successor states
        self.prio = {}   # state -> priority

        # Build the belief-support MDP via forward exploration
        self._build_belief_support_mdp()
        
        # Set priorities for each belief support state
        self._set_priorities()

    def _build_belief_support_mdp(self):
        """
        Build the belief-support MDP states and transitions via forward
        exploration from the initial belief support.
        """
        # 1. Define the initial belief support
        aut_init_idx = self.aut.get_init_state_number()
        initial_pomdp_states = {k for k, v in self.pomdp.start.items()
                               if v > 0}

        st = tuple(sorted([(s, aut_init_idx) for s in initial_pomdp_states]))

        self.states = [st]
        self.statesinv = {st: 0}
        self.start = 0

        explore = deque([st])

        # 2. Forward exploration to build state space and transitions
        while len(explore) > 0:
            current_bs = explore.popleft()
            current_bs_idx = self.statesinv[current_bs]
            self.trans[current_bs_idx] = {}

            for act_idx, act in enumerate(self.actions):
                self.trans[current_bs_idx][act_idx] = []

                # For each observation, collect successor belief support
                possible_succ_beliefs = {}  # obs_idx -> set of (s', q')

                # Iterate through product states in current belief support
                for s, q in current_bs:
                    if (s in self.pomdp.transitions and
                        act_idx in self.pomdp.transitions[s]):
                        trans_dict = self.pomdp.transitions[s][act_idx]
                        for (s_prime, o), prob in trans_dict.items():
                            if prob > 0:
                                q_prime = get_next_aut_state_by_observation(
                                    self.pomdp, self.aut, o, q
                                )
                                if o not in possible_succ_beliefs:
                                    possible_succ_beliefs[o] = set()
                                possible_succ_beliefs[o].add((s_prime, q_prime))

                # Sort observations for deterministic ordering
                sorted_obs = sorted(possible_succ_beliefs.keys())
                
                # For each observation, create successor belief support
                for o in sorted_obs:
                    next_bs_set = possible_succ_beliefs[o]
                    next_bs = tuple(sorted(list(next_bs_set)))

                    if not next_bs:
                        continue  # Skip empty belief supports

                    if next_bs in self.statesinv:
                        next_bs_idx = self.statesinv[next_bs]
                    else:
                        next_bs_idx = len(self.states)
                        self.statesinv[next_bs] = next_bs_idx
                        self.states.append(next_bs)
                        explore.append(next_bs)

                    if next_bs_idx not in self.trans[current_bs_idx][act_idx]:
                        self.trans[current_bs_idx][act_idx].append(next_bs_idx)

                # Sort successors for deterministic ordering
                self.trans[current_bs_idx][act_idx].sort()

    def _set_priorities(self):
        """
        Set priority for each belief support state.
        
        The priority of a belief support is the maximum priority of any
        automaton state within that support.
        """
        simplified_buchi = self.aut.acc().num_sets() == 1

        for i, bs in enumerate(self.states):
            max_prio = 0
            if not bs:  # Handle empty belief supports
                self.prio[i] = 0
                continue

            for _, q in bs:
                acc_sets = list(self.aut.state_acc_sets(q).sets())
                prio = 0
                if not acc_sets:
                    prio = 0 + (1 if simplified_buchi else 0)
                else:
                    # Map spot priorities to parity game convention
                    prio = max(acc_sets) + (2 if simplified_buchi else 0)

                if prio > max_prio:
                    max_prio = prio
            self.prio[i] = max_prio

    def prettyName(self, st):
        """Get pretty name for a belief support state."""
        return tuple([self.pomdp.states[q] for q in st])

    def show(self, outfname, reach=None, reachStrat=None,
             mecs_and_strats=None):
        """
        Render the belief-support MDP to a DOT file.
        
        Args:
            outfname: Output filename (if ends with .png/.svg, writes .dot)
            reach: Set of reachable states
            reachStrat: Reachability strategy dict
            mecs_and_strats: List of MEC strategy dicts
        """

        def belief_label(st):
            pairs = [f"{self.pomdp.states[s]}-{q}" for (s, q) in st]
            return "{" + ", ".join(pairs) + "}"

        # Collect all belief indices in MEC strategies
        all_mec_beliefs = set()
        if mecs_and_strats:
            for strat_dict in mecs_and_strats:
                all_mec_beliefs.update(strat_dict.keys())

        # Start DOT
        lines = ["digraph BeliefSupport {", "  rankdir=LR;"]

        # Nodes
        for i, st in enumerate(self.states):
            color = "white"
            if reach is not None and i in reach:
                color = "lightblue"
            if all_mec_beliefs and i in all_mec_beliefs:
                color = "palegreen"

            label = (
                f"B{i}: {belief_label(st)}"
                f"\\nprio={self.prio.get(i, 'N/A')}"
            )
            lines.append(
                f'  {i} [label="{label}", style="filled", '
                f'fillcolor="{color}"];'
            )

        # Edges
        for src in range(len(self.states)):
            # Determine which actions are chosen by strategies
            strat_actions = set()
            is_mec_strat = False
            if mecs_and_strats:
                for strat_dict in mecs_and_strats:
                    if src in strat_dict:
                        strat_actions = set(strat_dict[src])
                        is_mec_strat = True
                        break
            if not strat_actions and reachStrat and src in reachStrat:
                strat_actions = set(reachStrat[src])

            if src in self.trans:
                for a, act_name in enumerate(self.actions):
                    edge_color = "gray"
                    if a in strat_actions:
                        edge_color = "green" if is_mec_strat else "blue"
                    if a in self.trans[src]:
                        for dst in self.trans[src][a]:
                            lines.append(
                                f'  {src} -> {dst} [color="{edge_color}", '
                                f'label="{act_name}"];'
                            )

        lines.append("}")

        dot_content = "\n".join(lines)

        # If user asked for an image, write a .dot file instead
        if outfname.endswith('.png') or outfname.endswith('.svg'):
            outfname = outfname.rsplit('.', 1)[0] + '.dot'
        with open(outfname, 'w') as f:
            f.write(dot_content)