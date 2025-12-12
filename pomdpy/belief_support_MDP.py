from collections import deque
from pomdpy.pomdp import ParityPOMDP, ParityMDP, AtomicPropPOMDP
from typing import Any
import spot


def get_next_aut_state_by_observation(
    env: ParityPOMDP, aut: Any, obs_idx: int, state_aut_idx: int
):
    """
    Get the next automaton state based on observation.
    Assumes `env` is a Product/Parity POMDP built from an AtomicPropPOMDP,
    and uses the underlying base env's observation formula.
    """
    bdict = aut.get_dict()
    # ProductPOMDP exposes the base env via `_env`.
    base_env = getattr(env, "_env", None)
    if base_env is None:
        raise ValueError("Expected a ProductPOMDP with an underlying _env")
    formula = base_env.generate_observation_formula(obs_idx)
    for t in aut.out(state_aut_idx):
        if formula == spot.bdd_format_formula(bdict, t.cond):
            return t.dst
    raise ValueError(
        (
            "No matching automaton transition for obs "
            f"{obs_idx} from state {state_aut_idx}"
        )
    )


class BeliefSuppMDP(ParityMDP):
    """
    Belief-Support MDP constructed from a ParityPOMDP.
    
    This class builds the belief-support MDP where:
    - States are belief supports (sets of (pomdp_state, automaton_state) pairs)
    - Actions are the same as the POMDP actions
        - Transitions follow observations and update both POMDP
            and automaton states
    - Priorities are inherited from the automaton states
    """

    def __init__(self, env: ParityPOMDP, aut: Any = None):
        """
        Constructs the Belief-Support MDP from a ParityPOMDP.
        
        Args:
            env: ParityPOMDP. Can be:
                 - A ProductPOMDP (from AtomicPropPOMDP × automaton), or
                 - A ParityPOMDP directly (from PPS conversion)
            aut: spot parity automaton. Required if env is ProductPOMDP, 
                 None if env is a ParityPOMDP directly.
        """
        super().__init__()
        self.pomdp = env
        self.aut = aut
        self.actions = env.actions
        self.actionsinv = env.actionsinv

        # Build the belief-support MDP via forward exploration
        self._build_belief_support_mdp()
        
        # Set priorities for each belief support state
        self._set_priorities()

    def _build_belief_support_mdp(self):
        """
        Build the belief-support MDP states and transitions via forward
        exploration from the initial belief support.
        
        Handles two cases:
        1. ProductPOMDP (AtomicPropPOMDP × automaton): Uses automaton states
        2. ParityPOMDP directly: Uses POMDP states with their priorities
        """
        # Determine if we're dealing with a product or direct parity POMDP
        is_product = hasattr(self.pomdp, "_env") and self.aut is not None
        is_parity_direct = isinstance(self.pomdp, ParityPOMDP) and self.aut is None
        
        if is_product:
            self._build_belief_support_mdp_product()
        elif is_parity_direct:
            self._build_belief_support_mdp_parity()
        else:
            raise ValueError(
                "BeliefSuppMDP requires either a ProductPOMDP with "
                "an automaton, or a ParityPOMDP directly (aut=None)."
            )
    
    def _build_belief_support_mdp_product(self):
        """
        Build belief-support MDP for ProductPOMDP case (AtomicPropPOMDP × automaton).
        """
        # 1. Define the initial belief support
        aut_init_idx = self.aut.get_init_state_number()

        # Determine base POMDP and its state space size
        base_env = None
        if (hasattr(self.pomdp, "_env")
                and isinstance(self.pomdp._env, AtomicPropPOMDP)):
            base_env = self.pomdp._env
        elif isinstance(self.pomdp, AtomicPropPOMDP):
            base_env = self.pomdp

        if base_env is None:
            raise ValueError(
                "BeliefSuppMDP requires a ProductPOMDP built from an "
                "AtomicPropPOMDP or an AtomicPropPOMDP directly."
            )

        num_base_states = len(base_env.states)

        # Initial belief: use base POMDP start states with initial automaton state
        if hasattr(self.pomdp, "_env"):
            initial_base_states = {
                k for k, v in base_env.start.items() if v > 0
            }
        else:
            initial_base_states = {
                k for k, v in base_env.start.items() if v > 0
            }

        st = tuple(
            sorted([(s_base, aut_init_idx) for s_base in initial_base_states])
        )

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

                # Iterate through belief pairs (base_state, aut_state)
                for s_base, q in current_bs:
                    # Map (s_base, q) to product state index to
                    # use product transitions
                    src_idx = q * num_base_states + s_base
                    if (src_idx in self.pomdp.transitions and
                            act_idx in self.pomdp.transitions[src_idx]):
                        trans_dict = self.pomdp.transitions[src_idx][act_idx]
                        for (dst_idx, o), prob in trans_dict.items():
                            if prob > 0:
                                # Decode (s', q') from product dst index
                                s_prime_base = dst_idx % num_base_states
                                q_prime = dst_idx // num_base_states
                                if o not in possible_succ_beliefs:
                                    possible_succ_beliefs[o] = set()
                                possible_succ_beliefs[o].add(
                                    (s_prime_base, q_prime)
                                )

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
    
    def _build_belief_support_mdp_parity(self):
        """
        Build belief-support MDP for ParityPOMDP case (direct parity, no automaton).
        
        The belief support is just a set of POMDP states (no automaton component).
        """
        # Initial belief: POMDP start states
        initial_states = {
            k for k, v in self.pomdp.start.items() if v > 0
        }
        
        st = tuple(sorted(initial_states))
        
        self.states = [st]
        self.statesinv = {st: 0}
        self.start = 0
        
        explore = deque([st])
        
        # Forward exploration
        while len(explore) > 0:
            current_bs = explore.popleft()
            current_bs_idx = self.statesinv[current_bs]
            self.trans[current_bs_idx] = {}
            
            for act_idx, act in enumerate(self.actions):
                self.trans[current_bs_idx][act_idx] = []
                
                # For each observation, collect successor belief support
                possible_succ_beliefs = {}  # obs_idx -> set of next states
                
                # Iterate through states in current belief support
                for s in current_bs:
                    if (s in self.pomdp.transitions and
                            act_idx in self.pomdp.transitions[s]):
                        trans_dict = self.pomdp.transitions[s][act_idx]
                        for (dst_s, o), prob in trans_dict.items():
                            if prob > 0:
                                if o not in possible_succ_beliefs:
                                    possible_succ_beliefs[o] = set()
                                possible_succ_beliefs[o].add(dst_s)
                
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
        Assign priority to each belief-support state based on the
        original POMDP's parity objective.
        
        Handles two cases:
        1. ProductPOMDP: Get priority from automaton state acceptance set
        2. ParityPOMDP: Get priority from POMDP's prioinv dict
        """
        if self.aut is not None:
            # ProductPOMDP case: Use automaton state priorities
            self._set_priorities_product()
        elif isinstance(self.pomdp, ParityPOMDP):
            # ParityPOMDP case: Use direct priorities from POMDP
            self._set_priorities_parity()
        else:
            raise ValueError(
                "Cannot determine priority assignment: either provide "
                "an automaton (ProductPOMDP) or use ParityPOMDP directly."
            )
    
    def _set_priorities_product(self):
        """
        Set priorities for ProductPOMDP case using automaton state acceptance.
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
    
    def _set_priorities_parity(self):
        """
        Set priorities for ParityPOMDP case using direct state priorities.
        """
        # POMDP has prioinv: state_idx -> priority
        for i, bs in enumerate(self.states):
            max_prio = 0
            if not bs:  # Handle empty belief supports
                self.prio[i] = 0
                continue
            
            for s in bs:
                if s in self.pomdp.prioinv:
                    prio = self.pomdp.prioinv[s]
                    if prio > max_prio:
                        max_prio = prio
            self.prio[i] = max_prio

    def prettyName(self, st):
        """Get pretty name for a belief support state."""
        base_env = (
            self.pomdp._env if hasattr(self.pomdp, "_env") else self.pomdp
        )
        return tuple([f"{base_env.states[s]}-{q}" for (s, q) in st])

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
            base_env = (
                self.pomdp._env if hasattr(self.pomdp, "_env") else self.pomdp
            )
            pairs = [f"{base_env.states[s]}-{q}" for (s, q) in st]
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
 