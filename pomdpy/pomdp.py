from itertools import product
import numpy as np

class POMDP:
    """
    Base class for POMDPs with common functionality.
    Subclasses define specific specification types.
    """
    def __init__(self):
        self.states = []
        self.actions = []
        self.obs = []

        self.start = {} # state -> probability
        # dict(int, float)
        self.transitions = {}  # state x action -> Distribution(state x observation)
        # dict(int, dict(int, dict((int, int), float)))

        # inverse maps
        self.statesinv = {} # state_name -> index
        self.actionsinv = {} # action_name -> index
        self.obsinv = {} # observation_name -> index

        # for simulation purposes: current state
        self.curstate = None

        # for parsing purposes:
        self.T = {}  # state x action -> Distribution(state)
        # dict(int, dict(int, dict(int, float)))
        self.O = {}  # action x new_state -> Distribution(observation)
        # dict(int, dict(int, dict(int, float)))

    def __repr__(self) -> str:
        lines = ["POMDP("]
        
        # Basic structure
        lines.append(f"  states={len(self.states)} {self.states}")
        lines.append(f"  actions={len(self.actions)} {self.actions}")
        lines.append(f"  observations={len(self.obs)} {self.obs}")
        
        # Start distribution
        if self.start:
            start_info = []
            for state_id, prob in self.start.items():
                state_name = (self.states[state_id] if state_id < len(self.states)
                             else f"state_{state_id}")
                start_info.append(f"{state_name}:{prob:.3f}")
            lines.append(f"  start_distribution={{{', '.join(start_info)}}}")
        else:
            lines.append("  start_distribution=None")
        
        # Transition structure info
        trans_count = sum(len(self.transitions.get(s, {}))
                         for s in range(len(self.states)))
        if trans_count > 0:
            total = len(self.states) * len(self.actions)
            lines.append(f"  transitions_defined={trans_count}/{total}")
        
        # Current state (if simulating)
        if self.curstate is not None:
            current_state_name = (self.states[self.curstate]
                                 if self.curstate < len(self.states)
                                 else f"state_{self.curstate}")
            lines.append(f"  current_state={current_state_name}")
        
        lines.append(")")
        return "\n".join(lines)

    def reset(self):
        assert len(self.states) > 0
        assert sum(self.start.values()) == 1.0
        self.curstate = np.random.choice(list(self.start.keys()), p=list(self.start.values()))

    def step(self, action):
        assert len(self.states) > 0
        assert self.curstate is not None
        distr = self.transitions[self.curstate][action]
        # check whether we have a distribution
        s = sum(distr.values())
        assert abs(s - 1.0) < 1e-6, f"Probabilities don't sum to 1: {s}"

        # Convert the distribution to lists for sampling
        outcomes = list(distr.keys())  # List of (state, observation) tuples
        probabilities = list(distr.values())
        
        # Sample from the distribution
        choice_idx = np.random.choice(len(outcomes), p=probabilities)
        new_state, observation = outcomes[choice_idx]

        self.curstate = new_state
        return self.obs[observation]

# PARSING-RELATED METHODS

    def setStates(self, ids):
        if isinstance(ids, int):
            ids = list(map(str, range(0, ids)))
        self.states = ids
        for i, s in enumerate(ids):
            self.statesinv[s] = i

    def setActions(self, ids):
        if isinstance(ids, int):
            ids = list(map(str, range(0, ids)))
        self.actions = ids
        for i, s in enumerate(ids):
            self.actionsinv[s] = i

    def setObs(self, ids):
        if isinstance(ids, int):
            ids = list(map(str, range(0, ids)))
        self.obs = ids
        for i, s in enumerate(ids):
            self.obsinv[s] = i

    def _checkState(self, src):
        # src can be None, str, or int
        if src is None:
            src = list(range(len(self.states)))
        elif not isinstance(src, int):
            src = [self.statesinv[src]]
        else:
            assert src >= 0 and src < len(self.states)
            src = [src]
        return src

    def _checkAct(self, act):
        # act can be None, str, or int
        if act is None:
            act = list(range(len(self.actions)))
        elif not isinstance(act, int):
            act = [self.actionsinv[act]]
        else:
            assert act >= 0 and act < len(self.actions)
            act = [act]
        return act

    def addPriority(self, priority, states, ids=False):
        """Override in ParityPOMDP subclass"""
        raise NotImplementedError(
            "addPriority should only be used with ParityPOMDP"
        )

    def setUniformStart(self, inc=None, exc=None):
        # inc = includes, exc = excludes
        assert len(self.states) > 0
        if inc is None and exc is None:
            inc = list(range(len(self.states)))
        elif exc is not None:
            assert inc is None
            if isinstance(exc, str):
                inc = [i for i, s in enumerate(self.states) if s != exc]
            else:  # it's a list then
                inc = [i for i, s in enumerate(self.states) if s not in exc]
        elif inc is not None:
            assert exc is None
            if isinstance(inc, str):
                inc = [self.statesinv[inc]]
            else:  # it's a list then
                inc = [self.statesinv[i] if isinstance(i, str)
                       else i
                       for i in inc]
        else:
            assert False  # Both can't be true
        for i in inc:
            self.start[i] = 1.0 / len(inc)

    ## Adding transitions
    def _addOneTrans(self, src, act, dst, p):
        if src not in self.T:
            self.T[src] = {}
        if act not in self.T[src]:
            self.T[src][act] = {}
        self.T[src][act][dst] = p

    def addTrans(self, matrix, act=None, src=None):
        assert len(self.states) > 0
        act = self._checkAct(act)
        src = self._checkState(src)
        assert len(src) != 1 or len(matrix) == len(self.states)
        assert len(src) == 1 or\
            len(matrix) == len(self.states) * len(self.states)
        start = 0
        for s, a in product(src, act):
            end = start + len(self.states)
            for succ, psucc in enumerate(matrix[start:end]):
                self._addOneTrans(s, a, succ, psucc)
            start = (start + len(self.states)) % len(matrix)

    def _addOneIdentityTrans(self, act):
        for i, _ in enumerate(self.states):
            self._addOneTrans(i, act, i, 1.0)

    def addIdentityTrans(self, act=None):
        assert len(self.states) > 0
        act = self._checkAct(act)
        for a in act:
            self._addOneIdentityTrans(a)

    def _addOneUniformTrans(self, act, src):
        for i, _ in enumerate(self.states):
            self._addOneTrans(src, act, i, 1.0 / len(self.states))

    def addUniformTrans(self, act=None, src=None):
        assert len(self.states) > 0
        act = self._checkAct(act)
        src = self._checkState(src)
        for a, s in product(act, src):
            self._addOneUniformTrans(a, s)

    ## Adding observations
    def _addOneObs(self, act, dst, o, p):
        if act not in self.O:
            self.O[act] = {}
        if dst not in self.O[act]:
            self.O[act][dst] = {}
        self.O[act][dst][o] = p

    def _addOneUniformObs(self, act, dst):
        for i, _ in enumerate(self.obs):
            self._addOneObs(act, dst, i, 1.0 / len(self.obs))

    def addUniformObs(self, act=None, dst=None):
        assert len(self.states) > 0
        act = self._checkAct(act)
        dst = self._checkState(dst)
        for a, s in product(act, dst):
            self._addOneUniformObs(a, s)

    def addObs(self, matrix, act=None, dst=None):
        assert len(self.states) > 0
        act = self._checkAct(act)
        dst = self._checkState(dst)
        assert len(dst) != 1 or len(matrix) == len(self.obs)
        assert len(dst) == 1 or\
            len(matrix) == len(self.obs) * len(self.states)
        start = 0
        for a, d in product(act, dst):
            end = start + len(self.obs)
            for i, p in enumerate(matrix[start:end]):
                self._addOneObs(a, d, i, p)
            start = (start + len(self.obs)) % len(matrix)

    # Computing self.transitions from self.T and self.O
    def computeTrans(self):
        assert len(self.states) > 0
        assert len(self.actions) > 0
        assert len(self.obs) > 0
        if len(self.transitions) == 0 and len(self.T) > 0 and len(self.O) > 0:
            for s, _ in enumerate(self.states):
                if s not in self.transitions:
                    self.transitions[s] = {}
                for a, _ in enumerate(self.actions):
                    if a not in self.transitions[s]:
                        self.transitions[s][a] = {}
                    for next_s, obs in product(range(len(self.states)), range(len(self.obs))):
                        if s in self.T and a in self.T[s] and next_s in self.T[s][a] and a in self.O and next_s in self.O[a] and obs in self.O[a][next_s]:
                            self.transitions[s][a][(next_s,obs)] = self.T[s][a][next_s] * self.O[a][next_s][obs]
                        else:
                            self.transitions[s][a][(next_s,obs)] = 0.0

    def to_pomdp_string(self):
        """
        Generate POMDP file content as a string in standard .pomdp format.
        
        Returns:
            str: The POMDP file content
        """
        lines = []
        
        # Write header
        lines.append("# POMDP file")
        
        # Write states
        lines.append(f"states: {' '.join(self.states)}")
        
        # Write actions
        lines.append(f"actions: {' '.join(self.actions)}")
        
        # Write observations
        lines.append(f"observations: {' '.join(self.obs)}")
        lines.append("")
        
        # Write start distribution
        start_probs = []
        for state_id in range(len(self.states)):
            prob = self.start.get(state_id, 0.0)
            start_probs.append(f"{prob:.10f}")
        lines.append("start: " + " ".join(start_probs))
        lines.append("")
        
        # Write transitions in matrix format
        for action_id in range(len(self.actions)):
            lines.append(f"T:{self.actions[action_id]}")
            
            for state_id in range(len(self.states)):
                row = []
                for next_state_id in range(len(self.states)):
                    prob = 0.0
                    if state_id in self.T and action_id in self.T[state_id]:
                        prob = self.T[state_id][action_id].get(next_state_id, 0.0)
                    row.append(f"{prob:.2f}")
                lines.append(" ".join(row))
            lines.append("")
        
        # Write observations in matrix format
        for action_id in range(len(self.actions)):
            lines.append(f"O:{self.actions[action_id]}")
            
            for next_state_id in range(len(self.states)):
                row = []
                for obs_id in range(len(self.obs)):
                    prob = 0.0
                    if action_id in self.O and next_state_id in self.O[action_id]:
                        prob = self.O[action_id][next_state_id].get(obs_id, 0.0)
                    row.append(f"{prob:.2f}")
                lines.append(" ".join(row))
            lines.append("")
        
        return "\n".join(lines)

    def to_pomdp_file(self, filename):
        """
        Write POMDP to a file in the standard .pomdp format.
        
        Args:
            filename: Output filename
        """
        content = self.to_pomdp_string()
        with open(filename, 'w') as f:
            f.write(content)

    # Adding atomic propositions to observations
    def addAtom(self, atom, observations, ids=False):
        """Override in AtomicPropPOMDP subclass"""
        raise NotImplementedError(
            "addAtom should only be used with AtomicPropPOMDP"
        )

    def generate_observation_formula(self, obs):
        """Override in AtomicPropPOMDP subclass"""
        raise NotImplementedError(
            "generate_observation_formula should only be used with "
            "AtomicPropPOMDP"
        )

    def _generate_dot(self):
        """Generate DOT graph representation"""
        lines = ["digraph POMDP {", "  rankdir=LR;"]
        
        # Add nodes
        for i, s in enumerate(self.states):
            lines.append(f'  {i} [label="{s}"];')
        
        # Add edges
        for src in range(len(self.states)):
            if src not in self.transitions:
                continue
            for a, act in enumerate(self.actions):
                if a not in self.transitions[src]:
                    continue
                for (dst, o), p in self.transitions[src][a].items():
                    if p > 0:
                        label = f"{act}, {self.obs[o]} : {p:.3f}"
                        lines.append(f'  {src} -> {dst} [label="{label}"];')
        
        lines.append("}")
        return "\n".join(lines)

    def to_str(self, fmt: str = 'dot') -> str:
        """Return a string representation in the requested format.

        Currently supports:
        - 'dot': Graphviz DOT of the POMDP graph
        """
        if fmt == 'dot':
            return self._generate_dot()
        raise ValueError(f"Unsupported format: {fmt}")

    def show(self, outfname):
        """Generate visualization and save to file."""
        # Generate DOT content and save to file
        dot_content = self._generate_dot()
        if outfname.endswith('.png') or outfname.endswith('.svg'):
            # Change extension to .dot if no pygraphviz
            outfname = outfname.rsplit('.', 1)[0] + '.dot'
        with open(outfname, 'w') as f:
            f.write(dot_content)


class MDP:
    """Base class for MDPs (fully observable, no observation model)."""
    def __init__(self):
        self.states = []
        self.actions = []
        self.start = {}  # state -> probability
        self.trans = {}  # state -> action -> list of successor states
        
        # inverse maps
        self.statesinv = {}  # state_name -> index
        self.actionsinv = {}  # action_name -> index

    def __repr__(self) -> str:
        lines = ["MDP("]
        lines.append(f"  states={len(self.states)}")
        lines.append(f"  actions={len(self.actions)} {self.actions}")
        if self.start:
            start_info = []
            for state_id, prob in self.start.items():
                state_name = (self.states[state_id] if state_id < len(self.states)
                             else f"state_{state_id}")
                start_info.append(f"{state_name}:{prob:.3f}")
            lines.append(f"  start={{{', '.join(start_info)}}}")
        lines.append(")")
        return "\n".join(lines)


class ParityMDP(MDP):
    """MDP with parity objective specified via priority function on states."""
    def __init__(self):
        super().__init__()
        self.prio = {}  # state_idx -> priority

    def __repr__(self) -> str:
        lines = super().__repr__().split('\n')
        if self.prio:
            prio_summary = {}
            for state_idx, prio in self.prio.items():
                if prio not in prio_summary:
                    prio_summary[prio] = []
                prio_summary[prio].append(state_idx)
            prio_info = []
            for prio in sorted(prio_summary.keys()):
                state_names = [
                    self.states[s] if s < len(self.states) else f"state_{s}"
                    for s in prio_summary[prio]
                ]
                prio_info.append(f"{prio}:[{', '.join(state_names)}]")
            lines.insert(-1, f"  priorities={{{', '.join(prio_info)}}}")
        return "\n".join(lines)


class ParityPOMDP(POMDP):
    """
    POMDP with parity objective specified via priority function on states.
    """
    def __init__(self):
        super().__init__()
        self.prio = {}  # int -> Set[state]
        # dict(int, Set[int])
        self.prioinv = {}  # state -> int

    def __repr__(self) -> str:
        lines = super().__repr__().split('\n')
        # Insert priority info before the closing parenthesis
        if self.prio:
            prio_info = []
            for prio, state_set in sorted(self.prio.items()):
                state_names = [
                    self.states[s] if s < len(self.states) else f"state_{s}"
                    for s in state_set
                ]
                prio_info.append(f"{prio}:{{{', '.join(state_names)}}}")
            lines.insert(-1, f"  priorities={{{', '.join(prio_info)}}}")
        return "\n".join(lines)

    def addPriority(self, priority, states, ids=False):
        if priority not in self.prio:
            self.prio[priority] = set()
        for state in states:
            self.prio[priority].add(state if ids else self.statesinv[state])
            self.prioinv[state if ids else self.statesinv[state]] = priority

class AtomicPropPOMDP(POMDP):
    """
    POMDP with atomic propositions on observations.
    Used for LTL specifications via observation-based formulas.
    """
    def __init__(self):
        super().__init__()
        self.atoms = {}  # atom -> Set[observation]
        # dict(int, Set[int])

    def __repr__(self) -> str:
        lines = super().__repr__().split('\n')
        # Insert atomic propositions before the closing parenthesis
        if self.atoms:
            atom_info = []
            for atom_id, obs_set in sorted(self.atoms.items()):
                obs_names = [
                    self.obs[o] if o < len(self.obs) else f"obs_{o}"
                    for o in obs_set
                ]
                atom_info.append(f"p{atom_id}:{{{', '.join(obs_names)}}}")
            lines.insert(-1, f"  atoms={{{', '.join(atom_info)}}}")
        return "\n".join(lines)

    def addAtom(self, atom, observations, ids=False):
        # print(f"Adding {atom} for {observations}")
        if atom not in self.atoms:
            self.atoms[atom] = set()
        for obs in observations:
            self.atoms[atom].add(obs if ids else self.obsinv[obs])

    def generate_observation_formula(self, obs, atoms_to_use=None):
        """
        Generate observation formula using specified atoms.
        
        Args:
            obs: Observation index
            atoms_to_use: Optional list of atom IDs to include.
                         If None, uses all atoms in self.atoms
        """
        if atoms_to_use is None:
            atoms_to_use = sorted(self.atoms.keys())
        else:
            atoms_to_use = sorted(atoms_to_use)
        
        literals = []
        for prop in atoms_to_use:
            if prop in self.atoms and obs in self.atoms[prop]:
                literals.append("p" + str(prop))
            else:
                literals.append("!p" + str(prop))
        formula = " & ".join(literals)
        return formula

    def to_pomdp_string(self):
        """
        Generate POMDP file content including atomic propositions.
        
        Returns:
            str: The POMDP file content with atoms section
        """
        lines = []
        
        # Write header
        lines.append("# POMDP file with atomic propositions")
        
        # Write states
        lines.append(f"states: {' '.join(self.states)}")
        
        # Write actions
        lines.append(f"actions: {' '.join(self.actions)}")
        
        # Write observations
        lines.append(f"observations: {' '.join(self.obs)}")
        lines.append("")
        
        # Write atomic propositions
        if self.atoms:
            for atom_id in sorted(self.atoms.keys()):
                obs_names = [
                    self.obs[o] for o in sorted(self.atoms[atom_id])
                ]
                lines.append(f"atom {atom_id}: {' '.join(obs_names)}")
            lines.append("")
        
        # Write start distribution
        start_probs = []
        for state_id in range(len(self.states)):
            prob = self.start.get(state_id, 0.0)
            start_probs.append(f"{prob:.10f}")
        lines.append("start: " + " ".join(start_probs))
        lines.append("")
        
        # Write transitions in matrix format
        for action_id in range(len(self.actions)):
            lines.append(f"T:{self.actions[action_id]}")
            
            for state_id in range(len(self.states)):
                row = []
                for next_state_id in range(len(self.states)):
                    prob = 0.0
                    if state_id in self.T and action_id in self.T[state_id]:
                        prob = self.T[state_id][action_id].get(
                            next_state_id, 0.0
                        )
                    row.append(f"{prob:.2f}")
                lines.append(" ".join(row))
            lines.append("")
        
        # Write observations in matrix format
        for action_id in range(len(self.actions)):
            lines.append(f"O:{self.actions[action_id]}")
            
            for next_state_id in range(len(self.states)):
                row = []
                for obs_id in range(len(self.obs)):
                    prob = 0.0
                    if (action_id in self.O and
                            next_state_id in self.O[action_id]):
                        prob = self.O[action_id][next_state_id].get(
                            obs_id, 0.0
                        )
                    row.append(f"{prob:.2f}")
                lines.append(" ".join(row))
            lines.append("")
        
        return "\n".join(lines)
