from itertools import product
from pathlib import Path
from shutil import copymode
from tempfile import TemporaryDirectory
import numpy as np


def _format_probability_row(row, size):
    """Format a sparse row without repeatedly formatting missing zeros."""
    values = ["0"] * size
    for index, probability in row.items():
        if 0 <= index < size:
            values[index] = f"{probability:.14g}"
    return " ".join(values)


class POMDP:
    """
    Base class for POMDPs with common functionality.
    Subclasses define specific specification types.
    """
    def __init__(self):
        self.states = []
        self.actions = []
        self.obs = []

        self.start = {}  # state -> probability
        # dict(int, float)
        self.transitions = {}  # state x action -> Distribution(state x observation)
        # dict(int, dict(int, dict((int, int), float)))
        # Missing outcomes have probability zero.

        # inverse maps
        self.statesinv = {}  # state_name -> index
        self.actionsinv = {}  # action_name -> index
        self.obsinv = {}  # observation_name -> index

        # for simulation purposes: current state
        self.curstate = None

        # for parsing purposes:
        self.T = {}  # state x action -> Distribution(state)
        # dict(int, dict(int, dict(int, float)))
        self.O = {}  # action x new_state -> Distribution(observation)
        # dict(int, dict(int, dict(int, float)))
        # Missing entries in T and O also have probability zero.

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
        assert abs(sum(self.start.values()) - 1.0) < 1e-12
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
            if isinstance(exc, (str, int)):
                exc = [exc]
            excluded = {self._checkState(s)[0] for s in exc}
            inc = [i for i in range(len(self.states)) if i not in excluded]
        elif inc is not None:
            assert exc is None
            if isinstance(inc, (str, int)):
                inc = [inc]
            inc = [self._checkState(s)[0] for s in inc]
        else:
            assert False  # Both can't be true
        self.start.clear()
        for i in inc:
            self.start[i] = 1.0 / len(inc)

    ## Adding transitions
    def _addOneTrans(self, src, act, dst, p):
        if src not in self.T:
            self.T[src] = {}
        if act not in self.T[src]:
            self.T[src][act] = {}
        if p == 0:
            self.T[src][act].pop(dst, None)
        else:
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
            row = self.T.setdefault(s, {}).setdefault(a, {})
            if not row:
                # Empty rows need only their nonzero entries.
                row.update((i, p) for i, p in enumerate(matrix[start:end]) if p != 0)
            else:
                for succ, psucc in enumerate(matrix[start:end]):
                    self._addOneTrans(s, a, succ, psucc)
            start = (start + len(self.states)) % len(matrix)

    def _addOneIdentityTrans(self, act):
        for i, _ in enumerate(self.states):
            row = self.T.setdefault(i, {}).setdefault(act, {})
            row.clear()
            row[i] = 1.0

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
        if p == 0:
            self.O[act][dst].pop(o, None)
        else:
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
            row = self.O.setdefault(a, {}).setdefault(d, {})
            if not row:
                # Empty rows need only their nonzero entries.
                row.update((i, p) for i, p in enumerate(matrix[start:end]) if p != 0)
            else:
                for i, p in enumerate(matrix[start:end]):
                    self._addOneObs(a, d, i, p)
            start = (start + len(self.obs)) % len(matrix)

    # Computing self.transitions from self.T and self.O
    def computeTrans(self, force=False):
        """Compute joint transitions from T/O, keeping existing ones by default.

        Use force=True after editing T or O to rebuild existing joint transitions.
        Models with only joint transitions cannot be rebuilt from T/O.
        """
        assert len(self.states) > 0
        assert len(self.actions) > 0
        assert len(self.obs) > 0
        if force:
            if not self.T or not self.O:
                raise ValueError("Rebuilding joint transitions requires both T and O tables")
            self.transitions = {}
        if len(self.transitions) == 0 and len(self.T) > 0 and len(self.O) > 0:
            for s, _ in enumerate(self.states):
                self.transitions[s] = {}
                for a, _ in enumerate(self.actions):
                    row = {}
                    self.transitions[s][a] = row
                    for next_s, p_t in sorted(self.T.get(s, {}).get(a, {}).items()):
                        if p_t == 0:
                            continue
                        for obs, p_o in sorted(self.O.get(a, {}).get(next_s, {}).items()):
                            prob = p_t * p_o
                            if prob != 0:
                                row[next_s, obs] = prob

    def _pomdp_header_lines(self, comment="# POMDP file"):
        yield comment
        yield f"states: {' '.join(self.states)}"
        yield f"actions: {' '.join(self.actions)}"
        yield f"observations: {' '.join(self.obs)}"
        yield ""

    def _pomdp_lines(self):
        """Yield the shared POMDP format without buffering the full matrices."""
        # Products store only joint transitions, which this format cannot write directly.
        if not self.T or not self.O:
            raise ValueError(
                "POMDP export requires transition (T) and observation (O) tables; "
                "models with only joint transitions cannot be exported in this format."
            )
        yield from self._pomdp_header_lines()

        # Preserve the full initial distribution, including non-uniform weights.
        yield "start: " + _format_probability_row(self.start, len(self.states))
        yield ""

        # Write transitions in matrix format.
        for action_id, action in enumerate(self.actions):
            yield f"T:{action}"
            for state_id in range(len(self.states)):
                row = self.T.get(state_id, {}).get(action_id, {})
                yield _format_probability_row(row, len(self.states))
            yield ""

        # Write observations in matrix format.
        for action_id, action in enumerate(self.actions):
            yield f"O:{action}"
            observations = self.O.get(action_id, {})
            for next_state_id in range(len(self.states)):
                row = observations.get(next_state_id, {})
                yield _format_probability_row(row, len(self.obs))
            yield ""

    def to_pomdp_string(self):
        """Generate POMDP file content, including any objective declarations."""
        return "\n".join(self._pomdp_lines())

    def to_pomdp_file(self, filename):
        """Stream POMDP content, replacing the destination only on success."""
        filename = Path(filename).resolve()
        lines = self._pomdp_lines()
        # Keep temporary output on the same filesystem for atomic replacement.
        with TemporaryDirectory(prefix=".pomdp-export-", dir=filename.parent) as directory:
            temporary = Path(directory) / "output"
            with temporary.open('w') as f:
                # Use the same separators as to_pomdp_string(), without an extra newline.
                f.write(next(lines))
                f.writelines("\n" + line for line in lines)
            if filename.exists():
                copymode(filename, temporary)
            temporary.replace(filename)

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
                    str(self.states[s]) if s < len(self.states) else f"state_{s}"
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
        # The caller may pass one of the priority sets being updated.
        for state in tuple(states):
            state = state if ids else self.statesinv[state]
            previous = self.prioinv.get(state)
            if previous is not None and previous != priority:
                self.prio[previous].discard(state)
                if not self.prio[previous]:
                    del self.prio[previous]
            self.prio[priority].add(state)
            self.prioinv[state] = priority

    def _pomdp_header_lines(self):
        yield from super()._pomdp_header_lines("# POMDP file converted from PPS format\n")
        if self.prio:
            for prio in sorted(self.prio):
                state_names = [self.states[s] for s in sorted(self.prio[prio])]
                yield f"prio {prio}: {' '.join(state_names)}"
            yield ""


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
        formula = " & ".join(literals) or "1"
        return formula

    def _pomdp_header_lines(self):
        yield from super()._pomdp_header_lines("# POMDP file with atomic propositions")
        if self.atoms:
            for atom_id in sorted(self.atoms):
                obs_names = [self.obs[o] for o in sorted(self.atoms[atom_id])]
                yield f"atom {atom_id}: {' '.join(obs_names)}"
            yield ""
