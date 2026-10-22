"""
Almost-sure parity game solving algorithms for Belief-Support MDPs.

This module contains algorithms for solving parity games on MDPs,
particularly for computing almost-sure winning strategies and
maximal end components (MECs).
"""

from collections import defaultdict


class ParityMDPSolver:
    """
    Solver for parity objectives on MDPs (particularly Belief-Support MDPs).
    
    This class provides algorithms to:
    - Compute maximal end components (MECs)
    - Compute almost-sure reachable states
    - Compute almost-sure winning strategies for parity objectives
    """

    def __init__(self, mdp, verbose=False):
        """
        Initialize solver with an MDP.
        
        Args:
            mdp: Belief-Support MDP with attributes:
                - states: list of states
                - actions: list of actions
                - trans: dict mapping state -> action -> list of successors
                - prio: dict mapping state index -> priority
            verbose: If True, print detailed solver progress
        """
        self.mdp = mdp
        self.states = mdp.states
        self.actions = mdp.actions
        self.trans = mdp.trans
        self.prio = mdp.prio
        self.actionsinv = mdp.actionsinv
        self.verbose = verbose
        
        # Structures for reachability analysis
        self.pre = None  # Predecessor sets
        self.act = None  # Action counts

    def resetPreAct(self):
        """Build predecessor and action count structures."""
        self.pre = defaultdict(set)
        self.act = {}
        for i, _ in enumerate(self.states):
            self.act[i] = len(self.actions)
            for a, _ in enumerate(self.actions):
                preimage = (i, a)
                for dst in self.trans[i][a]:
                    self.pre[dst].add(preimage)

    def tarjanSCCs(self, states, actions):
        """
        Compute strongly connected components using Tarjan's algorithm.
        
        Args:
            states: Set of state indices to consider
            actions: Dict mapping state -> set of available actions
            
        Returns:
            List of SCCs (each SCC is a list of state indices)
        """
        assert len(states) > 0
        # A singleton is an SCC, whether or not it has a self-loop.
        if len(states) == 1:
            return [list(states)]
        idx = 0
        S = []
        m = max(states)
        stateIdx = [None] * (m + 1)
        stateLow = [None] * (m + 1)
        onStack = [False] * (m + 1)
        sccDecomp = []

        for s in states:
            if stateIdx[s] is not None:
                continue
            # Each DFS frame remembers where to resume its successor traversal.
            dfs = [(s, None)]
            while dfs:
                q, successors = dfs[-1]
                if successors is None:
                    stateIdx[q] = idx
                    stateLow[q] = idx
                    idx += 1
                    S.append(q)
                    onStack[q] = True

                    succ = set()
                    for a in actions[q]:
                        succ.update([dst for dst in self.trans[q][a]
                                     if dst in states])
                    successors = iter(succ)
                    dfs[-1] = (q, successors)

                for dst in successors:
                    if stateIdx[dst] is None:
                        dfs.append((dst, None))
                        break
                    elif onStack[dst] and stateIdx[dst] < stateLow[q]:
                        stateLow[q] = stateIdx[dst]
                else:
                    dfs.pop()
                    # Postprocess roots after all successors have been visited.
                    if stateIdx[q] == stateLow[q]:
                        scc = []
                        while True:
                            w = S.pop()
                            onStack[w] = False
                            scc.append(w)
                            if q == w:
                                break
                        sccDecomp.append(scc)

                    # Propagate the completed child's lowlink to its parent.
                    if dfs:
                        parent = dfs[-1][0]
                        if stateLow[q] < stateLow[parent]:
                            stateLow[parent] = stateLow[q]

        return sccDecomp

    def goodMECs(self, priority=0):
        """
        Compute Maximal End Components for given priority level.
        
        Implements Algorithm 47 from Baier & Katoen's "Principles of
        Model Checking".
        
        Args:
            priority: Even priority level to consider
            
        Returns:
            Tuple of (MECs, strategy_dict) where:
            - MECs: List of sets of state indices
            - strategy_dict: Dict mapping state -> set of valid actions
        """
        assert priority % 2 == 0
        self.resetPreAct()
        # An absent priority falls back to the nearest lower even priority.
        priority = max((p for p in self.prio.values() if p <= priority and p % 2 == 0),
                       default=None)
        if priority is None:
            return ([], {})
        return self._goodMECs(priority)

    def _goodMECs(self, priority):
        """Compute MECs for a present even priority using initialized predecessors."""
        if self.verbose:
            print(f"Computing MEC of priority {priority}")

        pre = self.pre
        new_act = {}

        for i, _ in enumerate(self.states):
            new_act[i] = {idx for _, idx in self.actionsinv.items()}

        # Get states with priority <= given priority.
        newMECs = [{i for i, _ in enumerate(self.states)
                    if self.prio[i] <= priority}]
        if self.verbose:
            print(f"Initial MEC: {newMECs}")

        MECs = []
        # Each refinement round takes its own snapshot before changing actions.
        act = new_act

        while (MECs != newMECs) or (act != new_act):
            if self.verbose:
                print(f"Refinement: \n{MECs=}, \n{newMECs=}, \n{act=}, "
                      f"\n{new_act=}")
            MECs = newMECs
            newMECs = []

            new_act = {s: actions.copy() for s, actions in act.items()}

            for mec in MECs:
                mec = set(mec)
                statesToRemove = set()

                # Split candidate MEC into SCCs
                SCCs = self.tarjanSCCs(mec, act)
                if self.verbose:
                    print(f"{SCCs=}")
                for C in SCCs:
                    C = set(C)
                    if self.verbose:
                        print(f"Considering SCC: {C}")
                    for s in C:
                        actsToRemove = []
                        for a in act[s]:
                            for dst in self.trans[s][a]:
                                if dst not in C:
                                    actsToRemove.append(a)
                                    if self.verbose:
                                        print(f"Removing action {a} from "
                                              f"state {s} (may get to {dst})")
                                    break
                        act[s] = act[s] - set(actsToRemove)
                        if len(act[s]) == 0:
                            statesToRemove.add(s)

                # Removing states
                while len(statesToRemove) > 0:
                    s = statesToRemove.pop()
                    mec.discard(s)
                    if self.verbose:
                        print(f"Removing state {s} completely")
                    if s in pre:
                        for t, a in pre[s]:
                            if t not in C:
                                continue
                            act[t].discard(a)
                            if len(act[t]) == 0 and t in mec:
                                statesToRemove.add(t)

                # Put split MEC candidates back in list
                for C in SCCs:
                    C = set(C)
                    res = C & mec
                    if len(res) == 0:
                        continue
                    if ((priority != 0) and
                        max([self.prio[o] for o in res]) < priority):
                        continue
                    newMECs.append(res)

        if self.verbose:
            print()
        return (MECs, act)

    def _goodMECs_up_to(self, max_priority):
        """Reuse fallback computations while keeping each returned result independent."""
        priorities = set(self.prio.values())
        results = []
        previous = ([], {})
        for priority in range(0, max_priority + 1, 2):
            if priority in priorities:
                previous = self._goodMECs(priority)
                results.append(previous)
            else:
                mecs, strategy = previous
                results.append(([mec.copy() for mec in mecs],
                                {s: actions.copy() for s, actions in strategy.items()}))
        return results

    def mecs(self):
        """Compute MECs for priority 0 and 2."""
        self.resetPreAct()
        return self._goodMECs_up_to(2)

    def cannotReach(self, targets, forbidden: set):
        """
        Find states that cannot reach any target state.
        
        Args:
            targets: Set of target state indices
            forbidden: Set of (state, action) pairs that cannot be used
            
        Returns:
            List of state indices that cannot reach targets
        """
        assert self.pre is not None
        visited = set(targets)
        tovisit = visited.copy()
        while len(tovisit) > 0:
            q = tovisit.pop()
            if q in self.pre:
                for i, a in self.pre[q]:
                    if (i, a) in forbidden:
                        continue
                    if i not in visited:
                        visited.add(i)
                        tovisit.add(i)
        return [i for i, _ in enumerate(self.states) if i not in visited]

    def almostSureReach(self, targets):
        """
        Compute almost-sure reachable states to targets.
        
        Implements Algorithm 45 from Baier & Katoen's "Principles of
        Model Checking".
        
        Args:
            targets: Set of target state indices
            
        Returns:
            List of state indices from which targets are almost-surely
            reachable
        """
        self.resetPreAct()
        return self._almostSureReach(targets)

    def _almostSureReach(self, targets):
        """Use the initialized predecessor map, pruning losing actions in place."""
        removed = set()
        U = set(self.cannotReach(targets, forbidden=set()))
        removed_sa_pairs: set = set()
        while True:
            R = set(U)
            while len(R) > 0:
                u = R.pop()
                if u in self.pre:
                    for t, a in self.pre[u]:
                        if t not in U and t not in removed:
                            if (t, a) not in removed_sa_pairs:
                                self.act[t] -= 1
                                removed_sa_pairs.add((t, a))
                            if self.act[t] == 0 and t not in targets:
                                R.add(t)
                                U.add(t)
                    del self.pre[u]
                removed.add(u)
            # Only process newly losing states in the next round.
            U = set(self.cannotReach(targets,
                                     forbidden=removed_sa_pairs)) - removed
            if len(U) == 0:
                break

        # Clean up removed transitions from predecessor sets
        for predecessors in self.pre.values():
            predecessors.difference_update(removed_sa_pairs)

        return [i for i, _ in enumerate(self.states) if i not in removed]

    def almostSureWin(self, max_priority=2, vis=None):
        """
        Compute almost-sure winning strategy for parity objective.
        
        Args:
            max_priority: Maximum priority to consider
            vis: Optional visualization filename
            
        Returns:
            Tuple of (reachable_states, reachability_strategy,
                     mec_strategies)
        """
        if self.verbose:
            print("\n=== Computing Almost-Sure Winning Strategy ===")
            print(f"Maximum priority to consider: {max_priority}")
            print(f"Even priorities (winning for Player 0): {list(range(0, max_priority + 1, 2))}")
            print()
            print("-- Step 1: Computing good MECs for each even priority --")
        
        # MEC computations only read predecessors; reachability then prunes them.
        self.resetPreAct()
        mecs_and_strats = self._goodMECs_up_to(max_priority)
        
        if self.verbose:
            print("\n-- Step 2: Extracting states from MECs --")
            for i, (mecs, strat) in enumerate(mecs_and_strats):
                priority = i * 2
                all_states = set().union(*mecs) if mecs else set()
                print(f"  Priority {priority}:")
                print(f"    Number of MECs: {len(mecs)}")
                print(f"    Total states covered: {len(all_states)}")
                if all_states:
                    print(f"    States: {sorted(all_states)}")
                print(f"    Strategy covers {len(strat)} states")
        
        states_and_strats = [
            (set().union(*mecs), strat)
            for (mecs, strat) in mecs_and_strats
        ]

        u = set().union(*(states for states, _ in states_and_strats))
        
        if self.verbose:
            print("\n-- Step 3: Computing almost-sure reachability to good MECs --")
            print(f"  Target set (union of all good MEC states): {sorted(u)}")
        
        r = self._almostSureReach(u)
        
        if self.verbose:
            print(f"  Almost-sure winning states: {sorted(r)}")
            print(f"  Total: {len(r)} out of {len(self.states)} states")
        
        # Get reachability strategy
        winning = set(r)
        reachStrat = {}
        for p in r:
            reachStrat[p] = []
            for a, _ in enumerate(self.actions):
                if all(q in winning for q in self.trans[p][a]):
                    reachStrat[p].append(a)
        
        if self.verbose:
            print("\n-- Step 4: Building reachability strategy --")
            print(f"  Strategy defined for {len(reachStrat)} states")
            for state in sorted(reachStrat.keys()):
                action_names = [self.actions[a] for a in reachStrat[state]]
                print(f"    State {state}: can use actions {reachStrat[state]} ({action_names})")
        
        # Clean MEC strategies
        for s, _ in enumerate(self.states):
            for states, strat in states_and_strats:
                if s in strat and s not in states:
                    del strat[s]
                if s in strat and len(strat[s]) == 0:
                    del strat[s]

        if self.verbose:
            print("\n-- Step 5: Cleaning MEC strategies --")
            for i, (states, strat) in enumerate(states_and_strats):
                priority = i * 2
                print(f"  Priority {priority}: strategy for {len(strat)} states")
                for state in sorted(strat.keys()):
                    action_names = [self.actions[a] for a in strat[state]]
                    print(f"    State {state}: can use actions {strat[state]} ({action_names})")

        # Visualize if needed
        mec_strats = [strat for (_, strat) in mecs_and_strats]
        if vis is not None:
            self.mdp.show(vis, r, reachStrat, mecs_and_strats=mec_strats)

        if self.verbose:
            print("\n=== Summary ===")
            print(f"Almost-sure winning states: {sorted(r)}")
            print(f"Number of states winning: {len(r)}/{len(self.states)}")
            print()
        
        return (r, reachStrat, mec_strats)

