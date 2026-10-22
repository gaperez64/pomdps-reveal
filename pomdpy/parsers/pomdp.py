import lark

from pomdpy.pomdp import POMDP, ParityPOMDP, AtomicPropPOMDP


class TreeSimplifier(lark.Transformer):
    def string(self, s):
        (s,) = s
        return str(s)

    def prob(self, n):
        (n,) = n
        n = float(n)
        assert n >= 0 and n <= 1, f"Probability must be between 0 and 1: {n}"
        return n

    def start_state_list(self, states):
        return [int(s) if s.type == "INTTOK" else str(s) for s in states]

    ident_list = list

    def prob_matrix(self, rows):
        values = []
        for row in rows:
            if isinstance(row, lark.Token):
                values.extend(map(float, row.split()))
            else:
                values.append(row)
        for value in values:
            assert 0 <= value <= 1, f"Probability must be between 0 and 1: {value}"
        return values


# This parses POMDP files with no discount factor
# and no rewards. The formal specification is a
# simplification of what you will find here:
#
# https://www.pomdp.org/code/pomdp-file-grammar.html

parser = lark.Lark(
    r"""
    pomdp_file      : preamble start_state param_list

    preamble        : param_type*

    ?param_type     : state_param | action_param | obs_param | prio_param | atom_param

    state_param     : "states" ":" state_tail

    ?state_tail     : int | ident_list

    action_param    : "actions" ":" action_tail

    ?action_tail    : int | ident_list

    obs_param       : "observations" ":" obs_param_tail

    ?obs_param_tail : int | ident_list

    prio_param : prio_entry+
    
    prio_entry : "prio" int ":" ident_list -> prio
                   
    atom_param : atom_entry+

    atom_entry : "atom" int ":" ident_list -> atom

    start_state     : "start" ":" u_matrix                   -> stmatrix
                    | "start" ":" string                     -> ststate
                    | "start" "include" ":" start_state_list -> include
                    | "start" "exclude" ":" start_state_list -> exclude

    // Keep indices distinct from probabilities, including after a state name.
    start_state_list: (INTTOK | STRINGTOK)+

    param_list      : param_spec*

    ?param_spec     : trans_prob_spec | obs_prob_spec

    ?trans_prob_spec: _T_HEADER trans_spec_tail

    trans_spec_tail : id ":" id ":" id prob -> trans_entry
                    | id ":" id u_matrix    -> trans_row
                    | id ui_matrix          -> trans_matrix

    ?obs_prob_spec  : _O_HEADER obs_spec_tail

    obs_spec_tail   : id ":" id ":" id prob
                    | id ":" id u_matrix
                    | id u_matrix           -> obs_matrix

    ?ui_matrix      : "uniform"    -> uniform
                    | "identity"   -> identity
                    | prob_matrix

    ?u_matrix       : "uniform"    -> uniform
                    | "reset"      -> reset
                    | prob_matrix

    prob_matrix     : (prob | PROB_ROW)+

    ?id             : int
                    | string
                    | "*"    -> asterisk

    ident_list      : string+

    prob            : int | float

    ?int            : INTTOK

    string          : STRINGTOK

    ?float          : FLOATTOK

    // Include the colon so that plain T and O can also be identifiers.
    _T_HEADER.2: /T(\s|#[^\n]*\n)*:/
    _O_HEADER.2: /O(\s|#[^\n]*\n)*:/
    // Batch two or more probabilities, leaving single values to the existing tokens.
    PROB_ROW: FLOATTOK (/[ \t]+/ FLOATTOK)+
    INTTOK: /0|[1-9][0-9]*/
    FLOATTOK: /(?:[0-9]+\.[0-9]*|\.[0-9]+|[0-9]+)(?:[eE][+-]?[0-9]+)?/
    STRINGTOK: /[a-zA-Z]([a-zA-Z0-9]|[\_\-])*/
    COMMENT: "#" /[^\n]*/ "\n"

    %import common.WS
    %ignore WS
    %ignore COMMENT
    """,
    start="pomdp_file",
    parser="lalr",
    # Simplify during parsing to avoid building a tree node per probability
    # and traversing the complete matrix data again afterwards.
    transformer=TreeSimplifier(),
)


class TreeToSets(lark.Visitor):
    def __init__(self, pomdp):
        self.pomdp = pomdp

    def state_param(self, tree):
        ids = tree.children[0]
        self.pomdp.setStates(int(ids) if isinstance(ids, lark.Token) else ids)

    def action_param(self, tree):
        ids = tree.children[0]
        self.pomdp.setActions(int(ids) if isinstance(ids, lark.Token) else ids)

    def obs_param(self, tree):
        ids = tree.children[0]
        self.pomdp.setObs(int(ids) if isinstance(ids, lark.Token) else ids)


class TreeToProbs(lark.Visitor):
    def __init__(self, pomdp):
        self.pomdp = pomdp

    def stmatrix(self, tree):
        child = tree.children[0]
        if isinstance(child, lark.Tree) and child.data == "uniform":
            self.pomdp.setUniformStart()
        elif isinstance(child, lark.Tree) and child.data == "reset":
            # Reset means start in state 0
            self.pomdp.start = {0: 1.0}
        elif isinstance(child, list):
            # Handle explicit probability distribution
            # TreeSimplifier converted prob_matrix to list of floats
            probs = child
            self.pomdp.start = {
                i: prob for i, prob in enumerate(probs) if prob > 0
            }
        else:
            raise ValueError(f"Unexpected start state format: {child}")

    def include(self, tree):
        child = tree.children[0]
        self.pomdp.setUniformStart(inc=child)

    def exclude(self, tree):
        child = tree.children[0]
        self.pomdp.setUniformStart(exc=child)

    def trans_matrix(self, tree):
        (action, matrix) = tree.children
        if isinstance(action, lark.Tree) and action.data == "asterisk":
            action = None
        if isinstance(matrix, lark.Tree):
            if matrix.data == "uniform":
                self.pomdp.addUniformTrans(act=action)
            elif matrix.data == "identity":
                self.pomdp.addIdentityTrans(act=action)
            else:
                assert False
        else:
            self.pomdp.addTrans(matrix, act=action)

    def obs_matrix(self, tree):
        (action, matrix) = tree.children
        if isinstance(action, lark.Tree) and action.data == "asterisk":
            action = None
        if isinstance(matrix, lark.Tree):
            if matrix.data == "uniform":
                self.pomdp.addUniformObs(act=action)
            else:
                assert False
        else:
            self.pomdp.addObs(matrix, act=action)

    def prio(self, tree):
        priority, states = tree.children
        self.pomdp.addPriority(int(priority), states)

    def atom(self, tree):
        atom_idx, observations = tree.children
        self.pomdp.addAtom(int(atom_idx), observations)


def parse(instr):
    ast = parser.parse(instr)

    # Detect which type of POMDP based on content
    has_prio = next(ast.find_data('prio'), None) is not None
    has_atoms = next(ast.find_data('atom'), None) is not None

    # Create appropriate POMDP subclass
    if has_prio and has_atoms:
        raise ValueError(
            "POMDP file contains both priorities and atoms. "
            "Please use either priorities (ParityPOMDP) or "
            "atoms (AtomicPropPOMDP), not both."
        )
    elif has_prio:
        pomdp = ParityPOMDP()
    elif has_atoms:
        pomdp = AtomicPropPOMDP()
    else:
        # No specification, use base POMDP
        pomdp = POMDP()
    
    TreeToSets(pomdp).visit(ast)
    TreeToProbs(pomdp).visit(ast)
    
    # Compute the transitions dictionary from T and O
    pomdp.computeTrans()
    
    return pomdp
