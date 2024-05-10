import lark


from pomdpy.pomdp import POMDP

# This parses POMDP files with no discount factor
# and no rewards. The formal specification is a
# simplication of what you will find here:
#
# https://www.pomdp.org/code/pomdp-file-grammar.html

parser = lark.Lark(r"""
    pomdp_file      : preamble start_state param_list

    preamble        : param_type*

    ?param_type     : state_param | action_param | obs_param

    state_param     : "states" ":" state_tail

    ?state_tail     : int | ident_list

    action_param    : "actions" ":" action_tail

    ?action_tail    : int | ident_list

    obs_param       : "observations" ":" obs_param_tail

    ?obs_param_tail : int | ident_list

    start_state     : "start" ":" u_matrix                   -> stmatrix
                    | "start" ":" string                     -> ststate
                    | "start" "include" ":" start_state_list -> include
                    | "start" "exclude" ":" start_state_list -> exclude

    start_state_list: id+

    param_list      : param_spec*

    ?param_spec     : trans_prob_spec | obs_prob_spec

    ?trans_prob_spec: "T" ":" trans_spec_tail

    trans_spec_tail : id ":" id ":" id prob -> trans_entry
                    | id ":" id u_matrix    -> trans_row
                    | id ui_matrix          -> trans_matrix

    ?obs_prob_spec  : "O" ":" obs_spec_tail

    obs_spec_tail   : id ":" id ":" id prob
                    | id ":" id u_matrix
                    | id u_matrix           -> obs_matrix

    ?ui_matrix      : "uniform"    -> uniform
                    | "identity"   -> identity
                    | prob_matrix

    ?u_matrix       : "uniform"    -> uniform
                    | "reset"      -> reset
                    | prob_matrix

    prob_matrix     : prob+

    ?id             : int
                    | string
                    | "*"    -> asterisk

    ident_list      : string+

    prob            : int | float

    ?int            : INTTOK

    string          : STRINGTOK

    ?float          : FLOATTOK

    INTTOK: /0|[1-9][0-9]*'/
    FLOATTOK: /([0-9]+\.[0-9]*|\.[0-9]+|[0-9]+)([eE][+-]?[0-9]+)?/
    STRINGTOK: /[a-zA-Z]([a-zA-Z0-9]|[\_\-])*/
    COMMENT: "#" /[^\n]*/ "\n"

    %import common.WS
    %ignore WS
    %ignore COMMENT
    """, start="pomdp_file", parser="lalr")


class TreeSimplifier(lark.Transformer):
    def string(self, s):
        (s,) = s
        return s[:]  # copies the string

    def prob(self, n):
        (n,) = n
        n = float(n)
        assert n >= 0 and n <= 1
        return n

    start_state_list = list
    ident_list = list
    prob_matrix = list


class TreeToSets(lark.Visitor):
    def __init__(self, pomdp):
        self.pomdp = pomdp

    def state_param(self, tree):
        self.pomdp.setStates(tree.children[0])

    def action_param(self, tree):
        self.pomdp.setActions(tree.children[0])

    def obs_param(self, tree):
        self.pomdp.setObs(tree.children[0])


class TreeToProbs(lark.Visitor):
    def __init__(self, pomdp):
        self.pomdp = pomdp

    def stmatrix(self, tree):
        child = tree.children[0]
        assert isinstance(child, lark.Tree)
        if child.data == "uniform":
            self.pomdp.setUniformStart()
        else:
            assert False

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


def parse(instr):
    cst = parser.parse(instr)
    ast = TreeSimplifier().transform(cst)
    pomdp = POMDP()
    TreeToSets(pomdp).visit(ast)
    TreeToProbs(pomdp).visit(ast)
    return pomdp
