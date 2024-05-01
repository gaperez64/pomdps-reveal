import lark

# This parses POMDP files with no discount factor
# and no rewards. The formal specification is a
# simplication of what you will find here:
#
# https://www.pomdp.org/code/pomdp-file-grammar.html

parser = lark.Lark(r"""
    pomdp_file      : preamble start_state param_list

    preamble        : preamble param_type | // empty

    param_type      : state_param | action_param | obs_param

    state_param     : STATESTOK COLONTOK state_tail

    state_tail      : INTTOK | ident_list

    action_param    : ACTIONSTOK COLONTOK action_tail

    action_tail     : INTTOK | ident_list

    obs_param       : OBSERVATIONSTOK COLONTOK obs_param_tail

    obs_param_tail  : INTTOK | ident_list

    start_state     : STARTTOK COLONTOK u_matrix
                    | STARTTOK COLONTOK STRINGTOK
                    | STARTTOK INCLUDETOK COLONTOK start_state_list
                    | STARTTOK EXCLUDETOK COLONTOK start_state_list
                    |  // empty

    start_state_list: start_state_list state | state

    param_list      : param_list param_spec | // empty

    param_spec      : trans_prob_spec | obs_prob_spec

    trans_prob_spec : TTOK COLONTOK trans_spec_tail

    trans_spec_tail : paction COLONTOK state COLONTOK state prob
                    | paction COLONTOK state u_matrix
                    |  paction ui_matrix

    obs_prob_spec   : OTOK COLONTOK obs_spec_tail

    obs_spec_tail   : paction COLONTOK state COLONTOK obs prob
                    | paction COLONTOK state u_matrix
                    | paction u_matrix

    ui_matrix       : UNIFORMTOK | IDENTITYTOK | prob_matrix

    u_matrix        : UNIFORMTOK | RESETTOK | prob_matrix

    prob_matrix     : prob_matrix prob | prob

    state           : INTTOK | STRINGTOK | ASTERICKTOK

    paction         : INTTOK | STRINGTOK | ASTERICKTOK

    obs             : INTTOK | STRINGTOK | ASTERICKTOK

    ident_list      : ident_list STRINGTOK | STRINGTOK

    prob            : INTTOK | FLOATTOK

    DISCOUNTTOK: "discount"
    VALUESTOK: "values"
    STATESTOK: "states"
    ACTIONSTOK: "actions"
    OBSERVATIONSTOK: "observations"
    TTOK: "T"
    OTOK: "O"
    RTOK: "R"
    UNIFORMTOK: "uniform"
    IDENTITYTOK: "identity"
    STARTTOK: "start"
    INCLUDETOK: "include"
    EXCLUDETOK: "exclude"
    RESETTOK: "reset"
    COLONTOK: ":"
    ASTERICKTOK: "*"
    INTTOK: /0 | [1-9][0-9]*'/
    FLOATTOK: /([0-9]+ \. [0-9]* | \. [0-9]+ | [0-9]+ ) ([eE] [+-]? [0-9]+)?/
    STRINGTOK: /[a-zA-Z] ( [a-zA-Z0-9] | [\_\-] )*/

    %import common.WS
    %ignore WS
    """, start="pomdp_file", parser="lalr")
