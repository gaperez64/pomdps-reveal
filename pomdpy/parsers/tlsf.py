import re


def parse_tlsf_file(tlsf_path):
    """
    Extract INPUTS and one-line GUARANTEES from the bundled TLSF format.

    Returns:
        dict with 'formulas' (list of LTL strings) and 'inputs' (list of proposition indices)
    """
    with open(tlsf_path, 'r') as f:
        content = f.read()

    # Remove line comments before finding sections, preserving quoted metadata.
    content = re.sub(
        r'"(?:\\.|[^"\\])*"|//[^\n]*',
        lambda match: match[0] if match[0].startswith('"') else '',
        content,
    )
    result = {'formulas': [], 'inputs': []}

    # Extract INPUTS section.
    inputs_match = re.search(r'INPUTS\s*\{([^}]+)\}', content, re.DOTALL)
    if inputs_match:
        input_matches = re.findall(r'p(\d+)\s*;', inputs_match.group(1))
        result['inputs'] = sorted(set(int(m) for m in input_matches))

    # Extract GUARANTEES section.
    guarantees_match = re.search(r'GUARANTEES\s*\{([^}]+)\}', content, re.DOTALL)
    if guarantees_match:
        for line in guarantees_match.group(1).splitlines():
            line = line.strip()
            if not line:
                continue
            if not line.endswith(';'):
                raise ValueError("TLSF guarantees must occupy one line and end with ';'")
            formula = line[:-1].strip()
            formula = formula.replace('&&', '&').replace('||', '|')
            formula = formula.replace('\\!', '!')
            formula = formula.replace('\\&', '&')
            formula = formula.replace('\\|', '|')
            if formula:
                result['formulas'].append(formula)

    return result
