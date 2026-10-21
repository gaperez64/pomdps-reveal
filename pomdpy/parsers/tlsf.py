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
    # Hide quoted text from section searches, keeping offsets into content.
    sections = re.sub(r'"(?:\\.|[^"\\])*"', lambda match: ' ' * len(match[0]), content)
    result = {'formulas': [], 'inputs': []}

    # Extract INPUTS section.
    inputs_match = re.search(r'\bINPUTS\s*\{([^{}]*)\}', sections, re.DOTALL)
    if inputs_match:
        input_matches = re.findall(r'p(\d+)\s*;', inputs_match.group(1))
        result['inputs'] = sorted(set(int(m) for m in input_matches))

    # Extract GUARANTEES section.
    guarantees_match = re.search(r'\bGUARANTEES\s*\{([^{}]*)\}', sections, re.DOTALL)
    if guarantees_match is None:
        raise ValueError("Missing or unterminated TLSF GUARANTEES section")
    guarantees = content[guarantees_match.start(1):guarantees_match.end(1)]
    for line in guarantees.splitlines():
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
