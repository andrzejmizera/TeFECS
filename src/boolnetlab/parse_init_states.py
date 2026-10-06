import re

_ASSIGN_RE = re.compile(r"M\.(\w+)\s*=\s*(true|false)")
_BLOCK_RE = re.compile(r"(\w+)\s*\n+(.*?);\s*\n+end\s+\1", re.DOTALL)


def _convert_assignment(match: re.Match) -> str:
    var, value = match.group(1), match.group(2)
    return var if value == "true" else f"~{var}"


def convert_expression(expr: str) -> str:
    """Convert a `M.v_NAME=true/false` expression into a Boolean string."""
    expr = _ASSIGN_RE.sub(_convert_assignment, expr)
    expr = re.sub(r"\s+", " ", expr).strip()
    expr = re.sub(r"\band\b", "&", expr)
    expr = re.sub(r"\bor\b", "|", expr)
    return expr


def parse_init_states_block(text: str) -> str:
    """Extract and convert a `<Name> ... end <Name>` boolean block."""
    match = _BLOCK_RE.search(text)
    if not match:
        raise ValueError("No '<Name> ... end <Name>' block found in input")
    return convert_expression(match.group(2))


def parse_target_states_block(text: str) -> str:
    return parse_init_states_block(text)


if __name__ == "__main__":
    example = """InitStates
		M.IL27RA=false and M.IL27_e=false and M.GP130=false and M.Galpha_QL=false and M.IL2RB=false and M.CGC=false and M.Galpha_iL=false and M.MHC_II=false and M.APC=false and M.IL18_e=false and M.IL9_e=false and M.IFNB_e=false and M.ECM=false and M.IL21_e=false and M.alpha_13L=false and M.IL10RA=false and M.IL10RB=false and M.IL10_e=false and M.IL15_e=false and M.B7=false and M.IFNGR1=false and M.IFNGR2=false and M.IFNG_e=false and (M.CAV1_ACTIVATOR=false or M.CAV1_ACTIVATOR=true) and M.GalphaS_L=false and M.IL4_e=false and M.IL6_e=false and M.IL6RA=false and M.TGFB_e=false and M.IL22_e=false and M.IL2_e=false and M.IL23_e=false and M.IL15RA=false and M.IL12_e=false;
end InitStates
"""
    print(parse_init_states_block(example))

