"""CNJ case-number parsing and DataJud tribunal alias derivation.

CNJ unified numbering (Resolution 65/2008): NNNNNNN-DD.AAAA.J.TR.OOOO
- J: judiciary segment (8 = state, 5 = labor, 4 = federal, 3 = STJ, ...)
- TR: tribunal within the segment (for state/electoral courts, the UF code)

The DataJud endpoint is per-tribunal: /api_publica_{alias}/_search.
"""

import re

CNJ_PATTERN = re.compile(r"(\d{7})-?(\d{2})\.?(\d{4})\.?(\d)\.?(\d{2})\.?(\d{4})")

# TR code -> UF for state courts (segment 8) and electoral (segment 6)
UF_BY_TR = {
    1: "ac", 2: "al", 3: "ap", 4: "am", 5: "ba", 6: "ce", 7: "df", 8: "es",
    9: "go", 10: "ma", 11: "mt", 12: "ms", 13: "mg", 14: "pa", 15: "pb",
    16: "pr", 17: "pe", 18: "pi", 19: "rj", 20: "rn", 21: "rs", 22: "ro",
    23: "rr", 24: "sc", 25: "se", 26: "sp", 27: "to",
}


def normalize_case_number(raw: str) -> str | None:
    """Return the 20-digit CNJ number, or None if it doesn't parse."""
    match = CNJ_PATTERN.search(raw)
    if not match:
        return None
    return "".join(match.groups())


def tribunal_alias(case_number: str) -> str | None:
    """Derive the DataJud alias (e.g. 'tjsp', 'trt15') from a CNJ number."""
    normalized = normalize_case_number(case_number)
    if normalized is None:
        return None
    segment = int(normalized[13])
    tr = int(normalized[14:16])

    if segment == 8:  # state courts
        uf = UF_BY_TR.get(tr)
        if uf is None:
            return None
        return "tjdft" if uf == "df" else f"tj{uf}"
    if segment == 5:  # labor
        return f"trt{tr}" if 1 <= tr <= 24 else None
    if segment == 4:  # federal
        return f"trf{tr}" if 1 <= tr <= 6 else None
    if segment == 6:  # electoral
        uf = UF_BY_TR.get(tr)
        return f"tre-{uf}" if uf else None
    if segment == 3:
        return "stj"
    return None  # unsupported segment - enrichment is skipped, not failed
