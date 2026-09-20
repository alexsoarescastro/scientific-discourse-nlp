import re, numpy as np, pandas as pd

NUM = r"([-+]?\d+(?:\.\d+)?)"

PATTERNS = {
    "current_density": [
        rf"{NUM}\s*(?:mA|A)\s*(?:m[-−]?\s*2|/m2|m\^-?2)",
    ],
    "power_density": [
        rf"{NUM}\s*(?:mW|W)\s*(?:m[-−]?\s*2|/m2|m\^-?2)",
    ],
    "applied_potential": [
        rf"(?:applied potential|applied voltage|cell voltage)[^\d+-]{{0,30}}{NUM}\s*(mV|V)",
    ],
    "hydrogen_efficiency": [
        rf"(?:hydrogen|H2)[^\n.%]{{0,50}}{NUM}\s*%",
    ],
    "pH": [
        rf"\bpH\s*(?:=|of|:)?\s*{NUM}\b",
    ],
}

SUBSTRATES = ["acetate","glycerol","glucose","wastewater","lactate","ethanol","co2","carbon dioxide"]

def first_numeric(text, patterns):
    for p in patterns:
        m = re.search(p, text, flags=re.I)
        if m:
            try: return float(m.group(1))
            except: pass
    return np.nan

def substrate_type(text):
    low = text.lower()
    found = [s for s in SUBSTRATES if s in low]
    return ";".join(found) if found else np.nan

def add_experimental_features(df):
    out = df.copy()
    for k,pats in PATTERNS.items():
        out[k] = out["text_minimal"].map(lambda x: first_numeric(x,pats))
    out["substrate_type"] = out["text_minimal"].map(substrate_type)
    # Missing values remain missing: no high/low contextual proxy.
    return out
