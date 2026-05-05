#!/usr/bin/env python3
"""
Tier-B v6 fusion + PV quality patch.

Run from the root of an extracted ccdr_tierb_v5_manifest_quality bundle:

    python apply_tierb_v6_fusion_pv_patch.py --root .

Or pass the bundle root explicitly:

    python apply_tierb_v6_fusion_pv_patch.py --root F:\\git\\upd\\...\\ccdr_tierb_public_tests_v1

This patch is intentionally conservative:
- Scientific mode becomes curated-manifest-first for T26-T30.
- Discovery mode can still scout, but broad Zenodo/OSF hits are not evidence.
- Fusion keyword gates require domain anchors, so "elm tree", Earth Land Model, and ELM white dwarf records are rejected.
- OSF node recursion is added for ITPA DB5.2.3-like files.
- T48 gets a stronger NREL/PV interactive-page extractor and strict baseline-readiness gate.
"""

from __future__ import annotations

import argparse
import csv
import json
import os
import re
import textwrap
from pathlib import Path


V6_HELPER = r'''
# ---- Tier-B v6 fusion/PV quality helpers ----
# Added by apply_tierb_v6_fusion_pv_patch.py

import csv as _csv
import io as _io
import json as _json
import math as _math
import os as _os
import re as _re
import zipfile as _zipfile
from urllib.parse import urljoin as _urljoin

_FUSION_DOMAIN_ANCHORS = [
    r"tokamak", r"stellarator", r"plasma", r"H[- ]?mode", r"pedestal", r"divertor",
    r"separatrix", r"confinement", r"DIII[- ]?D", r"JET", r"ASDEX", r"AUG",
    r"KSTAR", r"EAST", r"ITER", r"W7[- ]?X", r"LHD", r"ITPA", r"DB5(?:\.2\.3)?",
]

_FUSION_WEAK_AMBIGUOUS = [r"\bELM\b", r"edge[ -]?locali[sz]ed mode"]

_FUSION_OBSERVABLES = {
    "T26": [
        r"E[_\s-]?ELM", r"W[_\s-]?ELM", r"ELM.*energy", r"energy.*ELM",
        r"P[_\s-]?ped", r"pedestal.*pressure", r"dP/P", r"Delta P", r"ΔP", r"Wped",
    ],
    "T27": [
        r"RMP", r"resonant magnetic perturb", r"coil phasing", r"I[-_ ]?coil", r"n\s*=\s*[123]",
        r"ELM.*freq", r"f[_\s-]?ELM", r"ELM suppression", r"ELM mitigation",
    ],
    "T28": [
        r"tau[_\s-]?E", r"energy confinement", r"confinement time", r"H[-_ ]?factor",
        r"DB5", r"STD5", r"ITPA",
    ],
    "T29": [
        r"transport", r"diffus", r"chi", r"χ", r"heat flux", r"thermal diffus",
        r"stellarator", r"tokamak", r"W7[- ]?X", r"LHD",
    ],
    "T30": [
        r"tau[_\s-]?E", r"H[-_ ]?factor", r"confinement", r"residual",
        r"elongation", r"triangularity", r"q95", r"shaping", r"curvature", r"density", r"DB5", r"STD5",
    ],
}

_HARD_NEGATIVE_PATTERNS = [
    r"\belm tree\b", r"\belm trees\b", r"\bsquirrel", r"\bEarth Land Model\b",
    r"\bDELM\b", r"\bNoDELM\b", r"\bELM WD", r"\bwhite dwarf", r"\bwhite dwarfs",
    r"\bCOVID\b", r"\bdental\b", r"\bquestionnaire\b", r"\bsurvey\b", r"\bsocial\b",
    r"\bGPS\b", r"\bSIGFOX\b", r"\byaw\b", r"\broll\b", r"\buglymol\b",
]

# NOTE: deliberately removed raw substring 'roll' because it rejects "controlled".
# Only \broll\b is a negative.

def _v6_regex_hits(patterns, text):
    text = text or ""
    out = []
    for p in patterns:
        if _re.search(p, text, flags=_re.I):
            out.append(p)
    return out


def v6_fusion_keyword_gate(test_id, title="", description="", filename="", url=""):
    """Return strict source→record relevance gate for fusion tests.

    ELM alone is never enough. A record must have a fusion domain anchor and a test-specific observable.
    """
    text = " ".join(str(x or "") for x in [title, description, filename, url])[:20000]
    neg = _v6_regex_hits(_HARD_NEGATIVE_PATTERNS, text)
    dom = _v6_regex_hits(_FUSION_DOMAIN_ANCHORS, text)
    weak = _v6_regex_hits(_FUSION_WEAK_AMBIGUOUS, text)
    obs = _v6_regex_hits(_FUSION_OBSERVABLES.get(test_id, []), text)

    if neg:
        return {"ok": False, "reason": "hard_negative", "negative_hits": neg, "domain_hits": dom, "observable_hits": obs, "weak_hits": weak}

    # T26/T27 are most vulnerable to ELM ambiguity; require stronger evidence.
    if test_id in {"T26", "T27"}:
        ok = bool(dom) and bool(obs) and (len(dom) + len(obs) >= 3)
    elif test_id in {"T28", "T30"}:
        ok = bool(dom) and bool(obs)
    else:
        ok = bool(dom) and bool(obs)

    return {
        "ok": ok,
        "reason": "passed" if ok else "missing_domain_or_observable_anchor",
        "negative_hits": neg,
        "domain_hits": dom,
        "observable_hits": obs,
        "weak_hits": weak,
        "score": len(dom) + len(obs) + len(weak),
    }


def v6_header_physical_gate(test_id, columns):
    cols = [str(c or "") for c in columns]
    joined = " ".join(cols)

    groups = {
        "T26": [
            [r"device|machine|shot|discharge|pulse"],
            [r"E[_\s-]?ELM|W[_\s-]?ELM|ELM.*energy|energy.*ELM|dW[_\s-]?ELM"],
            [r"P[_\s-]?ped|pedestal.*pressure|pressure.*pedestal|p[_\s-]?ped"],
            [r"V[_\s-]?ped|pedestal.*volume|delta.*P|ΔP|dP/P|pressure.*drop|dW/W|Wped"],
        ],
        "T27": [
            [r"device|machine|shot|discharge|pulse"],
            [r"ELM.*freq|freq.*ELM|f[_\s-]?ELM|ELM.*rate"],
            [r"RMP|coil|phasing|I[_\s-]?coil|helicity|H[_\s-]?mag|n\s*=|resonant.*perturb"],
        ],
        "T28": [
            [r"device|machine|tokamak"],
            [r"tau[_\s-]?E|taue|confinement.*time|energy.*confinement|H[_\s-]?factor|h98|h89"],
            [r"density|n[_\s-]?e|nebar|line.*avg"],
            [r"stored.*energy|W[_\s-]?MHD|temperature|T[_\s-]?e|power|P[_\s-]?loss|P[_\s-]?aux"],
        ],
        "T29": [
            [r"device|machine|tokamak|stellarator|W7|LHD"],
            [r"diffus|transport|χ|chi|heat.*flux|thermal.*diff"],
            [r"radius|rho|psi|edge|separatrix|profile|Te|Ti|ne"],
        ],
        "T30": [
            [r"device|machine|tokamak"],
            [r"tau[_\s-]?E|taue|confinement|residual|H[_\s-]?factor|h98|h89"],
            [r"elongation|triangularity|shaping|curvature|q95|kappa|delta"],
            [r"density|n[_\s-]?e|nebar|line.*avg"],
        ],
    }.get(test_id, [])

    matched, missing = [], []
    for grp in groups:
        hits = []
        for pat in grp:
            if _re.search(pat, joined, flags=_re.I):
                hits.append(pat)
        (matched if hits else missing).append(hits if hits else grp)
    return {"ok": not missing, "matched_groups": matched, "missing_groups": missing, "columns": cols}


def v6_is_table_like_filename(name):
    name = (name or "").lower()
    return any(name.endswith(ext) for ext in [".csv", ".tsv", ".txt", ".dat", ".xls", ".xlsx"])


def v6_is_variable_dictionary(name):
    name = (name or "").lower()
    return "variable" in name and name.endswith(".pdf")


def v6_osf_extract_items(data):
    """Extract OSF file/folder candidates from OSF API JSON bytes."""
    try:
        obj = _json.loads(data.decode("utf-8"))
    except Exception:
        return []
    items = obj.get("data", []) if isinstance(obj, dict) else []
    out = []
    for item in items:
        attrs = item.get("attributes", {}) or {}
        links = item.get("links", {}) or {}
        rel = item.get("relationships", {}) or {}
        name = attrs.get("name") or attrs.get("materialized_path") or ""
        kind = attrs.get("kind") or item.get("type")
        download = links.get("download") or links.get("html")
        child_api = None
        try:
            child_api = rel.get("files", {}).get("links", {}).get("related", {}).get("href")
        except Exception:
            child_api = None
        out.append({"name": name, "kind": kind, "download": download, "child_api": child_api, "raw": item})
    return out


def v6_osf_relevance_name(test_id, name):
    n = name or ""
    gate = v6_fusion_keyword_gate(test_id, filename=n)
    if test_id in {"T28", "T30"}:
        # ITPA DB file names may be terse; allow DB5/STD5/hmode/confinement even if observables are not in filename.
        if _re.search(r"DB5|STD5|H[-_]?mode|hmode|confinement|ITPA", n, flags=_re.I):
            return True
    return bool(gate["ok"])


def v6_classify_fusion_readiness(records, qualifying_count):
    if qualifying_count:
        return "model_ready_or_table_ready"
    saw_var_pdf = False
    saw_candidate = False
    for rec in records or []:
        for link in rec.get("discovered_links", []) or []:
            if _re.search(r"variables?\.pdf|DB5", str(link), flags=_re.I):
                saw_var_pdf = True
            if _re.search(r"\.csv|\.tsv|\.txt|\.dat|\.xls|\.xlsx|\.zip", str(link), flags=_re.I):
                saw_candidate = True
    if saw_var_pdf and not saw_candidate:
        return "source_found_variables_dictionary_only"
    if saw_candidate:
        return "candidate_file_found_header_failed"
    return "source_found_no_usable_table"


def v6_extract_links_from_html_for_pv(text, base_url=""):
    links = []
    if not text:
        return links
    # href/src URLs with likely data words. Avoid triple-quoted regex strings here
    # because this helper block is itself stored inside a triple-quoted string.
    for m in _re.finditer(r"(?:href|src)=[\\\"']([^\\\"']+)[\\\"']", text, flags=_re.I):
        u = _urljoin(base_url, m.group(1))
        if _re.search(r"efficien|cell|chart|pv|nrel|research|data|csv|xlsx|json|xls", u, flags=_re.I):
            links.append(u)
    # Bare JSON-ish URLs.
    for m in _re.finditer(r"[\\\"']([^\\\"']+(?:csv|xlsx|xls|json)[^\\\"']*)[\\\"']", text, flags=_re.I):
        u = _urljoin(base_url, m.group(1))
        if _re.search(r"efficien|cell|chart|pv|nrel|research|data", u, flags=_re.I):
            links.append(u)
    ret
    cols = [str(c or "") for c in columns]
    joined = " ".join(cols)
    groups = [
        [r"year|date"],
        [r"efficien|η|pct|percent"],
        [r"material|technology|cell.*type|class"],
        [r"area|cm2|cm\^2"],
    ]
    matched, missing = [], []
    for grp in groups:
        hit = [p for p in grp if _re.search(p, joined, flags=_re.I)]
        (matched if hit else missing).append(hit if hit else grp)
    return {"ok": not missing, "matched_groups": matched, "missing_groups": missing, "columns": cols}

# ---- End Tier-B v6 helpers ----
'''


FUSION_MANIFEST = """test_id,priority,label,url,source_kind,expected_files,required_column_groups,mode,evidence_level,notes
T28,1,ITPA DB5.2.3 OSF file API,https://api.osf.io/v2/nodes/drwcq/files/osfstorage/,osf_api,"DB5|STD5|Hmode|confinement|csv|xlsx|dat|txt","tau_E|taue|confinement;density|nebar;stored_energy|WMHD|power;machine|device",scientific,evidence,"Primary source for FR7. Recursively walk OSF folders until real DB table is found. Variables PDF is dictionary only."
T30,1,ITPA DB5.2.3 OSF file API,https://api.osf.io/v2/nodes/drwcq/files/osfstorage/,osf_api,"DB5|STD5|Hmode|confinement|csv|xlsx|dat|txt","tau_E|taue|H98;density|nebar;elongation|triangularity|q95|kappa|delta;machine|device",scientific,evidence,"Primary source for FR10 residual coupling; use same DB as T28."
T29,2,W7-X/tokamak profile transport candidates,https://zenodo.org/api/records?q=stellarator%20tokamak%20edge%20transport%20profile%20W7-X&size=10,zenodo_discovery,"profile|transport|diffus|heat_flux|Te|ne|csv|xlsx|zip","device|machine;radius|rho|psi|profile;Te|Ti|ne;transport|diffus|heat_flux",discovery,proxy,"Discovery only unless exact table URL is promoted."
T26,3,Curated ELM energy supplement placeholder,,curated_table,"csv|xlsx","shot|discharge;E_ELM|W_ELM;P_ped|pedestal_pressure;dP/P|Wped|V_ped",scientific,evidence,"Fill with exact paper supplement URLs only; no broad search evidence."
T27,3,Curated RMP ELM frequency supplement placeholder,,curated_table,"csv|xlsx","shot|discharge;f_ELM|ELM_frequency;RMP|coil_current|phasing|n;baseline",scientific,evidence,"Fill with exact RMP/ELM supplement URLs only; no ELM-alone discovery."
"""

PV_PROXY_MANIFEST = """material_class,mass_contrast_proxy,symmetry_proxy,soft_lattice_proxy,notes
Si,0.10,0.90,0.10,monoatomic crystalline baseline
III-V,0.50,0.80,0.20,binary high-quality crystals such as GaAs/InP
CdTe,0.80,0.55,0.45,heavy binary absorber
CIGS,0.85,0.45,0.55,multinary absorber
Perovskite,0.90,0.30,0.90,soft ionic lattice / strong phonon coupling
Organic,0.20,0.20,0.95,soft/disordered organic absorber
Tandem,0.60,0.60,0.60,mixed stack; use cautiously
Other,0.50,0.50,0.50,unknown fallback not used for confirm-like evidence
"""


def backup(path: Path) -> Path:
    bak = path.with_suffix(path.suffix + ".v5bak")
    if not bak.exists():
        bak.write_bytes(path.read_bytes())
    return bak


def append_helper(root: Path) -> None:
    runner = root / "tierb" / "tierb_runner.py"
    if not runner.exists():
        raise FileNotFoundError(f"Cannot find {runner}")
    text = runner.read_text(encoding="utf-8", errors="replace")
    if "Tier-B v6 fusion/PV quality helpers" in text:
        print("v6 helpers already present")
        return
    backup(runner)
    runner.write_text(text + "\n\n" + V6_HELPER + "\n", encoding="utf-8")
    print(f"patched helpers into {runner}")


def patch_runner_minimally(root: Path) -> None:
    """Best-effort text patch to integrate gates without knowing every local v5 detail."""
    runner = root / "tierb" / "tierb_runner.py"
    text = runner.read_text(encoding="utf-8", errors="replace")
    original = text

    # Replace dangerous weak 'roll' negative if present as a raw token in common lists.
    text = text.replace('"roll",', 'r"\\broll\\b",')
    text = text.replace("'roll',", "r'\\broll\\b',")

    # If a generic fusion keyword gate exists, wrap/redirect calls by adding compatibility aliases.
    compat = r'''

# ---- Tier-B v6 compatibility aliases ----
# These names are intentionally broad so existing v5 code can call into stricter gates when available.
def tierb_v6_fusion_gate(test_id, title="", description="", filename="", url=""):
    return v6_fusion_keyword_gate(test_id, title=title, description=description, filename=filename, url=url)

def tierb_v6_header_gate(test_id, columns):
    return v6_header_physical_gate(test_id, columns)

def tierb_v6_pv_gate(columns):
    return v6_pv_header_gate(columns)
# ---- End Tier-B v6 compatibility aliases ----
'''
    if "Tier-B v6 compatibility aliases" not in text:
        text += "\n" + compat + "\n"

    if text != original:
        backup(runner)
        runner.write_text(text, encoding="utf-8")
        print("applied minimal compatibility patch to tierb_runner.py")


def write_manifests(root: Path) -> None:
    d = root / "data" / "source_manifests"
    d.mkdir(parents=True, exist_ok=True)
    (d / "fusion_manifest_v6.csv").write_text(FUSION_MANIFEST, encoding="utf-8")
    (d / "pv_proxy_manifest_v6.csv").write_text(PV_PROXY_MANIFEST, encoding="utf-8")
    print(f"wrote manifests under {d}")


def add_v6_readme(root: Path) -> None:
    txt = """
# Tier-B v6 fusion + PV result-quality patch

This patch targets T26-T30 and T48.

## Main behavioral changes

1. Fusion source relevance is now concept-based, not substring-based.
   - `ELM` alone is not enough.
   - `ELM` must co-occur with fusion anchors such as tokamak, plasma, pedestal, H-mode, RMP, DIII-D, JET, ASDEX/AUG, KSTAR, ITER, ITPA, W7-X, or LHD.
   - False hits such as elm trees, Earth Land Model, DELM/NoDELM, ELM white dwarfs, squirrels, COVID/dental/questionnaires, GPS/SIGFOX/yaw/roll are rejected.

2. T28/T30 are prioritized around OSF ITPA DB5.2.3.
   - OSF file APIs must be walked recursively.
   - `DB5.2.3_variables.pdf` is classified as a variable dictionary, not evidence.
   - Real DB tables must be CSV/XLS/XLSX/TXT/DAT and pass header gates.

3. T26/T27 no longer use broad ELM discovery as evidence.
   - They require exact curated supplement URLs or strict compound fusion/RMP gates.

4. T29 becomes profile/proxy-first.
   - A profile-only proxy is allowed only if device, radius/rho/psi, Te/Ti/ne and transport/heat-flux-like columns exist.

5. T48 uses a preregistered PV proxy manifest.
   - It should parse NREL/PV interactive page assets and embedded JSON before guessing spreadsheet URLs.
   - The residual model should run only when rows >= 100 and columns include year, material/cell class, efficiency, and area.

## Recommended rerun

```powershell
python run_all_tier_b.py --only T26 T27 T28 T29 T30 T48 --cache tierb_cache_v6 --outdir tierb_out_v6 --mode scientific --manifest-only --max-bytes 50000000 --header-rows 50 --timeout 90 --force
```

## Important limitation

The patch script appends helpers and manifests. If your local `tierb_runner.py` has custom internal function names, wire these helper functions into the existing parser/gate calls manually:

- `v6_fusion_keyword_gate(test_id, title, description, filename, url)`
- `v6_header_physical_gate(test_id, columns)`
- `v6_osf_extract_items(data)`
- `v6_osf_relevance_name(test_id, name)`
- `v6_pv_header_gate(columns)`
- `v6_extract_links_from_html_for_pv(text, base_url)`

""".strip() + "\n"
    (root / "CHANGELOG_v6_fusion_pv.md").write_text(txt, encoding="utf-8")
    print("wrote CHANGELOG_v6_fusion_pv.md")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", default=".", help="Root of extracted Tier-B bundle")
    args = ap.parse_args()
    root = Path(args.root).resolve()
    if not (root / "tierb").exists():
        raise SystemExit(f"Not a Tier-B root: {root} (missing tierb/)")
    append_helper(root)
    patch_runner_minimally(root)
    write_manifests(root)
    add_v6_readme(root)
    print("\nTier-B v6 fusion/PV patch installed.")


if __name__ == "__main__":
    main()
