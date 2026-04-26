#!/usr/bin/env python3
"""FR8 proxy: non-monotonic eta/s vs collisionality and lower stellarator M_KSS.

This is intentionally conservative. Public QGP and fusion-edge papers contain useful
statements, but not a common machine-readable table of eta/s, collisionality and device
geometry. The script downloads public sources, extracts only numeric table/range values that
are in text, and reports whether a real FR8 regression can be performed.
"""
from __future__ import annotations

import argparse
import json
import math
import re
from pathlib import Path
from typing import Any, Dict, List

import numpy as np

from _public_test_common import (
    download_to_cache,
    normalize_text,
    pdf_text,
    source_records,
    spearmanr,
    write_json,
)
from test_fr3_fr6_literature_proxy import parse_diiid_knolker_table2
from _public_improvements import evidence_for_status

SOURCES = {
    "QGP Heinz Shen Song arXiv abstract HTML": "https://arxiv.org/abs/1108.5323",
    "KSS bound status arXiv abstract HTML": "https://arxiv.org/abs/0804.2601",
    "DIII-D Knolker 2018 pedestal pressure collisionality PDF": "https://pure.mpg.de/rest/items/item_2622196_4/component/file_3017099/content",
    "W7-X optimization Dinklage 2018 PDF": "https://scipub.euro-fusion.org/wp-content/uploads/eurofusion/WPS1PR17_18393_submitted.pdf",
    "W7-X first divertor operation Pedersen 2018 PDF": "https://scipub.euro-fusion.org/wp-content/uploads/eurofusion/WPS1PR18_21092_submitted.pdf",
    "Tokamak-stellarator comparison Xu 2016 open HTML": "https://www.sciencedirect.com/science/article/pii/S2468080X16300322",
}


def read_source_text(path: Path) -> tuple[str, list[str]]:
    if path.suffix.lower() == ".pdf" or path.read_bytes()[:4] == b"%PDF":
        return pdf_text(path)
    raw = path.read_bytes()
    for enc in ("utf-8", "latin-1", "cp1252"):
        try:
            return raw.decode(enc, errors="replace"), []
        except Exception:
            pass
    return raw.decode("utf-8", errors="replace"), []


def qgp_kss_proxy(texts: Dict[str, str]) -> Dict[str, Any]:
    heinz = normalize_text(texts.get("QGP Heinz Shen Song arXiv abstract HTML", ""))
    kss = normalize_text(texts.get("KSS bound status arXiv abstract HTML", ""))
    return {
        "specific_shear_viscosity_extractable_from_flow_text_found": "specific shear viscosity" in heinz.lower() and "elliptic flow" in heinz.lower(),
        "same_eta_over_s_at_rhic_and_lhc_text_found": "same (eta/s)_QGP value" in heinz or "same (eta/s)" in heinz,
        "kss_bound_text_found": "1/(4 pi)" in kss or "1/4pi" in kss.replace(" ", ""),
        "note": "This is evidence that QGP eta/s extraction is public-literature anchored; it is not a machine-readable eta/s(T,collisionality) table.",
    }


def fusion_edge_proxy(knolker_rows: list[dict[str, Any]], texts: Dict[str, str]) -> Dict[str, Any]:
    if not knolker_rows:
        return {"status": "data_insufficient_public_proxy", "reason": "No DIII-D collisionality rows extracted from public PDF."}
    nu = [r["nu_e_star_mid"] for r in knolker_rows]
    f = [r["f_elm_hz_mid"] for r in knolker_rows]
    p = [r["p_ped_kpa_mid"] for r in knolker_rows]
    # This is not eta/s; it is a sanity check that edge observables are non-monotonic with collisionality.
    rank_f = spearmanr(nu, f)
    # Non-monotonic proxy: middle point higher/lower than both endpoints.
    nonmono_f = len(f) == 3 and ((f[1] > f[0] and f[1] > f[2]) or (f[1] < f[0] and f[1] < f[2]))
    dinklage = normalize_text(texts.get("W7-X optimization Dinklage 2018 PDF", ""))
    pedersen = normalize_text(texts.get("W7-X first divertor operation Pedersen 2018 PDF", ""))
    xu = normalize_text(texts.get("Tokamak-stellarator comparison Xu 2016 open HTML", ""))

    # FR8 cannot be computed without viscosity/entropy or a defensible M_KSS
    # conversion.  These flags keep track of whether we at least have stellarator
    # optimization/confinement context from public text.
    w7x_text = (dinklage + "\n" + pedersen).lower()
    w7x_text = (dinklage + "\n" + pedersen).lower()
    return {
        "status": "partial_proxy_only",
        "diiid_collisionality_range_rows": knolker_rows,
        "collisionality_vs_f_elm_spearman": rank_f,
        "nonmonotonic_f_elm_midpoint_proxy": bool(nonmono_f),
        "p_ped_kpa_midpoints": p,
        "stellarator_public_text_flags": {
            "w7x_energy_confinement_text_found": "energy confinement" in w7x_text,
            "w7x_optimisation_text_found": "optimisation" in w7x_text or "optimization" in w7x_text,
            "w7x_neoclassical_optimization_text_found": "neoclassical" in w7x_text and ("optimization" in w7x_text or "optimisation" in w7x_text),
            "w7x_divertor_or_edge_operation_text_found": "divertor" in w7x_text or "edge" in w7x_text,
            "tokamak_stellarator_comparison_text_found": "tokamak" in xu.lower() and "stellarator" in xu.lower(),
        },
        "proxy_caveat": "The DIII-D non-monotonic f_ELM/collisionality signal is not eta/s. W7-X text flags support the stellarator context only, not a numerical M_KSS comparison.",
        "why_not_decisive": "FR8 needs eta/s or M_KSS values on a common basis. Public sources here provide QGP eta/s context, DIII-D collisionality/f_ELM ranges, and W7-X optimization context, but not edge eta/s or stellarator M_KSS rows.",
    }



def split_fr8_subclaims(qgp: Dict[str, Any], edge: Dict[str, Any]) -> Dict[str, Any]:
    """v2.7 split FR8 into QGP context, fusion transport proxy, and true eta/M_KSS."""
    qgp_context = bool(qgp.get("kss_bound_text_found") and qgp.get("specific_shear_viscosity_extractable_from_flow_text_found"))
    fusion_partial = edge.get("status") == "partial_proxy_only"
    return {
        "FR8a_QGP_eta_over_s_context": {
            "status": "qualitative_context" if qgp_context else "data_missing",
            "evidence_ladder": evidence_for_status("qualitative_context" if qgp_context else "data_missing"),
            "interpretation": "QGP literature supports public eta/s/KSS context, but no machine-readable eta/s(T, collisionality) table is used here.",
            "qgp_flags": qgp,
        },
        "FR8b_fusion_transport_proxy": {
            "status": "partial_proxy" if fusion_partial else "data_missing",
            "evidence_ladder": evidence_for_status("partial_proxy" if fusion_partial else "data_missing"),
            "proxy_definition": "M_transport_proxy candidate = normalized heat diffusivity or gyro-Bohm-normalized transport when public profile rows exist; current output uses only DIII-D f_ELM/collisionality and W7-X context.",
            "computed_transport_rows": [],
            "edge_proxy_status": edge.get("status"),
        },
        "FR8c_true_edge_eta_or_MKSS": {
            "status": "data_insufficient_public_proxy",
            "evidence_ladder": evidence_for_status("data_insufficient_public_proxy"),
            "required_rows": [
                "device", "collisionality", "edge eta/s or M_KSS", "uncertainty", "method", "matched control parameters"
            ],
            "interpretation": "No public rows with comparable edge eta/s or M_KSS were found/downloaded, so the decisive FR8 claim remains untested.",
        },
    }

def fr8_split_evidence_summary(fr8_split: Dict[str, Any]) -> Dict[str, Any]:
    """v2.8: make split evidence levels explicit in the top-level summary.

    The decisive FR8c claim can remain level 0 while context/proxy branches are
    level 1+.  This prevents the top-level ladder from hiding useful but
    non-decisive split evidence.
    """
    levels: Dict[str, int] = {}
    for key, block in fr8_split.items():
        ladder = block.get("evidence_ladder") if isinstance(block, dict) else None
        if isinstance(ladder, dict):
            levels[key] = int(ladder.get("level", 0))
    decisive = levels.get("FR8c_true_edge_eta_or_MKSS", 0)
    context = max([levels.get("FR8a_QGP_eta_over_s_context", 0), levels.get("FR8b_fusion_transport_proxy", 0)], default=0)
    return {
        "FR8_decisive_evidence_level": decisive,
        "FR8_context_evidence_level": context,
        "FR8_split_evidence_levels": levels,
        "interpretation": "FR8c is the decisive eta/s or M_KSS claim; FR8a/FR8b are context/proxy branches and must not upgrade the decisive FR8 verdict.",
    }

def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--cache", default="fr8_cache")
    ap.add_argument("--out", default="out_fr8.json")
    ap.add_argument("--force", action="store_true")
    args = ap.parse_args()

    cache = Path(args.cache)
    downloads = [download_to_cache(label, url, cache, force=args.force) for label, url in SOURCES.items()]
    texts: Dict[str, str] = {}
    warnings: List[str] = []
    for src in downloads:
        if not src.ok:
            warnings.append(f"download failed for {src.label}: {src.error}")
            continue
        txt, ws = read_source_text(Path(src.path))
        texts[src.label] = txt
        warnings.extend([f"{src.label}: {w}" for w in ws])

    knolker = parse_diiid_knolker_table2(texts.get("DIII-D Knolker 2018 pedestal pressure collisionality PDF", ""))
    qgp = qgp_kss_proxy(texts)
    edge = fusion_edge_proxy(knolker.get("rows", []), texts)

    fr8_split = split_fr8_subclaims(qgp, edge)
    split_evidence_summary = fr8_split_evidence_summary(fr8_split)
    decisive_ready = False
    result = {
        "test_name": "FR8 eta/s-collisionality stellarator lower M_KSS public proxy",
        "downloaded_sources": source_records(downloads),
        "warnings": warnings,
        "prediction": "FR8: non-monotonic eta/s vs collisionality; stellarator edge transport should approach lower M_KSS than tokamaks.",
        "qgp_proxy": qgp,
        "fusion_edge_proxy": edge,
        "FR8_split": fr8_split,
        "FR8a_QGP_eta_over_s_context": fr8_split["FR8a_QGP_eta_over_s_context"],
        "FR8b_fusion_transport_proxy": fr8_split["FR8b_fusion_transport_proxy"],
        "FR8c_true_edge_eta_or_MKSS": fr8_split["FR8c_true_edge_eta_or_MKSS"],
        "FR8_decisive_evidence_level": split_evidence_summary["FR8_decisive_evidence_level"],
        "FR8_context_evidence_level": split_evidence_summary["FR8_context_evidence_level"],
        "FR8_split_evidence_summary": split_evidence_summary,
        "knolker_table2_extraction": {k: v for k, v in knolker.items() if k != "rows"},
        "decision": "data_insufficient_public_proxy" if not decisive_ready else "computed",
        "evidence_ladder": evidence_for_status("data_insufficient_public_proxy" if not decisive_ready else "computed"),
        "minimum_required_for_decisive_test": [
            ">=10 tokamak rows with collisionality and edge eta/s or M_KSS",
            ">=5 stellarator rows with comparable edge eta/s or M_KSS",
            "common units and method for entropy density/viscosity or diffusivity-to-M_KSS conversion",
        ],
        "falsification_logic": {
            "confirm_like": "eta/s or M_KSS has a clear minimum vs collisionality and stellarator rows lie below tokamak rows at matched edge parameters.",
            "falsify_like": "eta/s/M_KSS is monotonic or stellarator rows are not lower after matched controls and uncertainty propagation.",
            "current_script_policy": "Do not convert f_ELM/collisionality into eta/s unless a public paper gives the conversion or provides eta/s rows.",
        },
    }
    write_json(result, Path(args.out))
    print(json.dumps(result, indent=2, sort_keys=True, allow_nan=False))


if __name__ == "__main__":
    main()
