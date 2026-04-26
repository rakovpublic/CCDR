#!/usr/bin/env python3
"""QC5 public experimental benchmark proxy.

QC5 target: phononic crystal substrate / complete acoustic bandgap near the qubit
frequency should push T1 beyond 1 ms, with >2-5x improvement inside the bandgap
once non-phononic channels dominate.

This script downloads public arXiv/HTML sources and parses explicit reported T1 and
improvement statements. It separates:
  * transmon-qubit T1 benchmarks;
  * TLS/defect T1 inside phononic bandgap;
  * qubit-on-phononic-bandgap results.
The decisive QC5 condition is only true for the last category.
"""
from __future__ import annotations

import argparse
import json
import re
from pathlib import Path
from typing import Any, Dict, List, Optional

from _public_test_common import download_to_cache, normalize_text, read_text_any, source_records, write_json
from _public_improvements import evidence_for_status

SOURCES = {
    "Chen Painter 2024 phonon engineering arXiv": "https://arxiv.org/abs/2310.03929",
    "Odeh Sipahigil 2023 non-Markovian qubit phononic bandgap arXiv": "https://arxiv.org/abs/2312.01031",
    "Place Houck 2020 transmon 0.3 ms benchmark arXiv": "https://arxiv.org/abs/2003.00024",
    "Rosen 2019 phonon-mediated decay theory arXiv": "https://arxiv.org/abs/1903.06193",
}


def extract_qc5_claims(texts: Dict[str, str]) -> Dict[str, Any]:
    claims: List[Dict[str, Any]] = []
    chen = normalize_text(texts.get("Chen Painter 2024 phonon engineering arXiv", ""))
    chen_l = chen.lower()
    chen_has_order_gain = bool(re.search(r"two(?:\s|-)+orders?\s+of\s+magnitude|two(?:\s|-)+to(?:\s|-)+three(?:\s|-)+orders?", chen_l))
    chen_t1_gt_5ms = bool(re.search(r"(?:exceeding|exceeds|>|greater\s+than)\s*\$?\s*5\s*\$?\s*(?:ms|milliseconds?)", chen, re.IGNORECASE))
    if chen_has_order_gain and chen_t1_gt_5ms and "acoustic bandgap" in chen_l:
        claims.append(
            {
                "source": "Chen/Painter 2024",
                "category": "TLS_defect_inside_phononic_bandgap",
                "measurement_target": "embedded TLS strongly coupled to a transmon circuit, not transmon T1 itself",
                "reported_T1_ms": 5.0,
                "reported_improvement_factor_lower_bound": 100.0,
                "bandgap_condition": True,
                "qubit_T1": False,
                "evidence": "abstract reports two-orders relaxation-time increase and longest T1 exceeding 5 ms for embedded TLS inside acoustic bandgap",
            }
        )
    elif "acoustic bandgap" in chen_l:
        claims.append(
            {
                "source": "Chen/Painter 2024",
                "category": "phononic_bandgap_text_found",
                "reported_T1_ms": None,
                "reported_improvement_factor_lower_bound": None,
                "bandgap_condition": True,
                "qubit_T1": False,
            }
        )


    odeh = normalize_text(texts.get("Odeh Sipahigil 2023 non-Markovian qubit phononic bandgap arXiv", ""))
    # arXiv abstract says Purcell-engineered TLS lifetime of 34 microseconds.
    m = re.search(r"lifetime\s+of\s+(\d+(?:\.\d+)?)\s*\$?\\?mu\s*s|lifetime\s+of\s+(\d+(?:\.\d+)?)\s*microseconds", odeh, re.IGNORECASE)
    if m or "34" in odeh and "phononic bandgap" in odeh.lower():
        claims.append(
            {
                "source": "Odeh/Sipahigil 2023/2025",
                "category": "qubit_coupled_to_phononic_bandgap_TLS_bath_probe",
                "measurement_target": "engineered TLS lifetime / qubit non-Markovian dynamics, not standalone transmon T1",
                "reported_T1_ms": 0.034,
                "reported_improvement_factor_lower_bound": None,
                "bandgap_condition": True,
                "qubit_T1": False,
                "coupled_qubit_probe": True,
                "evidence": "abstract reports non-Markovian qubit dynamics due to Purcell-engineered TLS lifetime of 34 us inside phononic bandgap",
            }
        )

    place = normalize_text(texts.get("Place Houck 2020 transmon 0.3 ms benchmark arXiv", ""))
    if re.search(r"exceeding\s+0\.3\s*milliseconds", place, re.IGNORECASE):
        claims.append(
            {
                "source": "Place/Houck 2020/2021",
                "category": "non_phononic_transmon_material_baseline",
                "reported_T1_ms": 0.3,
                "reported_improvement_factor_lower_bound": None,
                "bandgap_condition": False,
                "qubit_T1": True,
                "evidence": "abstract reports transmon lifetimes/coherence exceeding 0.3 ms from tantalum material platform, not phononic bandgap",
            }
        )

    rosen = normalize_text(texts.get("Rosen 2019 phonon-mediated decay theory arXiv", ""))
    if "up to two orders of magnitude" in rosen.lower() or "phononic bandgap" in rosen.lower():
        claims.append(
            {
                "source": "Rosen 2019",
                "category": "phonon_mediated_decay_theory",
                "reported_T1_ms": None,
                "reported_improvement_factor_lower_bound": 100.0 if "up to two orders of magnitude" in rosen.lower() else None,
                "bandgap_condition": True,
                "qubit_T1": False,
                "evidence": "theory/model source for phononic bandgap suppression of phonon-mediated decay",
            }
        )
    return {"claims": claims}


def decide(claims: List[Dict[str, Any]]) -> Dict[str, Any]:
    """Split QC5 into mechanism support (QC5a) and original transmon target (QC5b)."""
    tls_claims = [
        c for c in claims
        if c.get("category") == "TLS_defect_inside_phononic_bandgap" and c.get("bandgap_condition")
    ]
    decisive_qubit_claims = [
        c for c in claims
        if c.get("qubit_T1")
        and c.get("bandgap_condition")
        and (c.get("reported_T1_ms") or 0) >= 1.0
    ]
    tls_pass = any(
        (c.get("reported_T1_ms") or 0) >= 1.0
        and (c.get("reported_improvement_factor_lower_bound") or 0) >= 2.0
        for c in tls_claims
    )
    best_tls_ms = max([c.get("reported_T1_ms") or 0 for c in tls_claims], default=0.0)
    best_tls_gain = max([c.get("reported_improvement_factor_lower_bound") or 0 for c in tls_claims], default=0.0)
    best_qubit = max([c.get("reported_T1_ms") or 0 for c in decisive_qubit_claims], default=0.0)
    qc5a = {
        "label": "QC5a_TLS_mechanism",
        "status": "mechanism_supported" if tls_pass else "not_established",
        "definition": "Complete acoustic/phononic bandgap suppresses phonon-mediated TLS/defect relaxation by >2-5x and can reach ms-scale T1.",
        "best_extracted_TLS_T1_ms_inside_bandgap": best_tls_ms if best_tls_ms else None,
        "best_extracted_TLS_improvement_factor_lower_bound": best_tls_gain if best_tls_gain else None,
        "confirmed_as_full_QC5": False,
    }
    qc5b = {
        "label": "QC5b_transmon_T1",
        "status": "confirmed" if decisive_qubit_claims else "unconfirmed",
        "definition": "Standalone transmon-qubit T1 > 1 ms with transition inside a complete acoustic bandgap and >2-5x over controls.",
        "best_extracted_qubit_T1_ms_inside_bandgap": best_qubit if best_qubit else None,
        "target_T1_ms": 1.0,
        "target_improvement_factor": ">2-5x",
    }
    qc5a["evidence_ladder"] = evidence_for_status(qc5a["status"])
    qc5b["evidence_ladder"] = evidence_for_status(qc5b["status"])
    overall = "pass" if decisive_qubit_claims else ("partial_support_TLS_not_qubit" if tls_pass else "inconclusive_or_null")
    overall_ladder = evidence_for_status(overall)
    return {
        "status": overall,
        "overall_status": overall,
        "evidence_ladder": overall_ladder,
        "QC5a_TLS_mechanism": qc5a,
        "QC5b_transmon_T1": qc5b,
        "tls_bandgap_support": bool(tls_pass),
        "decisive_qubit_T1_gt_1ms_inside_bandgap": bool(decisive_qubit_claims),
        "best_extracted_qubit_T1_ms_inside_bandgap": best_qubit if best_qubit else None,
        "metric": {
            "target_T1_ms": 1.0,
            "target_improvement_factor": ">2-5x",
        },
        "interpretation": (
            "QC5a is supported when public text contains the embedded-TLS bandgap result (>5 ms and >100x). "
            "QC5b, the original standalone-transmon T1 >1 ms target, remains unconfirmed until a public "
            "on/off-bandgap transmon experiment reports that metric."
        ),
    }

def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--cache", default="qc5_cache")
    ap.add_argument("--out", default="out_qc5.json")
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
        texts[src.label] = read_text_any(Path(src.path))

    extracted = extract_qc5_claims(texts)
    decision = decide(extracted["claims"])
    result = {
        "test_name": "QC5 phononic-crystal substrate T1 public benchmark",
        "downloaded_sources": source_records(downloads),
        "warnings": warnings,
        "prediction": "QC5 split: QC5a phononic-bandgap TLS/mechanism support; QC5b standalone transmon-qubit T1 >1 ms and >2-5x inside a complete bandgap near 4-8 GHz.",
        "extracted_claims": extracted["claims"],
        "decision": decision,
        "falsification_logic": {
            "confirm_like": "A public experiment reports transmon-qubit T1 >1 ms with the transition inside a complete acoustic bandgap and >2-5x over off-gap/control devices.",
            "falsify_like": "Comparable on/off-bandgap transmon devices repeatedly show <2x improvement or no trend once non-phononic channels are controlled.",
            "current_status_rule": "QC5a/TLS enhancement can be mechanism-supported; QC5b/original transmon T1 target requires standalone qubit T1 rows.",
        },
    }
    write_json(result, Path(args.out))
    print(json.dumps(result, indent=2, sort_keys=True, allow_nan=False))


if __name__ == "__main__":
    main()
