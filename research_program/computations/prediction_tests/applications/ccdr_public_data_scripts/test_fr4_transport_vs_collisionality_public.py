from __future__ import annotations

"""
FR4 numeric public-data proxy test.

Prediction: transport/viscosity proxy should be non-monotonic vs collisionality,
with a minimum. Public HDB data do not expose edge eta/s directly, so this uses
chi_eff ~ a^2/tau_E and a standard rough collisionality proxy.
"""

import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from _common_public_data import ensure_dir, json_dump, osf_find_download, save_plot, structured_report, read_public_table, best_matching_column_fuzzy, repair_header_if_needed

ROOT = Path(__file__).resolve().parent
OUT = ensure_dir(ROOT / "outputs" / "fr4")

COLS = {
    "tauE": ["TAUE", "TAU_E", "TAUTH", "TAU_E_TH", "tau e", "energy confinement time", "tauth s", "taue s"],
    "a": ["AMIN", "A_MINOR", "a", "minor radius", "amin m", "a m"],
    "R": ["RGEO", "R_MAJOR", "R", "major radius", "rgeo m"],
    "ne": ["NEBAR", "NEL", "nbar", "nebar", "line averaged density", "density"],
    "bt": ["BT", "BTOR", "BT0", "magnetic field", "btor", "ipb"],
    "q95": ["Q95", "q95", "Q_95", "q"] ,
    "te": ["TE0", "TE", "TEPED", "electron temperature", "te0", "teped"],
}


def find_cols(df):
    df = repair_header_if_needed(df, ["TAU", "AMIN", "RGEO", "NEBAR", "Q95", "BT"])
    return df, {k: best_matching_column_fuzzy(df, v) for k, v in COLS.items()}


def safe_num(df, col):
    return pd.to_numeric(df[col], errors="coerce")


def main():
    try:
        table = osf_find_download(["drwcq", "hrqcf"], r".*\.(xlsx|xls|csv|txt|dat)$", OUT / "HDB_public_table")
        df = read_public_table(table)
    except Exception as e:
        out = structured_report("FR4", "source_unavailable_or_unparseable", reason=repr(e), attempted_nodes=["drwcq", "hrqcf"])
        json_dump(out, OUT / "fr4_report.json"); print(json.dumps(out, indent=2)); return

    df, found = find_cols(df)
    sample_columns = [str(c) for c in list(df.columns)[:80]]
    required = ["tauE", "a", "ne", "bt", "q95", "te"]
    missing = [k for k in required if found.get(k) is None]
    if missing:
        out = structured_report(
            "FR4", "not_executable_missing_columns",
            n_rows=int(len(df)), columns_found=found, missing_required=missing, sample_columns=sample_columns,
            verdict="no_physics_verdict",
            note="A table was downloaded, but the columns needed to build a collisionality-vs-transport proxy were not recognisable. This is not a confirmation or falsification."
        )
        json_dump(out, OUT / "fr4_report.json"); print(json.dumps(out, indent=2)); return

    tau = safe_num(df, found["tauE"]); a = safe_num(df, found["a"])
    ne = safe_num(df, found["ne"]); bt = safe_num(df, found["bt"])
    q95 = safe_num(df, found["q95"]); te = safe_num(df, found["te"])
    mask = (tau > 0) & (a > 0) & (ne > 0) & (bt > 0) & (q95 > 0) & (te > 0)
    if int(mask.sum()) < 100:
        out = structured_report("FR4", "not_executable_too_few_rows", n_rows=int(len(df)), n_valid=int(mask.sum()), columns_found=found)
        json_dump(out, OUT / "fr4_report.json"); print(json.dumps(out, indent=2)); return
    tau=tau[mask].to_numpy(float); a=a[mask].to_numpy(float); ne=ne[mask].to_numpy(float); bt=bt[mask].to_numpy(float); q95=q95[mask].to_numpy(float); te=te[mask].to_numpy(float)
    chi = a*a/tau
    coll = ne*q95/(np.maximum(te,1e-12)**2*np.maximum(np.abs(bt),1e-12))
    good = np.isfinite(coll)&np.isfinite(chi)&(coll>0)&(chi>0)
    coll=coll[good]; chi=chi[good]
    nbins = min(14, max(6, len(coll)//500))
    qs = np.quantile(coll, np.linspace(0,1,nbins+1))
    mids=[]; meds=[]
    for lo,hi in zip(qs[:-1], qs[1:]):
        sel=(coll>=lo)&(coll<=hi)
        if sel.sum()>=20:
            mids.append(float(np.median(coll[sel]))); meds.append(float(np.median(chi[sel])))
    if len(mids)>=5:
        coeff = np.polyfit(np.log(mids), np.log(meds), 2)
        min_inside = bool(coeff[0] > 0 and min(np.log(mids)) < -coeff[1]/(2*coeff[0]) < max(np.log(mids)))
    else:
        coeff=[float('nan')]*3; min_inside=False
    fig, ax=plt.subplots(figsize=(7,4.6)); ax.scatter(coll,chi,s=5,alpha=.18); ax.plot(mids,meds,marker='o')
    ax.set_xscale('log'); ax.set_yscale('log'); ax.set_xlabel('collisionality proxy'); ax.set_ylabel('chi_eff ~ a^2/tauE'); ax.grid(alpha=.25)
    save_plot(fig, OUT/'fr4_proxy_plot.png'); plt.close(fig)
    out=structured_report(
        "FR4", "ok", n_rows=int(len(df)), n_used=int(len(coll)), columns_found=found,
        binned_medians={"collisionality_proxy":mids,"transport_proxy":meds}, quadratic_loglog_coeffs=[float(c) for c in coeff],
        verdict=("support_like_nonmonotonic_minimum" if min_inside else "no_support_for_nonmonotonic_minimum"),
        caveat="Proxy only: public HDB does not directly provide edge eta/s.")
    json_dump(out, OUT/'fr4_report.json'); print(json.dumps(out, indent=2))

if __name__ == '__main__': main()
