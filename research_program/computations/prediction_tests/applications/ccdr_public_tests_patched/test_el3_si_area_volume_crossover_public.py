from __future__ import annotations

import io
import json
import os
from pathlib import Path

import numpy as np
import pandas as pd
import requests

from _common_public_data import ensure_dir, json_dump, structured_report, TIMEOUT

ROOT = Path(__file__).resolve().parent
OUT = ensure_dir(ROOT / 'outputs' / 'el3')

CANDIDATE_COLS = {
    'node_nm': ['node_nm', 'linewidth_nm', 'line_width_nm', 'feature_size_nm', 'critical_dimension_nm', 'cd_nm'],
    'metric': ['resistance_ohm_um', 'resistivity_uohm_cm', 'delay_ps', 'rc_delay_ps', 'energy_per_bit_pj', 'edp', 'resistance', 'delay', 'energy'],
}


def norm(s: str) -> str:
    return ''.join(ch.lower() if ch.isalnum() else '_' for ch in str(s)).strip('_')


def pick(cols, names):
    nmap = {norm(c): c for c in cols}
    for wanted in names:
        if norm(wanted) in nmap:
            return nmap[norm(wanted)]
    for c in cols:
        nc = norm(c)
        if any(norm(w) in nc for w in names):
            return c
    return None


def read_table_from_url(url: str) -> pd.DataFrame:
    r = requests.get(url, timeout=TIMEOUT)
    r.raise_for_status()
    name = url.lower().split('?')[0]
    data = r.content
    if name.endswith(('.xlsx', '.xls')):
        return pd.read_excel(io.BytesIO(data))
    return pd.read_csv(io.BytesIO(data))


def fit_piecewise_log(x, y):
    x = np.asarray(x, float); y = np.asarray(y, float)
    m = np.isfinite(x) & np.isfinite(y) & (x > 0) & (y > 0)
    x = x[m]; y = y[m]
    order = np.argsort(x)
    x = x[order]; y = y[order]
    if len(x) < 8:
        return None
    lx = np.log(x); ly = np.log(y)
    best = None
    for i in range(3, len(x)-3):
        bp = x[i]
        left = slice(0, i); right = slice(i, len(x))
        a1, b1 = np.polyfit(lx[left], ly[left], 1)
        a2, b2 = np.polyfit(lx[right], ly[right], 1)
        pred = np.r_[a1*lx[left]+b1, a2*lx[right]+b2]
        sse = float(np.sum((ly - pred)**2))
        cand = {'breakpoint_nm': float(bp), 'sse': sse, 'slope_large_nm': float(a1), 'slope_small_nm': float(a2)}
        if best is None or sse < best['sse']:
            best = cand
    return best


def main():
    urls = [u.strip() for u in os.environ.get('EL3_NUMERIC_URLS', '').split(';') if u.strip()]
    if not urls:
        out = structured_report(
            'EL3', 'skipped_requires_numeric_table',
            prediction='volume-to-area information/transport scaling crossover at L≈10–100 nm for Si',
            required_columns=['node_or_linewidth_nm','delay_or_resistance_or_energy_per_bit','device_or_interconnect_class'],
            how_to_run='set EL3_NUMERIC_URLS to one or more CSV/XLSX URLs separated by semicolons, or run directly with a local edit',
            verdict='skipped_not_counted_in_batch',
            note='Removed from default batch unless EL3_NUMERIC_URLS is set. This avoids treating article lists as evidence.'
        )
        json_dump(out, OUT/'el3_report.json'); print(json.dumps(out, indent=2)); return

    results=[]
    for url in urls:
        try:
            df=read_table_from_url(url)
            node_col=pick(df.columns, CANDIDATE_COLS['node_nm'])
            metric_col=pick(df.columns, CANDIDATE_COLS['metric'])
            if not node_col or not metric_col:
                results.append({'url':url,'status':'missing_required_columns','columns':list(map(str,df.columns))[:50]}); continue
            x=pd.to_numeric(df[node_col], errors='coerce').to_numpy()
            y=pd.to_numeric(df[metric_col], errors='coerce').to_numpy()
            fit=fit_piecewise_log(x,y)
            if not fit:
                results.append({'url':url,'status':'too_few_numeric_points','node_col':node_col,'metric_col':metric_col}); continue
            support = 10 <= fit['breakpoint_nm'] <= 100
            results.append({'url':url,'status':'ok','node_col':node_col,'metric_col':metric_col,'fit':fit,'support_like_breakpoint_10_100nm':support})
        except Exception as e:
            results.append({'url':url,'status':'error','error':repr(e)})
    ok=[r for r in results if r.get('status')=='ok']
    out=structured_report('EL3','ok' if ok else 'not_executable_no_usable_numeric_table', results=results, verdict=('support_like_crossover' if any(r.get('support_like_breakpoint_10_100nm') for r in ok) else 'no_support_or_no_usable_data'))
    json_dump(out, OUT/'el3_report.json'); print(json.dumps(out, indent=2))

if __name__=='__main__': main()
