from __future__ import annotations

import io
import json
import os
from pathlib import Path

import numpy as np
import pandas as pd
import requests

from _common_public_data import ensure_dir, json_dump, structured_report

ROOT = Path(__file__).resolve().parent
OUT = ensure_dir(ROOT / 'outputs' / 'el8')

COLS = {
    'technology': ['technology','tech','type','interconnect_type','link_type'],
    'node_nm': ['node_nm','feature_size_nm','process_nm','linewidth_nm'],
    'energy': ['energy_per_bit_pj','pj_per_bit','pJ_bit','energy_bit','energy'],
    'bandwidth_density': ['bandwidth_density','gbps_per_mm','tbps_per_mm','gbps_per_lane','bandwidth'],
    'reach': ['reach_mm','distance_mm','length_mm','reach','distance'],
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
    r = requests.get(url, timeout=60)
    r.raise_for_status()
    name = url.lower().split('?')[0]
    data = r.content
    if name.endswith(('.xlsx', '.xls')):
        return pd.read_excel(io.BytesIO(data))
    return pd.read_csv(io.BytesIO(data))


def classify_tech(v):
    s=str(v).lower()
    if any(k in s for k in ['optical','photonic','silicon photonic','laser','waveguide']): return 'optical'
    if any(k in s for k in ['electrical','copper','cu','via','tsv','wire','metal']): return 'electrical'
    return 'unknown'


def main():
    urls = [u.strip() for u in os.environ.get('EL8_NUMERIC_URLS', '').split(';') if u.strip()]
    if not urls:
        out = structured_report(
            'EL8', 'skipped_requires_numeric_table',
            prediction='optical interconnects outperform electronic vias at sub-10 nm nodes',
            required_columns=['technology','node_or_feature_nm','energy_per_bit_pj','bandwidth_density_or_bandwidth','reach_or_distance'],
            how_to_run='set EL8_NUMERIC_URLS to one or more CSV/XLSX URLs separated by semicolons',
            verdict='skipped_not_counted_in_batch',
            note='Removed from default batch unless EL8_NUMERIC_URLS is set. This avoids treating article lists as evidence.'
        )
        json_dump(out, OUT/'el8_report.json'); print(json.dumps(out, indent=2)); return

    rows=[]; problems=[]
    for url in urls:
        try:
            df=read_table_from_url(url)
            tech_col=pick(df.columns, COLS['technology'])
            node_col=pick(df.columns, COLS['node_nm'])
            energy_col=pick(df.columns, COLS['energy'])
            bw_col=pick(df.columns, COLS['bandwidth_density'])
            if not (tech_col and node_col and energy_col):
                problems.append({'url':url,'status':'missing_required_columns','columns':list(map(str,df.columns))[:50]}); continue
            for _,r in df.iterrows():
                tech=classify_tech(r.get(tech_col,''))
                node=pd.to_numeric(r.get(node_col), errors='coerce')
                energy=pd.to_numeric(r.get(energy_col), errors='coerce')
                bw=pd.to_numeric(r.get(bw_col), errors='coerce') if bw_col else np.nan
                if tech!='unknown' and np.isfinite(node) and np.isfinite(energy):
                    rows.append({'source_url':url,'technology':tech,'node_nm':float(node),'energy_per_bit_pj':float(energy),'bandwidth_metric':None if not np.isfinite(bw) else float(bw)})
        except Exception as e:
            problems.append({'url':url,'status':'error','error':repr(e)})
    sub10=[r for r in rows if r['node_nm'] <= 10]
    optical=[r for r in sub10 if r['technology']=='optical']
    electrical=[r for r in sub10 if r['technology']=='electrical']
    verdict='no_support_or_no_usable_data'
    comparison=None
    if optical and electrical:
        med_o=float(np.median([r['energy_per_bit_pj'] for r in optical]))
        med_e=float(np.median([r['energy_per_bit_pj'] for r in electrical]))
        comparison={'n_optical_sub10':len(optical),'n_electrical_sub10':len(electrical),'median_optical_pj_bit':med_o,'median_electrical_pj_bit':med_e}
        verdict='support_like_optical_lower_energy_sub10' if med_o < med_e else 'no_support_electrical_not_worse_sub10'
    out=structured_report('EL8','ok' if rows else 'not_executable_no_usable_numeric_table', parsed_rows=rows[:200], problems=problems, comparison=comparison, verdict=verdict)
    json_dump(out, OUT/'el8_report.json'); print(json.dumps(out, indent=2))

if __name__=='__main__': main()
