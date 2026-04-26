from __future__ import annotations
import json, os
from pathlib import Path
from _common_public_data import ensure_dir, json_dump, structured_report, osf_find_download, read_public_table, cached_download
ROOT=Path(__file__).resolve().parent; OUT=ensure_dir(ROOT/'outputs'/'fr8')
def _env_table():
    url=os.environ.get('FR8_STELLARATOR_TABLE_URL','').strip()
    if not url: return None
    try:
        p=cached_download(url,OUT/'stellarator_external'/Path(url).name); df=read_public_table(p); cols=[str(c) for c in df.columns]; lower=' '.join(c.lower() for c in cols)
        return {'url':url,'n_rows':int(len(df)),'columns_sample':cols[:50],'ready_like':('tau' in lower and ('device' in lower or 'machine' in lower))}
    except Exception as e: return {'url':url,'error':repr(e)}
def main():
    try:
        table=osf_find_download(['drwcq','hrqcf'],r'.*\.(xlsx|xls|csv|txt|dat)$',OUT/'tokamak_HDB'); df=read_public_table(table); tok={'available':True,'n_rows':int(len(df)),'columns_sample':[str(c) for c in list(df.columns)[:50]]}
    except Exception as e: tok={'available':False,'reason':repr(e)}
    st=_env_table(); status='ready_for_user_supplied_stellarator_numeric_fit' if st and st.get('ready_like') else 'not_executable_missing_stellarator_machine_table'
    out=structured_report('FR8',status,tokamak_table=tok,optional_stellarator_table=st,required_stellarator_columns=['device','tau_E','minor_radius_or_transport_length','edge_transport_or_proxy'],verdict='can_fit_prediction' if status.startswith('ready') else 'no_physics_verdict',note='Improved script can ingest a discovered public stellarator CSV/XLSX via FR8_STELLARATOR_TABLE_URL; otherwise no article-list pseudo-test is used.')
    json_dump(out,OUT/'fr8_report.json'); print(json.dumps(out,indent=2))
if __name__=='__main__': main()
