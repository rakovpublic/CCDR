from __future__ import annotations
import json, os, math
from pathlib import Path
import numpy as np, pandas as pd
from _common_public_data import ensure_dir, json_dump, structured_report, cached_download, read_public_table
ROOT=Path(__file__).resolve().parent; OUT=ensure_dir(ROOT/'outputs'/'mat2')
def _fit(url):
    p=cached_download(url,OUT/'external'/Path(url).name); df=read_public_table(p); cols=[str(c) for c in df.columns]
    angle=next((c for c in cols if any(k in c.lower() for k in ['angle','misorientation','theta'])),None)
    val=next((c for c in cols if c!=angle and any(k in c.lower() for k in ['conduct','transmission','kappa','tbc','boundary'])),None)
    if not angle or not val: return {'url':url,'n_rows':int(len(df)),'columns_sample':cols[:50],'error':'required columns not recognized'}
    th=pd.to_numeric(df[angle],errors='coerce'); y=pd.to_numeric(df[val],errors='coerce'); m=th.notna()&y.notna()
    if m.sum()<8: return {'url':url,'error':'too few numeric rows'}
    rad=np.deg2rad(th[m].to_numpy(float)); yy=y[m].to_numpy(float); base=np.cos(rad/2)**2
    X0=np.column_stack([base,np.ones_like(base)]); X1=np.column_stack([base,base*np.cos(6*rad),np.ones_like(base)])
    b0=np.linalg.lstsq(X0,yy,rcond=None)[0]; b1=np.linalg.lstsq(X1,yy,rcond=None)[0]
    s0=float(np.sum((yy-X0@b0)**2)); s1=float(np.sum((yy-X1@b1)**2)); n=len(yy); a0=n*math.log(max(s0/n,1e-15))+4; a1=n*math.log(max(s1/n,1e-15))+6
    return {'url':url,'n_used':int(n),'angle_column':angle,'value_column':val,'aic_base':a0,'aic_cos6':a1,'delta_aic_cos6_minus_base':a1-a0,'epsilon_like_coeff':float(b1[1]),'verdict':'support_like_cos6' if a1+2<a0 else 'no_support_cos6'}
def main():
    urls=[u.strip() for u in os.environ.get('MAT2_NUMERIC_URLS','').split(',') if u.strip()]; fits=[]
    for u in urls:
        try: fits.append(_fit(u))
        except Exception as e: fits.append({'url':u,'error':repr(e)})
    status='ok_user_numeric_tables' if fits else 'not_executable_no_public_machine_table'; verdict='support_like_cos6' if any(f.get('verdict')=='support_like_cos6' for f in fits) else ('no_support_or_unusable_user_tables' if fits else 'no_physics_verdict')
    out=structured_report('MAT2',status,prediction='T(theta)=cos^2(theta/2)*(1+epsilon*cos(6 theta))',fits=fits,required_columns=['misorientation_angle_degrees','phonon_transmission_or_thermal_boundary_conductance','material_symmetry_or_lattice'],verdict=verdict,note='Improved script can run numeric fit from public CSV/XLSX URLs in MAT2_NUMERIC_URLS. No stable public machine table is bundled by default.')
    json_dump(out,OUT/'mat2_report.json'); print(json.dumps(out,indent=2))
if __name__=='__main__': main()
