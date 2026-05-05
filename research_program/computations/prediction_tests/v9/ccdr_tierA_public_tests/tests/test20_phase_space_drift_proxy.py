#!/usr/bin/env python3
from _common_public import *


def _extract_year_from_sources(att, idx):
    text=' '.join(str(a.get('url',''))+' '+str(a.get('path','')) for a in att)
    years=[int(y) for y in re.findall(r'20[0-9]{2}', text)]
    return years[idx % len(years)] if years else None


def main():
    args=build_parser('T20 phase-space/mass-peak drift readiness proxy').parse_args()
    cache=ensure_dir(args.cache); outdir=ensure_dir(args.outdir)
    res=result_template('T20',['P37'],'Compare public direct-detection curve extrema/coverage across releases as a proxy for live-DM mass drift, with metadata safeguards.')
    res['prediction_names']=['P37 — phase-space drift from live DM']
    res['metrics']['conservative_evidence_guard']=conservative_evidence_guard('direct_detection')
    res['falsification_logic']={'confirm_like':'non-endpoint extremum mass shifts coherently with verified release dates','falsify_like':'no drift beyond sensitivity/scan-grid changes'}
    tables,att=load_direct_detection_public_curves(cache,timeout=args.timeout,force=args.force); res['data_sources'].extend(att)
    rows=[]
    for idx,df in enumerate(tables):
        mcol,lcol=choose_mass_limit_columns(df)
        if not (mcol and lcol): continue
        mass=numeric_array(df,mcol); lim=numeric_array(df,lcol); q=np.isfinite(mass)&np.isfinite(lim)&(mass>0)&(lim>0); mass=mass[q]; lim=lim[q]
        if len(mass)<8: continue
        o=np.argsort(mass); mass=mass[o]; lim=lim[o]
        j=int(np.nanargmin(lim)); endpoint=j in (0,len(mass)-1)
        rows.append({'curve_index':idx,'release_year_guess':_extract_year_from_sources(att,idx),'best_mass_GeV':float(mass[j]),'best_limit':float(lim[j]),'best_is_endpoint':bool(endpoint),'mass_min':float(np.nanmin(mass)),'mass_max':float(np.nanmax(mass))})
    usable=[r for r in rows if r.get('release_year_guess') and not r.get('best_is_endpoint')]
    coef=None
    if len(usable)>=3:
        coef=safe_polyfit([r['release_year_guess'] for r in usable],[np.log10(r['best_mass_GeV']) for r in usable],1)
    res['metrics']={'release_order_best_masses':rows,'usable_nonendpoint_year_drift_points':usable,'log_best_mass_vs_release_year_coef':coef,'note':'v9.6 refuses to score drift from arbitrary curve index or endpoint minima.'}
    if coef and abs(coef[0])>0.02:
        res['status']='readiness_only_drift_proxy'
        res['warnings'].append('Release-year trend in limit-curve minima is readiness-only; no event-level drift evidence is claimed.')
    elif usable:
        res['status']='null'
    elif rows:
        res['status']='readiness_only'
    else:
        res['status']='data_limited'
    res.setdefault('notes',[]).append('v9.7 guardrail: limit-curve scans are readiness-only; no peak evidence is claimed without public event-level likelihoods or binned events.')
    write_result(res,outdir)
if __name__=='__main__': main()
