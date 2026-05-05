#!/usr/bin/env python3
from _common_public import *


def _curve_audit(df):
    mcol,lcol=choose_mass_limit_columns(df)
    if not (mcol and lcol): return None
    mass=numeric_array(df,mcol); lim=numeric_array(df,lcol)
    q=np.isfinite(mass)&np.isfinite(lim)&(mass>0)&(lim>0)
    if q.sum()<5: return None
    mass=mass[q]; lim=lim[q]
    o=np.argsort(mass); mass=mass[o]; lim=lim[o]
    window=(mass>=500)&(mass<=3000)
    best_idx=int(np.nanargmin(lim))
    endpoint_best=best_idx in (0,len(mass)-1)
    return {'mass_col':str(mcol),'limit_col':str(lcol),'n':int(len(mass)),'mass_min':float(np.nanmin(mass)),'mass_max':float(np.nanmax(mass)),'covers_500_3000_GeV':bool(np.nanmin(mass)<=500 and np.nanmax(mass)>=3000),'n_points_in_500_3000':int(window.sum()),'best_mass_GeV':float(mass[best_idx]),'best_limit':float(lim[best_idx]),'best_limit_is_endpoint':bool(endpoint_best),'best_limit_in_window':float(np.nanmin(lim[window])) if np.any(window) else None}


def main():
    args=build_parser('T18 direct-detection predicted-window readiness audit').parse_args()
    cache=ensure_dir(args.cache); outdir=ensure_dir(args.outdir)
    res=result_template('T18',['P10/P25','P27'],'Audit public direct-detection curve coverage of the CCDR 500-3000 GeV readiness window without claiming event-level peaks.')
    res['prediction_names']=['P10/P25 — DM peak-counting → N','P27 — geometric mass-tower ratios']
    res['metrics']['conservative_evidence_guard']=conservative_evidence_guard('direct_detection')
    res['falsification_logic']={'confirm_like':'event-level public data cover predicted mass window and show reproducible peak structure','falsify_like':'event-level public data fully cover window with no structure'}
    tables,att=load_direct_detection_public_curves(cache,timeout=args.timeout,force=args.force); res['data_sources'].extend(att)
    audits=[]
    for df in tables:
        a=_curve_audit(df)
        if a: audits.append(a)
    covers=sum(1 for a in audits if a['covers_500_3000_GeV'])
    adequate=sum(1 for a in audits if a['covers_500_3000_GeV'] and a['n_points_in_500_3000']>=3)
    res['metrics']={'curve_audits':audits,'n_curves_covering_window':covers,'n_curves_with_3plus_window_points':adequate,'event_level_data_available':False,'note':'v9.6 separates readiness/window coverage from peak evidence.'}
    if adequate>0:
        res['status']='readiness_only'
    elif audits:
        res['status']='not_ready'
    else:
        res['status']='data_limited'
    res.setdefault('notes',[]).append('v9.7 guardrail: limit-curve scans are readiness-only; no peak evidence is claimed without public event-level likelihoods or binned events.')
    write_result(res,outdir)
if __name__=='__main__': main()
