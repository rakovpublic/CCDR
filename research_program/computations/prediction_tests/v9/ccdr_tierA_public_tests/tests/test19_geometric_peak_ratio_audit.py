#!/usr/bin/env python3
from _common_public import *


def main():
    args=build_parser('T19 artifact-controlled geometric peak-ratio audit on public DD limit curves').parse_args()
    cache=ensure_dir(args.cache); outdir=ensure_dir(args.outdir)
    res=result_template('T19',['P27'],'Search public direct-detection limit curves for repeated mass-domain extrema, with endpoint/grid-artifact rejection.')
    res['prediction_names']=['P27 — geometric mass-tower ratios']
    res['metrics']['conservative_evidence_guard']=conservative_evidence_guard('direct_detection')
    res['falsification_logic']={'confirm_like':'same non-endpoint geometric extrema appear across independent public curves after smoothing/artifact controls','falsify_like':'extrema are absent or explained by endpoints, interpolation knots, or scan grids'}
    tables,att=load_direct_detection_public_curves(cache,timeout=args.timeout,force=args.force); res['data_sources'].extend(att)
    extrema=[]; nonartifact_masses=[]
    for curve_idx,df in enumerate(tables):
        mcol,lcol=choose_mass_limit_columns(df)
        if not (mcol and lcol): continue
        mass=numeric_array(df,mcol); lim=numeric_array(df,lcol); q=np.isfinite(mass)&np.isfinite(lim)&(mass>0)&(lim>0)
        mass=mass[q]; lim=lim[q]
        if len(mass)<25: continue
        o=np.argsort(mass); x=np.log10(mass[o]); y=np.log10(lim[o])
        # Remove duplicate mass grid points.
        uniq=np.r_[True, np.diff(x)>1e-8]; x=x[uniq]; y=y[uniq]
        if len(x)<25: continue
        coef=np.polyfit(x,y,min(3,len(x)-2)); resi=y-np.polyval(coef,x)
        smooth=resi.copy()
        if signal is not None and len(resi)>=9:
            try: smooth=signal.savgol_filter(resi, 9 if len(resi)>=9 else len(resi)//2*2+1, 2)
            except Exception: pass
            peaks,_=signal.find_peaks(-smooth,prominence=max(np.nanstd(smooth),1e-12))
        else:
            peaks=np.array([i for i in range(1,len(smooth)-1) if smooth[i]<smooth[i-1] and smooth[i]<smooth[i+1]])
        rows=[]
        for i in peaks:
            # artifact controls: exclude endpoints/near-endpoints and sparse-grid neighbors.
            endpoint = i < 2 or i > len(x)-3
            grid_left = x[i]-x[i-1] if i>0 else np.nan
            grid_right = x[i+1]-x[i] if i<len(x)-1 else np.nan
            sparse = bool(np.isfinite(grid_left) and np.isfinite(grid_right) and max(grid_left,grid_right)>3*max(min(grid_left,grid_right),1e-12))
            artifact = endpoint or sparse
            mass_i=float(10**x[i])
            rows.append({'mass_GeV':mass_i,'residual':float(smooth[i]),'endpoint_or_near_endpoint':bool(endpoint),'sparse_grid_neighbor':bool(sparse),'artifact_candidate':bool(artifact)})
            if not artifact:
                nonartifact_masses.append(mass_i)
        good=[r['mass_GeV'] for r in rows if not r['artifact_candidate']]
        ratios=(np.array(good[1:])/np.array(good[:-1])).tolist() if len(good)>1 else []
        extrema.append({'curve_index':curve_idx,'mass_col':str(mcol),'limit_col':str(lcol),'n_points':int(len(x)),'n_raw_extrema':int(len(rows)),'n_nonartifact_extrema':int(len(good)),'nonartifact_masses_GeV':[float(v) for v in good[:20]],'consecutive_nonartifact_ratios':[float(v) for v in ratios[:20]],'raw_extrema_first20':rows[:20]})
    # Cross-curve reproducibility: count masses recurring within 0.15 dex.
    reproducible=0
    if len(nonartifact_masses)>=2:
        lx=np.sort(np.log10(nonartifact_masses))
        for i,v in enumerate(lx):
            if np.sum(np.abs(lx-v)<0.15)>=2:
                reproducible+=1
    res['metrics']={'extrema_audit':extrema,'n_nonartifact_masses_total':len(nonartifact_masses),'reproducible_within_0p15dex_count':int(reproducible),'note':'Limit-curve extrema are not event-level peaks; v9.6 requires non-endpoint recurrence before suggestive status.'}
    if reproducible>=2:
        res['status']='readiness_only_artifact_controlled_pattern'
        res['warnings'].append('Non-endpoint extrema recur in limit curves, but this remains readiness-only without event-level likelihoods or binned event data.')
    elif extrema:
        res['status']='artifact_candidate_or_null'
    else:
        res['status']='data_limited'
    res.setdefault('notes',[]).append('v9.7 guardrail: limit-curve scans are readiness-only; no peak evidence is claimed without public event-level likelihoods or binned events.')
    write_result(res,outdir)
if __name__=='__main__': main()
