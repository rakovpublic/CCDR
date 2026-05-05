#!/usr/bin/env python3
from _common_public import *


def main():
    args = build_parser('T03 Pantheon+ low-z systematic isolation for covariance ν-like fit').parse_args()
    cache=ensure_dir(args.cache); outdir=ensure_dir(args.outdir)
    res=result_template('T03',['P1','P39','CL3'],'Repeat covariance-aware ν-like Pantheon+ fit under low-z cuts to test whether signal is low-z-systematics carried.')
    res['prediction_names']=['P1 — RVM BAO scale shift','P39 — secular Λ-drift from bulk cooling','CL3 — ν cross-link']
    res['falsification_logic']={'confirm_like':'ν-like estimate remains positive, non-bound-hit, and non-low-z-carried under cuts','falsify_like':'ν-like estimate is bound-dominated or carried entirely by low-z choices'}
    df, att=load_pantheon(cache,timeout=args.timeout,force=args.force,max_rows=None)
    res['data_sources'].extend(att)
    cov, catt=load_pantheon_covariance(cache,timeout=args.timeout,force=args.force)
    res['data_sources'].extend(catt)
    if df is None or pd is None:
        res['warnings'].append('Pantheon+ could not be parsed.')
        write_result(res,outdir); return
    cuts=[0.005,0.01,0.02,0.03,0.05,0.08]
    fits=[]
    for zmin in cuts:
        fit=fit_pantheon_nu_cov_like(df,cov,zmin=zmin)
        fit['zmin']=zmin
        fits.append(fit)
    res['metrics']['cut_fits_covariance']=fits
    # Also keep diagonal proxy for backward comparison, but do not score it.
    diag=[]
    zc, _, _ = pantheon_columns(df)
    if zc:
        for zmin in cuts:
            sub=df[pd.to_numeric(df[zc],errors='coerce')>=zmin].copy()
            f=fit_pantheon_nu_like(sub); f['zmin']=zmin; diag.append(f)
    res['metrics']['cut_fits_diagonal_proxy_crosscheck']=diag
    nus=[f.get('nu_like') for f in fits if isinstance(f.get('nu_like'),float) and np.isfinite(f.get('nu_like'))]
    bound_hits=sum(1 for f in fits if is_bound_hit_metric(f))
    if len(nus)>=3:
        res['metrics']['nu_median']=float(np.nanmedian(nus)); res['metrics']['nu_std']=float(np.nanstd(nus))
        res['metrics']['bound_hit_fraction']=float(bound_hits/len(fits))
        res['metrics']['nu_change_zmin_0p005_to_0p08']=float(abs(nus[-1]-nus[0])) if len(nus)==len(fits) else None
    if bound_hits:
        res['status']='diagnostic_only'
        res['warnings'].append(f'{bound_hits}/{len(fits)} low-z cuts hit the ν bound; no P39 support claimed.')
    elif len(nus)>=3 and np.nanmedian(nus)>0 and np.nanstd(nus)<0.05:
        res['status']='covariance_suggestive'
    elif len(nus)>=3:
        res['status']='null'
    else:
        res['status']='data_limited'
    write_result(res,outdir)

if __name__=='__main__': main()
