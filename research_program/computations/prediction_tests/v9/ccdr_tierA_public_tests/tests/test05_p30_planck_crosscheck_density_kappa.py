#!/usr/bin/env python3
from _common_public import *


def main():
    args=build_parser('T05 Euclid Q1 density vs Planck PR4 κ map cross-check').parse_args()
    cache=ensure_dir(args.cache); outdir=ensure_dir(args.outdir)
    res=result_template('T05',['P14/P30'],'Cross-check Euclid density-κ sign using public Planck PR3/PR4 lensing products; harmonic klm/alm requires healpy.alm2map.')
    res['falsification_logic']={'confirm_like':'same-sign density-κ correlation as ACT, weaker but positive','falsify_like':'Planck κ map gives opposite sign or no correlation; masks/constant maps are data_limited, not null'}
    eu,att=load_euclid_q1_sample(cache,timeout=args.timeout,max_rows=args.max_rows,force=args.force); res['data_sources'].extend(att)
    cand,att2=download_planck_lensing_map_candidates(cache,timeout=args.timeout,allow_large=args.allow_large,force=args.force); res['data_sources'].extend(att2)
    if eu is None or not cand:
        res['status']='data_limited'; res['warnings'].append('Requires Euclid Q1 sample and Planck κ/klm map; use --allow-large for map-level public product.')
        write_result(res,outdir); return
    ra=numeric_array(eu,'ra'); dec=numeric_array(eu,'dec'); m=np.isfinite(ra)&np.isfinite(dec)
    ra=ra[m][:args.max_rows]; dec=dec[m][:args.max_rows]
    dens=nearest_density(ra,dec,k=10)
    mp, vals, info = sample_first_valid_kappa_candidate(cand, ra, dec, max_points=min(args.max_rows,5000), prefer_healpy=getattr(args,'prefer_healpy',False), no_harmonic=getattr(args,'no_harmonic',False), min_finite=20)
    res['metrics']['candidate_map_selection']=info
    res['metrics']['candidate_diagnostic_summary']=info.get('candidate_diagnostic_summary', {}) if isinstance(info, dict) else {}
    if vals is None or mp is None:
        res['status']='data_limited'
        if info.get('requires_healpy'): res['warnings'].append('Only harmonic Planck products were usable; install healpy or use a public pixel κ map.')
        else: res['warnings'].append(info.get('error','No valid Planck κ map candidate after validation.'))
        write_result(res,outdir); return
    vcheck=info.get('map_validation',{}); label=vcheck.get('signal_label','kappa')
    res['metrics']['kappa_coordinate_validation']=kappa_coordinate_validation(mp, info.get('map_sampling',{}), ra, dec)
    n=len(vals); good=np.isfinite(vals)&np.isfinite(dens[:n]); d=dens[:n][good]; v=vals[good]
    if len(v)<20:
        res['status']='data_limited'; res['warnings'].append('Too few finite Planck κ samples after candidate validation.'); write_result(res,outdir); return
    q=np.nanmedian(d); delta=bootstrap_mean_delta(v[d>=q],v[d<q],seed=args.seed)
    rng=np.random.default_rng(args.seed); null=[]
    for _ in range(120):
        vv=rng.permutation(v); null.append(float(np.nanmean(vv[d>=q])-np.nanmean(vv[d<q])))
    null=np.asarray(null,float)
    res['metrics'][f'density_planck_{label}']={'selected_map':str(mp),'spearman':safe_corr(d,v),f'high_minus_low_{label}':delta,'shuffle_null':{'n':int(len(null)),'mean':float(np.mean(null)),'sigma':float(np.std(null,ddof=1)) if len(null)>1 else None,'p_one_sided_positive':float((np.sum(null>=delta.get('delta',np.nan))+1)/(len(null)+1)) if delta.get('delta') is not None else None}}
    res['status']=classify_by_sign(delta.get('delta'),positive_confirm=True,sigma=delta.get('boot_sigma')) if delta.get('delta') is not None else 'data_limited'
    write_result(res,outdir)
if __name__=='__main__': main()
