#!/usr/bin/env python3
from _common_public import *


def _rotation_null(ra, dec, dens, vals, seed=12345):
    rng=np.random.default_rng(seed); n=len(vals); good=np.isfinite(vals)&np.isfinite(dens[:n])
    if good.sum()<20: return {'n':0}
    d=dens[:n][good]; v=vals[good]; q=np.nanmedian(d)
    obs=float(np.nanmean(v[d>=q])-np.nanmean(v[d<q]))
    null=[]
    for _ in range(120):
        vv=rng.permutation(v)
        null.append(float(np.nanmean(vv[d>=q])-np.nanmean(vv[d<q])))
    null=np.asarray(null,float)
    return {'n':int(len(null)),'observed_delta':obs,'mean':float(np.nanmean(null)),'sigma':float(np.nanstd(null,ddof=1)) if len(null)>1 else None,'p_one_sided_positive':float((np.sum(null>=obs)+1)/(len(null)+1))}


def main():
    args=build_parser('T04 Euclid Q1 density vs ACT DR6 κ map').parse_args()
    cache=ensure_dir(args.cache); outdir=ensure_dir(args.outdir)
    res=result_template('T04',['P14/P30'],'Cross-correlate Euclid Q1 source density with ACT DR6 CMB-lensing convergence map using candidate pixel maps or healpy alm2map.')
    res['falsification_logic']={'confirm_like':'high Euclid-density positions have higher ACT κ and positive density-κ Spearman correlation after shuffle/null controls','falsify_like':'correlation disappears/flips under density splits/nulls; masks/constant maps are data_limited'}
    eu,att=load_euclid_q1_sample(cache,timeout=args.timeout,max_rows=args.max_rows,force=args.force); res['data_sources'].extend(att)
    cand,att2=download_act_dr6_lensing_map_candidates(cache,timeout=args.timeout,allow_large=args.allow_large,force=args.force); res['data_sources'].extend(att2)
    if eu is None or not cand:
        res['status']='data_limited'; res['warnings'].append('Need Euclid Q1 public catalog and ACT κ candidate map; rerun with --allow-large for ACT map.')
        write_result(res,outdir); return
    ra=numeric_array(eu,'ra'); dec=numeric_array(eu,'dec'); m=np.isfinite(ra)&np.isfinite(dec)
    ra=ra[m][:args.max_rows]; dec=dec[m][:args.max_rows]
    dens=nearest_density(ra,dec,k=10)
    mp, vals, info = sample_first_valid_kappa_candidate(cand, ra, dec, max_points=min(args.max_rows,5000), prefer_healpy=getattr(args,'prefer_healpy',False), no_harmonic=getattr(args,'no_harmonic',False), min_finite=20)
    res['metrics']['candidate_map_selection']=info
    res['metrics']['candidate_diagnostic_summary']=info.get('candidate_diagnostic_summary', {}) if isinstance(info, dict) else {}
    if vals is None or mp is None:
        res['status']='data_limited'
        if info.get('requires_healpy'): res['warnings'].append('Only harmonic ACT products were usable; install healpy or use a public pixel κ map.')
        else: res['warnings'].append(info.get('error','No valid ACT κ map candidate after validation.'))
        write_result(res,outdir); return
    vcheck=info.get('map_validation',{}); label=vcheck.get('signal_label','kappa')
    res['metrics']['kappa_coordinate_validation']=kappa_coordinate_validation(mp, info.get('map_sampling',{}), ra, dec)
    n=len(vals); good=np.isfinite(vals)&np.isfinite(dens[:n])
    if good.sum()<20:
        res['status']='data_limited'; res['warnings'].append('Too few finite ACT κ samples after candidate validation.'); write_result(res,outdir); return
    corr=safe_corr(dens[:n][good],vals[good]); q=np.nanmedian(dens[:n][good])
    delta=bootstrap_mean_delta(vals[good][dens[:n][good]>=q],vals[good][dens[:n][good]<q],seed=args.seed)
    null=_rotation_null(ra[:n],dec[:n],dens[:n],vals,seed=args.seed)
    res['metrics'][f'density_{label}']={'selected_map':str(mp),'spearman':corr,f'high_minus_low_{label}':delta,'shuffle_null':null}
    if delta.get('delta') is not None and delta.get('boot_sigma'):
        pnull=null.get('p_one_sided_positive') if isinstance(null,dict) else None
        if delta['delta']>0 and delta['delta']/delta['boot_sigma']>=2 and (pnull is None or pnull<0.05): res['status']='suggestive'
        elif delta['delta']<=0: res['status']='null'
        else: res['status']='weak_or_null'
    else: res['status']='data_limited'
    write_result(res,outdir)
if __name__=='__main__': main()
