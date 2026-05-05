#!/usr/bin/env python3
from _common_public import *


def main():
    args=build_parser('T16 PTA × κ sky-position cross-link').parse_args()
    cache=ensure_dir(args.cache); outdir=ensure_dir(args.outdir)
    res=result_template('T16',['P8c','CL2'],'Use NANOGrav pulsar sky positions and public ACT/Planck κ products to test a PTA×κ sky cross-link proxy.')
    res['falsification_logic']={'confirm_like':'pulsar-associated PTA proxy correlates positively with κ map values','falsify_like':'random sky rotations match or exceed observed κ correlation; masks are data_limited'}
    arc,att=load_nanograv_archive(cache,timeout=args.timeout,force=args.force,allow_large=args.allow_large); res['data_sources'].extend(att)
    if arc is None:
        res['status']='data_limited'; res['warnings'].append('NANOGrav archive requires --allow-large.'); write_result(res,outdir); return
    root=extract_archive(arc,cache/'nanograv'/'extract')
    pos,info=parse_par_positions(root); res['metrics']['pulsar_position_parse']=info
    if pos is None or 'ra' not in pos.columns:
        res['status']='data_limited'; res['warnings'].append('Need parsed pulsar RA/Dec from public NANOGrav .par files.'); write_result(res,outdir); return
    ra=numeric_array(pos,'ra'); dec=numeric_array(pos,'dec')
    cand=[]; attall=[]
    c1,a1=download_act_dr6_lensing_map_candidates(cache,timeout=args.timeout,allow_large=args.allow_large,force=args.force); cand.extend(c1); attall.extend(a1)
    # If ACT candidates fail validation, Planck candidates are also tried below, not only when ACT is absent.
    c2,a2=download_planck_lensing_map_candidates(cache,timeout=args.timeout,allow_large=args.allow_large,force=args.force); cand.extend(c2); attall.extend(a2)
    res['data_sources'].extend(attall)
    if not cand:
        res['status']='data_limited'; res['warnings'].append('Need ACT or Planck κ candidate map; rerun with --allow-large.'); write_result(res,outdir); return
    mp, vals, mi = sample_first_valid_kappa_candidate(cand, ra, dec, max_points=len(pos), prefer_healpy=getattr(args,'prefer_healpy',False), no_harmonic=getattr(args,'no_harmonic',False), min_finite=5)
    res['metrics']['candidate_map_selection']=mi
    if vals is None or mp is None:
        res['status']='data_limited'
        if mi.get('requires_healpy'): res['warnings'].append('Only harmonic κ products were available or reconstruction failed; install healpy and inspect alm diagnostics.')
        else: res['warnings'].append(mi.get('error','No valid κ candidate for PTA sky positions.'))
        write_result(res,outdir); return
    vcheck=mi.get('map_validation',{}); label=vcheck.get('signal_label','kappa')
    obs=float(np.nanmean(vals)); rng=np.random.default_rng(args.seed); null=[]
    for _ in range(120):
        vv,_info=sample_map_values_for_points(mp,(ra+rng.uniform(0,360))%360,dec,max_points=len(pos),prefer_healpix=getattr(args,'prefer_healpy',False) or healpy_available())
        vc=validate_map_sample_values(vv,_info,mp,min_finite=5)
        if vv is not None and vc.get('ok'):
            null.append(float(np.nanmean(vv)))
    z=(obs-np.mean(null))/(np.std(null,ddof=1) or np.nan) if null else None
    res['metrics'][f'pulsar_{label}_mean_vs_ra_rotation_null']={'selected_map':str(mp),'observed_mean':obs,'null_mean':float(np.mean(null)) if null else None,'null_sigma':float(np.std(null,ddof=1)) if len(null)>1 else None,'z':float(z) if z is not None and np.isfinite(z) else None,'n_null':int(len(null))}
    res['status']='suggestive' if z is not None and z>2 else ('null' if z is not None else 'data_limited')
    write_result(res,outdir)
if __name__=='__main__': main()
