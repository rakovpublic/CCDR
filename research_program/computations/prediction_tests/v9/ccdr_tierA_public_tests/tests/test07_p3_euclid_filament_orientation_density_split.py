#!/usr/bin/env python3
from _common_public import *


def _field_labels(ra):
    order=np.argsort(ra); labels=np.full(len(ra),-1,dtype=int)
    if len(ra)==0: return labels
    sr=ra[order]; gaps=np.diff(sr)
    cut=max(10.0,5.0*np.nanmedian(gaps[gaps>0]) if np.any(gaps>0) else 10.0)
    starts=[0]+[int(i+1) for i,g in enumerate(gaps) if g>cut]+[len(ra)]
    lab=0
    for a,b in zip(starts[:-1],starts[1:]):
        idx=order[a:b]
        if len(idx)>=50:
            labels[idx]=lab; lab+=1
    if lab==0: labels[:]=0
    return labels


def _paired_mean_diff(hi,lo,min_pairs=200):
    diffs=[]; weights=[]
    for h,l in zip(hi,lo):
        if h.get('corr') is not None and l.get('corr') is not None and h.get('n_pairs',0)>min_pairs and l.get('n_pairs',0)>min_pairs:
            w=min(h.get('n_pairs',0),l.get('n_pairs',0))
            diffs.append(h['corr']-l['corr']); weights.append(w)
    if not diffs: return None,0
    return float(np.average(diffs,weights=weights)), int(np.sum(weights))


def _residualize_density_against_depth(dens, depth):
    if depth is None or len(depth)!=len(dens) or np.isfinite(depth).sum()<20 or np.nanstd(depth[np.isfinite(depth)])==0:
        return dens, {'depth_residualized': False}
    m=np.isfinite(dens)&np.isfinite(depth)
    X=np.vstack([np.ones(np.sum(m)), depth[m]]).T
    try:
        beta=np.linalg.lstsq(X, np.log(np.maximum(dens[m],1e-12)), rcond=None)[0]
        out=np.asarray(dens,float).copy()
        pred=X@beta
        out[m]=np.log(np.maximum(dens[m],1e-12))-pred
        return out, {'depth_residualized': True, 'beta': [float(x) for x in beta], 'n': int(np.sum(m))}
    except Exception as e:
        return dens, {'depth_residualized': False, 'error': str(e)}


def main():
    args=build_parser('T07 field/depth-aware Euclid Q1 filament orientation-density split').parse_args()
    cache=ensure_dir(args.cache); outdir=ensure_dir(args.outdir)
    res=result_template('T07',['P3','CL4'],'Estimate local filament axes from Euclid Q1 source positions and compare orientation correlations in high-vs-low residual-density regions.')
    res['falsification_logic']={'confirm_like':'high residual-density orientation correlation exceeds low-density correlation inside connected Q1 fields and beats within-field density/depth-shuffle null','falsify_like':'density-stratified orientation signal is absent/opposite or indistinguishable from within-field null'}
    df,att=load_euclid_q1_sample(cache,timeout=args.timeout,max_rows=args.max_rows,force=args.force); res['data_sources'].extend(att)
    if df is None:
        res['warnings'].append('No Euclid Q1 sample from IRSA TAP.'); write_result(res,outdir); return
    ra0=numeric_array(df,'ra'); dec0=numeric_array(df,'dec')
    good0=np.isfinite(ra0)&np.isfinite(dec0); ra=ra0[good0]; dec=dec0[good0]
    dproxy,dinfo=euclid_depth_proxy(df)
    depth=dproxy[good0] if dproxy is not None and len(dproxy)==len(df) else None
    labels=_field_labels(ra)
    photoz = numeric_array(df,'z')[good0] if 'z' in getattr(df,'columns',[]) else None
    matched_controls = euclid_field_depth_photoz_matched_controls(ra,dec,labels,depth,photoz,seed=args.seed)
    bins=[0.0,0.05,0.1,0.2,0.5,1.0,2.0,4.0]
    all_hi=[]; all_lo=[]; summaries=[]
    rng=np.random.default_rng(args.seed)
    null_diffs=[]; residualization=[]
    for lab in sorted(set(labels)):
        if lab<0: continue
        mm=labels==lab
        if mm.sum()<100: continue
        r=ra[mm]; d=dec[mm]; dep=depth[mm] if depth is not None else None
        dens=nearest_density(r,d,k=10)
        dens_use, rinfo=_residualize_density_against_depth(dens,dep); residualization.append({'field':int(lab),**rinfo})
        ang=local_orientation_angles(r,d,k=12)
        q=np.nanmedian(dens_use); high=dens_use>=q; low=dens_use<q
        hi=orientation_correlation(r[high],d[high],ang[high],bins)
        lo=orientation_correlation(r[low],d[low],ang[low],bins)
        md,w=_paired_mean_diff(hi,lo)
        all_hi.extend(hi); all_lo.extend(lo)
        summaries.append({'field':int(lab),'n':int(mm.sum()),'density_split_median':float(q),'field_mean_high_minus_low':md,'field_weight':w})
        for _ in range(50):
            sh=np.zeros(len(r),dtype=bool)
            if dep is not None and np.isfinite(dep).sum()>=20:
                # Depth-stratified reassignment: preserve number of high-density objects in quartiles.
                qs=np.nanquantile(dep[np.isfinite(dep)], [0,0.25,0.5,0.75,1.0])
                for a,b in zip(qs[:-1],qs[1:]):
                    binm=np.isfinite(dep)&(dep>=a)&(dep<=b)
                    nh=int(np.sum(high&binm))
                    idx=np.where(binm)[0]
                    if len(idx) and nh>0:
                        sh[rng.choice(idx,size=min(nh,len(idx)),replace=False)]=True
            else:
                sh[rng.choice(len(r),size=int(np.sum(high)),replace=False)]=True
            if sh.sum()<10 or (~sh).sum()<10: continue
            nhi=orientation_correlation(r[sh],d[sh],ang[sh],bins)
            nlo=orientation_correlation(r[~sh],d[~sh],ang[~sh],bins)
            nd,_=_paired_mean_diff(nhi,nlo)
            if nd is not None: null_diffs.append(nd)
    mean_diff,total_w=_paired_mean_diff(all_hi,all_lo)
    null=np.asarray(null_diffs,float)
    z=None; p_one_sided=None
    if mean_diff is not None and len(null)>5 and np.nanstd(null)>0:
        z=float((mean_diff-np.nanmean(null))/np.nanstd(null,ddof=1))
        p_one_sided=float((np.sum(null>=mean_diff)+1)/(len(null)+1))
    res['metrics']={'n_sources':int(len(ra)),'field_summaries':summaries,'bins_deg':bins,'high_density_corr':all_hi,'low_density_corr':all_lo,'mean_high_minus_low_corr':mean_diff,'weighted_pair_count':total_w,'depth_proxy':dinfo,'density_residualization':residualization,'density_depth_stratified_null':{'n':int(len(null)),'mean':float(np.nanmean(null)) if len(null) else None,'sigma':float(np.nanstd(null,ddof=1)) if len(null)>1 else None,'z_vs_null':z,'p_one_sided_high_gt_low':p_one_sided},'field_depth_photoz_matched_controls':matched_controls}
    if mean_diff is None:
        res['status']='data_limited'
    else:
        # v9.8: negative density split is formal tension unless an independent endpoint/skeleton catalogue reverses it.
        res['status']=t07_formal_status(mean_diff, p_one_sided, independent_endpoint_support=False)
    if depth is None:
        res['warnings'].append('No public depth/magnitude proxy available; density split may still carry survey-depth systematics.')
    if mean_diff is not None and mean_diff<0:
        res['warnings'].append('High-density minus low-density orientation correlation is negative after v9.6 controls; this is formal tension for P3 unless a 3D/photo-z/skeleton or independent endpoint catalogue reverses it.')
    write_result(res,outdir)
if __name__=='__main__': main()
