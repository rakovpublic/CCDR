#!/usr/bin/env python3
from _common_public import *


def _field_labels(ra, dec):
    ra = np.asarray(ra, float)
    order = np.argsort(ra)
    labels = np.full(len(ra), -1, dtype=int)
    if len(ra) == 0:
        return labels
    sorted_ra = ra[order]
    gaps = np.diff(sorted_ra)
    cut = max(10.0, 5.0 * np.nanmedian(gaps[gaps > 0]) if np.any(gaps > 0) else 10.0)
    starts = [0] + [int(i+1) for i,g in enumerate(gaps) if g > cut] + [len(ra)]
    lab = 0
    for a,b in zip(starts[:-1], starts[1:]):
        idx = order[a:b]
        if len(idx) >= 20:
            labels[idx] = lab; lab += 1
    if lab == 0:
        labels[:] = 0
    return labels


def _patch_cv_within_fields(ra, dec, labels, nbin=4):
    cvs = []
    patch_counts = []
    for lab in sorted(set(labels)):
        if lab < 0:
            continue
        m = labels == lab
        if m.sum() < nbin * nbin * 3:
            continue
        r = ra[m]; d = dec[m]
        rb = np.linspace(np.nanmin(r), np.nanmax(r), nbin+1)
        db = np.linspace(np.nanmin(d), np.nanmax(d), nbin+1)
        counts = []
        for i in range(nbin):
            for j in range(nbin):
                mm = (r >= rb[i]) & ((r < rb[i+1]) if i < nbin-1 else (r <= rb[i+1])) & (d >= db[j]) & ((d < db[j+1]) if j < nbin-1 else (d <= db[j+1]))
                counts.append(int(mm.sum()))
        counts = np.asarray(counts, float)
        if counts.mean() > 0:
            cvs.append(float(counts.std(ddof=1) / counts.mean()))
            patch_counts.extend(counts.tolist())
    return float(np.nanmean(cvs)) if cvs else None, patch_counts


def main():
    args=build_parser('T06 Euclid Q1 field/depth-aware patch-spread audit').parse_args()
    cache=ensure_dir(args.cache); outdir=ensure_dir(args.outdir)
    res=result_template('T06',['OP13','P30'],'Measure source-density patch spread across Euclid Q1 fields while auditing depth/mask proxies.')
    res['falsification_logic']={'confirm_like':'patch spread exceeds randomized within-field null and is not explained by public depth/magnitude proxies','falsify_like':'patch spread matches mask/depth/random null'}
    df,att=load_euclid_q1_sample(cache,timeout=args.timeout,max_rows=args.max_rows,force=args.force); res['data_sources'].extend(att)
    if df is None:
        res['warnings'].append('No Euclid Q1 sample from IRSA TAP.')
        write_result(res,outdir); return
    ra=numeric_array(df,'ra'); dec=numeric_array(df,'dec')
    m=np.isfinite(ra)&np.isfinite(dec); ra=ra[m]; dec=dec[m]
    dproxy,dinfo=euclid_depth_proxy(df)
    depth=dproxy[m] if dproxy is not None and len(dproxy)==len(df) else None
    labels=_field_labels(ra,dec)
    cv, counts = _patch_cv_within_fields(ra,dec,labels)
    rng=np.random.default_rng(args.seed)
    null=[]
    for _ in range(200):
        rr=ra.copy(); dd=dec.copy()
        for lab in sorted(set(labels)):
            mm=labels==lab
            if mm.sum()<5: continue
            rr[mm]=rng.uniform(np.nanmin(ra[mm]),np.nanmax(ra[mm]),mm.sum())
            dd[mm]=rng.uniform(np.nanmin(dec[mm]),np.nanmax(dec[mm]),mm.sum())
        c,_counts=_patch_cv_within_fields(rr,dd,labels)
        if c is not None and np.isfinite(c): null.append(c)
    null=np.asarray(null,float)
    z=None
    if cv is not None and len(null)>10 and np.std(null)>0:
        z=float((cv-np.mean(null))/np.std(null,ddof=1))
    photoz = numeric_array(df,'z')[m] if 'z' in getattr(df,'columns',[]) else None
    depth_diag=patch_count_depth_diagnostic(ra,dec,labels,depth)
    matched_controls=euclid_field_depth_photoz_matched_controls(ra,dec,labels,depth,photoz,seed=args.seed)
    res['metrics']={'n_sources':int(len(ra)),'n_fields':int(len(set(labels[labels>=0]))),'field_counts':{str(int(l)):int((labels==l).sum()) for l in sorted(set(labels)) if l>=0},'patch_cv_field_aware':cv,'uniform_within_field_null_mean':float(np.mean(null)) if len(null) else None,'uniform_within_field_null_sigma':float(np.std(null,ddof=1)) if len(null)>1 else None,'z_vs_uniform_null':z,'depth_proxy':dinfo,'depth_or_mask_diagnostic':depth_diag,'field_depth_photoz_matched_controls':matched_controls}
    depth_corr=depth_diag.get('count_depth_spearman',{}).get('rho') if isinstance(depth_diag.get('count_depth_spearman'),dict) else None
    if z is None:
        res['status']='data_limited'
    elif depth is None:
        res['status']='mask_risk_suggestive' if z>2 else 'null'
        res['warnings'].append('No Euclid depth/magnitude proxy was available; large patch spread can still be survey mask/depth, not CCDR.')
    else:
        res['status']=t06_matched_status(z, depth_corr, matched_controls)
        if res['status']=='depth_confounded':
            res['warnings'].append('Patch count spread is strongly correlated with public depth/magnitude proxy; do not count as P30 support.')
        elif res['status']=='matched_statistic_suggestive':
            res['warnings'].append('Status is based on field/depth/photo-z matched controls; raw patch spread alone is not counted.')
    write_result(res,outdir)
if __name__=='__main__': main()
