#!/usr/bin/env python3
from _common_public import *

def _read_planck_spectrum(path, max_rows=None):
    if pd is None:
        return None
    candidates=[]
    for kwargs in [dict(sep=r"\s+", comment="#", header=None), dict(sep=r"\s+", comment="#")]:
        try:
            df=pd.read_csv(path,engine="python",nrows=max_rows,**kwargs)
            candidates.append(df)
        except Exception:
            pass
    try:
        candidates.append(read_planck_lowell_numeric_table(path,max_rows=max_rows))
    except Exception:
        pass
    try:
        candidates.append(read_table_any(path,max_rows=max_rows))
    except Exception:
        pass
    d=first_nonempty_dataframe(*candidates)
    if d is not None and getattr(d,'shape',(0,0))[1] >= 2 and getattr(d,'shape',(0,0))[0] > 10:
        return d
    return None

def main():
    args=build_parser('T22 Planck low-l large-angle no-map proxy').parse_args()
    cache=ensure_dir(args.cache); outdir=ensure_dir(args.outdir)
    res=result_template('T22',['P4'],'Use public Planck PR3 spectra/tables, not HEALPix maps, to screen low-l large-angle anomalies.')
    res['falsification_logic']={'confirm_like':'low-l power/residual pattern differs from high-l continuation in predicted anomaly direction','falsify_like':'low-l table values are consistent with smooth ΛCDM continuation'}
    paths,att=download_planck_spectrum_candidates(cache,timeout=args.timeout,force=args.force,max_files=8); res['data_sources'].extend(att)
    if not paths:
        write_result(res,outdir); return
    p=None; df=None; parse_attempts=[]
    for cand in paths:
        d=_read_planck_spectrum(cand,max_rows=args.max_rows)
        parse_attempts.append({'path':str(cand),'parsed': bool(d is not None and d.shape[1]>=2), 'shape': list(d.shape) if d is not None else None})
        if d is not None and d.shape[1]>=2:
            p=cand; df=d; break
    res['metrics']['parse_attempts']=parse_attempts
    if df is None or df.shape[1]<2:
        res['warnings'].append('Downloaded Planck spectrum candidates but none were parseable as simple numeric tables.'); write_result(res,outdir); return
    nums=find_numeric_columns(df)
    if len(nums)<2: write_result(res,outdir); return
    ell=numeric_array(df,nums[0]); cl=numeric_array(df,nums[1]); m=np.isfinite(ell)&np.isfinite(cl)&(ell>1)
    ell=ell[m]; cl=cl[m]
    low=cl[(ell>=2)&(ell<=30)]; mid=cl[(ell>30)&(ell<=200)]
    approx=planck_lowell_approx_likelihood(ell, cl)
    res['metrics']={'path':str(p),'ell_col':str(nums[0]),'cl_col':str(nums[1]),'n_low':int(len(low)),'n_mid':int(len(mid)),'low_mean':float(np.mean(low)) if len(low) else None,'mid_mean':float(np.mean(mid)) if len(mid) else None,'low_over_mid':float(np.mean(low)/np.mean(mid)) if len(low) and len(mid) and np.mean(mid)!=0 else None,'approx_lowell_likelihood':approx}
    ratio=res['metrics']['low_over_mid']; papprox=approx.get('p_value_chi2_approx') if isinstance(approx,dict) else None; zapprox=approx.get('chi2_zscore_approx') if isinstance(approx,dict) else None
    if papprox is not None and papprox<0.05:
        res['status']='approx_likelihood_suggestive'
    else:
        res['status']='diagnostic_suggestive' if ratio is not None and (ratio<0.8 or ratio>1.2) else ('null' if ratio is not None else 'data_limited')
    if res['status'] in ('diagnostic_suggestive','approx_likelihood_suggestive'):
        res['warnings'].append('Low-l screen uses approximate cosmic-variance likelihood, not the official Planck likelihood; do not treat as hard evidence.')
    write_result(res,outdir)
if __name__=='__main__': main()
