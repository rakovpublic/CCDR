#!/usr/bin/env python3
from _common_public import *

def _best_void_radius_column(df):
    nums=find_numeric_columns(df)
    if not nums: return None
    for c in nums:
        if re.search(r'(^r$|radius|rad|reff|eff|r_eff)',str(c),re.I): return c
    candidates=[]
    for c in nums:
        x=numeric_array(df,c); x=x[np.isfinite(x)]
        if len(x)>20 and np.nanmedian(x)>0 and np.nanmax(x)>np.nanmedian(x):
            candidates.append((np.nanmedian(x),np.nanmax(x)-np.nanmin(x),c))
    candidates=[q for q in candidates if q[0]<500]
    return sorted(candidates,key=lambda q:(q[0]<1,-q[1]))[0][2] if candidates else nums[-1]

def _kurtosis_controls(x,seed=12345,nboot=300):
    rng=np.random.default_rng(seed)
    x=np.asarray(x,float); x=x[np.isfinite(x)]
    if len(x)<20: return {}
    resid=(x-np.median(x))/(mad(x) or np.std(x) or 1)
    k=simple_kurtosis(resid)
    # Null controls: Gaussian and lognormal with matched sample size / log moments.
    g=[]; l=[]
    lx=np.log(x[x>0])
    for _ in range(nboot):
        gg=rng.normal(size=len(x)); g.append(simple_kurtosis((gg-np.median(gg))/(mad(gg) or np.std(gg) or 1)))
        if len(lx)>10:
            yy=rng.lognormal(mean=float(np.mean(lx)),sigma=float(np.std(lx) or 1e-6),size=len(lx))
            l.append(simple_kurtosis((yy-np.median(yy))/(mad(yy) or np.std(yy) or 1)))
    return {'kurtosis':k,'gaussian_null_p95':float(np.nanpercentile(g,95)) if g else None,'lognormal_null_p95':float(np.nanpercentile(l,95)) if l else None,'exceeds_gaussian_p95':bool(k is not None and g and k>np.nanpercentile(g,95)),'exceeds_lognormal_p95':bool(k is not None and l and k>np.nanpercentile(l,95))}

def main():
    args=build_parser('T09 P38 VAST void-wall Cauchy-tail/kurtosis test').parse_args()
    cache=ensure_dir(args.cache); outdir=ensure_dir(args.outdir)
    res=result_template('T09',['P38','CL4'],'Compute excess kurtosis/cauchy-tail proxy from public VAST SDSS DR7 void catalogues.')
    res['falsification_logic']={'confirm_like':'void radius/wall proxy has k4>4 and exceeds matched Gaussian/lognormal null controls','falsify_like':'kurtosis is Gaussian/lognormal-like or below k4 threshold'}
    tables,att=load_vast_void_tables(cache,timeout=args.timeout,force=args.force,allow_large=True); res['data_sources'].extend(att)
    rows=[]
    for i,df in enumerate(tables):
        col=_best_void_radius_column(df)
        if col is None: continue
        x=numeric_array(df,col); x=x[np.isfinite(x)]; x=x[(x>0)&(x<500)]
        if len(x)>20:
            rctrl=_kurtosis_controls(x,seed=args.seed+i)
            lx=np.log(x)
            lctrl=_kurtosis_controls(lx-lx.min()+1e-9,seed=args.seed+1000+i) if len(lx)>20 else {}
            rows.append({'table_index':int(i),'rows':int(len(x)),'column':str(col),'median_radius_like':float(np.median(x)),'radius_controls':rctrl,'log_radius_kurtosis':simple_kurtosis((lx-np.median(lx))/(mad(lx) or np.std(lx) or 1))})
    valid=[r for r in rows if r.get('radius_controls',{}).get('kurtosis') is not None]
    kmax=max([r['radius_controls']['kurtosis'] for r in valid],default=None)
    robust=[r for r in valid if r['radius_controls'].get('kurtosis',0)>4 and r['radius_controls'].get('exceeds_lognormal_p95')]
    res['metrics']={'void_tables':rows,'max_radius_kurtosis':kmax,'n_robust_tables':len(robust),'note':'v9.3 requires k4>4 and excess over matched lognormal null; log-radius-only excess is reported separately, not used as P38 confirmation.'}
    res['status']='suggestive' if robust else ('null' if valid else 'data_limited')
    if valid and not robust:
        res['warnings'].append('No radius-like VAST void statistic passed both k4>4 and matched-lognormal-null controls; current P38 implementation remains null/tension.')
    write_result(res,outdir)
if __name__=='__main__': main()
