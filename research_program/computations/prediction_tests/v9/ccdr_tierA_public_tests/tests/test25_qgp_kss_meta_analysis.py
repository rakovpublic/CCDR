#!/usr/bin/env python3
from _common_public import *

def _column_score_for_eta(c):
    cl=str(c).lower().replace(' ','')
    score=0
    if 'eta/s' in cl or 'η/s' in cl or 'etas' in cl or 'eta_over_s' in cl: score+=20
    if 'shear' in cl and 'visc' in cl: score+=15
    if 'visc' in cl: score+=6
    if 'v2' in cl or 'flow' in cl or 'centrality' in cl or 'deta' in cl or 'deltaeta' in cl: score-=20
    if 'err' in cl or 'stat' in cl or 'sys' in cl: score-=5
    return score

def main():
    args=build_parser('T25 QGP KSS η/s meta-analysis from public HEPData/literature tables').parse_args()
    cache=ensure_dir(args.cache); outdir=ensure_dir(args.outdir)
    res=result_template('T25',['P5'],'Download public HEPData/literature tables and test whether η/s approaches the KSS bound non-monotonically.')
    res['metrics']['conservative_evidence_guard']=conservative_evidence_guard('kss')
    res['falsification_logic']={'confirm_like':'η/s(T) has a minimum near 1/(4π) in natural units and non-monotonic structure','falsify_like':'verified η/s columns show no KSS-adjacent minimum'}
    parsed, attempts = discover_kss_eta_tables(cache, timeout=args.timeout, force=args.force)
    res['data_sources'].extend(attempts)
    used=[]; rejected=[]
    for u,df in parsed:
        nums=find_numeric_columns(df)
        if len(nums)<2:
            rejected.append({'source':u,'reason':'too_few_numeric_columns','columns':[str(c) for c in df.columns[:20]]}); continue
        scored=sorted([(_column_score_for_eta(c),c) for c in nums],reverse=True,key=lambda x:x[0])
        best_score,ycol=scored[0]
        if best_score < 10:
            rejected.append({'source':u,'reason':'no_verified_eta_over_s_column','best_column':str(ycol),'best_score':int(best_score),'columns':[str(c) for c in df.columns[:30]],'note':'Rejecting v2/flow tables prevents false KSS nulls.'})
            continue
        xcol=next((c for c in nums if c!=ycol and re.search(r'T|temp|temperature|centrality|cent',str(c),re.I)), next((c for c in nums if c!=ycol), nums[0]))
        x=numeric_array(df,xcol); y=numeric_array(df,ycol); m=np.isfinite(x)&np.isfinite(y)&(y>0)&(y<10)
        if m.sum()>=3:
            ymin=float(np.nanmin(y[m])); used.append({'source':u,'x_col':str(xcol),'eta_over_s_col':str(ycol),'n':int(m.sum()),'min_eta_over_s':ymin,'kss_natural_1_over_4pi':float(1/(4*np.pi)),'min_over_kss':float(ymin/(1/(4*np.pi))),'nonmonotonic_screen':bool(np.nanargmin(y[m]) not in (0,m.sum()-1))})
    res['metrics']={'tables_used':used,'tables_rejected':rejected,'kss_eta_over_s_SI':KSS_ETA_OVER_S,'note':'v9.6 discovers HEPData candidates but still requires explicit η/s-like column names; v2/flow tables are not treated as viscosity.'}
    near=any(0.5<=u['min_over_kss']<=3 for u in used)
    res['status']='suggestive' if near else ('data_limited' if not used else 'null')
    if not used: res['warnings'].append('No verified public η/s column was found after HEPData discovery; no KSS falsification is claimed.')
    write_result(res,outdir)
if __name__=='__main__': main()
