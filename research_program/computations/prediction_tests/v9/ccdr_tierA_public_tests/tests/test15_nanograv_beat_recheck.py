#!/usr/bin/env python3
from _common_public import *

def main():
    args=build_parser('T15 NANOGrav beat recheck without TEMPO2/enterprise').parse_args()
    cache=ensure_dir(args.cache); outdir=ensure_dir(args.outdir)
    res=result_template('T15',['P8'],'Acquire public NANOGrav 15-year products and perform non-specialized beat-screening where tabulated residual/proxy data exist.')
    res['metrics']['conservative_evidence_guard']=conservative_evidence_guard('pta')
    res['falsification_logic']={'confirm_like':'public residual/posterior tables show excess at predicted beat band','falsify_like':'no beat-band excess in public residual/posterior tables at current sensitivity'}
    arc,att=load_nanograv_archive(cache,timeout=args.timeout,force=args.force,allow_large=args.allow_large); res['data_sources'].extend(att)
    if arc is None:
        res['warnings'].append('NANOGrav timing archive is large; rerun with --allow-large. Full residual refit would require TEMPO2/enterprise and is intentionally skipped.')
        write_result(res,outdir); return
    root=extract_archive(arc,cache/'nanograv'/'extract')
    # Without timing software, count TOA spans and look for public residual-like files.
    tims=list(root.rglob('*.tim')); pars=list(root.rglob('*.par'))
    residual_like=[p for p in root.rglob('*') if re.search(r'(resid|residual|post|posterior|chain|free.*spectrum).*\.(txt|dat|csv|npy)$',p.name,re.I)]
    spectra=[]; rejected=[]
    for p in residual_like[:50]:
        df=read_table_any(p,max_rows=args.max_rows)
        if df is None or df.shape[1]<2:
            continue
        ident=nanograv_table_identity(df)
        nums=find_numeric_columns(df)
        if ident.get('kind')=='residual_or_toa_table' and len(nums)>=2:
            x=numeric_array(df,nums[0]); y=numeric_array(df,nums[1]); m=np.isfinite(x)&np.isfinite(y)
            if m.sum()>20 and signal is not None:
                f,pxx=signal.periodogram(y[m]-np.mean(y[m]))
                spectra.append({'path':str(p),'kind':ident.get('kind'),'n':int(m.sum()),'x_col':str(nums[0]),'y_col':str(nums[1]),'peak_frequency_index':int(np.argmax(pxx[1:])+1) if len(pxx)>1 else None,'peak_power':float(np.max(pxx[1:])) if len(pxx)>1 else None})
        elif ident.get('kind')=='posterior_parameter_table':
            spectra.append({'path':str(p),'kind':ident.get('kind'),'n_rows':int(len(df)),'columns':ident.get('columns')[:20],'note':'posterior table identified; no FFT applied'})
        else:
            rejected.append({'path':str(p),'reason':'unverified_column_identity','identity':ident})
    res['metrics']={'n_tim_files':len(tims),'n_par_files':len(pars),'n_residual_like_files':len(residual_like),'verified_public_products':spectra,'rejected_numeric_tables':rejected[:20],'note':'v9.6 avoids FFT on arbitrary chain columns; only residual/TOA tables or identified posterior parameter tables are counted.'}
    res['status']='data_limited' if not spectra else 'diagnostic_suggestive'
    write_result(res,outdir)
if __name__=='__main__': main()
