#!/usr/bin/env python3
from _common_public import *


def _read_bk18_table(path: Path):
    if pd is None:
        return None, {"error":"pandas_missing"}
    try:
        df=pd.read_csv(path,sep=r"\s+",comment="#",header=None,engine="python")
        if df.shape[1] >= 4 and df.shape[0] > 3:
            return df, {"format":"bk18_pair_columns","n_columns":int(df.shape[1])}
    except Exception as e:
        return None,{"error":str(e)}
    return None,{"error":"could_not_parse_bk18"}


def _bk18_header_labels(path: Path):
    labels=[]
    try:
        for line in path.read_text(errors="ignore").splitlines()[:80]:
            if not line.lstrip().startswith('#'):
                continue
            clean=line.lstrip('#').strip()
            # Capture tokens such as BB, EE, TE, 95x150 etc.  BK tables vary;
            # this is metadata only and not used for hard inference.
            if any(k in clean.upper() for k in [' BB', 'BB ', 'B-MODE', 'BMODE', 'BICEP', 'KECK']):
                labels.append(clean[:300])
    except Exception:
        pass
    return labels[:20]


def main():
    args=build_parser('T23 P40 bulk-Weyl B-mode bandpower public-table screen').parse_args()
    cache=ensure_dir(args.cache); outdir=ensure_dir(args.outdir)
    res=result_template('T23',['P40','CL5'],'Screen public B-mode bandpower tables for low-l residual room for a bulk-Weyl component.')
    res['prediction_names']=['P40 — bulk-Weyl CMB B-mode component','CL5 — joint (ν_bulk, c_W) constraint from P39 + P40']
    res['falsification_logic']={'confirm_like':'low-l BB bandpowers leave residual room for a P40-shaped contribution after uncertainties','falsify_like':'published BB bandpowers/uncertainties exclude predicted amplitude/shape'}
    urls=[
        'https://lambda.gsfc.nasa.gov/data/suborbital/BICEPK_2021/BK18_bandpowers_20210607.txt',
        'https://bicepkeck.org/BK18_datarelease/BK18_bandpowers_20210607.txt',
    ]
    attempts=[]; path=None
    for u in urls:
        try:
            p=download_file(u,cache/'cmb_bmode',timeout=args.timeout,force=args.force,max_bytes=25*1024*1024)
            attempts.append({'url':u,'ok':True,'path':str(p)})
            path=p; break
        except Exception as e:
            attempts.append({'url':u,'ok':False,'error':str(e)})
    res['data_sources'].extend(attempts)
    if path is None:
        write_result(res,outdir); return
    df,parse_info=_read_bk18_table(path)
    parse_info['header_label_hints']=_bk18_header_labels(path)
    parse_info['column_label_map']=bk18_column_label_map(path, df.shape[1] if df is not None else 0)
    res['metrics']['parse_info']=parse_info
    if df is None:
        res['warnings'].append('Could not parse BK18 bandpower pair columns.'); write_result(res,outdir); return
    ell=pd.to_numeric(df.iloc[:,1],errors='coerce').to_numpy(float)
    pairs=[]
    for j in range(2,df.shape[1]-1,2):
        bp=pd.to_numeric(df.iloc[:,j],errors='coerce').to_numpy(float)
        er=pd.to_numeric(df.iloc[:,j+1],errors='coerce').to_numpy(float)
        m=np.isfinite(ell)&np.isfinite(bp)&np.isfinite(er)&(er>0)&(ell>0)
        if m.sum()>=3:
            snr=np.abs(bp[m]/er[m])
            low=m&(ell<=150)
            label_map=parse_info.get('column_label_map',{}) if isinstance(parse_info,dict) else {}
            label=str(label_map.get(j+1,'') or label_map.get(j+2,'')).upper()
            bb_verified=bool(re.search(r'(^|[^A-Z])BB([^A-Z]|$)|B[- ]?MODE', label))
            pairs.append({
                'spectrum_identity_status':'verified_BB_from_header' if bb_verified else 'unverified_pair_columns',
                'spectrum_label_hint': label[:80],
                'bandpower_col_index_1based':int(j+1),
                'error_col_index_1based':int(j+2),
                'n':int(m.sum()),
                'median_error':float(np.nanmedian(er[m])),
                'max_abs_snr':float(np.nanmax(snr)),
                'low_ell_n':int(low.sum()),
                'low_ell_weighted_mean_bp':float(np.average(bp[low],weights=1/(er[low]**2))) if low.sum() else None,
                'low_ell_weighted_sigma':float(1/np.sqrt(np.sum(1/(er[low]**2)))) if low.sum() else None,
                'low_ell_max_abs_snr':float(np.nanmax(np.abs(bp[low]/er[low]))) if low.sum() else None,
            })
    bb_pairs=[r for r in pairs if r.get('spectrum_identity_status')=='verified_BB_from_header']
    best=min(bb_pairs or pairs,key=lambda r:r['median_error']) if pairs else None
    p40_bound=None
    if best is not None:
        j=int(best['bandpower_col_index_1based'])-1
        k=int(best['error_col_index_1based'])-1
        bp=pd.to_numeric(df.iloc[:,j],errors='coerce').to_numpy(float)
        er=pd.to_numeric(df.iloc[:,k],errors='coerce').to_numpy(float)
        m=np.isfinite(ell)&np.isfinite(bp)&np.isfinite(er)&(er>0)&(ell>0)&(ell<=150)
        if m.sum()>=3:
            templ=1.0/(ell[m]*(ell[m]+1.0))
            w=1.0/(er[m]**2)
            ahat=float(np.sum(w*templ*bp[m])/np.sum(w*templ*templ))
            asig=float(1.0/np.sqrt(np.sum(w*templ*templ)))
            p40_bound={'template':'A/[ell(ell+1)] on ell<=150 for best-error BK18 pair','A_best':ahat,'A_95_abs_limit':float(abs(ahat)+1.96*asig),'n_low_ell_bins':int(m.sum()),'note':'Generic amplitude bound only; a real P40 test needs verified BB spectrum identity and a theory-predicted ell-shape.'}
    res['metrics']['bk18_pair_summary']={'n_pairs':len(pairs),'best_measured_pair':best,'pair_summaries_first10':pairs[:10],'p40_generic_template_bound':p40_bound,'note':'Column 2 is ell; columns 3+ are bandpower/error pairs. v9.6 preserves header hints and treats spectrum identity as unverified unless labels prove BB.'}
    if not pairs:
        res['status']='data_limited'
    else:
        res['status']='consistent_bound_only' if bb_pairs else 'consistent_bound_only_unverified_spectrum'
        if not bb_pairs:
            res['warnings'].append('No BK18 pair was verified as BB from headers; P40 interpretation remains metadata-limited.')
        if best and best.get('low_ell_max_abs_snr') is not None and best['low_ell_max_abs_snr']>5:
            res['warnings'].append('Large low-ell |bandpower/error| exists in at least one selected pair; require verified BB identity before interpreting P40.')
    write_result(res,outdir)
if __name__=='__main__': main()
