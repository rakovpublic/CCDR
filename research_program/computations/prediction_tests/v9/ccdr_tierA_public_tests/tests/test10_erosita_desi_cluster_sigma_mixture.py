#!/usr/bin/env python3
from _common_public import *


def _tables_from_download(path, cache, max_rows):
    out=[]
    try:
        if zipfile.is_zipfile(path) or tarfile.is_tarfile(path):
            ex=extract_archive(path, cache/'erosita'/(path.stem+'_extract'))
            for q in list(ex.rglob('*.fits'))+list(ex.rglob('*.fit'))+list(ex.rglob('*.csv'))+list(ex.rglob('*.txt'))+list(ex.rglob('*.dat')):
                df=read_table_any(q,max_rows=max_rows)
                if df is not None and len(df)>50:
                    out.append((str(q),df))
        else:
            df=read_table_any(path,max_rows=max_rows)
            if df is not None and len(df)>50:
                out.append((str(path),df))
    except Exception:
        pass
    return out


def _select_redshift_col(df):
    nums=find_numeric_columns(df)
    for c in nums:
        cl=str(c).lower()
        if cl in ('z','redshift','z_lambda','z_best','photoz') or 'redshift' in cl:
            arr=numeric_array(df,c); finite=arr[np.isfinite(arr)]
            if len(finite)>20 and np.nanmin(finite)>=0 and np.nanmax(finite)<3:
                return c
    return None


def _candidate_physical_proxies(df):
    nums=find_numeric_columns(df)
    candidates=[]
    # Prefer mass/richness/luminosity/extent proxies over redshift itself.
    positive_words=['lambda','lambda_chisq','rich','richness','m500','m_500','m200','m_200','mass','lum','l500','lx','l_x','extent','ext','ext_like','det_like','rate','flux','count_rate','ml_flux','cr','cts','sigma','temperature','tx','y_x','ysz']
    reject_words=['z_lambda','redshift','photoz','ra','dec','glon','glat','err','e_','sigma_z','srcid','name','id']
    for c in nums:
        cl=str(c).lower()
        if any(rw in cl for rw in reject_words):
            continue
        score=sum(1 for w in positive_words if w in cl)
        if score>0:
            candidates.append((score,c))
    candidates=sorted(candidates,reverse=True,key=lambda x:x[0])
    return [c for _,c in candidates]


def _selection_control_columns(df, zc, proxy):
    nums=find_numeric_columns(df)
    controls=[]
    for c in nums:
        if c in (zc, proxy):
            continue
        cl=str(c).lower()
        if any(k in cl for k in ['det_like','extent','ext_like','flux','rate','count','exposure','exptime','ml_flux','ra','dec','glon','glat']):
            arr=numeric_array(df,c)
            finite=arr[np.isfinite(arr)]
            if len(finite)>50 and np.nanstd(finite)>0:
                controls.append(c)
    return controls[:4]


def _residualize_against_controls(y_masked, z_full, df, controls, mask):
    zz=z_full[mask]
    cols=[np.ones(len(y_masked)), zz, zz**2]
    used=[]
    for c in controls:
        arr=numeric_array(df,c)[mask]
        if np.isfinite(arr).sum()<0.8*len(y_masked) or np.nanstd(arr[np.isfinite(arr)])==0:
            continue
        med=np.nanmedian(arr); sig=np.nanstd(arr) or 1.0
        cols.append((np.where(np.isfinite(arr),arr,med)-med)/sig)
        used.append(str(c))
    X=np.vstack(cols).T
    beta=np.linalg.lstsq(X,y_masked,rcond=None)[0]
    return y_masked-X@beta, {'controls_used':used,'n_controls':len(used),'design_cols':int(X.shape[1])}


def _sky_region_labels(df, mask):
    ra_col=next((c for c in df.columns if str(c).lower() in ('ra','raj2000','ra_deg') or 'ra'==str(c).lower()), None)
    dec_col=next((c for c in df.columns if str(c).lower() in ('dec','dej2000','dec_deg') or 'dec'==str(c).lower()), None)
    if ra_col is None or dec_col is None:
        return None
    ra=numeric_array(df,ra_col)[mask]; dec=numeric_array(df,dec_col)[mask]
    if np.isfinite(ra).sum()<50 or np.isfinite(dec).sum()<50:
        return None
    return (ra>np.nanmedian(ra)).astype(int)+2*(dec>np.nanmedian(dec)).astype(int)


def _shuffle_null_delta(resid, n=80, seed=12345):
    rng=np.random.default_rng(seed)
    vals=[]
    for _ in range(n):
        yy=np.array(resid,copy=True)
        rng.shuffle(yy)
        mm=gaussian_mixture_bic_1d(yy)
        if mm.get('delta_bic_single_minus_mix') is not None:
            vals.append(mm['delta_bic_single_minus_mix'])
    return {'n':len(vals),'mean_delta_bic':float(np.mean(vals)) if vals else None,'p95_delta_bic':float(np.percentile(vals,95)) if vals else None}


def main():
    args=build_parser('T10 eROSITA cluster multi-component residual proxy').parse_args()
    cache=ensure_dir(args.cache); outdir=ensure_dir(args.outdir)
    res=result_template('T10',['P34'],'Fit single-vs-mixture models to public eROSITA cluster proxy residuals after redshift control.')
    res['prediction_names']=['P34 — multi-component cluster σv']
    res['falsification_logic']={'confirm_like':'two-component mixture preferred in mass/richness/X-ray residuals after redshift control','falsify_like':'single Gaussian/lognormal explains residuals within redshift bins'}
    seed_urls=[
      'https://erosita.mpe.mpg.de/dr1/AllSkySurveyData_dr1/Catalogues_dr1/BulbulE_DR1/erass1cl_primary_v3.2.fits.tgz',
      'https://erosita.mpe.mpg.de/dr1/AllSkySurveyData_dr1/Catalogues_dr1/BulbulE_DR1/erass1cl_cosmology_v1.1.fits.tgz',
    ]
    attempts=[]; tables=[]
    for u in seed_urls:
        try:
            p=download_file(u,cache/'erosita',timeout=args.timeout,force=args.force,max_bytes=None if args.allow_large else 300*1024*1024)
            attempts.append({'url':u,'ok':True,'path':str(p)})
            tables.extend(_tables_from_download(p, cache, args.max_rows))
        except Exception as e:
            attempts.append({'url':u,'ok':False,'error':str(e)})
    res['data_sources'].extend(attempts)
    best=None; tried=[]
    for u,df in tables:
        zc=_select_redshift_col(df)
        proxies=_candidate_physical_proxies(df)
        if not zc or not proxies:
            tried.append({'source':u,'reason':'missing_redshift_or_physical_proxy','z_col':str(zc) if zc else None,'n_proxy_candidates':len(proxies),'numeric_column_profile':_numeric_column_profile(df)})
            continue
        z=numeric_array(df,zc)
        for proxy in proxies[:8]:
            x=numeric_array(df,proxy)
            m=np.isfinite(z)&np.isfinite(x)&(z>=0)&(z<3)
            if int(m.sum()) == 0:
                continue
            if np.nanmin(x[m]) <= 0:
                m=m&(x>0)
            if m.sum()<100:
                continue
            yy=np.log10(x[m]) if np.nanmax(x[m])/max(np.nanmin(x[m]),1e-99)>20 else x[m].astype(float)
            zz=z[m]
            # Residualize proxy versus redshift plus selection observables (detection/extent/flux/sky proxies).
            controls=_selection_control_columns(df, zc, proxy)
            resid,control_info=_residualize_against_controls(yy, z, df, controls, m)
            mix=gaussian_mixture_bic_1d(resid)
            null=_shuffle_null_delta(resid, n=60, seed=args.seed)
            # Within-redshift-bin mixture check.
            bin_checks=[]
            try:
                qs=np.quantile(zz,[0,0.33,0.66,1.0])
                for a,b in zip(qs[:-1],qs[1:]):
                    mm=(zz>=a)&(zz<=b)
                    if mm.sum()>=80:
                        bm=gaussian_mixture_bic_1d(resid[mm]); bm['z_bin']=[float(a),float(b)]; bin_checks.append(bm)
            except Exception:
                pass
            regions=_sky_region_labels(df,m)
            region_checks=[]
            if regions is not None:
                for lab in sorted(set(regions)):
                    rr=resid[regions==lab]
                    if len(rr)>=80:
                        gm=gaussian_mixture_bic_1d(rr); gm['region']=int(lab); region_checks.append(gm)
            cand={'source':u,'proxy_col':str(proxy),'redshift_col':str(zc),'n':int(m.sum()),'selection_controls':control_info,'shuffle_null':null,'selection_residual_mixture':mix,'within_redshift_bin_checks':bin_checks,'sky_region_checks':region_checks,'note':'v9.6 residualizes physical proxy against redshift plus available selection/sky proxies; redshift itself is never used as the mixture observable.'}
            cand['delta_bic_single_minus_mix']=mix.get('delta_bic_single_minus_mix')
            if best is None or (cand['delta_bic_single_minus_mix'] is not None and cand['delta_bic_single_minus_mix']>(best.get('delta_bic_single_minus_mix') or -1e99)):
                best=cand
    res['metrics']={'best_residual_mixture':best,'tables_tried':tried[:20], 'parser_upgrade_note':'v10 expands eROSITA proxy aliases and exposes control-gate breakdowns on failures.'}
    status, warn = t10_control_status(best)
    if best is not None and isinstance(best, dict):
        res['metrics']['control_gate_breakdown']=best.get('v10_control_gate_breakdown') or best.get('v9_9_control_summary')
    res['status']=status
    res['warnings'].extend(warn)
    write_result(res,outdir)
if __name__=='__main__': main()
