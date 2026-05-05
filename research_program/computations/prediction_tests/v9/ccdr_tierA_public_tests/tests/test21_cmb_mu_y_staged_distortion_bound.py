#!/usr/bin/env python3
from _common_public import *


def _numeric_table(path):
    if pd is None: return None
    for kwargs in [dict(sep=r"\s+", comment="#", header=None), dict(sep=r"\s+", comment="#")]:
        try:
            df=pd.read_csv(path,engine="python",**kwargs)
            if df.shape[1]>=3 and df.shape[0]>10:
                return df
        except Exception:
            pass
    return read_table_any(path)


def _planck_nu_mjy_sr(freq_ghz, T=2.7255):
    # Return spectral radiance in MJy/sr using SI constants.
    h=6.62607015e-34; k=1.380649e-23; c=299792458.0
    nu=np.asarray(freq_ghz,float)*1e9
    x=h*nu/(k*T)
    B=2*h*nu**3/c**2/(np.exp(x)-1.0)  # W/m2/Hz/sr
    return B/1e-20  # MJy/sr


def main():
    args=build_parser('T21 FIRAS μ/y staged-distortion physical bound screen').parse_args()
    cache=ensure_dir(args.cache); outdir=ensure_dir(args.outdir)
    res=result_template('T21',['P28'],'Fit physical μ/y-like spectral distortion templates to public COBE/FIRAS monopole spectrum.')
    res['prediction_names']=['P28 — staged CMB μ/y distortions']
    res['falsification_logic']={'confirm_like':'a physically normalized staged distortion amplitude is nonzero and below FIRAS limits','falsify_like':'predicted staged amplitude is above FIRAS residual limits'}
    df,att=load_firas_spectrum(cache,timeout=args.timeout,force=args.force); res['data_sources'].extend(att)
    good_paths=[a.get('path') for a in att if a.get('ok') and a.get('path')]
    if good_paths:
        df=_numeric_table(Path(good_paths[0]))
    if df is None or df.shape[1]<3:
        write_result(res,outdir); return
    nums=find_numeric_columns(df)
    if len(nums)<3:
        res['warnings'].append('FIRAS table parsed but fewer than 3 numeric columns found.'); write_result(res,outdir); return
    freq=numeric_array(df,nums[0]); I=numeric_array(df,nums[1]); err=np.abs(numeric_array(df,nums[2]))
    m=np.isfinite(freq)&np.isfinite(I)&np.isfinite(err)&(err>0)&(freq>0)
    freq=freq[m]; I=I[m]; err=err[m]
    if len(freq)<10:
        write_result(res,outdir); return
    # FIRAS monopole file is usually frequency in cm^-1; convert to GHz if values look like 2-25 cm^-1.
    unit='GHz'
    fghz=freq.copy()
    if np.nanmedian(freq)<100:
        fghz=freq*29.9792458
        unit='cm^-1 converted to GHz'
    B=_planck_nu_mjy_sr(fghz)
    # Fit a free blackbody normalization and smooth calibration polynomial, then μ/y-like residual shapes.
    x=(fghz-np.mean(fghz))/(np.std(fghz) or 1)
    xx=6.62607015e-34*(fghz*1e9)/(1.380649e-23*2.7255)
    mu_shape=B*(xx/np.maximum(1-np.exp(-xx),1e-12)-2.1923)
    y_shape=B*(xx*(np.exp(xx)+1)/np.maximum(np.exp(xx)-1,1e-12)-4)
    A=np.vstack([B, np.ones_like(x), x, x**2, mu_shape, y_shape]).T
    W=np.diag(1/err)
    try:
        coef,cov_res,rank,s=np.linalg.lstsq(W@A,W@I,rcond=None)
        model=A@coef; chi2=float(np.sum(((I-model)/err)**2)); dof=int(len(I)-A.shape[1])
        # Linear covariance approximation.
        cov=np.linalg.pinv((A.T/(err**2))@A)
        mu_hat=float(coef[-2]); y_hat=float(coef[-1])
        mu_sig=float(np.sqrt(abs(cov[-2,-2]))); y_sig=float(np.sqrt(abs(cov[-1,-1])))
        res['metrics']={'n_points':int(len(freq)),'unit_audit':firas_unit_audit(freq,I,err),'frequency_unit':unit,'columns':list(map(str,nums[:3])),'mu_best':mu_hat,'mu_95_abs_limit':float(abs(mu_hat)+1.96*mu_sig),'y_best':y_hat,'y_95_abs_limit':float(abs(y_hat)+1.96*y_sig),'chi2':chi2,'dof':dof,'chi2_over_dof':float(chi2/dof) if dof>0 else None,'official_covariance_used':False,'note':'Physical-template screen with nuisance blackbody/calibration terms; v9.6 adds unit/fit sanity audit, but this is still not the official FIRAS residual covariance likelihood.'}
        res['status']='consistent_bound_only_official_likelihood_missing'
        res['warnings'].append('No official FIRAS residual covariance likelihood is used; this remains a staged physical-template bound only.')
    except Exception as e:
        res['warnings'].append(str(e))
    write_result(res,outdir)
if __name__=='__main__': main()
