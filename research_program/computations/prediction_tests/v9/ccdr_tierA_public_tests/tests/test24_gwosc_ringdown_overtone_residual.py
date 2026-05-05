#!/usr/bin/env python3
from _common_public import *

def main():
    args=build_parser('T24 GWOSC ringdown overtone residual simple fit').parse_args()
    cache=ensure_dir(args.cache); outdir=ensure_dir(args.outdir)
    res=result_template('T24',['P32'],'Download public GWOSC event data and run a simple damped-sinusoid ringdown residual screen without LALSuite/Bilby.')
    res['falsification_logic']={'confirm_like':'two-mode damped fit leaves systematic overtone-like residual at predicted scale','falsify_like':'Kerr-like one/two-mode residuals are noise-like in public strain'}
    obj,att=get_gwosc_event_json(timeout=args.timeout); res['data_sources'].extend(att)
    if obj is None: write_result(res,outdir); return
    urls=extract_gwosc_file_urls(obj)
    # v9.3: h5py can only read HDF5. Prefer HDF5 over GWF, and prefer 4 kHz/32 s H1.
    urls=sorted(set(urls), key=lambda u: (not re.search(r'\.hdf5$|\.h5$',u,re.I), 'H-H1' not in u and 'H1' not in u, '4KHZ' not in u.upper(), '32' not in u, len(u)))
    res['metrics']['gwosc_file_candidates']=urls[:10]
    if not args.allow_large:
        res['warnings'].append('GWOSC strain files can be large; rerun with --allow-large to download and fit.')
        write_result(res,outdir); return
    p,att2=download_first(urls[:10],cache/'gwosc',timeout=args.timeout,force=args.force,max_bytes=None); res['data_sources'].extend(att2)
    if p is None: write_result(res,outdir); return
    if not re.search(r'\.hdf5$|\.h5$',str(p),re.I):
        res['warnings'].append('Downloaded a non-HDF5 file; install/use a GWF reader such as gwpy, or rerun after HDF5 URL is reachable.')
        write_result(res,outdir); return
    try:
        import h5py
        with h5py.File(p,'r') as h:
            strain=np.array(h['strain']['Strain'])
            dur=float(h['meta']['Duration'][()]); fs=len(strain)/dur
        j=int(np.argmax(np.abs(strain))); pre=int(0.005*fs); post=int(0.15*fs)
        a=max(0,j-pre); b=min(len(strain),j+post); win=strain[a:b]
        t=np.arange(len(win))/fs
        if len(win)<100 or optimize is None:
            res['warnings'].append('Insufficient strain samples or scipy unavailable.'); write_result(res,outdir); return
        y=win-np.mean(win)
        def model1(t,A,f,tau,phi,c): return A*np.exp(-t/abs(tau))*np.sin(2*np.pi*f*t+phi)+c
        p0=[np.max(np.abs(y)),150,0.05,0,0]
        popt,_=optimize.curve_fit(model1,t,y,p0=p0,maxfev=30000)
        r=y-model1(t,*popt)
        # Minimal two-frequency screen: fit residual with another damped mode and
        # report improvement; not a physical Kerr/overtone inference.
        def model2(t,A1,f1,tau1,phi1,A2,f2,tau2,phi2,c):
            return A1*np.exp(-t/abs(tau1))*np.sin(2*np.pi*f1*t+phi1)+A2*np.exp(-t/abs(tau2))*np.sin(2*np.pi*f2*t+phi2)+c
        two=None
        try:
            p02=[popt[0],popt[1],popt[2],popt[3],0.3*popt[0],300,0.02,0,popt[4]]
            popt2,_=optimize.curve_fit(model2,t,y,p0=p02,maxfev=50000)
            r2=y-model2(t,*popt2)
            two={'two_mode_params':list(map(float,popt2)),'two_mode_residual_rms':float(np.std(r2)),'improvement_fraction':float(1-np.std(r2)/(np.std(r) or np.nan))}
        except Exception as e:
            two={'two_mode_fit_error':str(e)}
        res['metrics'].update({'file':str(p),'fs_Hz':float(fs),'window_samples':int(len(y)),'one_mode_params':list(map(float,popt)),'one_mode_residual_rms':float(np.std(r)),'signal_rms':float(np.std(y)),'one_mode_residual_over_signal':float(np.std(r)/(np.std(y) or np.nan)),'two_mode_screen':two,'note':'Simple damped-sinusoid screen only; not a LALSuite/Bilby ringdown posterior.'})
        res['status']='diagnostic_only'
    except Exception as e:
        res['warnings'].append(f'GWOSC HDF5 fit failed: {e}')
    write_result(res,outdir)
if __name__=='__main__': main()
