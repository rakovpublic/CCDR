#!/usr/bin/env python3
from _common_public import *

def main():
    args=build_parser('T08 independent public filament-catalogue replication').parse_args()
    cache=ensure_dir(args.cache); outdir=ensure_dir(args.outdir)
    res=result_template('T08',['P3'],'Attempt independent filament-orientation replication using public filament catalogues; endpoint/position-angle catalogues are primary, kNN is secondary.')
    res['falsification_logic']={'confirm_like':'independent public filament catalogue shows positive orientation correlation','falsify_like':'orientation statistic is null/opposite in independent catalogue'}
    urls=filament_endpoint_catalogue_registry()
    attempts=[]; best=None
    for i,u in enumerate(urls):
        try:
            p=download_file(u,cache/'filaments',filename=f'vizier_filaments_{i}.tsv',timeout=args.timeout,force=args.force)
            df,info=parse_vizier_filament_table(p,max_rows=args.max_rows)
            info.update({'url':u,'ok': df is not None, 'path':str(p)})
            attempts.append(info)
            if df is None or info.get('error') or info.get('n_finite',0)<50: continue
            m=info['finite_mask']; ra=info['ra'][m]; dec=info['dec'][m]
            if info.get('angle') is not None:
                ang=info['angle'][m]
                ok=np.isfinite(ang); ra=ra[ok]; dec=dec[ok]; ang=ang[ok]
            else:
                # v9.8: endpoint catalogue angle is primary; kNN reconstruction is secondary fallback only.
                ang=local_orientation_angles(ra,dec,k=10)
            if len(ra)<50: continue
            bins=[0,0.2,0.5,1,2,5,10]
            corr=orientation_correlation(ra,dec,ang,bins)
            best={'source':u,'n':int(len(ra)),'ra_col':info.get('ra_col'),'dec_col':info.get('dec_col'),'ra2_col':info.get('ra2_col'),'dec2_col':info.get('dec2_col'),'pa_col':info.get('pa_col'),'angle_mode':info.get('angle_mode'),'corr':corr,'coordinate_parse':info.get('coordinate_parse'),'endpoint_or_pa_primary': bool(info.get('angle_mode') in ('catalogue_endpoint_angle','catalogue_position_angle'))}
            break
        except Exception as e:
            attempts.append({'url':u,'ok':False,'error':str(e)})
    res['data_sources'].extend(attempts); res['metrics']['filament_catalogue_result']=best
    vals=[r['corr'] for r in best['corr'] if r.get('corr') is not None and r.get('n_pairs',0)>100] if best else []
    mean=float(np.mean(vals)) if vals else None; res['metrics']['mean_corr']=mean
    res['metrics']['catalogue_priority_note']='Endpoint/spine or catalogue position-angle geometry is primary. kNN angle reconstruction remains secondary and cannot override a negative Euclid T07 result.'
    res['status']=t08_status_from_mode(mean, best.get('angle_mode') if best else None)
    if res['status']=='knn_secondary_suggestive':
        res['warnings'].append('Positive result uses kNN-reconstructed angles, not endpoint/spine geometry; keep as weak secondary support only.')
    if res['status']=='data_limited':
        res['warnings'].append('No parseable public filament table with usable coordinate/endpoint/position-angle columns found.')
    write_result(res,outdir)
if __name__=='__main__': main()
