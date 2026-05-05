#!/usr/bin/env python3
from _common_public import *

def main():
    args=build_parser('T17 GW spectral-index proxy from NANOGrav public new-physics files').parse_args()
    cache=ensure_dir(args.cache); outdir=ensure_dir(args.outdir)
    res=result_template('T17',['P8b'],'Search public NANOGrav new-physics/tabulated posterior files for spectral-index constraints near δn≈ν/3.')
    res['falsification_logic']={'confirm_like':'public posterior constraints contain predicted δn≈ν/3 shift','falsify_like':'predicted shift excluded by public posterior summaries'}
    paths,att=download_zenodo_matching('8092761',cache,[r'\.(csv|txt|dat|json|npy)$',r'model',r'data'],timeout=args.timeout,force=args.force,allow_large=True); res['data_sources'].extend(att)
    parsed=[]; spectral=[]
    for p in paths[:60]:
        df=read_table_any(p,max_rows=args.max_rows)
        if df is not None and df.shape[1]>=2:
            info=parse_spectral_index_constraints(df)
            info.update({'path':str(p),'rows':int(len(df))})
            parsed.append(info)
            if info.get('has_verified_spectral_index'):
                spectral.append(info)
    target=5e-3/3
    res['metrics']={'parsed_public_new_physics_tables':parsed[:30],'verified_spectral_index_tables':spectral,'target_delta_n_for_nu_5e-3':target,'note':'v9.6 requires spectral-index/gamma column identity; full PTA likelihood is not run.'}
    res['status']='data_limited' if not spectral else 'diagnostic_suggestive'
    write_result(res,outdir)
if __name__=='__main__': main()
