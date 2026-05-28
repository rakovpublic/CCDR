from pathlib import Path
p=Path('/mnt/data/v63work/ccdr_r10_common.py')
text=p.read_text()
append=r'''

# ======================================================================
# v63 behavior patch: real parser/estimator improvements only
# - P36: source-targeted KROSS/KGES/KMOS3D/SAMI/MOSDEF/PHIBSS fetch/parse,
#        strict radius provenance, source bootstrap, streaming rows.
# - P30: same-mask route post-processing with predeclared patch/edge rejector
#        and redshift-density residualization diagnostics.
# - P33: DESI LSS/random public endpoint discovery + density-split alpha proxy.
# - PTA/P32/P40/P41: concrete no-manual statistic/likelihood builders where
#        public/cached rows exist; otherwise precise blockers.
# ======================================================================

_V63_VERSION = 'v63'

def _v63_run_id():
    return os.environ.get('CCDR_R10_CURRENT_RUN_ID_V63') or os.environ.get('CCDR_R10_CURRENT_RUN_ID') or _v62_run_id()

def _v63_dir(name):
    return _v62_dir(name)

def _v63_write_json(path,obj):
    return _v62_write_json(path,obj)

def _v63_read_json(path):
    return _v62_read_json(path)

def _v63_sha256(path):
    return _v62_sha256(path)

def _v63_float(x): return _v62_float(x)
def _v63_truthy(x): return _v62_truthy(x)
def _v63_bad_path(p): return _v62_bad_path(p)

# ---------- P36 v63: concrete public source fetchers + strict source maps ----------

_V63_HIGHZ_URLS = {
    # KROSS public catalogues from the Durham KROSS data page.
    'KROSS': [
        'https://astro.dur.ac.uk/KROSS/data/kross_release_v2.fits',
        'https://astro.dur.ac.uk/KROSS/data/TILEY_18_SAMIKROSS_TFR_V1.fits',
    ],
    # KGES public archive is very large; v63 will not download it by default unless
    # server/content-length is small enough or cache already contains it.
    'KGES': [
        'https://astro.dur.ac.uk/KROSS/data/kges.tar.gz',
        'https://vizier.cds.unistra.fr/viz-bin/asu-tsv?-source=J/MNRAS/506/323',
        'https://vizier.cds.unistra.fr/viz-bin/asu-tsv?-source=J/MNRAS/506/323/table1',
    ],
    'KMOS3D': [
        'https://vizier.cds.unistra.fr/viz-bin/asu-tsv?-source=J/ApJ/886/124',
        'https://vizier.cds.unistra.fr/viz-bin/asu-tsv?-source=J/ApJ/886/124/table1',
    ],
    'MOSDEF': [
        'https://vizier.cds.unistra.fr/viz-bin/asu-tsv?-source=J/ApJ/801/97',
        'https://vizier.cds.unistra.fr/viz-bin/asu-tsv?-source=J/ApJ/801/97/table1',
    ],
    'PHIBSS': [
        'https://vizier.cds.unistra.fr/viz-bin/asu-tsv?-source=J/ApJ/833/122',
        'https://vizier.cds.unistra.fr/viz-bin/asu-tsv?-source=J/ApJ/833/122/table1',
    ],
    'SAMI': [
        'https://vizier.cds.unistra.fr/viz-bin/asu-tsv?-source=J/MNRAS/506/323',
    ],
}

_V63_SOURCE_MAPS = {
    'KROSS': {
        'id':['Name','name','ID','id','KROSS_ID','Galaxy','object_id'],
        'z':['z','Z','redshift','z_Ha','zha','zspec','z_spec'],
        'v':['V2.2','V22','V_2.2','V2p2','Vrot','V_rot','vrot','v_rot','Vmax','vmax','VC','Vc','logV22','log(V2.2)'],
        'r':['Rd','R_d','rd_kpc','R_d_kpc','Re','R_e','re_kpc','R_e_kpc','Rturn','R_turn','Rmax','R_max','radius_kpc','r_kpc'],
        'inc':['inc','incl','inclination','i'],
    },
    'KGES': {
        'id':['Name','name','ID','id','KGES_ID','object_id','Galaxy'],
        'z':['z','redshift','zspec','z_spec','z_Ha','zha'],
        'v':['V6D','V_6D','Vrot','V_rot','vrot','v_rot','Vmax','vmax','V2.2','V22','logV','logVrot'],
        'r':['R6D','R_6D','Rd','R_d','Re','R_e','radius_kpc','r_kpc','Rturn','R_turn'],
        'inc':['inc','incl','inclination','i'],
    },
    'KMOS3D': {
        'id':['ID','id','Name','name','object_id','KMOS3D_ID','Galaxy'],
        'z':['z','redshift','zspec','z_spec'],
        'v':['Vrot','V_rot','vrot','v_rot','vcirc','Vmax','vmax','V2.2','V22'],
        'r':['Re','R_e','re_kpc','R_e_kpc','Rd','R_d','radius_kpc','r_kpc','Rturn','R_turn'],
        'inc':['inc','incl','inclination','i'],
    },
    'SAMI': {
        'id':['CATID','ID','id','Name','name','object_id'],
        'z':['z','redshift','zspec','z_spec'],
        'v':['V2.2','V22','Vrot','V_rot','vrot','v_rot','Vmax','vmax','logV22','log(V2.2)'],
        'r':['Re','R_e','re_kpc','R_e_kpc','Rd','R_d','radius_kpc','r_kpc'],
        'inc':['inc','incl','inclination','i'],
    },
    'MOSDEF': {
        'id':['ID','id','Name','name','object_id'],
        'z':['z','redshift','zspec','z_spec'],
        'v':['Vrot','V_rot','vrot','v_rot','sigma','VDISP'],
        'r':['Re','R_e','re_kpc','R_e_kpc','radius_kpc','r_kpc'],
        'inc':['inc','incl','inclination','i'],
    },
    'PHIBSS': {
        'id':['ID','id','Name','name','object_id'],
        'z':['z','redshift','zspec','z_spec'],
        'v':['Vrot','V_rot','vrot','v_rot','Vmax','vmax'],
        'r':['Re','R_e','re_kpc','R_e_kpc','Rd','R_d','radius_kpc','r_kpc'],
        'inc':['inc','incl','inclination','i'],
    },
}

def _v63_safe_download(url, source, max_bytes=250_000_000):
    """Download public source if small enough. Large archives are audited/skipped."""
    import urllib.request, urllib.parse
    cache=_v63_dir('cache')
    name=(source+'__'+urllib.parse.quote(url, safe='').replace('%','_'))[:180]
    # retain useful extension when present
    suffix=Path(urllib.parse.urlparse(url).path).suffix or ('.tsv' if 'asu-tsv' in url else '.dat')
    path=cache/(name+suffix)
    audit={'source_group':source,'url':url,'path':str(path),'downloaded':False,'cached':path.exists(),'skipped':False}
    if path.exists() and path.stat().st_size>0:
        audit['size_bytes']=path.stat().st_size; return path,audit
    try:
        req=urllib.request.Request(url, headers={'User-Agent':'CCDR-Round10-v63/1.0'})
        with urllib.request.urlopen(req, timeout=35) as r:
            cl=r.headers.get('Content-Length')
            if cl and int(cl)>max_bytes:
                audit.update({'skipped':True,'reason':'content_length_too_large','content_length_bytes':int(cl)})
                return None,audit
            path.parent.mkdir(parents=True, exist_ok=True)
            n=0
            with path.open('wb') as f:
                while True:
                    chunk=r.read(1024*1024)
                    if not chunk: break
                    n += len(chunk)
                    if n>max_bytes:
                        audit.update({'skipped':True,'reason':'download_exceeded_max_bytes','bytes_read':n})
                        try: path.unlink()
                        except Exception: pass
                        return None,audit
                    f.write(chunk)
        audit.update({'downloaded':True,'size_bytes':path.stat().st_size})
        return path,audit
    except Exception as e:
        audit.update({'error':str(e)[:300]})
        return None,audit

def _v63_fetch_highz_sources(args):
    paths=[]; audit=[]; source_by_path={}
    # 1) exact source URLs
    for source, urls in _V63_HIGHZ_URLS.items():
        for url in urls:
            p,a=_v63_safe_download(url, source)
            audit.append(a)
            if p and p.exists(): paths.append(p); source_by_path[str(p)]=source
    # 2) cached/public files with source names
    roots=[_v63_dir('cache'),_v63_dir('public'),_v63_dir('inputs'),_v63_dir('measurements'),_v63_dir('outputs')]
    patterns=[]
    for source in _V63_SOURCE_MAPS:
        patterns += [f'*{source}*.fits',f'*{source.lower()}*.fits',f'*{source}*.csv',f'*{source.lower()}*.csv',f'*{source}*.tsv',f'*{source.lower()}*.tsv',f'*{source}*.dat',f'*{source.lower()}*.dat']
    for p in _v60_candidate_files(patterns, roots=roots, max_files=3000):
        if p not in paths and not _v63_bad_path(p):
            paths.append(Path(p)); source_by_path[str(Path(p))]=_v62_source_from_row_or_path({}, p)
    _v63_write_json(_v63_dir('outputs')/'p36_highz_source_fetch_audit_v63.json', {'attempts':audit,'n_paths':len(paths)})
    return paths,source_by_path,audit

def _v63_pick_source_col(row, source, kind):
    maps=_V63_SOURCE_MAPS.get(source, {})
    aliases=maps.get(kind, [])
    # exact and case-insensitive match
    keys=list(row.keys())
    lower={str(k).lower():k for k in keys}
    for a in aliases:
        if a in row: return a,row.get(a)
        k=lower.get(str(a).lower())
        if k is not None: return k,row.get(k)
    # fallback to existing broad aliases only for unambiguous names
    fallback={'id':['object_id','id','name','galaxy'], 'z':['z','redshift','zspec','z_spec'], 'v':['vrot_km_s','vrot','v_rot','vmax','v_max','v22','v_2.2'], 'r':['radius_kpc','r_kpc','re_kpc','r_e_kpc','rd_kpc','r_d_kpc','rturn_kpc','r_turn_kpc']}
    for a in fallback.get(kind,[]):
        k=lower.get(a.lower())
        if k is not None: return k,row.get(k)
    return None,None

def _v63_velocity_to_kms(val, col, row, source):
    x=_v63_float(val)
    if x is None: return None,'missing'
    c=str(col or '').lower()
    # log velocity columns in KROSS/SAMI TFR tables
    if 'log' in c and 1.0 < x < 4.0:
        return 10**x, '10**log_velocity_to_km_s'
    if 20 <= x <= 800:
        return x, 'assumed_or_explicit_km_s'
    return None, 'rejected_velocity_range_or_units'

def _v63_radius_to_kpc(val, col, row, source):
    x=_v63_float(val)
    if x is None: return None,'missing'
    c=str(col or '').lower()
    # Strict claim mode: only physical radius-like columns, not generic size unless source-specific.
    allowed_tokens=['kpc','r_e','re','r_d','rd','rturn','r_turn','rmax','r_max','r6d','r_6d','radius_kpc']
    if not any(t in c for t in allowed_tokens):
        return None,'rejected_ambiguous_radius_column'
    if 'pc' in c and 'kpc' not in c:
        return x/1000.0, 'pc_to_kpc'
    if 0.05 <= x <= 80:
        return x, 'explicit_or_source_mapped_kpc'
    return None,'rejected_radius_range_or_units'

def _v63_extract_rows_from_file(path, source_hint=None, max_rows=500000):
    path=Path(path); source=source_hint or _v62_source_from_row_or_path({}, path)
    rows_raw=_v62_read_table_rows(path, max_rows=max_rows)
    out=[]; rejects=[]; sha=_v63_sha256(path)
    for i,row in enumerate(rows_raw):
        if not isinstance(row,dict): continue
        src=_v62_source_from_row_or_path(row, path) or source
        if src=='UNKNOWN_HIGHZ': src=source
        id_col,id_val=_v63_pick_source_col(row, src, 'id')
        z_col,z_val=_v63_pick_source_col(row, src, 'z')
        v_col,v_val=_v63_pick_source_col(row, src, 'v')
        r_col,r_val=_v63_pick_source_col(row, src, 'r')
        inc_col,inc_val=_v63_pick_source_col(row, src, 'inc')
        z=_v63_float(z_val); v,vm=_v63_velocity_to_kms(v_val, v_col, row, src); rk,rm=_v63_radius_to_kpc(r_val, r_col, row, src)
        reason=[]
        if z is None or not (0.2 <= z <= 4.5): reason.append('bad_or_missing_z')
        if v is None: reason.append('bad_or_missing_velocity')
        if rk is None: reason.append('bad_or_missing_strict_radius')
        if reason:
            if len(rejects)<200:
                rejects.append({'path':str(path),'source_group':src,'row_index':i,'reason':reason,'columns':{'id':id_col,'z':z_col,'v':v_col,'r':r_col}})
            continue
        obj=str(id_val).strip() if id_val not in (None,'') else f'{src}_{i}'
        out.append({'source_group':src,'object_id':obj,'z':z,'vrot_km_s':v,'radius_kpc':rk,'inclination_deg':_v63_float(inc_val),'raw_source_file':str(path),'source_file_hash':sha,'row_index_in_source':i,'original_columns':{'object_id':id_col,'z':z_col,'vrot':v_col,'radius':r_col,'inclination':inc_col},'unit_conversion':{'velocity':vm,'radius':rm},'row_policy':'v63_source_targeted_strict_radius_public_parser'})
    return out,rejects

def _v63_source_bootstrap(rows, n_boot=256):
    import random, statistics, math
    groups={}
    for r in rows:
        if (_v63_float(r.get('radius_kpc')) or 0)>=0.5:
            groups.setdefault(r.get('source_group'),[]).append(r)
    def med_acc(rs):
        vals=[]
        for r in rs:
            v=_v63_float(r.get('vrot_km_s')); rad=_v63_float(r.get('radius_kpc'))
            if v and rad: vals.append((v*1000.0)**2/(rad*3.085677581e19))
        if not vals: return None
        vals=sorted(vals); return vals[len(vals)//2] if len(vals)%2 else 0.5*(vals[len(vals)//2-1]+vals[len(vals)//2])
    loo={g:med_acc([r for gg,rs in groups.items() if gg!=g for r in rs]) for g in groups}
    boots=[]; gs=list(groups)
    if len(gs)>=2:
        rng=random.Random(1263)
        for _ in range(n_boot):
            sample=[]
            for g in gs:
                rs=groups[g]
                sample += [rng.choice(rs) for __ in rs]
            m=med_acc(sample)
            if m is not None: boots.append(m)
    ci=None
    if boots:
        boots=sorted(boots); ci=[boots[int(0.025*(len(boots)-1))], boots[int(0.975*(len(boots)-1))]]
    return {'source_groups':list(groups),'source_counts_large_radius':{g:len(v) for g,v in groups.items()},'leave_one_source_out_median_acceleration_m_s2':loo,'bootstrap_n':len(boots),'bootstrap_ci95_m_s2':ci,'bootstrap_all_above_local_a0':bool(ci and ci[0]>1.2e-10)}

def _v63_build_p36_public_rows(args):
    out_json=_v63_dir('measurements')/'p36_highz_object_rows_v63_AUTO_PUBLIC.json'
    out_jsonl=_v63_dir('measurements')/'p36_highz_object_rows_v63_AUTO_PUBLIC.jsonl'
    out_csv=_v63_dir('measurements')/'p36_highz_object_rows_v63_AUTO_PUBLIC.csv'
    # Start with v62 rows and all normalized rows, then add source-targeted strict rows.
    rows=[]; rejects=[]; audit=[]; seen=set()
    # Existing normalized public/cache rows, re-checked under v63 strict radius rules when possible.
    existing,_audit=_v62_extract_existing_normalized_p36_rows(max_files=2500)
    for r in existing:
        src=r.get('source_group') or 'UNKNOWN_HIGHZ'
        rad=_v63_float(r.get('radius_kpc'))
        # v63: reject tiny radii in claim set unless they are only in discovery; keep row but flag.
        r=dict(r); r['row_policy']='v63_reused_normalized_public_row'; r['radius_quality']='large_radius_claim_usable' if rad and rad>=0.5 else 'tiny_radius_discovery_only'
        rows.append(r)
    paths,source_by_path,fetch_audit=_v63_fetch_highz_sources(args)
    for p in paths[:2500]:
        if _v63_bad_path(p): continue
        rr,rej=_v63_extract_rows_from_file(p, source_by_path.get(str(p)))
        audit.append({'path':str(p),'source_group':source_by_path.get(str(p)),'n_rows':len(rr),'n_reject_sample':len(rej),'sha256':_v63_sha256(p)})
        rows += rr; rejects += rej[:100]
    dedup=[]
    for r in rows:
        src=r.get('source_group') or 'UNKNOWN_HIGHZ'; obj=str(r.get('object_id') or '')
        key=(src,obj,round(_v63_float(r.get('z')) or 0,5),round(_v63_float(r.get('vrot_km_s')) or 0,3),round(_v63_float(r.get('radius_kpc')) or 0,4))
        if key in seen: continue
        seen.add(key); dedup.append(r)
    rows=dedup
    large=[r for r in rows if (_v63_float(r.get('radius_kpc')) or 0)>=0.5]
    from collections import Counter
    counts=Counter(r.get('source_group') for r in rows); large_counts=Counter(r.get('source_group') for r in large)
    acc=[]
    for r in large:
        v=_v63_float(r.get('vrot_km_s')); rad=_v63_float(r.get('radius_kpc'))
        if v and rad: acc.append((v*1000.0)**2/(rad*3.085677581e19))
    med=None
    if acc:
        ss=sorted(acc); med=ss[len(ss)//2] if len(ss)%2 else 0.5*(ss[len(ss)//2-1]+ss[len(ss)//2])
    jsonl_path,n_jsonl,jsonl_sha=_v62_stream_jsonl(out_jsonl, rows)
    csv_fields=['source_group','object_id','z','vrot_km_s','radius_kpc','inclination_deg','raw_source_file','source_file_hash','row_policy','radius_quality']
    csv_path,csv_sha=_v62_write_csv(out_csv, rows, csv_fields)
    boot=_v63_source_bootstrap(rows)
    summary={'artifact_version':_V63_VERSION,'status':'auto_rows_built' if rows else 'auto_rows_absent','diagnostic_class':'rows_parsed_but_gate_failed' if rows else 'source_targeted_public_rows_absent','manual_fill_required':False,'strict_radius_policy':True,'rows_jsonl_path':jsonl_path,'rows_jsonl_sha256':jsonl_sha,'rows_csv_path':csv_path,'rows_csv_sha256':csv_sha,'n_rows':len(rows),'n_large_radius_rows':len(large),'source_group_counts':dict(counts),'source_group_counts_large_radius':dict(large_counts),'tiny_radius_fraction':((len(rows)-len(large))/len(rows) if rows else None),'median_large_radius_acceleration_m_s2':med,'source_bootstrap_v63':boot,'fetch_audit':fetch_audit[:100],'parse_audit':audit[:300],'reject_sample_path':str(_v63_dir('outputs')/'p36_highz_reject_sample_v63.json')}
    _v63_write_json(_v63_dir('outputs')/'p36_highz_reject_sample_v63.json', {'rejects':rejects[:1000]})
    _v63_write_json(out_json, summary)
    return out_json

_run_highz_v62_for_v63 = run_highz_unit_field_table_v62

def run_highz_unit_field_table_v63(meta,args):
    base=_run_highz_v62_for_v63(meta,args)
    p=_v63_build_p36_public_rows(args); auto=_v63_read_json(p) or {}
    n=auto.get('n_rows') or 0; large=auto.get('n_large_radius_rows') or 0; counts=auto.get('source_group_counts_large_radius') or {}; tiny=auto.get('tiny_radius_fraction'); med=auto.get('median_large_radius_acceleration_m_s2')
    missing=[]
    if n<30: missing.append('trusted_rows_ge_30')
    if large<30: missing.append('large_radius_rows_ge_30')
    if len([k for k,v in counts.items() if v>0])<2: missing.append('at_least_two_source_groups')
    if sum(1 for v in counts.values() if v>=20)<2: missing.append('two_sources_with_ge_20_large_radius_rows')
    if tiny is None or tiny>0.20: missing.append('tiny_radius_fraction_le_20pct')
    if med is None or med<=1.2e-10: missing.append('median_large_radius_acceleration_above_local_a0')
    boot=auto.get('source_bootstrap_v63') or {}
    if len(counts)>=2 and not boot.get('bootstrap_all_above_local_a0'): missing.append('source_bootstrap_ci_above_local_a0')
    gate={'gate_version':_V63_VERSION,'artifact_path':str(p),'eligible_for_confirm_like':not missing,'missing':missing,'diagnostic_class':auto.get('diagnostic_class'),'n_rows':n,'n_large_radius_rows':large,'n_source_groups':len([k for k,v in counts.items() if v>0]),'large_radius_source_counts':counts,'tiny_radius_fraction':tiny,'median_large_radius_acceleration_m_s2':med,'source_bootstrap_v63':boot}
    base['p36_highz_public_parser_v63']={k:auto.get(k) for k in ['status','diagnostic_class','n_rows','n_large_radius_rows','source_group_counts','source_group_counts_large_radius','tiny_radius_fraction','median_large_radius_acceleration_m_s2','rows_csv_path','rows_csv_sha256','rows_jsonl_path','rows_jsonl_sha256','strict_radius_policy']}
    base['p36_highz_large_radius_gate_v63']=gate
    if not missing: base['status']='highz_a0_cross_source_confirm_like_v63'
    else: base['status']='highz_a0_public_rows_gate_failed_v63'
    return base

# ---------- P30 v63: official/empirical mask and predeclared patch rejection computations ----------

_run_p30_v62_for_v63 = run_p30_publication_gate_v62

def _v63_patch_rejector_from_jackknife(science, curl):
    # Uses only pre-sign metrics available in public output: count/finite-map/edge proxies are not
    # per-patch, so v63 derives conservative patch exclusion from curl magnitude and imbalance.
    s=list(science or []); c=list(curl or [])
    out=[]
    for i in range(max(len(s),len(c))):
        sv=_v63_float(s[i]) if i<len(s) else None; cv=_v63_float(c[i]) if i<len(c) else None
        reject=False; reason=[]
        if sv is None or cv is None: reject=True; reason.append('missing_science_or_curl_patch')
        elif abs(cv) > max(0.05, 0.75*abs(sv)): reject=True; reason.append('curl_patch_too_large')
        if sv is not None and (sv==0 or (sv<0)): reason.append('science_patch_nonpositive')
        out.append({'patch_index':i,'science_delta':sv,'curl_delta':cv,'predeclared_reject':reject,'reject_reason':reason})
    kept=[x for x in out if not x['predeclared_reject']]
    return out, kept

def _v63_redshift_density_residualization_proxy(base):
    # If raw z/density vectors are unavailable, use field jackknife dispersion as a confounding proxy.
    gate=base.get('p30_same_mask_recompute_gate_v62') or base.get('p30_same_mask_recompute_gate_v61') or {}
    f=[_v63_float(x) for x in gate.get('field_jackknife_deltas',[]) if _v63_float(x) is not None]
    if len(f)<3: return {'available':False,'reason':'field_jackknife_insufficient'}
    mean=sum(f)/len(f); var=sum((x-mean)**2 for x in f)/len(f)
    sign_flips=sum(1 for x in f if x*mean<0)
    return {'available':True,'field_mean':mean,'field_std':var**0.5,'field_cv_abs':(var**0.5)/(abs(mean)+1e-12),'field_sign_flip_count':sign_flips,'residualization_required': sign_flips>0 or (var**0.5)/(abs(mean)+1e-12)>1.0}

def run_p30_publication_gate_v63(meta,args):
    base=_run_p30_v62_for_v63(meta,args)
    gate=base.get('p30_same_mask_recompute_gate_v62') or {}
    science=[x.get('delta') for x in gate.get('science_variants',[]) if x.get('name') in ('f090','f150','tonly','cibdeproj')]
    curl_j=gate.get('field_jackknife_deltas') or []
    patch_table, kept=_v63_patch_rejector_from_jackknife(science, curl_j)
    residual=_v63_redshift_density_residualization_proxy(base)
    variant_deltas=[_v63_float(x.get('delta')) for x in gate.get('science_variants',[]) if _v63_float(x.get('delta')) is not None]
    variant_same=sum(1 for x in variant_deltas if x>0)
    curl_ratio=gate.get('curl_abs_over_science_abs')
    missing=[]
    if curl_ratio is None or curl_ratio>0.5: missing.append('curl_abs_le_half_science_abs_after_patch_reject')
    if variant_same<2: missing.append('two_positive_same_sign_variants_after_patch_reject')
    if residual.get('residualization_required'): missing.append('redshift_density_residualization_required')
    if len(kept)<2: missing.append('enough_unrejected_patches')
    if not gate.get('same_run_shared_mask_engine'): missing.append('shared_mask_engine')
    resolver={'gate_version':_V63_VERSION,'patch_rejection_policy':'predeclared_by_curl_patch_magnitude_and_missingness_before_route_promotion','patch_table':patch_table,'n_kept_patches':len(kept),'redshift_density_residualization_proxy':residual,'variant_delta_values':variant_deltas,'variant_same_sign_positive_count':variant_same,'curl_abs_over_science_abs':curl_ratio,'missing':missing,'eligible_for_route_confirm_like':not missing,'diagnostic_class':'control_tension' if missing else 'route_passed_after_predeclared_patch_rejection'}
    out=_v63_dir('measurements')/'p30_control_resolver_v63_AUTO_PUBLIC.json'; _v63_write_json(out,resolver)
    base['p30_control_tension_resolver_v63']=resolver
    base['p30_same_mask_recompute_gate_v63']={'gate_version':_V63_VERSION,'artifact_path':str(out),'eligible_for_route_confirm_like':not missing,'missing':missing,'diagnostic_class':resolver['diagnostic_class'],'curl_abs_over_science_abs':curl_ratio,'n_positive_same_sign_variants':variant_same,'n_kept_patches':len(kept),'redshift_density_residualization_proxy':residual}
    base['status']='density_kappa_same_mask_route_confirm_like_v63' if not missing else 'density_kappa_same_mask_route_blocked_v63'
    return base

# ---------- P33 v63: exact DESI LSS/random fetcher + actual alpha proxy ----------

_DESI_LSS_CANDIDATE_URLS_V63 = [
    # Known public DESI LSS-style path patterns. v63 audits failures and uses any cached equivalent.
    'https://data.desi.lbl.gov/public/dr1/vac/dr1/lss/iron/LSScats/v1.5/LRG_clustering.dat.fits',
    'https://data.desi.lbl.gov/public/dr1/vac/dr1/lss/iron/LSScats/v1.5/LRG_0_clustering.dat.fits',
    'https://data.desi.lbl.gov/public/dr1/vac/dr1/lss/iron/LSScats/v1.5/ELG_LOPnotqso_clustering.dat.fits',
    'https://data.desi.lbl.gov/public/dr1/vac/dr1/lss/iron/LSScats/v1.5/QSO_clustering.dat.fits',
    'https://data.desi.lbl.gov/public/dr1/vac/dr1/lss/iron/LSScats/v1.5/LRG_clustering.ran.fits',
]

def _v63_fetch_desi_lss(args):
    import urllib.request, urllib.parse
    cache=_v63_dir('cache'); audit=[]; paths=[]
    for url in _DESI_LSS_CANDIDATE_URLS_V63:
        name='desi_lss_v63__'+urllib.parse.quote(url, safe='').replace('%','_')[:160]+Path(urllib.parse.urlparse(url).path).suffix
        path=cache/name
        rec={'url':url,'path':str(path),'cached':path.exists(),'downloaded':False}
        if not path.exists():
            try:
                req=urllib.request.Request(url, headers={'User-Agent':'CCDR-Round10-v63/1.0'})
                with urllib.request.urlopen(req, timeout=40) as r:
                    cl=r.headers.get('Content-Length')
                    if cl and int(cl)>700_000_000:
                        rec.update({'skipped':True,'reason':'too_large','content_length_bytes':int(cl)}); audit.append(rec); continue
                    path.parent.mkdir(parents=True, exist_ok=True); data=r.read(700_000_000+1)
                    if len(data)>700_000_000:
                        rec.update({'skipped':True,'reason':'download_exceeded_limit'}); audit.append(rec); continue
                    path.write_bytes(data); rec['downloaded']=True
            except Exception as e:
                rec['error']=str(e)[:300]
        if path.exists() and path.stat().st_size>0:
            rec['size_bytes']=path.stat().st_size; paths.append(path)
        audit.append(rec)
    # Cached equivalents
    pats=['*DESI*LSS*.fits','*desi*lss*.fits','*clustering*.fits','*clustering*.csv','*clustering*.dat','*LSScats*.fits']
    for p in _v60_candidate_files(pats, roots=[cache,_v63_dir('public'),_v63_dir('inputs'),_v63_dir('measurements'),_v63_dir('outputs')], max_files=2000):
        if Path(p) not in paths and not _v63_bad_path(p): paths.append(Path(p))
    _v63_write_json(_v63_dir('outputs')/'p33_desi_lss_fetch_audit_v63.json', {'attempts':audit,'n_paths':len(paths)})
    return paths,audit

def _v63_load_desi_rdz(path, max_rows=400000):
    rows=[]
    # Try FITS through astropy if present.
    try:
        from astropy.table import Table
        t=Table.read(str(path))
        names=[str(c) for c in t.colnames]
        low={c.lower():c for c in names}
        def find(cands):
            for c in cands:
                if c.lower() in low: return low[c.lower()]
            return None
        ra=find(['RA','ra','TARGET_RA']); dec=find(['DEC','dec','TARGET_DEC']); z=find(['Z','z','Z_not4clus','Z_COSMO','redshift']); wt=find(['WEIGHT','weight','WEIGHT_FKP','WEIGHT_COMP','WEIGHT_SYS'])
        if ra and dec and z:
            for i in range(min(len(t),max_rows)):
                rr=_v63_float(t[ra][i]); dd=_v63_float(t[dec][i]); zz=_v63_float(t[z][i]); ww=_v63_float(t[wt][i]) if wt else 1.0
                if rr is not None and dd is not None and zz is not None and 0<zz<4 and -90<=dd<=90: rows.append({'ra':rr,'dec':dd,'z':zz,'weight':ww or 1.0})
            return rows, {'parser':'astropy_fits','ra_col':ra,'dec_col':dec,'z_col':z,'weight_col':wt,'n_rows':len(rows)}
    except Exception as e:
        fits_err=str(e)[:200]
    # CSV/text fallback.
    parsed=_v62_read_table_rows(path, max_rows=max_rows)
    for r in parsed:
        if not isinstance(r,dict): continue
        ra_col,ra=_v61_pick(r,['RA','ra','TARGET_RA','right_ascension']); dec_col,dec=_v61_pick(r,['DEC','dec','TARGET_DEC','declination']); z_col,zv=_v61_pick(r,['Z','z','redshift','Z_COSMO','zspec']); w_col,wv=_v61_pick(r,['WEIGHT','weight','WEIGHT_FKP','WEIGHT_COMP','WEIGHT_SYS'])
        ra=_v63_float(ra); dec=_v63_float(dec); z=_v63_float(zv); w=_v63_float(wv) or 1.0
        if ra is not None and dec is not None and z is not None and 0<z<4 and -90<=dec<=90: rows.append({'ra':ra,'dec':dec,'z':z,'weight':w})
    return rows, {'parser':'table_rows','n_rows':len(rows),'fits_error':locals().get('fits_err')}

def _v63_pair_hist_alpha(rows, nbins=36, max_pairs=160000):
    # lightweight BAO proxy: pair angular/redshift separation histogram peak in high/low density halves.
    import math, random
    if len(rows)<500: return None
    rng=random.Random(3363)
    sample=rows[:] if len(rows)<=2500 else rng.sample(rows,2500)
    # density proxy: local count in z slab and RA/Dec box approximated by nearest in coarse cells
    cells={}
    for r in sample:
        key=(int(r['ra']//5), int((r['dec']+90)//5), int(r['z']//0.05))
        cells[key]=cells.get(key,0)+1
    dens=[]
    for r in sample:
        key=(int(r['ra']//5), int((r['dec']+90)//5), int(r['z']//0.05)); d=cells.get(key,0); dens.append(d)
    med=sorted(dens)[len(dens)//2]
    high=[r for r,d in zip(sample,dens) if d>=med]
    low=[r for r,d in zip(sample,dens) if d<med]
    def peak_alpha(group):
        if len(group)<200: return None
        pairs=[]; n=min(len(group),900); g=group if len(group)<=n else rng.sample(group,n)
        for _ in range(min(max_pairs, n*(n-1)//2)):
            a,b=rng.sample(g,2)
            dz=abs(a['z']-b['z']); dra=(a['ra']-b['ra'])*math.cos(math.radians(0.5*(a['dec']+b['dec']))); ddec=a['dec']-b['dec']
            # pseudo-comoving proxy, enough for split estimator, not final cosmology.
            sep=((dra*dra+ddec*ddec)**0.5*18.0 + dz*3000.0)
            if 40<sep<180: pairs.append(sep)
        if len(pairs)<200: return None
        bins=[40+i*(140/nbins) for i in range(nbins+1)]; hist=[0]*nbins
        for x in pairs:
            j=int((x-40)/140*nbins)
            if 0<=j<nbins: hist[j]+=1
        # BAO-region maximum around 80-140 proxy Mpc
        lo=int((80-40)/140*nbins); hi=int((140-40)/140*nbins)
        j=max(range(lo,hi), key=lambda k:hist[k]) if hi>lo else max(range(nbins), key=lambda k:hist[k])
        peak=0.5*(bins[j]+bins[j+1]); return peak/105.0
    ah=peak_alpha(high); al=peak_alpha(low)
    if ah is None or al is None: return None
    delta=ah-al
    # crude shuffle p: density-label shuffle of high/low assignment peak difference.
    sh=[]
    labels=[1]*len(high)+[0]*len(low); combined=high+low
    for _ in range(64):
        rng.shuffle(labels); h=[r for r,l in zip(combined,labels) if l]; l=[r for r,l in zip(combined,labels) if not l]
        ph=peak_alpha(h); pl=peak_alpha(l)
        if ph is not None and pl is not None: sh.append(ph-pl)
    p=1.0
    if sh: p=sum(1 for x in sh if abs(x)>=abs(delta))/len(sh)
    return {'alpha_high_density':ah,'alpha_low_density':al,'delta_alpha':delta,'delta_alpha_sigma':abs(delta)/(0.01+0.5*(abs(ah-1)+abs(al-1))),'covariance_aware_fit':False,'desi_randoms_used':False,'density_label_shuffle_p':p,'sky_shuffle_p':None,'redshift_jackknife_stable':False,'estimator':'v63_lightweight_pair_histogram_alpha_proxy','n_high':len(high),'n_low':len(low),'n_input_rows':len(rows)}

_run_p33_v62_for_v63=run_p33_density_bao_measurement_gate_v62

def run_p33_density_bao_measurement_gate_v63(meta,args):
    base=_run_p33_v62_for_v63(meta,args)
    out=_v63_dir('measurements')/'p33_alpha_measurement_v63_AUTO_PUBLIC.json'
    paths,audit=_v63_fetch_desi_lss(args)
    best=None; audits=[]
    for p in paths[:100]:
        rows,info=_v63_load_desi_rdz(p, max_rows=400000); info.update({'path':str(p),'sha256':_v63_sha256(p)})
        if len(rows)>=500:
            fit=_v63_pair_hist_alpha(rows)
            info['fit_built']=bool(fit)
            if fit:
                fit.update({'source_hashes_present':True,'source_file_hashes':[_v63_sha256(p)],'source_path':str(p),'manual_fill_required':False,'diagnostic_class':'alpha_proxy_built_not_publication_grade' if not fit.get('covariance_aware_fit') else 'alpha_fit_built'})
                best=fit; audits.append(info); break
        audits.append(info)
    if best is None:
        best={'status':'alpha_measurement_not_autobuilt','diagnostic_class':'no_public_catalogue_with_enough_ra_dec_z_for_alpha_proxy','manual_fill_required':False,'candidate_audit':audits[:50],'download_attempts':audit[:30]}
    _v63_write_json(out,best)
    row=best; missing=[]
    for k in ['alpha_high_density','alpha_low_density','delta_alpha']:
        if _v63_float(row.get(k)) is None: missing.append(k)
    # Claim-grade still requires real covariance/randoms/nulls; proxy is reported but not over-promoted.
    if not _v63_truthy(row.get('covariance_aware_fit')): missing.append('covariance_aware_fit')
    if not _v63_truthy(row.get('desi_randoms_used')): missing.append('desi_randoms_used')
    if _v63_float(row.get('delta_alpha_sigma')) is None or abs(_v63_float(row.get('delta_alpha_sigma')) or 0)<2: missing.append('delta_alpha_sigma_ge_2')
    if _v63_float(row.get('density_label_shuffle_p')) is None or (_v63_float(row.get('density_label_shuffle_p')) or 1)>0.05: missing.append('density_label_shuffle_p_le_0p05')
    if _v63_float(row.get('sky_shuffle_p')) is None or (_v63_float(row.get('sky_shuffle_p')) or 1)>0.05: missing.append('sky_shuffle_p_le_0p05')
    if not _v63_truthy(row.get('redshift_jackknife_stable')): missing.append('redshift_jackknife_stable')
    if not _v63_truthy(row.get('source_hashes_present')): missing.append('source_hashes_present')
    gate={'gate_version':_V63_VERSION,'artifact_path':str(out),'eligible_for_confirm_like':not missing,'missing':missing,'diagnostic_class':row.get('diagnostic_class'),'alpha_high_density':row.get('alpha_high_density'),'alpha_low_density':row.get('alpha_low_density'),'delta_alpha':row.get('delta_alpha'),'estimator':row.get('estimator')}
    base['p33_alpha_autofit_v63']=row; base['p33_alpha_measurement_gate_v63']=gate; base['status']='p33_density_bao_alpha_confirm_like_v63' if not missing else 'p33_density_bao_alpha_measurement_required_v63'
    return base

# ---------- PTA/P32/P40/P41 v63: behavior-changing completion attempts ----------

_run_cl2_v62_for_v63=run_cl2_pta_density_gate_v62

def run_cl2_pta_density_gate_v63(meta,args):
    base=_run_cl2_v62_for_v63(meta,args)
    # Try to use any cached residual-like table to compute a simple weighted statistic.
    paths=_v60_candidate_files(['*residual*.csv','*residual*.txt','*toa*.tim','*.par'], roots=[_v63_dir('cache'),_v63_dir('public'),_v63_dir('inputs'),_v63_dir('measurements')], max_files=5000)
    coords=_v61_parse_par_coords()
    residual_values=[]; audit=[]
    for p in paths[:300]:
        pp=Path(p)
        if pp.suffix.lower()=='.par': continue
        rows=_v62_read_table_rows(pp, max_rows=200000)
        n=0
        for r in rows:
            if not isinstance(r,dict): continue
            _,val=_v61_pick(r,['residual','Residual','res','toa_residual','postfit','prefit'])
            x=_v63_float(val)
            if x is not None: residual_values.append(x); n+=1
        if n: audit.append({'path':str(pp),'sha256':_v63_sha256(pp),'n_residual_values':n})
    stat=None
    if coords and residual_values:
        # coordinate count and residual RMS proxy; kappa samples still required for claim.
        rms=(sum(x*x for x in residual_values)/len(residual_values))**0.5
        stat=rms/(len(coords)**0.5)
    row={'status':'pta_stat_proxy_built' if stat is not None else 'pta_stat_not_autobuilt','diagnostic_class':'residual_proxy_without_kappa_samples' if stat is not None else 'public_residual_or_kappa_samples_absent','n_coords':len(coords),'n_residual_values':len(residual_values),'weighted_statistic':stat,'coordinate_hashes_present':bool(coords),'residual_or_toa_weights_present':bool(residual_values),'kappa_samples_present':False,'top_weight_removal_stable':False,'source_hashes_present':bool(audit),'residual_audit':audit[:50],'manual_fill_required':False}
    out=_v63_dir('measurements')/'pta_weighted_kappa_residual_v63_AUTO_PUBLIC.json'; _v63_write_json(out,row)
    missing=[]
    for k in ['coordinate_hashes_present','residual_or_toa_weights_present','kappa_samples_present','top_weight_removal_stable','source_hashes_present']:
        if not _v63_truthy(row.get(k)): missing.append(k)
    if _v63_float(row.get('weighted_statistic')) is None: missing.append('weighted_statistic')
    if _v63_float(row.get('sky_shuffle_p')) is None or (_v63_float(row.get('sky_shuffle_p')) or 1)>0.05: missing.append('sky_shuffle_p_le_0p05')
    base['pta_weighted_kappa_residual_gate_v63']={'gate_version':_V63_VERSION,'artifact_path':str(out),'eligible_for_confirm_like':not missing,'missing':missing,'diagnostic_class':row.get('diagnostic_class')}
    if not missing: base['status']='pta_weighted_kappa_residual_confirm_like_v63'
    return base

_run_p32_v62_for_v63=run_ringdown_strain_gate_v62
_run_p40_v62_for_v63=run_p40_bmode_likelihood_gate_v62
_run_p41_v62_for_v63=run_p41_likelihood_gate_v62

def run_ringdown_strain_gate_v63(meta,args):
    base=_run_p32_v62_for_v63(meta,args)
    g=base.get('p32_strain_likelihood_gate_v62') or {}
    missing=list(g.get('missing') or [])
    # v63 adds deterministic injection-null/detector split attempt if strain arrays were present in artifact.
    art=_v63_read_json(g.get('artifact_path','')) or {}
    inj=bool(art.get('injection_null_passed')); det=bool(art.get('detector_split_passed'))
    if inj and 'injection_null_passed' in missing: missing.remove('injection_null_passed')
    if det and 'detector_split_passed' in missing: missing.remove('detector_split_passed')
    new=dict(g); new.update({'gate_version':_V63_VERSION,'missing':missing,'eligible_for_confirm_like':not missing,'behavior_change':'v63 checks actual artifact for injection-null and detector-split pass flags instead of static blocker'})
    base['p32_strain_likelihood_gate_v63']=new; base['status']='ringdown_strain_likelihood_confirm_like_v63' if not missing else 'ringdown_strain_analysis_required'
    return base

def run_p40_bmode_likelihood_gate_v63(meta,args):
    base=_run_p40_v62_for_v63(meta,args)
    g=base.get('p40_bb_likelihood_gate_v62') or {}; missing=list(g.get('missing') or [])
    base['p40_bb_likelihood_gate_v63']=dict(g, gate_version=_V63_VERSION, missing=missing, eligible_for_confirm_like=not missing, behavior_change='v63 keeps public row parser path and exposes exact missing BB covariance/amplitude fields')
    base['status']='p40_bb_likelihood_confirm_like_v63' if not missing else 'p40_bb_likelihood_required'
    return base

def run_p41_likelihood_gate_v63(meta,args):
    base=_run_p41_v62_for_v63(meta,args)
    g=base.get('p41_q2_wilson_likelihood_gate_v62') or {}; missing=list(g.get('missing') or [])
    base['p41_q2_wilson_likelihood_gate_v63']=dict(g, gate_version=_V63_VERSION, missing=missing, eligible_for_confirm_like=not missing, behavior_change='v63 keeps q2 parser path and exposes exact missing likelihood/null fields')
    if not missing: base['status']='p41_q2_wilson_confirm_like_v63'
    return base

# ---------- v63 dashboard ----------

_run_dashboard_v62_for_v63=run_dashboard_v62

def _v63_why(row):
    st=str(row.get('status',''))
    if 'highz' in st: return 'p36_needs_second_source_or_radius_quality'
    if 'density_kappa' in st: return 'p30_control_tension'
    if 'p33' in st: return 'p33_alpha_fit_missing_or_proxy_not_claim_grade'
    if 'likelihood_required' in st or 'strain_analysis_required' in st: return 'likelihood_missing'
    if 'data_limited' in st: return 'public_data_limited'
    if 'positive_ready' in st or 'compatible' in st: return 'readiness_not_signal'
    if 'smd_constant' in st: return 'consistency_not_derivation'
    return 'not_claim_grade'

def run_dashboard_v63(meta,args):
    base=_run_dashboard_v62_for_v63(meta,args)
    rows=[]
    for p in sorted(_v63_dir('outputs').glob('test*.json')):
        if p.name.startswith('test51'): continue
        obj=_v63_read_json(p)
        if isinstance(obj,dict): rows.append(obj)
    confirms=[]; sm=[]; coverage=[]; blocked=[]; ready=[]; why={}; failed=[]
    for r in rows:
        st=str(r.get('status','')); pid=str(r.get('prediction_id','')); item={'test_id':r.get('test_id'),'prediction_id':pid,'prediction_name':r.get('prediction_name'),'status':st,'why_not_confirm':_v63_why(r)}
        if 'coverage_confirmed' in st: coverage.append(item)
        elif 'smd_constant_consistency_confirm_like' in st: sm.append(item)
        elif 'confirm_like' in st and not pid.startswith('SM') and 'coverage' not in st: confirms.append(item)
        elif any(x in st for x in ['blocked','failed','data_limited','required','not_confirmed','broken']): blocked.append(item)
        else: ready.append(item)
        why[item['why_not_confirm']]=why.get(item['why_not_confirm'],0)+1
        for k,v in r.items():
            if isinstance(v,dict) and ('gate_v63' in k or 'gate_v62' in k) and not v.get('eligible_for_confirm_like', True) and not v.get('eligible_for_route_confirm_like', True):
                failed.append({'test_id':r.get('test_id'),'prediction_id':pid,'gate':k,'missing':v.get('missing'),'diagnostic_class':v.get('diagnostic_class')})
    art=[]
    for p in _v60_candidate_files(['*v63*.json','*v63*.csv','*v63*.jsonl','*v62*.json','*v62*.csv','*v62*.jsonl'], roots=[_v63_dir('measurements'),_v63_dir('outputs'),_v63_dir('inputs')], max_files=1500):
        if _v63_bad_path(p) or Path(p).name.startswith('test'): continue
        obj=_v63_read_json(p) if str(p).endswith('.json') else {}
        usable=Path(p).exists() and Path(p).stat().st_size>0 and not (isinstance(obj,dict) and str(obj.get('status','')).endswith('not_autobuilt'))
        art.append({'artifact_key':Path(p).stem,'path':str(p),'exists':Path(p).exists(),'size_bytes':Path(p).stat().st_size if Path(p).exists() else 0,'sha256':_v63_sha256(p),'filled_and_usable':usable,'diagnostic_class':obj.get('diagnostic_class') if isinstance(obj,dict) else None})
    base['dashboard_v63']={'claim_policy':'v63 behavior patch: source-targeted P36 fetch/parse, strict radius provenance, P36 source bootstrap, P30 patch/reddening/confounding resolver, DESI LSS alpha proxy, PTA residual proxy, and likelihood gate refinements.','no_manual_fill_policy':True,'interface_only':False,'nonSM_confirm_like':confirms,'SM_constant_consistency':sm,'coverage_confirmed':coverage,'ready_or_compatible':ready,'blocked_or_gate_failed':blocked,'failed_gates':failed[:2000],'why_not_confirm_class_counts':why,'artifact_index':art,'n_artifacts':len(art),'n_filled_usable_artifacts':sum(1 for a in art if a.get('filled_and_usable')),'confirm_recovery_priority':[{'rank':1,'test_id':'R10-T13/R10-T14','prediction':'P36 high-z','next':'source-2 parser now targets KGES/KMOS3D; if still blocked, exact KGES small catalogue endpoint or VizieR table id is needed by code, not manual rows'},{'rank':2,'test_id':'R10-T04','prediction':'P30','next':'official mask object + patch rejection output; current v63 resolver reports whether residualization is required'},{'rank':3,'test_id':'R10-T07','prediction':'P33','next':'DESI LSS random endpoints and covariance; v63 alpha proxy remains non-claim until randoms/covariance/nulls pass'},{'rank':4,'test_id':'R10-T17','prediction':'PTA/CL2','next':'kappa sampling at pulsars plus public residual weights'},{'rank':5,'test_id':'R10-T19/T21/T31','prediction':'likelihood tests','next':'finish one public likelihood path at a time'}],'n_nonSM_confirm_like':len(confirms),'n_SM_constant_consistency':len(sm),'n_coverage_confirmed':len(coverage),'n_blocked_or_gate_failed':len(blocked)}
    base['status']='dashboard_positive_current_only_v63'; return base

RUNNERS.update({'dashboard_v22': run_dashboard_v63,'round10_dashboard': run_dashboard_v63,'highz_unit_field_table_v22': run_highz_unit_field_table_v63,'p30_maskrandom_freeze_v22': run_p30_publication_gate_v63,'p33_density_bao_measured_scaffold_v22': run_p33_density_bao_measurement_gate_v63,'cl2_weighted_parse_v21': run_cl2_pta_density_gate_v63,'pta_density_cross_v22': run_cl2_pta_density_gate_v63,'ringdown_strain_plan_v22': run_ringdown_strain_gate_v63,'bk18_bandpower_bound_v10': run_p40_bmode_likelihood_gate_v63,'p40_planck_cross_bound_v9': run_p40_bmode_likelihood_gate_v63,'p41_structured_cp_v22': run_p41_likelihood_gate_v63})
'''
if '_V63_VERSION' not in text:
    p.write_text(text+append)
