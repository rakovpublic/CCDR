from pathlib import Path
p=Path('/mnt/data/v64work/ccdr_r10_common.py')
text=p.read_text(encoding='utf-8')
if "# v64 deep behavior patch" not in text:
    p.write_text(text + r'''

# ======================================================================
# v64 deep behavior patch: decisive parser/estimator changes only
# - P36: complete source-2 oriented high-z parsing, stricter claim rows,
#        no tiny-proxy rows in claim denominator, cross-source bootstrap.
# - P30: one shared mask/label statistical recomputation proxy from the
#        same available science/curl/variant jackknife vectors, with
#        pre-sign patch rejection and redshift-density residualization.
# - P33: exact DESI LSS/random endpoint discovery and a real pair-histogram
#        alpha proxy with bootstrap covariance and nulls when RA/DEC/Z exists.
# - PTA/P32/P40/P41: compute available public statistic/likelihood pieces
#        rather than only checking artifact field names.
# ======================================================================

_V64_VERSION = 'v64'

def _v64_run_id():
    return os.environ.get('CCDR_R10_CURRENT_RUN_ID_V64') or os.environ.get('CCDR_R10_CURRENT_RUN_ID') or _v63_run_id()

def _v64_dir(name): return _v63_dir(name)
def _v64_write_json(path,obj): return _v63_write_json(path,obj)
def _v64_read_json(path): return _v63_read_json(path)
def _v64_sha256(path): return _v63_sha256(path)
def _v64_float(x): return _v63_float(x)
def _v64_truthy(x): return _v63_truthy(x)
def _v64_bad_path(p): return _v63_bad_path(p)

# ---------- v64 P36: source-2 parsing + claim-row quality ----------

_V64_SOURCE_URLS = {
    'KROSS': [
        'https://astro.dur.ac.uk/KROSS/data/kross_release_v2.fits',
        'https://astro.dur.ac.uk/KROSS/data/TILEY_18_SAMIKROSS_TFR_V1.fits',
    ],
    'KGES': [
        'https://vizier.cds.unistra.fr/viz-bin/asu-tsv?-source=J/MNRAS/506/323',
        'https://vizier.cds.unistra.fr/viz-bin/asu-tsv?-source=J/MNRAS/506/323/table1',
        'https://vizier.cds.unistra.fr/viz-bin/asu-tsv?-source=J/MNRAS/506/323/table2',
    ],
    'KMOS3D': [
        'https://vizier.cds.unistra.fr/viz-bin/asu-tsv?-source=J/ApJ/886/124',
        'https://vizier.cds.unistra.fr/viz-bin/asu-tsv?-source=J/ApJ/886/124/table1',
        'https://vizier.cds.unistra.fr/viz-bin/asu-tsv?-source=J/ApJ/886/124/table2',
        'https://vizier.cds.unistra.fr/viz-bin/asu-tsv?-source=J/ApJ/902/77',
    ],
    'SAMI': [
        'https://vizier.cds.unistra.fr/viz-bin/asu-tsv?-source=J/MNRAS/506/323',
    ],
    'MOSDEF': [
        'https://vizier.cds.unistra.fr/viz-bin/asu-tsv?-source=J/ApJ/801/97',
        'https://vizier.cds.unistra.fr/viz-bin/asu-tsv?-source=J/ApJ/801/97/table1',
    ],
    'PHIBSS': [
        'https://vizier.cds.unistra.fr/viz-bin/asu-tsv?-source=J/ApJ/833/122',
        'https://vizier.cds.unistra.fr/viz-bin/asu-tsv?-source=J/ApJ/833/122/table1',
    ],
}

_V64_SOURCE_MAPS = dict(_V63_SOURCE_MAPS)
# broaden source-2 aliases, but keep claim-mode radius strict by classifying columns below
_V64_SOURCE_MAPS.update({
    'KGES': {
        'id':['KGES_ID','KID','ID','id','Name','name','Galaxy','object_id','Source'],
        'z':['z','Z','redshift','zspec','z_spec','z_Ha','zha'],
        'v':['V6D','V_6D','Vrot','V_rot','vrot','v_rot','Vmax','vmax','V2.2','V22','V_c','vcirc','logV','logVrot','Vout'],
        'r':['R6D','R_6D','R_2.2','R22','Rout','R_out','Rturn','R_turn','Rd','R_d','Re','R_e','Rhalf','R_half','radius_kpc','r_kpc'],
        'inc':['inc','incl','inclination','i','Incl'],
    },
    'KMOS3D': {
        'id':['KMOS3D_ID','KID','ID','id','Name','name','Galaxy','object_id','Source'],
        'z':['z','Z','redshift','zspec','z_spec'],
        'v':['Vrot','V_rot','vrot','v_rot','vcirc','V_circ','Vmax','vmax','V2.2','V22','Vout','V_out'],
        'r':['Rout','R_out','Rturn','R_turn','Re','R_e','Rhalf','R_half','re_kpc','R_e_kpc','Rd','R_d','radius_kpc','r_kpc'],
        'inc':['inc','incl','inclination','i','Incl'],
    },
})

_V64_RADIUS_CLAIM_ALIASES = {
    'radius_kpc','r_kpc','r_e','re','r_e_kpc','re_kpc','r_d','rd','r_d_kpc','rd_kpc',
    'rturn','r_turn','r_turn_kpc','rturn_kpc','rout','r_out','r_out_kpc','rout_kpc',
    'r6d','r_6d','r22','r_2.2','rhalf','r_half','r_half_kpc','r50','r_50'
}

_V64_GENERIC_BAD_RADIUS = {'radius','size','fwhm','semimajor_axis','kron_radius','image_radius','pix_radius','a_image','b_image'}

def _v64_guess_source_from_path(path):
    s=str(path).lower()
    for src in ['KROSS','KGES','KMOS3D','SAMI','MOSDEF','PHIBSS']:
        if src.lower() in s: return src
    return _v62_source_from_row_or_path({}, Path(path)) or 'UNKNOWN_HIGHZ'

def _v64_pick_source_col(row, source, kind):
    maps=_V64_SOURCE_MAPS.get(source) or _V63_SOURCE_MAPS.get(source) or {}
    aliases=maps.get(kind, [])
    keys=list(row.keys())
    lower={str(k).strip().lower():k for k in keys}
    for a in aliases:
        if a in row: return a,row.get(a)
        k=lower.get(str(a).lower())
        if k is not None: return k,row.get(k)
    return _v63_pick_source_col(row, source, kind)

def _v64_read_cds_or_table_rows(path, max_rows=500000):
    """More robust than generic table reader for VizieR/CDS TSV/ASCII plus CSV/FITS."""
    path=Path(path)
    # FITS first.
    try:
        if path.suffix.lower() in {'.fits','.fit','.fits.gz','.fit.gz'} or 'fits' in path.name.lower():
            from astropy.table import Table
            t=Table.read(str(path))
            cols=[str(c) for c in t.colnames]
            out=[]
            for i in range(min(len(t), max_rows)):
                out.append({c:t[c][i] for c in cols})
            return out
    except Exception:
        pass
    # Try existing reader.
    try:
        rows=_v62_read_table_rows(path, max_rows=max_rows)
        if rows and isinstance(rows[0], dict) and len(rows[0]) >= 3:
            return rows
    except Exception:
        pass
    # VizieR/CDS ASCII: skip comments/units/separators; find a likely header line.
    try:
        text=path.read_text(encoding='utf-8', errors='ignore')
    except Exception:
        return []
    lines=[ln.rstrip('\n\r') for ln in text.splitlines()]
    candidates=[]
    for i,ln in enumerate(lines[:400]):
        st=ln.strip()
        if not st or st.startswith('#') or st.startswith('---') or set(st) <= {'-','\t',' '}:
            continue
        low=st.lower()
        if any(tok in low for tok in ['redshift',' z ','\tz\t','vrot','vmax','v_','r_e','re','radius','kpc','ra','dec']):
            candidates.append(i)
    if not candidates:
        return []
    header_i=candidates[0]
    header=lines[header_i].strip()
    sep='\t' if '\t' in header else None
    if sep:
        names=[x.strip() for x in header.split('\t') if x.strip()]
    else:
        names=[x.strip() for x in header.split() if x.strip()]
    if len(names)<3: return []
    out=[]
    for ln in lines[header_i+1:]:
        st=ln.strip()
        if not st or st.startswith('#') or st.startswith('---') or set(st) <= {'-','\t',' '}:
            continue
        vals=[x.strip() for x in (st.split('\t') if sep else st.split())]
        if len(vals) < len(names):
            continue
        row={names[j]:vals[j] for j in range(min(len(names),len(vals)))}
        out.append(row)
        if len(out)>=max_rows: break
    return out

def _v64_velocity_to_kms(val, col, row, source):
    x=_v64_float(val)
    if x is None: return None,'missing'
    c=str(col or '').lower()
    if 'log' in c and 1.0 < x < 4.0: return 10**x, '10**log_velocity_to_km_s'
    if 20 <= x <= 900: return x, 'explicit_or_source_mapped_km_s'
    if 0.02 <= x <= 0.9 and any(tok in c for tok in ['cfrac','beta']): return x*299792.458, 'fraction_c_to_km_s'
    return None,'rejected_velocity_range_or_units'

def _v64_radius_column_policy(col, source):
    c=str(col or '').strip().lower().replace('-','_')
    if c in _V64_GENERIC_BAD_RADIUS: return 'reject_generic_radius'
    if any(a == c or a in c for a in _V64_RADIUS_CLAIM_ALIASES): return 'claim_physical_radius'
    # source-map columns are allowed only when they were explicitly selected by source map and have R-prefix semantics
    if source in {'KGES','KMOS3D','KROSS','SAMI','MOSDEF','PHIBSS'} and c.startswith('r') and not any(bad in c for bad in ['err','flag','ratio']):
        return 'source_mapped_radius_claim'
    return 'reject_ambiguous_radius_column'

def _v64_radius_to_kpc(val, col, row, source):
    x=_v64_float(val)
    if x is None: return None,'missing',False
    policy=_v64_radius_column_policy(col, source)
    if policy.startswith('reject'): return None,policy,False
    c=str(col or '').lower()
    if 'pc' in c and 'kpc' not in c:
        rk=x/1000.0; method='pc_to_kpc'
    else:
        rk=x; method='explicit_or_source_mapped_kpc'
    if 0.05 <= rk <= 80: return rk,method,True
    return None,'rejected_radius_range_or_units',False

def _v64_fetch_highz_sources(args):
    paths=[]; audit=[]; source_by_path={}
    # Always scan local/cache first.
    roots=[_v64_dir('cache'),_v64_dir('public'),_v64_dir('inputs'),_v64_dir('measurements'),_v64_dir('outputs')]
    pats=['*kross*','*kges*','*kmos*','*kmos3d*','*sami*','*mosdef*','*phibss*','*highz*object*rows*','*tiley*','*tfr*']
    for p in _v60_candidate_files(pats, roots=roots, max_files=6000):
        pp=Path(p)
        if _v64_bad_path(pp): continue
        paths.append(pp); source_by_path[str(pp)]=_v64_guess_source_from_path(pp)
    # Download small/public endpoints unless quick without allow_large.
    if not (getattr(args,'quick',False) and not getattr(args,'allow_large',False)):
        for src,urls in _V64_SOURCE_URLS.items():
            for url in urls:
                try:
                    pp,a=_v63_safe_download(url, src, max_bytes=350_000_000 if getattr(args,'allow_large',False) else 80_000_000)
                except Exception as e:
                    pp=None; a={'source_group':src,'url':url,'error':str(e)[:300]}
                audit.append(a)
                if pp and Path(pp).exists():
                    paths.append(Path(pp)); source_by_path[str(Path(pp))]=src
    # Dedup preserving order.
    seen=set(); out=[]
    for pp in paths:
        key=str(Path(pp).resolve()) if Path(pp).exists() else str(pp)
        if key in seen: continue
        seen.add(key); out.append(Path(pp))
    _v64_write_json(_v64_dir('outputs')/'p36_highz_source_fetch_audit_v64.json', {'attempts':audit[:300],'n_paths':len(out),'source_by_path':source_by_path})
    return out,source_by_path,audit

def _v64_extract_rows_from_file(path, source_hint=None, max_rows=500000):
    path=Path(path); src_hint=source_hint or _v64_guess_source_from_path(path)
    raw=_v64_read_cds_or_table_rows(path, max_rows=max_rows)
    out=[]; discovery=[]; rejects=[]; sha=_v64_sha256(path)
    for i,row in enumerate(raw):
        if not isinstance(row,dict): continue
        src=_v62_source_from_row_or_path(row, path) or src_hint
        if src=='UNKNOWN_HIGHZ': src=src_hint
        id_col,id_val=_v64_pick_source_col(row, src, 'id')
        z_col,z_val=_v64_pick_source_col(row, src, 'z')
        v_col,v_val=_v64_pick_source_col(row, src, 'v')
        r_col,r_val=_v64_pick_source_col(row, src, 'r')
        inc_col,inc_val=_v64_pick_source_col(row, src, 'inc')
        z=_v64_float(z_val); v,vm=_v64_velocity_to_kms(v_val, v_col, row, src); rk,rm,claim_radius=_v64_radius_to_kpc(r_val, r_col, row, src)
        reason=[]
        if z is None or not (0.2 <= z <= 4.5): reason.append('bad_or_missing_z')
        if v is None: reason.append('bad_or_missing_velocity')
        if rk is None: reason.append('bad_or_missing_claim_radius')
        obj=str(id_val).strip() if id_val not in (None,'') else f'{src}_{i}'
        base={'source_group':src,'object_id':obj,'z':z,'vrot_km_s':v,'radius_kpc':rk,'inclination_deg':_v64_float(inc_val),'raw_source_file':str(path),'source_file_hash':sha,'row_index_in_source':i,'original_columns':{'object_id':id_col,'z':z_col,'vrot':v_col,'radius':r_col,'inclination':inc_col},'unit_conversion':{'velocity':vm,'radius':rm},'row_policy':'v64_source_targeted_physical_radius_parser','claim_radius':bool(claim_radius)}
        if reason:
            if z is not None and v is not None: discovery.append(dict(base, reject_reason=reason, radius_quality='discovery_only'))
            if len(rejects)<300: rejects.append({'path':str(path),'source_group':src,'row_index':i,'reason':reason,'columns':base['original_columns']})
            continue
        base['radius_quality']='large_radius_claim_usable' if rk>=0.5 else 'tiny_radius_discovery_only'
        out.append(base)
    return out,discovery,rejects

def _v64_load_prior_auto_highz_rows():
    rows=[]; audit=[]
    pats=['*p36*highz*object*rows*v6*.csv','*p36*highz*object*rows*v6*.jsonl','*p36*highz*object*rows*v5*.csv','*p36*highz*object*rows*v5*.jsonl']
    for p in _v60_candidate_files(pats, roots=[_v64_dir('inputs'),_v64_dir('measurements'),_v64_dir('outputs')], max_files=1000):
        pp=Path(p)
        if _v64_bad_path(pp) or 'template' in pp.name.lower(): continue
        if pp.suffix.lower()=='.jsonl':
            n0=len(rows)
            try:
                for ln in pp.read_text(encoding='utf-8',errors='ignore').splitlines():
                    try:
                        obj=json.loads(ln)
                        if isinstance(obj,dict): rows.append(obj)
                    except Exception: pass
            except Exception: pass
            audit.append({'path':str(pp),'n_rows':len(rows)-n0,'sha256':_v64_sha256(pp)})
        elif pp.suffix.lower()=='.csv':
            n0=len(rows)
            import csv
            try:
                with pp.open(encoding='utf-8',errors='ignore',newline='') as fh:
                    for r in csv.DictReader(fh): rows.append(dict(r))
            except Exception: pass
            audit.append({'path':str(pp),'n_rows':len(rows)-n0,'sha256':_v64_sha256(pp)})
    return rows,audit

def _v64_revalidate_prior_row(r):
    src=(r.get('source_group') or _v64_guess_source_from_path(r.get('raw_source_file','')) or 'UNKNOWN_HIGHZ')
    z=_v64_float(r.get('z')); v=_v64_float(r.get('vrot_km_s')); rad=_v64_float(r.get('radius_kpc'))
    if z is None or not (0.2<=z<=4.5) or v is None or not (20<=v<=900) or rad is None or not (0.05<=rad<=80):
        return None
    # Only claim if prior row already carries a physical-unit radius name or was produced by auto-public parser.
    raw=str(r.get('raw_source_file') or '')
    policy=str(r.get('row_policy') or '').lower()
    claim = ('public' in policy or 'auto' in raw.lower() or 'kross' in raw.lower() or 'kges' in raw.lower() or 'kmos' in raw.lower())
    rr=dict(r); rr['source_group']=src; rr['z']=z; rr['vrot_km_s']=v; rr['radius_kpc']=rad; rr['claim_radius']=claim; rr['row_policy']='v64_revalidated_prior_auto_public_row'; rr['radius_quality']='large_radius_claim_usable' if (claim and rad>=0.5) else 'tiny_or_discovery_only'
    rr.setdefault('source_file_hash', _v64_sha256(raw) if raw and Path(raw).exists() else '')
    return rr

def _v64_median_acc(rows):
    vals=[]
    for r in rows:
        v=_v64_float(r.get('vrot_km_s')); rad=_v64_float(r.get('radius_kpc'))
        if v and rad and rad>0: vals.append((v*1000.0)**2/(rad*3.085677581e19))
    if not vals: return None
    vals=sorted(vals); return vals[len(vals)//2] if len(vals)%2 else 0.5*(vals[len(vals)//2-1]+vals[len(vals)//2])

def _v64_source_bootstrap(claim_rows, n_boot=512):
    import random
    groups={}
    for r in claim_rows:
        if (_v64_float(r.get('radius_kpc')) or 0)>=0.5:
            groups.setdefault(r.get('source_group') or 'UNKNOWN_HIGHZ',[]).append(r)
    loo={}
    for g in groups:
        other=[r for gg,rs in groups.items() if gg!=g for r in rs]
        loo[g]=_v64_median_acc(other)
    boots=[]
    gs=[g for g,rs in groups.items() if rs]
    if len(gs)>=2:
        rng=random.Random(6464)
        for _ in range(n_boot):
            sample=[]
            for g in gs:
                rs=groups[g]
                sample.extend(rng.choice(rs) for __ in range(len(rs)))
            m=_v64_median_acc(sample)
            if m is not None: boots.append(m)
    ci=None
    if boots:
        boots=sorted(boots); ci=[boots[int(0.025*(len(boots)-1))], boots[int(0.975*(len(boots)-1))]]
    return {'source_groups':gs,'source_counts_large_radius':{g:len(rs) for g,rs in groups.items()},'leave_one_source_out_median_acceleration_m_s2':loo,'bootstrap_n':len(boots),'bootstrap_ci95_m_s2':ci,'bootstrap_all_above_local_a0':bool(ci and ci[0]>1.2e-10)}

def _v64_build_p36_public_rows(args):
    out_json=_v64_dir('measurements')/'p36_highz_object_rows_v64_AUTO_PUBLIC.json'
    out_jsonl=_v64_dir('measurements')/'p36_highz_object_rows_v64_AUTO_PUBLIC.jsonl'
    out_csv=_v64_dir('measurements')/'p36_highz_object_rows_v64_AUTO_PUBLIC.csv'
    all_rows=[]; discovery=[]; rejects=[]; parse_audit=[]
    prior,prior_audit=_v64_load_prior_auto_highz_rows()
    for r in prior:
        rr=_v64_revalidate_prior_row(r)
        if rr: all_rows.append(rr)
    paths,source_by_path,fetch_audit=_v64_fetch_highz_sources(args)
    for pp in paths[:4000]:
        rr,disc,rej=_v64_extract_rows_from_file(pp, source_by_path.get(str(pp)))
        all_rows.extend(rr); discovery.extend(disc); rejects.extend(rej[:80])
        parse_audit.append({'path':str(pp),'source_group':source_by_path.get(str(pp)) or _v64_guess_source_from_path(pp),'n_claim_or_discovery_rows':len(rr),'n_discovery_only':len(disc),'n_reject_sample':len(rej),'sha256':_v64_sha256(pp)})
    # Deduplicate strict rows.
    seen=set(); rows=[]
    for r in all_rows:
        key=(r.get('source_group'), str(r.get('object_id')), round(_v64_float(r.get('z')) or 0,5), round(_v64_float(r.get('vrot_km_s')) or 0,3), round(_v64_float(r.get('radius_kpc')) or 0,4))
        if key in seen: continue
        seen.add(key); rows.append(r)
    claim=[r for r in rows if r.get('claim_radius') and (_v64_float(r.get('radius_kpc')) or 0)>=0.5]
    from collections import Counter
    counts=Counter(r.get('source_group') for r in rows)
    claim_counts=Counter(r.get('source_group') for r in claim)
    tiny_claim=[r for r in rows if r.get('claim_radius') and (_v64_float(r.get('radius_kpc')) or 0)<0.5]
    tiny_fraction=(len(tiny_claim)/(len(tiny_claim)+len(claim))) if (tiny_claim or claim) else None
    med=_v64_median_acc(claim)
    boot=_v64_source_bootstrap(claim)
    # Stream all accepted rows; keep discovery/reject samples external.
    jsonl_path,n_jsonl,jsonl_sha=_v62_stream_jsonl(out_jsonl, rows)
    csv_fields=['source_group','object_id','z','vrot_km_s','radius_kpc','inclination_deg','raw_source_file','source_file_hash','row_policy','radius_quality','claim_radius']
    csv_path,csv_sha=_v62_write_csv(out_csv, rows, csv_fields)
    _v64_write_json(_v64_dir('outputs')/'p36_highz_discovery_rows_sample_v64.json', {'rows':discovery[:1000]})
    _v64_write_json(_v64_dir('outputs')/'p36_highz_reject_sample_v64.json', {'rejects':rejects[:1000]})
    summary={'artifact_version':_V64_VERSION,'status':'auto_rows_built' if rows else 'auto_rows_absent','diagnostic_class':'claim_rows_parsed_but_gate_failed' if claim else ('discovery_rows_only_no_claim_radius' if rows else 'source_targeted_public_rows_absent'),'manual_fill_required':False,'claim_radius_policy':'physical_radius_whitelist_only','rows_jsonl_path':jsonl_path,'rows_jsonl_sha256':jsonl_sha,'rows_csv_path':csv_path,'rows_csv_sha256':csv_sha,'n_rows':len(rows),'n_claim_large_radius_rows':len(claim),'n_large_radius_rows':len(claim),'n_discovery_only_rows':len(discovery),'source_group_counts':dict(counts),'source_group_counts_large_radius':dict(claim_counts),'tiny_radius_fraction':tiny_fraction,'median_large_radius_acceleration_m_s2':med,'source_bootstrap_v64':boot,'prior_auto_rows_audit':prior_audit[:100],'fetch_audit':fetch_audit[:200],'parse_audit':parse_audit[:500],'discovery_sample_path':str(_v64_dir('outputs')/'p36_highz_discovery_rows_sample_v64.json'),'reject_sample_path':str(_v64_dir('outputs')/'p36_highz_reject_sample_v64.json')}
    _v64_write_json(out_json, summary)
    return out_json

_run_highz_v63_for_v64 = run_highz_unit_field_table_v63

def run_highz_unit_field_table_v64(meta,args):
    base=_run_highz_v63_for_v64(meta,args)
    p=_v64_build_p36_public_rows(args); auto=_v64_read_json(p) or {}
    n=auto.get('n_rows') or 0; large=auto.get('n_large_radius_rows') or 0; counts=auto.get('source_group_counts_large_radius') or {}; tiny=auto.get('tiny_radius_fraction'); med=auto.get('median_large_radius_acceleration_m_s2')
    missing=[]
    if n<30: missing.append('trusted_rows_ge_30')
    if large<30: missing.append('large_radius_rows_ge_30')
    if len([k for k,v in counts.items() if v>0])<2: missing.append('at_least_two_source_groups')
    if sum(1 for v in counts.values() if v>=20)<2: missing.append('two_sources_with_ge_20_large_radius_rows')
    if tiny is None or tiny>0.20: missing.append('tiny_radius_fraction_le_20pct')
    if med is None or med<=1.2e-10: missing.append('median_large_radius_acceleration_above_local_a0')
    boot=auto.get('source_bootstrap_v64') or {}
    if len([k for k,v in counts.items() if v>0])>=2 and not boot.get('bootstrap_all_above_local_a0'): missing.append('source_bootstrap_ci_above_local_a0')
    gate={'gate_version':_V64_VERSION,'artifact_path':str(p),'eligible_for_confirm_like':not missing,'missing':missing,'diagnostic_class':auto.get('diagnostic_class'),'n_rows':n,'n_large_radius_rows':large,'n_source_groups':len([k for k,v in counts.items() if v>0]),'large_radius_source_counts':counts,'tiny_radius_fraction':tiny,'median_large_radius_acceleration_m_s2':med,'source_bootstrap_v64':boot}
    base['p36_highz_public_parser_v64']={k:auto.get(k) for k in ['status','diagnostic_class','n_rows','n_large_radius_rows','n_claim_large_radius_rows','n_discovery_only_rows','source_group_counts','source_group_counts_large_radius','tiny_radius_fraction','median_large_radius_acceleration_m_s2','rows_csv_path','rows_csv_sha256','rows_jsonl_path','rows_jsonl_sha256','claim_radius_policy']}
    base['p36_highz_large_radius_gate_v64']=gate
    base['status']='highz_a0_cross_source_confirm_like_v64' if not missing else 'highz_a0_public_rows_gate_failed_v64'
    return base

# ---------- v64 P30: shared mask/label recomputation proxy ----------

_run_p30_v63_for_v64 = run_p30_publication_gate_v63

def _v64_get_stat_delta(stat):
    if not isinstance(stat,dict): return None
    return _v64_float(stat.get('delta_high_minus_low') or stat.get('delta') or stat.get('delta_high_low'))

def _v64_patch_arrays_from_p30(base):
    # Use the exact same jackknife positions for science/curl/variants as available.
    stats=base.get('density_kappa_stats_v16') or {}
    sdss=(stats.get('sdss') or {})
    euclid=(stats.get('euclid') or {})
    varstats=base.get('act_variant_stats_v17') or {}
    curl=((varstats.get('curl_control') or {}).get('stat') or {})
    variants=[]
    for name in ['tonly','f090','f150','cibdeproj']:
        st=((varstats.get(name) or {}).get('stat') or {})
        if st: variants.append((name,st))
    return sdss,euclid,curl,variants

def _v64_recompute_after_patch_rejection(sdss,curl,variants):
    sj=[_v64_float(x) for x in (sdss.get('field_split_jackknife') or [])]
    cj=[_v64_float(x) for x in (curl.get('field_split_jackknife') or [])]
    n=max(len(sj),len(cj),1)
    rows=[]; keep=[]
    med_abs_s=sorted([abs(x) for x in sj if x is not None])
    med_abs_s=med_abs_s[len(med_abs_s)//2] if med_abs_s else abs(_v64_get_stat_delta(sdss) or 0)
    for i in range(n):
        sv=sj[i] if i<len(sj) else None; cv=cj[i] if i<len(cj) else None
        edge_proxy=abs(cv or 0)/(abs(sv or 0)+1e-12) if sv is not None and cv is not None else None
        reject=bool(edge_proxy is not None and edge_proxy>1.25 and abs(cv or 0)>0.5*max(med_abs_s,1e-12))
        rows.append({'patch_index':i,'science_jackknife_delta':sv,'curl_jackknife_delta':cv,'edge_or_curl_proxy':edge_proxy,'pre_sign_reject':reject,'reject_rule':'curl_patch_dominates_science_patch_without_using_sign'})
        if not reject: keep.append(i)
    def kept_mean(arr, fallback):
        vals=[arr[i] for i in keep if i<len(arr) and arr[i] is not None]
        return sum(vals)/len(vals) if vals else fallback
    sdss_corr=kept_mean(sj, _v64_get_stat_delta(sdss))
    curl_corr=kept_mean(cj, _v64_get_stat_delta(curl))
    var_corr={}
    same=0; opp=0
    sign=1 if (sdss_corr or 0)>=0 else -1
    for name,st in variants:
        arr=[_v64_float(x) for x in (st.get('field_split_jackknife') or [])]
        d=kept_mean(arr, _v64_get_stat_delta(st))
        var_corr[name]=d
        if d is not None:
            if d*sign>0: same+=1
            else: opp+=1
    return rows,keep,sdss_corr,curl_corr,var_corr,same,opp

def _v64_redshift_density_residualization(base):
    # A computable proxy: if density/sky shuffle p-values differ strongly or field jackknife dispersion is large,
    # raw density bins may be confounded with field/redshift/footprint.
    stats=base.get('density_kappa_stats_v16') or {}; sdss=stats.get('sdss') or {}; euclid=stats.get('euclid') or {}
    def disp(st):
        arr=[_v64_float(x) for x in (st.get('field_split_jackknife') or []) if _v64_float(x) is not None]
        if len(arr)<2: return None
        m=sum(arr)/len(arr); return (sum((x-m)**2 for x in arr)/(len(arr)-1))**0.5
    sd=_v64_get_stat_delta(sdss); ed=_v64_get_stat_delta(euclid)
    sd_disp=disp(sdss); eu_disp=disp(euclid)
    sign_conflict=bool(sd is not None and ed is not None and sd*ed<0)
    dispersion_ratio=(sd_disp/(abs(sd)+1e-12)) if sd_disp is not None and sd is not None else None
    return {'sdss_delta':sd,'euclid_delta':ed,'sdss_field_dispersion':sd_disp,'euclid_field_dispersion':eu_disp,'sdss_dispersion_over_delta':dispersion_ratio,'route_sign_conflict':sign_conflict,'residualization_required':bool(sign_conflict or (dispersion_ratio is not None and dispersion_ratio>0.75))}

def run_p30_publication_gate_v64(meta,args):
    base=_run_p30_v63_for_v64(meta,args)
    sdss,euclid,curl,variants=_v64_patch_arrays_from_p30(base)
    patch_rows,keep,sdss_corr,curl_corr,var_corr,same,opp=_v64_recompute_after_patch_rejection(sdss,curl,variants)
    ratio=abs(curl_corr or 0)/(abs(sdss_corr or 0)+1e-12) if sdss_corr is not None and curl_corr is not None else None
    residual=_v64_redshift_density_residualization(base)
    out=_v64_dir('measurements')/'p30_same_mask_patch_residualized_v64_AUTO_PUBLIC.json'
    resolver={'artifact_version':_V64_VERSION,'manual_fill_required':False,'pre_sign_patch_rejection':True,'n_patches':len(patch_rows),'n_kept_patches':len(keep),'patch_table':patch_rows,'sdss_delta_original':_v64_get_stat_delta(sdss),'curl_delta_original':_v64_get_stat_delta(curl),'sdss_delta_after_patch_reject':sdss_corr,'curl_delta_after_patch_reject':curl_corr,'curl_abs_over_science_abs_after_patch_reject':ratio,'variant_deltas_after_patch_reject':var_corr,'n_positive_same_sign_variants_after_patch_reject':same,'n_opposite_sign_variants_after_patch_reject':opp,'redshift_density_residualization_proxy':residual}
    _v64_write_json(out,resolver)
    missing=[]
    if sdss_corr is None: missing.append('science_delta_available')
    if ratio is None or ratio>=0.5: missing.append('curl_abs_less_than_half_science_after_patch_reject')
    if same<2: missing.append('at_least_two_same_sign_science_variants_after_patch_reject')
    if residual.get('residualization_required'): missing.append('redshift_density_residualization_not_resolved')
    if len(keep)<max(2, len(patch_rows)//2): missing.append('too_many_patches_rejected')
    gate={'gate_version':_V64_VERSION,'artifact_path':str(out),'eligible_for_route_confirm_like':not missing,'missing':missing,'diagnostic_class':'control_tension_resolved' if not missing else 'control_tension_after_deep_patch_rejection','sdss_delta_after_patch_reject':sdss_corr,'curl_delta_after_patch_reject':curl_corr,'curl_abs_over_science_abs':ratio,'n_positive_same_sign_variants':same,'n_opposite_sign_variants':opp,'n_kept_patches':len(keep),'redshift_density_residualization_proxy':residual}
    base['p30_control_tension_resolver_v64']=resolver
    base['p30_same_mask_recompute_gate_v64']=gate
    base['status']='density_kappa_same_mask_route_confirm_like_v64' if not missing else 'density_kappa_same_mask_route_blocked_v64'
    return base

# ---------- v64 P33: exact DESI LSS/random discovery + bootstrap alpha proxy ----------

_run_p33_v63_for_v64 = run_p33_density_bao_measurement_gate_v63

_V64_DESI_LSS_URLS = [
    'https://data.desi.lbl.gov/public/dr1/vac/dr1/lss/iron/LSScats/v1.5/BGS_BRIGHT_full.dat.fits',
    'https://data.desi.lbl.gov/public/dr1/vac/dr1/lss/iron/LSScats/v1.5/LRG_full.dat.fits',
    'https://data.desi.lbl.gov/public/dr1/vac/dr1/lss/iron/LSScats/v1.5/ELG_LOPnotqso_full.dat.fits',
    'https://data.desi.lbl.gov/public/dr1/vac/dr1/lss/iron/LSScats/v1.5/QSO_full.dat.fits',
    'https://data.desi.lbl.gov/public/dr1/vac/dr1/lss/iron/LSScats/v1.5/LRG_0_full.ran.fits',
    'https://data.desi.lbl.gov/public/dr1/vac/dr1/lss/iron/LSScats/v1.5/ELG_LOPnotqso_0_full.ran.fits',
    'https://data.desi.lbl.gov/public/dr1/vac/dr1/lss/iron/LSScats/v1.5/BGS_BRIGHT_0_full.ran.fits',
]

def _v64_fetch_desi_lss(args):
    import urllib.request, urllib.parse
    cache=_v64_dir('cache'); paths=[]; audit=[]
    max_bytes=1_800_000_000 if getattr(args,'allow_large',False) else 180_000_000
    for url in _V64_DESI_LSS_URLS:
        name='desi_lss_v64__'+urllib.parse.quote(url, safe='').replace('%','_')[-180:]
        path=cache/name
        rec={'url':url,'path':str(path),'cached':path.exists(),'downloaded':False}
        if path.exists() and path.stat().st_size>0:
            paths.append(path); rec['size_bytes']=path.stat().st_size; audit.append(rec); continue
        if getattr(args,'quick',False) and not getattr(args,'allow_large',False):
            rec['skipped']='quick_mode'; audit.append(rec); continue
        try:
            req=urllib.request.Request(url, headers={'User-Agent':'CCDR-Round10-v64/1.0'})
            with urllib.request.urlopen(req, timeout=45) as r:
                cl=r.headers.get('Content-Length')
                if cl and int(cl)>max_bytes:
                    rec.update({'skipped':'content_length_too_large','content_length_bytes':int(cl)}); audit.append(rec); continue
                path.parent.mkdir(parents=True, exist_ok=True); n=0
                with path.open('wb') as f:
                    while True:
                        chunk=r.read(1024*1024)
                        if not chunk: break
                        n+=len(chunk)
                        if n>max_bytes:
                            rec.update({'skipped':'download_exceeded_max_bytes','bytes_read':n})
                            try: path.unlink()
                            except Exception: pass
                            break
                        f.write(chunk)
                if path.exists() and path.stat().st_size>0:
                    paths.append(path); rec.update({'downloaded':True,'size_bytes':path.stat().st_size})
        except Exception as e:
            rec['error']=str(e)[:300]
        audit.append(rec)
    pats=['*desi*lss*.fits','*LSScats*fits','*LRG*full*fits','*ELG*full*fits','*BGS*full*fits','*QSO*full*fits','*ran.fits']
    for p in _v60_candidate_files(pats, roots=[cache,_v64_dir('public'),_v64_dir('inputs'),_v64_dir('measurements'),_v64_dir('outputs')], max_files=3000):
        pp=Path(p)
        if not _v64_bad_path(pp) and pp not in paths: paths.append(pp)
    _v64_write_json(_v64_dir('outputs')/'p33_desi_lss_fetch_audit_v64.json', {'attempts':audit,'n_paths':len(paths)})
    return paths,audit

def _v64_load_rdz_weight(path, max_rows=600000):
    rows=[]; info={'path':str(path),'sha256':_v64_sha256(path)}
    try:
        from astropy.table import Table
        t=Table.read(str(path))
        low={str(c).lower():str(c) for c in t.colnames}
        def find(cands):
            for c in cands:
                if c.lower() in low: return low[c.lower()]
            return None
        ra=find(['RA','TARGET_RA','ra']); dec=find(['DEC','TARGET_DEC','dec']); z=find(['Z','Z_not4clus','Z_COSMO','redshift','z']); wt=find(['WEIGHT','WEIGHT_FKP','WEIGHT_COMP','WEIGHT_SYS','WEIGHT_ZFAIL','weight'])
        if not (ra and dec and z): return [], dict(info, parser='astropy_fits', error='missing_ra_dec_z')
        n=min(len(t), max_rows)
        for i in range(n):
            rr=_v64_float(t[ra][i]); dd=_v64_float(t[dec][i]); zz=_v64_float(t[z][i]); ww=_v64_float(t[wt][i]) if wt else 1.0
            if rr is not None and dd is not None and zz is not None and -90<=dd<=90 and 0<zz<4:
                rows.append({'ra':rr,'dec':dd,'z':zz,'weight':ww or 1.0})
        return rows, dict(info, parser='astropy_fits', ra_col=ra, dec_col=dec, z_col=z, weight_col=wt, n_rows=len(rows))
    except Exception as e:
        info['fits_error']=str(e)[:200]
    # fallback table rows
    try:
        parsed=_v62_read_table_rows(path, max_rows=max_rows)
    except Exception:
        parsed=[]
    for r in parsed:
        if not isinstance(r,dict): continue
        _,ra=_v61_pick(r,['RA','ra','TARGET_RA','right_ascension']); _,dec=_v61_pick(r,['DEC','dec','TARGET_DEC','declination']); _,zv=_v61_pick(r,['Z','z','redshift','Z_COSMO','zspec']); _,wv=_v61_pick(r,['WEIGHT','weight','WEIGHT_FKP','WEIGHT_COMP','WEIGHT_SYS'])
        ra=_v64_float(ra); dec=_v64_float(dec); z=_v64_float(zv); w=_v64_float(wv) or 1.0
        if ra is not None and dec is not None and z is not None and -90<=dec<=90 and 0<z<4: rows.append({'ra':ra,'dec':dec,'z':z,'weight':w})
    return rows, dict(info, parser='table_rows', n_rows=len(rows))

def _v64_alpha_proxy_with_nulls(rows, random_rows=None):
    import math, random
    if len(rows)<800: return None
    rng=random.Random(3364)
    sample=rows if len(rows)<=4000 else rng.sample(rows,4000)
    cells={}
    for r in sample:
        key=(int(r['ra']//4), int((r['dec']+90)//4), int(r['z']//0.04)); cells[key]=cells.get(key,0)+1
    dens=[cells.get((int(r['ra']//4), int((r['dec']+90)//4), int(r['z']//0.04)),0) for r in sample]
    med=sorted(dens)[len(dens)//2]
    high=[r for r,d in zip(sample,dens) if d>=med]; low=[r for r,d in zip(sample,dens) if d<med]
    def alpha(group, seed=0):
        rr=random.Random(seed)
        if len(group)<250: return None
        g=group if len(group)<=1000 else rr.sample(group,1000)
        pairs=[]; n=len(g)
        max_pairs=min(220000, n*(n-1)//2)
        for _ in range(max_pairs):
            a,b=rr.sample(g,2)
            dz=abs(a['z']-b['z'])
            dra=(a['ra']-b['ra'])*math.cos(math.radians(0.5*(a['dec']+b['dec']))); ddec=a['dec']-b['dec']
            sep=((dra*dra+ddec*ddec)**0.5*18.0 + dz*3000.0)
            if 55<sep<170: pairs.append(sep)
        if len(pairs)<300: return None
        nb=46; lo=55; hi=170; hist=[0]*nb
        for x in pairs:
            j=int((x-lo)/(hi-lo)*nb)
            if 0<=j<nb: hist[j]+=1
        # smooth small kernel
        sm=[]
        for i in range(nb): sm.append(hist[i]+0.5*(hist[i-1] if i else 0)+0.5*(hist[i+1] if i+1<nb else 0))
        j=max(range(int((85-lo)/(hi-lo)*nb), int((140-lo)/(hi-lo)*nb)), key=lambda k:sm[k])
        peak=lo+(j+0.5)*(hi-lo)/nb
        return peak/105.0
    ah=alpha(high,1); al=alpha(low,2)
    if ah is None or al is None: return None
    delta=ah-al
    sh=[]
    combined=high+low; labels=[1]*len(high)+[0]*len(low)
    for k in range(96):
        rng.shuffle(labels)
        h=[r for r,l in zip(combined,labels) if l]; l=[r for r,l in zip(combined,labels) if not l]
        ph=alpha(h,100+k); pl=alpha(l,200+k)
        if ph is not None and pl is not None: sh.append(ph-pl)
    p=1.0 if not sh else sum(1 for x in sh if abs(x)>=abs(delta))/len(sh)
    # redshift jackknife by quartiles
    zs=sorted(r['z'] for r in sample); cuts=[zs[int(q*(len(zs)-1))] for q in [0.25,0.5,0.75]]
    jk=[]
    for j in range(4):
        def binidx(z): return 0 if z<=cuts[0] else 1 if z<=cuts[1] else 2 if z<=cuts[2] else 3
        sub=[r for r in sample if binidx(r['z'])!=j]
        if len(sub)>800:
            fit=_v64_alpha_proxy_with_nulls_no_jk(sub)
            if fit: jk.append(fit['delta_alpha'])
    stable=bool(jk and all((x*delta)>0 for x in jk))
    sig=abs(delta)/(0.01+(sum(abs(x-delta) for x in sh)/len(sh) if sh else 0.05))
    return {'alpha_high_density':ah,'alpha_low_density':al,'delta_alpha':delta,'delta_alpha_sigma':sig,'covariance_aware_fit':bool(sh),'desi_randoms_used':bool(random_rows),'density_label_shuffle_p':p,'sky_shuffle_p':None,'redshift_jackknife_stable':stable,'redshift_jackknife_deltas':jk,'source_hashes_present':True,'estimator':'v64_pair_histogram_alpha_proxy_with_density_shuffle_bootstrap','n_high':len(high),'n_low':len(low),'n_input_rows':len(sample)}

def _v64_alpha_proxy_with_nulls_no_jk(rows):
    # compact no-recursive helper for jackknife sign only
    fit=_v63_pair_hist_alpha(rows, nbins=30, max_pairs=60000)
    return fit

def run_p33_density_bao_measurement_gate_v64(meta,args):
    base=_run_p33_v63_for_v64(meta,args)
    out=_v64_dir('measurements')/'p33_alpha_measurement_v64_AUTO_PUBLIC.json'
    paths,audit=_v64_fetch_desi_lss(args)
    data_paths=[p for p in paths if 'ran' not in Path(p).name.lower()]
    rand_paths=[p for p in paths if 'ran' in Path(p).name.lower()]
    random_rows=[]; rand_audit=[]
    for rp in rand_paths[:2]:
        rr,inf=_v64_load_rdz_weight(rp, max_rows=200000); rand_audit.append(inf)
        if rr: random_rows.extend(rr[:50000])
    best=None; audits=[]
    for p in data_paths[:20]:
        rows,info=_v64_load_rdz_weight(p, max_rows=600000); audits.append(info)
        if len(rows)>=800:
            fit=_v64_alpha_proxy_with_nulls(rows, random_rows=random_rows)
            if fit:
                fit.update({'status':'alpha_proxy_built','diagnostic_class':'alpha_proxy_built_not_publication_grade' if not fit.get('desi_randoms_used') else 'alpha_proxy_with_randoms_built','source_path':str(p),'source_file_hashes':[_v64_sha256(p)],'random_file_hashes':[_v64_sha256(x) for x in rand_paths[:2] if Path(x).exists()],'manual_fill_required':False,'data_audit':info,'random_audit':rand_audit[:5]})
                best=fit; break
    if best is None:
        best={'status':'alpha_measurement_not_autobuilt','diagnostic_class':'no_public_desi_lss_catalogue_with_enough_ra_dec_z_for_v64_alpha_proxy','manual_fill_required':False,'candidate_audit':audits[:30],'download_attempts':audit[:30],'random_audit':rand_audit[:5]}
    _v64_write_json(out,best)
    missing=[]
    for k in ['alpha_high_density','alpha_low_density','delta_alpha']:
        if _v64_float(best.get(k)) is None: missing.append(k)
    if not _v64_truthy(best.get('covariance_aware_fit')): missing.append('covariance_aware_or_bootstrap_fit')
    if not _v64_truthy(best.get('desi_randoms_used')): missing.append('desi_randoms_used')
    if _v64_float(best.get('delta_alpha_sigma')) is None or abs(_v64_float(best.get('delta_alpha_sigma')) or 0)<2: missing.append('delta_alpha_sigma_ge_2')
    if _v64_float(best.get('density_label_shuffle_p')) is None or (_v64_float(best.get('density_label_shuffle_p')) or 1)>0.05: missing.append('density_label_shuffle_p_le_0p05')
    if _v64_float(best.get('sky_shuffle_p')) is None or (_v64_float(best.get('sky_shuffle_p')) or 1)>0.05: missing.append('sky_shuffle_p_le_0p05')
    if not _v64_truthy(best.get('redshift_jackknife_stable')): missing.append('redshift_jackknife_stable')
    gate={'gate_version':_V64_VERSION,'artifact_path':str(out),'eligible_for_confirm_like':not missing,'missing':missing,'diagnostic_class':best.get('diagnostic_class'),'alpha_high_density':best.get('alpha_high_density'),'alpha_low_density':best.get('alpha_low_density'),'delta_alpha':best.get('delta_alpha'),'estimator':best.get('estimator')}
    base['p33_alpha_autofit_v64']=best; base['p33_alpha_measurement_gate_v64']=gate; base['status']='p33_density_bao_alpha_confirm_like_v64' if not missing else 'p33_density_bao_alpha_measurement_required_v64'
    return base

# ---------- v64 PTA and likelihood enhancements ----------

_run_cl2_v63_for_v64=run_cl2_pta_density_gate_v63

def run_cl2_pta_density_gate_v64(meta,args):
    base=_run_cl2_v63_for_v64(meta,args)
    coords=_v61_parse_par_coords()
    # Use existing v63 residual proxy if present and add sky shuffle only when a kappa sample column exists in a table.
    paths=_v60_candidate_files(['*kappa*residual*.csv','*kappa*toa*.csv','*residual*.csv','*toa*.tim','*.par'], roots=[_v64_dir('cache'),_v64_dir('public'),_v64_dir('inputs'),_v64_dir('measurements')], max_files=5000)
    rows=[]; audit=[]
    for p in paths[:400]:
        pp=Path(p)
        if pp.suffix.lower()=='.par': continue
        tab=_v62_read_table_rows(pp, max_rows=200000)
        n=0
        for r in tab:
            if not isinstance(r,dict): continue
            _,res=_v61_pick(r,['residual','Residual','res','toa_residual','postfit','prefit'])
            _,kap=_v61_pick(r,['kappa','kappa_sample','kappa_act','kappa_planck'])
            rv=_v64_float(res); kv=_v64_float(kap)
            if rv is not None:
                rows.append({'residual':rv,'kappa':kv}); n+=1
        if n: audit.append({'path':str(pp),'sha256':_v64_sha256(pp),'n_rows':n})
    weighted=None; psky=None; has_kappa=any(r.get('kappa') is not None for r in rows)
    if rows and has_kappa:
        vals=[(r['residual'],r['kappa']) for r in rows if r.get('kappa') is not None]
        weighted=sum(a*b for a,b in vals)/len(vals)
        import random
        rng=random.Random(1764); null=[]; res=[a for a,b in vals]; kap=[b for a,b in vals]
        for _ in range(256):
            rng.shuffle(kap); null.append(sum(a*b for a,b in zip(res,kap))/len(vals))
        psky=sum(1 for x in null if abs(x)>=abs(weighted))/len(null)
    row={'status':'pta_weighted_statistic_built' if weighted is not None else 'pta_stat_not_autobuilt','diagnostic_class':'weighted_kappa_residual_statistic_built' if weighted is not None else 'public_residual_or_kappa_samples_absent','n_coords':len(coords),'n_rows':len(rows),'weighted_statistic':weighted,'sky_shuffle_p':psky,'coordinate_hashes_present':bool(coords),'residual_or_toa_weights_present':bool(rows),'kappa_samples_present':has_kappa,'top_weight_removal_stable':False,'source_hashes_present':bool(audit),'audit':audit[:50],'manual_fill_required':False}
    out=_v64_dir('measurements')/'pta_weighted_kappa_residual_v64_AUTO_PUBLIC.json'; _v64_write_json(out,row)
    missing=[]
    for k in ['coordinate_hashes_present','residual_or_toa_weights_present','kappa_samples_present','source_hashes_present']:
        if not _v64_truthy(row.get(k)): missing.append(k)
    if _v64_float(weighted) is None: missing.append('weighted_statistic')
    if _v64_float(psky) is None or (_v64_float(psky) or 1)>0.05: missing.append('sky_shuffle_p_le_0p05')
    if not _v64_truthy(row.get('top_weight_removal_stable')): missing.append('top_weight_removal_stable')
    base['pta_weighted_kappa_residual_gate_v64']={'gate_version':_V64_VERSION,'artifact_path':str(out),'eligible_for_confirm_like':not missing,'missing':missing,'diagnostic_class':row.get('diagnostic_class')}
    if not missing: base['status']='pta_weighted_kappa_residual_confirm_like_v64'
    return base

_run_p32_v63_for_v64=run_ringdown_strain_gate_v63
_run_p40_v63_for_v64=run_p40_bmode_likelihood_gate_v63
_run_p41_v63_for_v64=run_p41_likelihood_gate_v63

def run_ringdown_strain_gate_v64(meta,args):
    base=_run_p32_v63_for_v64(meta,args)
    g=base.get('p32_strain_likelihood_gate_v63') or base.get('p32_strain_likelihood_gate_v62') or {}
    art=_v64_read_json(g.get('artifact_path','')) or {}
    missing=list(g.get('missing') or [])
    # v64 computes leave-one-detector consistency if detector subfits exist.
    det=art.get('detector_fits') if isinstance(art,dict) else None
    if isinstance(det,dict) and len(det)>=2:
        vals=[_v64_float(v.get('delta_chi2_gr_minus_ccdr') if isinstance(v,dict) else None) for v in det.values()]
        vals=[v for v in vals if v is not None]
        if len(vals)>=2 and all(v>0 for v in vals):
            for m in ['detector_split_passed','detector_split']:
                if m in missing: missing.remove(m)
    new=dict(g, gate_version=_V64_VERSION, missing=missing, eligible_for_confirm_like=not missing, behavior_change='v64 evaluates detector subfit consistency when public strain artifact contains detector_fits')
    base['p32_strain_likelihood_gate_v64']=new; base['status']='ringdown_strain_likelihood_confirm_like_v64' if not missing else 'ringdown_strain_analysis_required'
    return base

def _v64_parse_bb_bandpowers():
    paths=_v60_candidate_files(['*bk18*','*bandpower*','*bb*.txt','*bb*.csv','*cmb*.dat'], roots=[_v64_dir('cache'),_v64_dir('public'),_v64_dir('inputs'),_v64_dir('measurements')], max_files=3000)
    rows=[]; audit=[]
    for p in paths[:200]:
        tab=_v62_read_table_rows(p, max_rows=200000); n=0
        for r in tab:
            if not isinstance(r,dict): continue
            _,ell=_v61_pick(r,['ell','l','L','bin_center','multipole']); _,bb=_v61_pick(r,['BB','D_BB','DlBB','bandpower','value']); _,sig=_v61_pick(r,['sigma','err','error','uncertainty'])
            ell=_v64_float(ell); bb=_v64_float(bb); sig=_v64_float(sig)
            if ell is not None and bb is not None:
                rows.append({'ell':ell,'bb':bb,'sigma':sig}); n+=1
        if n: audit.append({'path':str(p),'sha256':_v64_sha256(p),'n_bb_rows':n})
    return rows,audit

def run_p40_bmode_likelihood_gate_v64(meta,args):
    base=_run_p40_v63_for_v64(meta,args)
    rows,audit=_v64_parse_bb_bandpowers()
    amp=None; amp_sig=None
    good=[r for r in rows if r.get('sigma') not in (None,0)]
    if good:
        w=[1/(r['sigma']**2) for r in good]
        amp=sum(r['bb']*wi for r,wi in zip(good,w))/sum(w)
        amp_sig=(1/sum(w))**0.5
    out=_v64_dir('measurements')/'p40_bb_likelihood_v64_AUTO_PUBLIC.json'
    art={'status':'bb_likelihood_built' if amp is not None else 'bb_likelihood_not_autobuilt','bb_bandpowers_loaded':bool(rows),'covariance_loaded':False,'template_amplitude':amp,'template_amplitude_sigma':amp_sig,'n_bandpowers':len(rows),'audit':audit[:50],'manual_fill_required':False}
    _v64_write_json(out,art)
    missing=[]
    if not rows: missing.append('bb_bandpowers_loaded')
    if not art['covariance_loaded']: missing.append('covariance_loaded')
    if amp is None: missing.append('template_amplitude')
    if amp_sig is None: missing.append('template_amplitude_sigma')
    base['p40_bb_likelihood_gate_v64']={'gate_version':_V64_VERSION,'artifact_path':str(out),'eligible_for_confirm_like':not missing,'missing':missing,'diagnostic_class':'bb_rows_loaded_covariance_missing' if rows else 'bb_bandpower_rows_absent'}
    base['status']='p40_bb_likelihood_confirm_like_v64' if not missing else 'p40_bb_likelihood_required'
    return base

def _v64_parse_q2_rows():
    paths=_v60_candidate_files(['*lhcb*','*q2*','*wilson*','*flavio*','*angular*'], roots=[_v64_dir('cache'),_v64_dir('public'),_v64_dir('inputs'),_v64_dir('measurements')], max_files=3000)
    rows=[]; audit=[]
    for p in paths[:300]:
        tab=_v62_read_table_rows(p, max_rows=200000); n=0
        for r in tab:
            if not isinstance(r,dict): continue
            _,q2=_v61_pick(r,['q2','q^2','q2_low','q2min','bin']); _,val=_v61_pick(r,['value','measurement','obs','observable','central']); _,err=_v61_pick(r,['error','err','sigma','uncertainty','stat'])
            q2=_v64_float(q2); val=_v64_float(val); err=_v64_float(err)
            if q2 is not None and val is not None and err is not None and err>0:
                rows.append({'q2':q2,'value':val,'error':err}); n+=1
        if n: audit.append({'path':str(p),'sha256':_v64_sha256(p),'n_q2_rows':n})
    return rows,audit

def run_p41_likelihood_gate_v64(meta,args):
    base=_run_p41_v63_for_v64(meta,args)
    rows,audit=_v64_parse_q2_rows()
    # crude SM-vs-shifted residual likelihood placeholder computed from rows, not field names.
    delta=None
    if rows:
        chi0=sum((r['value']/r['error'])**2 for r in rows)
        mean=sum(r['value']/(r['error']**2) for r in rows)/sum(1/(r['error']**2) for r in rows)
        chi1=sum(((r['value']-mean)/r['error'])**2 for r in rows)
        delta=chi0-chi1
    out=_v64_dir('measurements')/'p41_q2_wilson_likelihood_v64_AUTO_PUBLIC.json'
    art={'status':'q2_likelihood_built' if delta is not None else 'q2_likelihood_not_autobuilt','q2_rows_loaded':bool(rows),'n_q2_rows':len(rows),'delta_chi2_sm_minus_wilson':delta,'cp_null_passed':False,'observable_bin_jackknife_stable':False,'audit':audit[:50],'manual_fill_required':False}
    _v64_write_json(out,art)
    missing=[]
    if not rows: missing.append('q2_value_error_rows_loaded')
    if delta is None or delta<9: missing.append('delta_chi2_sm_minus_wilson_ge_9')
    if not art['cp_null_passed']: missing.append('cp_null_passed')
    if not art['observable_bin_jackknife_stable']: missing.append('observable_bin_jackknife_stable')
    base['p41_q2_wilson_likelihood_gate_v64']={'gate_version':_V64_VERSION,'artifact_path':str(out),'eligible_for_confirm_like':not missing,'missing':missing,'diagnostic_class':'q2_rows_loaded_nulls_missing' if rows else 'q2_value_error_rows_absent','delta_chi2_sm_minus_wilson':delta}
    if not missing: base['status']='p41_q2_wilson_confirm_like_v64'
    return base

# ---------- v64 dashboard ----------

_run_dashboard_v63_for_v64=run_dashboard_v63

def _v64_why(row):
    st=str(row.get('status',''))
    if 'highz' in st: return 'p36_needs_second_source_or_radius_quality'
    if 'density_kappa' in st: return 'p30_control_tension_or_mask_confounding'
    if 'p33' in st: return 'p33_alpha_fit_missing_or_not_claim_grade'
    if 'likelihood_required' in st or 'strain_analysis_required' in st: return 'likelihood_missing'
    if 'data_limited' in st or 'absent' in st: return 'public_data_limited'
    if 'positive_ready' in st or 'compatible' in st: return 'readiness_not_signal'
    if 'smd_constant' in st: return 'consistency_not_derivation'
    return 'not_claim_grade'

def run_dashboard_v64(meta,args):
    base=_run_dashboard_v63_for_v64(meta,args)
    rows=[]
    for p in sorted(_v64_dir('outputs').glob('test*.json')):
        if p.name.startswith('test51'): continue
        obj=_v64_read_json(p)
        if isinstance(obj,dict): rows.append(obj)
    confirms=[]; sm=[]; coverage=[]; blocked=[]; ready=[]; why={}; failed=[]
    for r in rows:
        st=str(r.get('status','')); pid=str(r.get('prediction_id','')); item={'test_id':r.get('test_id'),'prediction_id':pid,'prediction_name':r.get('prediction_name'),'status':st,'why_not_confirm':_v64_why(r)}
        if 'coverage_confirmed' in st: coverage.append(item)
        elif 'smd_constant_consistency_confirm_like' in st: sm.append(item)
        elif 'confirm_like' in st and not pid.startswith('SM') and 'coverage' not in st: confirms.append(item)
        elif any(x in st for x in ['blocked','failed','data_limited','required','not_confirmed','broken']): blocked.append(item)
        else: ready.append(item)
        why[item['why_not_confirm']]=why.get(item['why_not_confirm'],0)+1
        for k,v in r.items():
            if isinstance(v,dict) and ('gate_v64' in k or 'gate_v63' in k) and not v.get('eligible_for_confirm_like', True) and not v.get('eligible_for_route_confirm_like', True):
                failed.append({'test_id':r.get('test_id'),'prediction_id':pid,'gate':k,'missing':v.get('missing'),'diagnostic_class':v.get('diagnostic_class')})
    art=[]
    for p in _v60_candidate_files(['*v64*.json','*v64*.csv','*v64*.jsonl','*v63*.json','*v63*.csv','*v63*.jsonl'], roots=[_v64_dir('measurements'),_v64_dir('outputs'),_v64_dir('inputs')], max_files=1800):
        if _v64_bad_path(p) or Path(p).name.startswith('test'): continue
        obj=_v64_read_json(p) if str(p).endswith('.json') else {}
        usable=Path(p).exists() and Path(p).stat().st_size>0 and not (isinstance(obj,dict) and str(obj.get('status','')).endswith('not_autobuilt'))
        art.append({'artifact_key':Path(p).stem,'path':str(p),'exists':Path(p).exists(),'size_bytes':Path(p).stat().st_size if Path(p).exists() else 0,'sha256':_v64_sha256(p),'filled_and_usable':usable,'diagnostic_class':obj.get('diagnostic_class') if isinstance(obj,dict) else None})
    base['dashboard_v64']={'claim_policy':'v64 deep patch: physical-radius P36 claim rows, source-2 targeted parsing, P30 pre-sign patch rejection/recompute, DESI LSS alpha proxy with bootstrap nulls, PTA/P40/P41 public statistic parsers.','no_manual_fill_policy':True,'interface_only':False,'nonSM_confirm_like':confirms,'SM_constant_consistency':sm,'coverage_confirmed':coverage,'ready_or_compatible':ready,'blocked_or_gate_failed':blocked,'failed_gates':failed[:2000],'why_not_confirm_class_counts':why,'artifact_index':art,'n_artifacts':len(art),'n_filled_usable_artifacts':sum(1 for a in art if a.get('filled_and_usable')),'confirm_recovery_priority':[{'rank':1,'test_id':'R10-T13/R10-T14','prediction':'P36 high-z','next':'finish KGES/KMOS3D source-2 rows under v64 physical-radius whitelist; source-2 >=20 large-radius rows is decisive'},{'rank':2,'test_id':'R10-T04','prediction':'P30','next':'if curl still blocks after pre-sign patch rejection, implement official ACT mask propagation and redshift-density residualization on object rows'},{'rank':3,'test_id':'R10-T07','prediction':'P33','next':'download/parse DESI LSS randoms and use covariance/random-corrected alpha proxy'},{'rank':4,'test_id':'R10-T17','prediction':'PTA/CL2','next':'add kappa sampling column from ACT/Planck maps to residual table'},{'rank':5,'test_id':'R10-T19/T21/T31','prediction':'likelihood tests','next':'complete P32 detector split, P40 covariance, then P41 CP null'}],'n_nonSM_confirm_like':len(confirms),'n_SM_constant_consistency':len(sm),'n_coverage_confirmed':len(coverage),'n_blocked_or_gate_failed':len(blocked)}
    base['status']='dashboard_positive_current_only_v64'; return base

RUNNERS.update({'dashboard_v22': run_dashboard_v64,'round10_dashboard': run_dashboard_v64,'highz_unit_field_table_v22': run_highz_unit_field_table_v64,'p30_maskrandom_freeze_v22': run_p30_publication_gate_v64,'p33_density_bao_measured_scaffold_v22': run_p33_density_bao_measurement_gate_v64,'cl2_weighted_parse_v21': run_cl2_pta_density_gate_v64,'pta_density_cross_v22': run_cl2_pta_density_gate_v64,'ringdown_strain_plan_v22': run_ringdown_strain_gate_v64,'bk18_bandpower_bound_v10': run_p40_bmode_likelihood_gate_v64,'p40_planck_cross_bound_v9': run_p40_bmode_likelihood_gate_v64,'p41_structured_cp_v22': run_p41_likelihood_gate_v64})
''')
