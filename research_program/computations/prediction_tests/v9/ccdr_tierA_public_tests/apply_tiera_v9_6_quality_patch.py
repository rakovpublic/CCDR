#!/usr/bin/env python3
"""Apply Tier-A v9.6 quality patch.

Run from the root of ccdr_tierA_public_tests:
    python apply_tiera_v9_6_quality_patch.py --apply

The patch is conservative:
- It writes tiera_quality_v96.py and run_tiera_v96_quality.py.
- It fixes the T03 _pantheon_columns NameError if the file exists.
- It writes extra automated source manifests/contracts.
- It does not delete existing tests.
"""
from __future__ import annotations
import argparse, re, shutil
from pathlib import Path


def resolve_root(arg_root: str, srcdir: Path) -> Path:
    """Resolve the Tier-A root even when the patcher is launched from tiera_v96_patch.

    Common safe cases:
    - run from ccdr_tierA_public_tests: root=.
    - run from v9/tiera_v96_patch: auto-select sibling v9/ccdr_tierA_public_tests
    - explicit --root path: use that path.
    """
    root = Path(arg_root).resolve()
    if (root / "tests").exists() and (root / "run_all_tierA.py").exists():
        return root
    if root == srcdir or root.name.lower() == "tiera_v96_patch":
        parent = root.parent
        sibling = parent / "ccdr_tierA_public_tests"
        if sibling.exists() and (sibling / "tests").exists():
            print(f"auto-detected Tier-A root: {sibling}")
            return sibling.resolve()
        if (parent / "tests").exists() and (parent / "run_all_tierA.py").exists():
            print(f"auto-detected Tier-A root: {parent}")
            return parent.resolve()
    return root


def copy_self(root: Path, srcdir: Path):
    for fn in ["tiera_quality_v96.py", "run_tiera_v96_quality.py", "apply_tiera_v9_6_quality_patch.py"]:
        src = (srcdir / fn).resolve()
        dst = (root / fn).resolve()
        if src == dst:
            print("skip same file", dst)
            continue
        try:
            if dst.exists() and src.samefile(dst):
                print("skip same file", dst)
                continue
        except Exception:
            pass
        shutil.copy2(src, dst)
        print("wrote", dst)


def patch_t03(root: Path):
    p = root / "tests" / "test03_pantheon_lowz_systematic_isolation.py"
    if not p.exists():
        print("skip T03 patch: file not found", p); return
    s = p.read_text(encoding="utf-8", errors="ignore")
    if "from tiera_quality_v96 import pantheon_columns_v96" not in s:
        insert = "\n# v9.6 patch: robust Pantheon column helper\nfrom tiera_quality_v96 import pantheon_columns_v96\n\n"
        # insert after first import block or at top
        m = list(re.finditer(r"^(?:import|from) .*$", s, flags=re.M))
        if m:
            pos = m[-1].end()
            s = s[:pos] + insert + s[pos:]
        else:
            s = insert + s
    # Define compatibility alias if function missing.
    if "_pantheon_columns = pantheon_columns_v96" not in s and "def _pantheon_columns" not in s:
        s = s.replace("from tiera_quality_v96 import pantheon_columns_v96", "from tiera_quality_v96 import pantheon_columns_v96\n_pantheon_columns = pantheon_columns_v96")
    p.write_text(s, encoding="utf-8")
    print("patched", p)


def write_contracts(root: Path):
    data = root / "data"; data.mkdir(exist_ok=True)
    extra = data / "tiera_v96_extra_sources.json"
    extra.write_text('''{
  "T04": ["https://lambda.gsfc.nasa.gov/product/act/actadv_dr6_lensing_maps_get.html", "https://pla.esac.esa.int/pla/#cosmology"],
  "T05": ["https://pla.esac.esa.int/pla/#cosmology"],
  "T08": ["https://cdsarc.cds.unistra.fr/viz-bin/cat/J/A+A/671/A48", "https://cdsarc.cds.unistra.fr/viz-bin/cat/J/MNRAS/502/2369"],
  "T15": ["https://data.nanograv.org/", "https://zenodo.org/api/records/?q=nanograv%2015-year%20posterior"],
  "T17": ["https://data.nanograv.org/", "https://zenodo.org/api/records/?q=stochastic%20gravitational%20wave%20spectral%20index%20posterior"],
  "T21": ["https://lambda.gsfc.nasa.gov/product/cobe/firas_products.html", "https://lambda.gsfc.nasa.gov/data/cobe/firas/"],
  "T23": ["https://hepdata.net/search/?q=BK18%20B-mode%20bandpower&format=json", "https://bicepkeck.org/"],
  "T24": ["https://www.gw-openscience.org/eventapi/html/GWTC/", "https://zenodo.org/api/records/?q=ringdown%20overtone%20posterior"],
  "T25": ["https://hepdata.net/search/?q=eta%2Fs%20Bayesian%20posterior%20heavy%20ion&format=json", "https://zenodo.org/api/records/?q=QGP%20eta%2Fs%20posterior"]
}
''', encoding="utf-8")
    print("wrote", extra)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--apply", action="store_true")
    ap.add_argument("--root", default=".")
    args = ap.parse_args()
    srcdir = Path(__file__).resolve().parent
    root = resolve_root(args.root, srcdir)
    if not args.apply:
        print("Dry run. Use --apply to patch", root); return
    if not (root / "tests").exists():
        raise SystemExit(f"Tier-A root not found or has no tests/ directory: {root}. Pass --root F:\\...\\ccdr_tierA_public_tests")
    copy_self(root, srcdir)
    patch_t03(root)
    write_contracts(root)
    changelog = root / "CHANGELOG_v9_6_quality_patch.md"
    changelog.write_text('''# Tier-A v9.6 quality patch

Implemented improvements:

1. Fixed T03 `_pantheon_columns` NameError by importing a robust Pantheon+/SH0ES column selector.
2. Added robust kappa map product classifier/sampler helpers: WCS maps, HEALPix maps, and ALM detection with optional healpy alm2map.
3. Added Euclid Q1 depth/quality proxy helper for T06/T07 residualization.
4. Added VizieR/CDS parser fallback for T08 filament catalogues.
5. Added supplemental BAO/SN likelihood-support helpers and posterior/chain readers for PTA/GW/ringdown tests.
6. Added automated source seeds for FIRAS/Planck/BK/HEPData/NANOGrav/GWOSC/eta-s posterior discovery.
7. Added strict artifact typing so metadata records are link sources, not physical evidence tables.
8. Added run_tiera_v96_quality.py supplemental diagnostics.

Run:

```powershell
python run_tiera_v96_quality.py --cache .cache --outdir out_v9_6_quality --allow-large
```
''', encoding="utf-8")
    print("wrote", changelog)

if __name__ == "__main__":
    main()
