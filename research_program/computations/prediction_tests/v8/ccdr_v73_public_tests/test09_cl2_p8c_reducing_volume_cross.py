#!/usr/bin/env python3
from __future__ import annotations

import io
import tarfile
from pathlib import Path

import numpy as np
import pandas as pd
from scipy import stats

from _common_public_data import (
    CACHE_DIR,
    add_source,
    build_argparser,
    density_proxy_at_targets,
    finalize_result,
    json_result_template,
    load_act_dr6_kappa_sampler,
    load_euclid_q1_sample,
    load_nanograv_archive,
    nanograv_path,
    ensure_package,
    query_sdss_galaxies,
    robust_pearson,
    robust_spearman,
)


def extract_pulsar_positions(tar_path: Path, max_pulsars: int = 7) -> pd.DataFrame:
    names = []
    ras = []
    decs = []
    seen = set()
    with tarfile.open(tar_path, "r:*") as tf:
        for member in tf.getmembers():
            lname = member.name.lower()
            if not any(lname.endswith(ext) for ext in [".par", ".tim", ".txt", ".dat"]):
                continue
            f = tf.extractfile(member)
            if f is None:
                continue
            text = io.TextIOWrapper(f, encoding="utf-8", errors="ignore").read()
            ra = dec = None
            elon = elat = None
            name = Path(member.name).stem
            for ln in text.splitlines():
                ls = ln.strip().split()
                if len(ls) < 2:
                    continue
                key = ls[0].upper()
                if key in {"PSRJ", "PSR", "PSRB"}:
                    name = ls[1].strip()
                elif key in {"RAJ", "RA"}:
                    ra = ls[1]
                elif key in {"DECJ", "DEC"}:
                    dec = ls[1]
                elif key in {"ELONG", "LAMBDA"}:
                    elon = ls[1]
                elif key in {"ELAT", "BETA"}:
                    elat = ls[1]
            try:
                if ra and dec:
                    ra_deg = hms_to_deg(ra)
                    dec_deg = dms_to_deg(dec)
                elif elon is not None and elat is not None:
                    ra_deg, dec_deg = ecliptic_to_icrs(float(elon), float(elat))
                else:
                    continue
                if name in seen:
                    continue
                seen.add(name)
                ras.append(ra_deg)
                decs.append(dec_deg)
                names.append(name)
            except Exception:
                continue
            if len(names) >= max_pulsars:
                break
    if not names:
        raise RuntimeError("No pulsar positions extracted from NANOGrav archive")
    return pd.DataFrame({"name": names, "ra": ras, "dec": decs})




def ecliptic_to_icrs(lon_deg: float, lat_deg: float) -> tuple[float, float]:
    ensure_package("astropy")
    from astropy import units as u
    from astropy.coordinates import BarycentricTrueEcliptic, SkyCoord

    c = SkyCoord(lon=lon_deg * u.deg, lat=lat_deg * u.deg, frame=BarycentricTrueEcliptic)
    icrs = c.icrs
    return float(icrs.ra.deg), float(icrs.dec.deg)


def hms_to_deg(s: str) -> float:
    parts = [float(x) for x in s.replace(':', ' ').split()[:3]]
    while len(parts) < 3:
        parts.append(0.0)
    h, m, sec = parts
    return 15.0 * (h + m / 60.0 + sec / 3600.0)


def dms_to_deg(s: str) -> float:
    sign = -1.0 if s.strip().startswith('-') else 1.0
    parts = [abs(float(x)) for x in s.replace(':', ' ').split()[:3]]
    while len(parts) < 3:
        parts.append(0.0)
    d, m, sec = parts
    return sign * (d + m / 60.0 + sec / 3600.0)


def main() -> None:
    parser = build_argparser("T9 — CL2 P8c reducing-volume cross")
    parser.add_argument("--max-galaxies", type=int, default=12000)
    parser.add_argument("--max-pulsars", type=int, default=30)
    args = parser.parse_args()

    result = json_result_template(
        "T9 — CL2 P8c reducing-volume cross",
        "Use public NANOGrav pulsar positions, a public SDSS galaxy sample, and a public ACT lensing proxy to test whether the PTA-proxy field tracks the same sign as the P30 density–κ field.",
    )

    ng_path = load_nanograv_archive()
    pulsars = extract_pulsar_positions(ng_path, max_pulsars=args.max_pulsars)

    try:
        gal = query_sdss_galaxies(args.max_galaxies, 0.05, 0.7, cache_key=f"sdss_gal_{args.max_galaxies}")
        gal_source = "SDSS DR17 SkyServer SQL"
        gal_url = "https://skyserver.sdss.org/dr17/SkyServerWS/SearchTools/SqlSearch"
    except Exception:
        # Fallback to Euclid Q1 positions so the CL2 proxy can still run on public data
        gal = load_euclid_q1_sample(max_rows=args.max_galaxies, seed=args.seed)
        gal_source = "Euclid Q1 public PHZ sample fallback"
        gal_url = "https://irsa.ipac.caltech.edu/data/Euclid/q1/"

    act = load_act_dr6_kappa_sampler()
    # Use the ACT footprint itself to define the kappa field; this avoids the previous
    # all-NaN / all-zero failure from averaging over a sparse random Euclid draw.
    p30_field = np.asarray(act.sample(pulsars["ra"], pulsars["dec"]), dtype=float)
    pta_field = density_proxy_at_targets(gal["ra"], gal["dec"], pulsars["ra"], pulsars["dec"], k=min(64, len(gal)))
    good = np.isfinite(p30_field) & np.isfinite(pta_field)
    pulsars = pulsars.loc[good].reset_index(drop=True)
    p30_field = p30_field[good]
    pta_field = pta_field[good]
    result["pulsars"] = pulsars.to_dict(orient="records")
    result["cross_correlation"] = {
        "pearson": robust_pearson(pta_field, p30_field),
        "spearman": robust_spearman(pta_field, p30_field),
        "same_sign_as_p30": bool(np.sign(np.nanmedian(pta_field)) == np.sign(np.nanmedian(p30_field))),
        "mean_pta_field": float(np.nanmean(pta_field)),
        "mean_kappa_field": float(np.nanmean(p30_field)),
        "n_used": int(len(p30_field)),
    }
    if len(p30_field) < 3:
        result["notes"].append("Too few pulsars landed inside the ACT footprint for a stable correlation estimate.")
    add_source(result, "NANOGrav 15-year archive", nanograv_path())
    add_source(result, gal_source, gal_url)
    add_source(result, "ACT DR6 lensing release", "https://lambda.gsfc.nasa.gov/data/suborbital/ACT/ACT_dr6/dr6_lensing_release.tar.gz")
    finalize_result(__file__, result, args.out)


if __name__ == "__main__":
    main()
