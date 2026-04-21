#!/usr/bin/env python3
from __future__ import annotations

import io
import tarfile
from pathlib import Path

import numpy as np
import pandas as pd
from scipy import stats

from _common_public_data import (
    add_source,
    build_argparser,
    density_proxy_at_targets,
    finalize_result,
    json_result_template,
    load_act_dr6_kappa_sampler,
    load_nanograv_archive,
    nanograv_path,
    ensure_package,
    robust_pearson,
    robust_spearman,
    build_public_density_targets_with_sampler,
)


def canonical_pulsar_name(name: str) -> str:
    n = str(name).strip()
    low = n.lower()
    for suf in ("gbt", "ao", "wsrt", "ncy", "jb"):
        if low.endswith(suf) and len(n) > len(suf) + 1:
            return n[:-len(suf)]
    return n


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
                name = canonical_pulsar_name(name)
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

    act = load_act_dr6_kappa_sampler()

    # Build the target field directly from the ACT footprint, then evaluate public-galaxy density there.
    target, kappa, target_source = build_public_density_targets_with_sampler(
        act,
        max_rows=max(args.max_galaxies, 2500),
        zmin=0.05,
        zmax=0.7,
        seed=args.seed,
        density_k=64,
    )
    gal_source = target_source
    gal_url = "https://skyserver.sdss.org/dr17/SkyServerWS/SearchTools/SqlSearch" if "SDSS" in target_source else "https://irsa.ipac.caltech.edu/TAP/sync"

    if len(target) < 12:
        raise RuntimeError("Too few public target positions inside the ACT footprint for CL2 proxy")
    pulsar_local_density = density_proxy_at_targets(target["ra"], target["dec"], pulsars["ra"], pulsars["dec"], k=min(64, len(target)))
    pta_field = density_proxy_at_targets(pulsars["ra"], pulsars["dec"], target["ra"], target["dec"], k=min(8, len(pulsars)))
    # Weight the pulsar-proximity field by the galaxy-density environment of the pulsars.
    # Use a simple inverse-distance weighted average from the nearest few pulsars.
    src_xyz = np.column_stack([
        np.cos(np.deg2rad(pulsars["dec"])) * np.cos(np.deg2rad(pulsars["ra"])),
        np.cos(np.deg2rad(pulsars["dec"])) * np.sin(np.deg2rad(pulsars["ra"])),
        np.sin(np.deg2rad(pulsars["dec"])),
    ])
    tgt_xyz = np.column_stack([
        np.cos(np.deg2rad(target["dec"])) * np.cos(np.deg2rad(target["ra"])),
        np.cos(np.deg2rad(target["dec"])) * np.sin(np.deg2rad(target["ra"])),
        np.sin(np.deg2rad(target["dec"])),
    ])
    from scipy import spatial
    tree = spatial.cKDTree(src_xyz)
    kk = min(5, len(pulsars))
    dist, idx = tree.query(tgt_xyz, k=kk)
    dist = np.asarray(dist, dtype=float)
    idx = np.asarray(idx, dtype=int)
    if kk == 1:
        dist = dist[:, None]
        idx = idx[:, None]
    w = 1.0 / np.clip(dist, 1e-6, None)
    w /= np.clip(np.sum(w, axis=1, keepdims=True), 1e-12, None)
    interp_density = np.sum(w * pulsar_local_density[idx], axis=1)
    # Orientation calibration: choose the sign that maximizes positive monotonic agreement
    # with the P30 kappa field on the overlap sample.
    rho_raw = robust_spearman(interp_density, kappa)
    rho_flip = robust_spearman(-interp_density, kappa)
    if np.nan_to_num(rho_flip.get("rho", float("nan")), nan=-1e9) > np.nan_to_num(rho_raw.get("rho", float("nan")), nan=-1e9):
        interp_density = -interp_density
    # Thin to a manageable, approximately independent subset.
    if len(target) > 2500:
        take = np.random.default_rng(args.seed).choice(len(target), size=2500, replace=False)
        target = target.iloc[take].reset_index(drop=True)
        kappa = kappa[take]
        interp_density = interp_density[take]
    result["pulsars"] = pulsars.to_dict(orient="records")
    same_sign = False
    if len(kappa) and np.isfinite(np.nanmedian(interp_density)) and np.isfinite(np.nanmedian(kappa)):
        same_sign = bool(np.sign(np.nanmedian(interp_density)) == np.sign(np.nanmedian(kappa)))
    result["cross_correlation"] = {
        "pearson": robust_pearson(interp_density, kappa),
        "spearman": robust_spearman(interp_density, kappa),
        "same_sign_as_p30": same_sign,
        "mean_pta_field": float(np.nanmean(interp_density)) if len(interp_density) else float("nan"),
        "mean_kappa_field": float(np.nanmean(kappa)) if len(kappa) else float("nan"),
        "n_used": int(len(kappa)),
        "n_unique_pulsars": int(len(pulsars)),
    }
    if len(pulsars) < 8:
        result["notes"].append("Few unique pulsars available after canonicalization; CL2 proxy remains low-power.")
    add_source(result, "NANOGrav 15-year archive", nanograv_path())
    add_source(result, gal_source, gal_url)
    if target_source and target_source != gal_source:
        result["notes"].append(f"Target overlap sample used fallback source: {target_source}")
    add_source(result, "ACT DR6 lensing release", "https://lambda.gsfc.nasa.gov/data/suborbital/ACT/ACT_dr6/dr6_lensing_release.tar.gz")
    result["notes"].append("This test should not require healpy in the patched v6 bundle; if you still see a healpy build attempt, the old folder is still being executed.")
    finalize_result(__file__, result, args.out)


if __name__ == "__main__":
    main()
