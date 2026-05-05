"""Tier-B test metadata from uploaded CCDR v7.5 application-test table."""
from __future__ import annotations

TESTS = {
    "T26": {"name": "fusion ELM energy scaling", "predictions": ["FR3"], "prediction_names": ["ELM energy scaling E_ELM ≈ P_ped · V · (ΔP/P)^2"], "family": "fusion", "mode": "literature", "queries": ["JET DIII-D ASDEX Upgrade ELM energy pedestal pressure pedestal volume delta P P table", "edge localized mode energy pedestal pressure public data JET DIII-D AUG"]},
    "T27": {"name": "ELM helicity proxy", "predictions": ["FR6"], "prediction_names": ["ELM frequency proportional to separatrix magnetic helicity proxy |H_mag|"], "family": "fusion", "mode": "literature", "queries": ["RMP coil phasing ELM frequency DIII-D JET ASDEX Upgrade table", "resonant magnetic perturbation ELM suppression coil current phasing ELM frequency"]},
    "T28": {"name": "global H-mode confinement/KSS margin", "predictions": ["FR7", "FR10"], "prediction_names": ["M_KSS correlates with energy confinement time", "density plus curvature residual coupling in confinement scalings"], "family": "fusion", "mode": "literature", "queries": ["ITPA H-mode confinement database 14153 records OSF", "international tokamak physics activity H mode confinement database raw data"]},
    "T29": {"name": "stellarator vs tokamak edge transport", "predictions": ["FR8"], "prediction_names": ["stellarator edge transport closer to KSS than tokamak edge transport"], "family": "fusion", "mode": "literature", "queries": ["W7-X LHD stellarator edge transport diffusivity tokamak comparison table", "stellarator tokamak edge heat diffusivity public data"]},
    "T30": {"name": "fusion residual curvature coupling", "predictions": ["FR10"], "prediction_names": ["combined density plus shaping/curvature reduces confinement residuals by 10-20 percent"], "family": "fusion", "mode": "literature", "queries": ["tokamak confinement scaling shaping elongation triangularity density residual database", "ITER98y2 confinement scaling shape density residual public table"]},
    "T31": {"name": "cryogenic kappa CCDR vs Casimir", "predictions": ["MAT1", "MAT3"], "prediction_names": ["CCDR grain-boundary thermal conductivity modifier", "low-temperature boundary scattering differs from Casimir"], "family": "materials", "mode": "cmbs4_thermal_ccdr"},
    "T32": {"name": "low-T kappa exponent", "predictions": ["MAT3"], "prediction_names": ["CCDR low-temperature nanocrystalline κ exponent near T^1/2"], "family": "materials", "mode": "cmbs4_lowt_exponent"},
    "T33": {"name": "diamond/hBN thermal ceiling audit", "predictions": ["MAT6", "MAT7"], "prediction_names": ["acoustic-optical orthogonality maximizes thermal conductivity", "thermoelectric/grain-boundary optimization trend"], "family": "materials", "mode": "literature", "queries": ["isotopically pure diamond thermal conductivity hBN mass ratio acoustic optical gap table", "diamond boron nitride thermal conductivity isotope purity literature table"]},
    "T34": {"name": "Bi2Te3 ZT angle meta-analysis", "predictions": ["MAT4"], "prediction_names": ["hexagonal grain-boundary cos(6 theta) fingerprint and θ≈30° ZT improvement"], "family": "materials", "mode": "literature", "queries": ["Bi2Te3 grain boundary angle EBSD thermoelectric ZT table", "bismuth telluride nanocomposite grain boundary misorientation angle ZT supplementary"]},
    "T35": {"name": "Kibble-Zurek grain-size exponent", "predictions": ["MAT5", "MAT10"], "prediction_names": ["grain size scales with cooling/quench rate by Kibble-Zurek exponent", "KZ processing profile improves materials"], "family": "materials", "mode": "literature", "queries": ["grain size cooling rate additive manufacturing Kibble Zurek exponent dataset", "annealing cooling rate grain size public csv"]},
    "T36": {"name": "density-stratified grain scattering", "predictions": ["MAT11", "CL4"], "prediction_names": ["dislocation-density stratification changes grain-boundary scattering", "CL4 density-stratified texture analogue"], "family": "materials", "mode": "literature", "queries": ["friction stir welding dislocation density thermal conductivity hardness local EBSD dataset", "laser shock peening dislocation density hardness thermal conductivity public data"]},
    "T37": {"name": "auxetic phonon transport", "predictions": ["MAT12"], "prediction_names": ["auxetic/Weyl-inspired phonon transport κ enhancement near 12 percent"], "family": "materials", "mode": "literature", "queries": ["auxetic material thermal conductivity Poisson ratio dataset", "negative Poisson ratio phonon thermal transport experimental table"]},
    "T38": {"name": "skyrmion lifetime literature audit", "predictions": ["MAT8"], "prediction_names": ["room-temperature/topologically protected skyrmion lifetime exceeds 1e-6 s envelope"], "family": "materials", "mode": "literature", "queries": ["skyrmion lifetime temperature helicity material table", "room temperature skyrmion lifetime nanoseconds microseconds supplementary data"]},
    "T39": {"name": "moire phononic bandgap twist scaling", "predictions": ["MAT9", "QC10"], "prediction_names": ["moiré twist angle tunes phononic bandgap", "twist-angle tunable phononic qubit substrate"], "family": "materials_quantum", "mode": "literature", "queries": ["moire phononic crystal twist angle bandgap table", "twisted bilayer phononic bandgap twist angle supplementary data"]},
    "T40": {"name": "transmon phononic-substrate T1 audit", "predictions": ["QC5"], "prediction_names": ["phononic crystal substrate transmon T1 trajectory above 1 ms"], "family": "quantum", "mode": "literature", "queries": ["transmon phononic bandgap substrate T1 table", "superconducting qubit phononic crystal bandgap lifetime T1 data"]},
    "T41": {"name": "qubit T2 plateau meta-analysis", "predictions": ["QC11"], "prediction_names": ["ν_bulk noise floor creates asymptotic transmon T2 cap near 1 s"], "family": "quantum", "mode": "literature", "queries": ["superconducting qubit T2 best coherence time year transmon table", "IBM Google superconducting qubit T2 coherence time public data"]},
    "T42": {"name": "spin-qubit T2 in isotopically pure Si", "predictions": ["QC8"], "prediction_names": ["isotopically pure silicon spin qubits reach T2 > 10 s envelope"], "family": "quantum", "mode": "literature", "queries": ["isotopically enriched silicon spin qubit T2 seconds table", "Si-28 spin qubit coherence time T2 isotope fraction data"]},
    "T43": {"name": "DTC qubit error-per-cycle trend", "predictions": ["QC7"], "prediction_names": ["discrete time-crystal qubit error per cycle falls exponentially with N"], "family": "quantum", "mode": "literature", "queries": ["discrete time crystal qubit number cycles error per cycle table", "time crystal experiment qubits cycles fidelity supplementary data"]},
    "T44": {"name": "3D NAND area vs volume scaling", "predictions": ["EL1", "EL3"], "prediction_names": ["inter-layer area rather than volume controls scaling", "3D electronics vertical volume crossover"], "family": "electronics", "mode": "literature", "queries": ["3D NAND layer count die area capacity table Samsung Micron SK Hynix", "ISSCC 3D NAND layers die size capacity public table"]},
    "T45": {"name": "optical interconnect trend", "predictions": ["EL8"], "prediction_names": ["optical surface links improve energy/bit and bandwidth/mm versus electronic vias"], "family": "electronics", "mode": "literature", "queries": ["optical interconnect energy per bit bandwidth per mm advanced node table", "ITRS optical interconnect energy bit bandwidth density public data"]},
    "T46": {"name": "LDPC burst-channel benchmark", "predictions": ["EL6"], "prediction_names": ["CDT-like/random graph code improves burst-channel LDPC capacity proxy"], "family": "electronics", "mode": "ldpc_synthetic"},
    "T47": {"name": "neuromorphic graph energy audit", "predictions": ["EL7"], "prediction_names": ["graph topology metrics correlate with neuromorphic inference energy"], "family": "electronics", "mode": "literature", "queries": ["neuromorphic benchmark inference energy graph topology table", "Loihi TrueNorth SpiNNaker energy per inference benchmark dataset"]},
    "T48": {"name": "photovoltaic acoustic-optical proxy", "predictions": ["EN?"], "prediction_names": ["material symmetry/mass-contrast residual proxy for PV efficiency"], "family": "energy", "mode": "nrel_pv"},
    "T49": {"name": "battery/thermoelectric materials symmetry audit", "predictions": ["EN?", "MAT4", "MAT7"], "prediction_names": ["acoustic-optical gap proxy predicts high ZT/low thermal loss"], "family": "energy_materials", "mode": "literature", "queries": ["thermoelectric dataset band gap thermal conductivity ZT material symmetry", "battery thermoelectric materials thermal conductivity band gap public csv"]},
    "T50": {"name": "Casimir residual public-table audit", "predictions": ["SE2", "SE6"], "prediction_names": ["ν_bulk-like constant floor in precision Casimir residuals"], "family": "sensors", "mode": "literature", "queries": ["precision Casimir force residual table separation experiment", "Casimir force residuals public data torsion oscillator table"]},
    "T51": {"name": "optical-clock drift literature bound", "predictions": ["SE3", "AE3"], "prediction_names": ["long-baseline frequency-ratio drift bounds ν_bulk-like drift"], "family": "sensors", "mode": "literature", "queries": ["optical clock frequency ratio drift long baseline public data", "atomic clock comparison frequency ratio drift table year"]},
    "T52": {"name": "atom-interferometer noise-floor audit", "predictions": ["SE6", "SE7"], "prediction_names": ["atom interferometer residual noise floor near 1e-15 systematic target"], "family": "sensors", "mode": "literature", "queries": ["atom interferometer residual noise floor sensitivity table", "cold atom interferometer acceleration noise floor public data"]},
    "T53": {"name": "biological symmetry/protein-folding proxy", "predictions": ["BI?"], "prediction_names": ["high-symmetry biomolecular assemblies show anomalous stability/folding proxies"], "family": "biotech", "mode": "pdb_symmetry"},
    "T54": {"name": "photosynthetic coherence meta-analysis", "predictions": ["BI?"], "prediction_names": ["photosynthetic coherence lifetime tracks symmetry/excitonic coupling protection"], "family": "biotech", "mode": "literature", "queries": ["photosynthetic coherence lifetime symmetry excitonic coupling table", "FMO complex coherence lifetime temperature 2D spectroscopy data table"]},
    "T55": {"name": "radiation/spacecraft anomaly audit", "predictions": ["AE5"], "prediction_names": ["spacecraft residual acceleration bounds after radiothermal/solar controls"], "family": "aerospace", "mode": "literature", "queries": ["Pioneer 10 11 trajectory residual acceleration data public PDS", "Cassini radiometric residual acceleration public data anomaly"]},
    "T56": {"name": "solar-sail residual proxy", "predictions": ["AE4"], "prediction_names": ["solar-sail residual acceleration vs sail normal and galactic-DM-wind proxy"], "family": "aerospace", "mode": "literature", "queries": ["LightSail telemetry residual acceleration public data sail normal", "IKAROS solar sail trajectory telemetry acceleration residual dataset"]},
    "T57": {"name": "cosmic-ray cross-section enhancement", "predictions": ["AE1"], "prediction_names": [">1 TeV cosmic-ray residual at ~1e-6 level relative to standard spectra"], "family": "aerospace", "mode": "hepdata_cosmic"},
    "T58": {"name": "exoplanet/stellar chronometry null", "predictions": ["AE3 analogue"], "prediction_names": ["decade-scale timing drift after astrophysical controls"], "family": "aerospace", "mode": "nasa_exoplanet"},
    "T59": {"name": "public HEP anomaly ledger", "predictions": ["P9b", "P9e", "P9f"], "prediction_names": ["MET near EW threshold, Drell-Yan ~1 TeV, di-Higgs threshold audit"], "family": "hep", "mode": "hepdata_ledger"},
    "T60": {"name": "Koide sector-distance audit", "predictions": ["§6.3"], "prediction_names": ["sector-dependent Koide distance/scales persist with public PDG/FLAG values"], "family": "particle", "mode": "koide_public"},
}

VALUE_TERMS = {
    "fusion": [r"E[_ -]?ELM", r"pedestal", r"ELM frequency", r"RMP", r"confinement", r"tau[_ ]?E", r"diffusivity", r"triangularity", r"elongation"],
    "materials": [r"thermal conductivity", r"grain", r"misorientation", r"cooling rate", r"dislocation", r"skyrmion", r"lifetime", r"bandgap", r"Poisson"],
    "materials_quantum": [r"twist angle", r"bandgap", r"phononic", r"moir"],
    "quantum": [r"T1", r"T2", r"coherence", r"lifetime", r"qubit", r"cycles", r"fidelity", r"error"],
    "electronics": [r"layer", r"die area", r"capacity", r"energy per bit", r"bandwidth", r"inference energy"],
    "energy_materials": [r"ZT", r"thermal conductivity", r"band gap", r"Seebeck"],
    "sensors": [r"residual", r"drift", r"noise floor", r"frequency ratio", r"Casimir"],
    "biotech": [r"coherence lifetime", r"folding", r"symmetry", r"stability"],
    "aerospace": [r"residual acceleration", r"telemetry", r"trajectory", r"drift"],
}

def get_test(test_id: str):
    return TESTS[test_id.upper()]



# v2: named physical column requirements. Each test is an AND of groups; within
# each group, regex alternatives are OR. If no table satisfies these groups, the
# result must be data_limited, not partial.
STRICT_COLUMN_RULES = {
    "T26": [[r"E[_\s-]?ELM|W[_\s-]?ELM|ELM.*energy|energy.*ELM"], [r"P[_\s-]?ped|pedestal.*pressure|pressure.*pedestal"], [r"V[_\s-]?ped|pedestal.*volume|delta.*P|ΔP|dP/P|pressure.*drop"]],
    "T27": [[r"ELM.*freq|freq.*ELM|f[_\s-]?ELM"], [r"RMP|coil|phasing|n[_\s-]?=|current|I[_\s-]?coil|helicity|H[_\s-]?mag"]],
    "T28": [[r"tau[_\s-]?E|confinement.*time|energy.*confinement"], [r"density|n[_eip]?|temperature|T[_eip]?|transport|diffus|viscos|eta|χ|chi"]],
    "T29": [[r"diffus|transport|χ|chi|heat.*flux|thermal.*diff"], [r"stellarator|tokamak|W7|LHD|JET|DIII|AUG|device|machine"]],
    "T30": [[r"tau[_\s-]?E|confinement|residual|H[_\s-]?factor"], [r"elongation|triangularity|shaping|curvature|q95|kappa|delta"], [r"density|n[_eip]?"]],
    "T33": [[r"thermal.*conduct|kappa|κ"], [r"isotope|purity|mass|gap|phonon|diamond|hBN|BN|boron"]],
    "T34": [[r"ZT|zT|figure.*merit"], [r"angle|misorientation|EBSD|grain.*boundary|theta|θ"]],
    "T35": [[r"grain.*size|grain.*diam|cell.*size|domain.*size"], [r"cooling.*rate|quench|scan.*speed|solidification|dT/dt"]],
    "T36": [[r"dislocation|hardness|thermal.*conduct|kappa|κ"], [r"position|distance|local|region|density|strain"]],
    "T37": [[r"thermal.*conduct|kappa|κ"], [r"Poisson|auxetic|negative.*Poisson|anisotropy"]],
    "T38": [[r"lifetime|time|duration|decay"], [r"skyrmion|helicity|DMI|temperature|field"]],
    "T39": [[r"twist|angle|θ|theta"], [r"bandgap|gap|phononic|frequency"]],
    "T40": [[r"T1|lifetime|relaxation"], [r"phononic|bandgap|substrate|transmon|qubit"]],
    "T41": [[r"T2|coherence"], [r"year|date|channel|device|qubit|transmon"]],
    "T42": [[r"T2|coherence"], [r"Si|silicon|isotope|28Si|enrich|fraction"]],
    "T43": [[r"cycle|cycles|N|qubits|system.*size"], [r"error|fidelity|decay|lifetime"]],
    "T44": [[r"layer|layers"], [r"capacity|Gb|Tb|bit"], [r"die.*area|area|mm2|mm\^2"]],
    "T45": [[r"energy.*bit|pJ/bit|fJ/bit"], [r"bandwidth|Gb/s|Tb/s|mm"]],
    "T47": [[r"energy|power|J|mJ|uJ|µJ"], [r"inference|benchmark|accuracy|topology|graph"]],
    "T49": [[r"ZT|zT|thermal.*conduct|Seebeck"], [r"band.*gap|symmetry|space.*group|material"]],
    "T50": [[r"residual|force|pressure|gradient"], [r"Casimir|separation|distance"]],
    "T51": [[r"drift|frequency.*ratio|fractional"], [r"year|date|baseline|clock"]],
    "T52": [[r"noise|sensitivity|residual|Allan"], [r"atom|interferometer|acceleration|strain"]],
    "T54": [[r"coherence|lifetime|dephasing|oscillation"], [r"temperature|symmetry|complex|FMO|photosystem"]],
    "T55": [[r"residual.*accel|acceleration|doppler|range"], [r"Pioneer|Cassini|trajectory|radiothermal"]],
    "T56": [[r"residual|acceleration|delta.*v|trajectory"], [r"sail|normal|attitude|LightSail|IKAROS"]],
}

def strict_rules_for(test_id: str, family: str = ""):
    if test_id in STRICT_COLUMN_RULES:
        return STRICT_COLUMN_RULES[test_id]
    # Family fallback: still requires a physical named column, never text-window numerics.
    family_terms = VALUE_TERMS.get(family, [])
    if family_terms:
        return [family_terms]
    return []
