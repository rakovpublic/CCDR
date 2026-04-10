#!/usr/bin/env python3
"""
SMD5_koide.py
Verify Koide formula and compute quark extensions with QCD corrections.
"""
import numpy as np

# PDG 2024 masses (GeV) -- pole masses for leptons, MSbar for quarks
LEPTONS = {
    'e':   0.000510999,
    'mu':  0.105658,
    'tau': 1.77686
}

# Quark MSbar masses at mu = 2 GeV (light quarks) and pole masses (heavy)
QUARKS_MSBAR = {
    'u': 0.00216,   # +/- 0.00049
    'd': 0.00467,   # +/- 0.00048
    's': 0.0934,    # +/- 0.0086
    'c': 1.27,      # +/- 0.02 (at mu = m_c)
    'b': 4.18,      # +/- 0.03 (at mu = m_b)
    't': 172.69,    # +/- 0.30 (pole mass)
}

# Running masses at mu = M_Z = 91.1876 GeV (for uniform comparison)
QUARKS_MZ = {
    'u': 0.00127,
    'd': 0.00284,
    's': 0.0555,
    'c': 0.619,
    'b': 2.855,
    't': 171.7,   # near pole
}


def koide_Q(m1, m2, m3):
    """Compute the Koide ratio Q = (m1+m2+m3) / (sqrt(m1)+sqrt(m2)+sqrt(m3))^2."""
    numerator = m1 + m2 + m3
    denominator = (np.sqrt(m1) + np.sqrt(m2) + np.sqrt(m3))**2
    return numerator / denominator


def qcd_correction(alpha_s, n_loops=1):
    """QCD correction to Koide for quarks: dQ = c * alpha_s/pi."""
    if n_loops == 1:
        return (4.0 / 3.0) * alpha_s / np.pi  # leading colour factor CF
    return 0


def main():
    print("=" * 60)
    print("SM-D5: KOIDE FORMULA VERIFICATION")
    print("=" * 60)

    # 1. Charged leptons (the original Koide)
    m_e, m_mu, m_tau = LEPTONS['e'], LEPTONS['mu'], LEPTONS['tau']
    Q_lep = koide_Q(m_e, m_mu, m_tau)
    print(f"\nCHARGED LEPTONS:")
    print(f"  m_e  = {m_e:.9f} GeV")
    print(f"  m_mu = {m_mu:.6f} GeV")
    print(f"  m_tau = {m_tau:.5f} GeV")
    print(f"  Q = {Q_lep:.10f}")
    print(f"  2/3 = {2/3:.10f}")
    print(f"  |Q - 2/3| = {abs(Q_lep - 2/3):.2e}")
    print(f"  |Q - 2/3| / (2/3) = {abs(Q_lep - 2/3) / (2/3) * 100:.4f}%")
    if abs(Q_lep - 2/3) < 0.001:
        print(f"  [PASS] Koide Q = 2/3 confirmed to "
              f"{-np.log10(abs(Q_lep - 2/3)):.0f} significant figures")

    # 2. Up-type quarks (u, c, t)
    print(f"\nUP-TYPE QUARKS (MSbar at M_Z):")
    m_u, m_c, m_t = QUARKS_MZ['u'], QUARKS_MZ['c'], QUARKS_MZ['t']
    Q_up = koide_Q(m_u, m_c, m_t)
    alpha_s_MZ = 0.1179
    dQ_qcd = qcd_correction(alpha_s_MZ)
    Q_up_predicted = 2 / 3 + dQ_qcd
    print(f"  m_u = {m_u} GeV, m_c = {m_c} GeV, m_t = {m_t} GeV")
    print(f"  Q_up = {Q_up:.6f}")
    print(f"  Predicted: 2/3 + alpha_s CF/pi = {Q_up_predicted:.6f}")
    print(f"  Deviation from 2/3: {(Q_up - 2/3):.6f}")
    print(f"  QCD correction: {dQ_qcd:.6f}")

    # 3. Down-type quarks (d, s, b)
    print(f"\nDOWN-TYPE QUARKS (MSbar at M_Z):")
    m_d, m_s, m_b = QUARKS_MZ['d'], QUARKS_MZ['s'], QUARKS_MZ['b']
    Q_down = koide_Q(m_d, m_s, m_b)
    print(f"  m_d = {m_d} GeV, m_s = {m_s} GeV, m_b = {m_b} GeV")
    print(f"  Q_down = {Q_down:.6f}")
    print(f"  Deviation from 2/3: {(Q_down - 2/3):.6f}")

    # 4. Mixed triplets
    print(f"\nMIXED TRIPLETS:")
    triplets = [
        ('e', 'mu', 'tau', LEPTONS),
        ('u', 'c', 't', QUARKS_MZ),
        ('d', 's', 'b', QUARKS_MZ),
        ('e', 'u', 'd', {**LEPTONS, **QUARKS_MZ}),
        ('mu', 'c', 's', {**LEPTONS, **QUARKS_MZ}),
        ('tau', 't', 'b', {**LEPTONS, **QUARKS_MZ}),
    ]
    for n1, n2, n3, masses in triplets:
        Q = koide_Q(masses[n1], masses[n2], masses[n3])
        print(f"  ({n1}, {n2}, {n3}): Q = {Q:.6f}, delta = {Q - 2/3:+.6f}")

    # 5. Scale dependence: run quarks to different mu and check Q
    print(f"\nSCALE DEPENDENCE (Q_up at different mu):")
    # Running mass: m(mu) = m(M_Z) * [alpha_s(mu)/alpha_s(M_Z)]^(gamma0/beta0)
    # gamma0 = 8, beta0 = 23/3 for n_f = 5
    gamma0 = 8.0
    beta0 = 23.0 / 3.0
    mu_values = [2, 5, 10, 50, 91.2, 200, 1000]  # GeV
    for mu in mu_values:
        # Approximate alpha_s running
        alpha_s = 0.1179 / (1 + 0.1179 * beta0 / (2 * np.pi) * np.log(mu / 91.2))
        # Running mass ratio
        ratio = (alpha_s / 0.1179)**(gamma0 / (2 * beta0))
        m_u_run = QUARKS_MZ['u'] * ratio
        m_c_run = QUARKS_MZ['c'] * ratio
        m_t_run = QUARKS_MZ['t'] * ratio
        Q_run = koide_Q(m_u_run, m_c_run, m_t_run)
        dQ = qcd_correction(alpha_s)
        print(f"  mu = {mu:6.1f} GeV: alpha_s = {alpha_s:.4f}, "
              f"Q = {Q_run:.6f}, pred = {2/3 + dQ:.6f}")

    # 6. Test: does Q -> 2/3 at any scale?
    print(f"\nSEARCH FOR Q = 2/3 SCALE:")
    from scipy.optimize import brentq  # noqa: E402

    def Q_minus_twothirds(log_mu):
        mu = np.exp(log_mu)
        alpha_s = 0.1179 / (1 + 0.1179 * beta0 / (2 * np.pi) * np.log(mu / 91.2))
        if alpha_s <= 0 or alpha_s > 1:
            return 1.0
        ratio = (alpha_s / 0.1179)**(gamma0 / (2 * beta0))
        Q = koide_Q(QUARKS_MZ['u'] * ratio, QUARKS_MZ['c'] * ratio,
                     QUARKS_MZ['t'] * ratio)
        return Q - 2 / 3

    # Scan
    found = False
    for log_mu in np.linspace(0, 15, 100):
        val = Q_minus_twothirds(log_mu)
        if abs(val) < 0.001:
            mu_star = np.exp(log_mu)
            print(f"  Q ~ 2/3 at mu ~ {mu_star:.0f} GeV (hexagonal BZ scale?)")
            found = True
    if not found:
        print("  No scale found where Q_up = 2/3 exactly within scan range.")


if __name__ == '__main__':
    main()
