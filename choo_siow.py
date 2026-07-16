"""
Choo & Siow (2006), "Who Marries Whom and Why", JPE 114(1):175-201.

A static, transferable-utility marriage-matching model with McFadden (1974)
extreme-value taste shocks. This module implements the paper's two core objects:

1. estimate_gains()   -- the closed-form gains-to-marriage estimator.
2. solve_equilibrium() -- given structural gains + population supplies, solve the
                          market-clearing matching by IPFP (Galichon-Salanie).

Notation (paper's, eqs 4-12):
  types            i = 1..I men,  j = 1..J women
  mu_ij            number of (i,j) marriages
  mu_i0            number of UNMARRIED type-i men     (j = 0 outcome)
  mu_0j            number of UNMARRIED type-j women
  m_i = mu_i0 + sum_j mu_ij   available men of type i
  f_j = mu_0j + sum_i mu_ij   available women of type j

Man g of type i marrying type-j woman:  V_ijg = a_ij - t_ij + eps_ijg     (eq 4)
Woman of type j marrying type-i man:    receives transfer t_ij            (eq 9)
  a_ij = systematic gross return to the man,  t_ij = equilibrium transfer.

Quasi-demand (men, eq 7):    ln mu_ij = ln mu_i0 + (a_ij - a_i0) - t_ij
Quasi-supply (women, eq 9):  ln mu_ij = ln mu_0j + (g_ij - g_0j) + t_ij
Add them (transfer t_ij cancels) -> marriage matching function (eq 10-11):

        pi_ij = ln mu_ij - 0.5*ln mu_i0 - 0.5*ln mu_0j
              = (alpha_ij + gamma_ij) / 2       [total systematic gain / partner]

  where alpha_ij = a_ij - a_i0, gamma_ij = g_ij - g_0j are the gross gains vs single.
Equivalently the matching function:  mu_ij = sqrt(mu_i0 * mu_0j) * exp(pi_ij).

Net gains, still transfer-laden but identified (eq 12):
        n_ij = ln(mu_ij / mu_i0) = alpha_ij - t_ij     (man's net gain)
        N_ij = ln(mu_ij / mu_0j) = gamma_ij + t_ij     (woman's net gain)
        n_ij + N_ij = 2 * pi_ij.

Expected value of *participating* in the market (McFadden inclusive value, eq 8):
        q_i = ln(m_i / mu_i0)   (men)     Q_j = ln(f_j / mu_0j)  (women)
These are the natural type-level "market value": how much a type gains from being
in the market. Higher q => scarcer/more-in-demand type => fewer stay single.

IMPORTANT: pi_ij is STRUCTURAL (a property of preferences/technology) and does NOT
depend on the population vectors. That is what licenses counterfactuals: hold pi
fixed, change the supplies (m, f) -- e.g. a skewed sex ratio -- and re-solve the
equilibrium. Every type's outcome moves, because the whole market re-clears
(the "spillover" a scale-free reduced form cannot produce).
"""

import numpy as np


# --------------------------------------------------------------------------
# 1. ESTIMATION  (eqs 11-12) -- closed form, nonparametric in the matching dist.
# --------------------------------------------------------------------------
def estimate_gains(mu_ij, mu_i0, mu_0j):
    """Recover gains-to-marriage from observed matching counts.

    Parameters
    ----------
    mu_ij : (I, J) array of (i,j) marriage counts
    mu_i0 : (I,)  array of unmarried type-i men
    mu_0j : (J,)  array of unmarried type-j women

    Returns dict with:
      pi   (I,J) total systematic gain per partner   = (alpha+gamma)/2   (eq 11)
      n    (I,J) man's net gain   = alpha - transfer                     (eq 12)
      N    (I,J) woman's net gain = gamma + transfer                     (eq 12)
      q    (I,)  men's expected market payoff   ln(m_i/mu_i0)            (eq 8)
      Q    (J,)  women's expected market payoff ln(f_j/mu_0j)            (eq 8)
    """
    mu_ij = np.asarray(mu_ij, float)
    mu_i0 = np.asarray(mu_i0, float)
    mu_0j = np.asarray(mu_0j, float)

    L_ij = np.log(mu_ij)
    Li0  = np.log(mu_i0)[:, None]
    L0j  = np.log(mu_0j)[None, :]

    pi = L_ij - 0.5 * Li0 - 0.5 * L0j          # eq (11):  ln mu_ij - .5 ln mu_i0 - .5 ln mu_0j
    n  = L_ij - Li0                            # eq (12):  ln(mu_ij / mu_i0)
    N  = L_ij - L0j                            # eq (12):  ln(mu_ij / mu_0j)

    m_i = mu_i0 + mu_ij.sum(axis=1)
    f_j = mu_0j + mu_ij.sum(axis=0)
    q = np.log(m_i / mu_i0)                     # eq (8) market participation value, men
    Q = np.log(f_j / mu_0j)                     #                                    women
    return dict(pi=pi, n=n, N=N, q=q, Q=Q, m_i=m_i, f_j=f_j)


# --------------------------------------------------------------------------
# 2. EQUILIBRIUM  -- solve the matching for given gains + supplies, via IPFP.
# --------------------------------------------------------------------------
def solve_equilibrium(pi, m_i, f_j, tol=1e-12, max_iter=100000):
    """Market-clearing match given structural gains pi and supplies (m_i, f_j).

    Writes mu_ij = a_i * b_j * K_ij with a_i = sqrt(mu_i0), b_j = sqrt(mu_0j),
    K_ij = exp(pi_ij). The accounting constraints (paper eqs 1-2)
        m_i = a_i^2 + a_i * sum_j b_j K_ij
        f_j = b_j^2 + b_j * sum_i a_i K_ij
    are each a scalar quadratic in a_i (resp. b_j) given the other side, giving the
    Galichon-Salanie iterated-proportional-fitting update (positive root):
        a_i = (-S_i + sqrt(S_i^2 + 4 m_i)) / 2,   S_i = sum_j b_j K_ij
        b_j = (-T_j + sqrt(T_j^2 + 4 f_j)) / 2,   T_j = sum_i a_i K_ij
    """
    pi = np.asarray(pi, float)
    m_i = np.asarray(m_i, float)
    f_j = np.asarray(f_j, float)
    K = np.exp(pi)

    a = np.sqrt(m_i / 2.0)                      # init
    b = np.sqrt(f_j / 2.0)
    for it in range(max_iter):
        S = K @ b                               # S_i = sum_j K_ij b_j
        a_new = (-S + np.sqrt(S * S + 4.0 * m_i)) / 2.0
        T = K.T @ a_new                         # T_j = sum_i K_ij a_i
        b_new = (-T + np.sqrt(T * T + 4.0 * f_j)) / 2.0
        if max(np.max(np.abs(a_new - a)), np.max(np.abs(b_new - b))) < tol:
            a, b = a_new, b_new
            break
        a, b = a_new, b_new

    mu_i0 = a * a
    mu_0j = b * b
    mu_ij = (a[:, None] * b[None, :]) * K
    return dict(mu_ij=mu_ij, mu_i0=mu_i0, mu_0j=mu_0j, iters=it + 1)


# --------------------------------------------------------------------------
# 3. DEMONSTRATION -- age-typed market, recovery check, and a sex-ratio counterfactual.
# --------------------------------------------------------------------------
def _demo():
    rng_ages = np.arange(20, 50)               # men & women both typed by age 20..49
    A = len(rng_ages)

    # --- "true" structural gains: assortative in age with a small man-older gap ---
    # pi_ij large when ages are close (preferred gap ~ +2 for the man). log-gain.
    gap_pref = 2.0
    ai = rng_ages[:, None]
    aj = rng_ages[None, :]
    pi_true = -0.045 * (ai - aj - gap_pref) ** 2 - 1.2   # level shift sets overall marriage rate

    # --- population supplies: mild decline with age, sexes balanced at baseline ---
    base_pop = np.exp(-0.02 * (rng_ages - 20))
    m_i = 100.0 * base_pop
    f_j = 100.0 * base_pop

    # Baseline equilibrium = our "observed data"
    base = solve_equilibrium(pi_true, m_i, f_j)

    # --- Recovery check: estimate pi from the simulated counts, compare to truth ---
    est = estimate_gains(base['mu_ij'], base['mu_i0'], base['mu_0j'])
    max_err = np.max(np.abs(est['pi'] - pi_true))
    print(f"[recovery] max |pi_hat - pi_true| = {max_err:.2e}   "
          f"(should be ~0: the estimator inverts the equilibrium exactly)")
    print(f"[equilibrium] IPFP converged in {base['iters']} iters; "
          f"accounting residual = {_resid(base, m_i, f_j):.2e}")

    def single_rate(res):
        return res['mu_i0'] / m_i, res['mu_0j'] / f_j

    mS0, wS0 = single_rate(base)

    # --- Counterfactual: male-skewed market (15% fewer women), SAME structural gains ---
    f_scarce = 0.85 * f_j
    cf = solve_equilibrium(pi_true, m_i, f_scarce)
    mS1 = cf['mu_i0'] / m_i
    wS1 = cf['mu_0j'] / f_scarce
    qi0, Qj0 = est['q'], est['Q']
    cf_est = estimate_gains(cf['mu_ij'], cf['mu_i0'], cf['mu_0j'])
    qi1, Qj1 = cf_est['q'], cf_est['Q']

    print("\nSex-ratio counterfactual: women's supply cut 15% (male-skewed market),")
    print("structural gains pi held fixed. Share never-marrying, by age:\n")
    print(f"{'age':>4} | {'MEN single base':>15} {'-> skewed':>10} | "
          f"{'WOMEN single base':>17} {'-> skewed':>10}")
    print("-" * 70)
    for k, age in enumerate(rng_ages):
        if age % 5 == 0:
            print(f"{age:>4} | {mS0[k]:>15.3f} {mS1[k]:>10.3f} | "
                  f"{wS0[k]:>17.3f} {wS1[k]:>10.3f}")

    print("\nEvery MEN's-side cell got worse and every WOMEN's-side cell improved,")
    print("even for ages whose OWN numbers did not change -- that is the equilibrium")
    print("spillover. Aggregate market payoff (McFadden inclusive value, eq 8):")
    print(f"  men   : mean q  {qi0.mean():+.3f} -> {qi1.mean():+.3f}   "
          f"(dating-market value falls)")
    print(f"  women : mean Q  {Qj0.mean():+.3f} -> {Qj1.mean():+.3f}   "
          f"(dating-market value rises)")


def _resid(res, m_i, f_j):
    r1 = np.abs(res['mu_i0'] + res['mu_ij'].sum(1) - m_i).max()
    r2 = np.abs(res['mu_0j'] + res['mu_ij'].sum(0) - f_j).max()
    return max(r1, r2)


if __name__ == "__main__":
    _demo()
