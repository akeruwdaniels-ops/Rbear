"""
+=========================================================================+
|  DERIV RISE / FALL BOT  v3  —  Monte Carlo + Jump-Diffusion            |
|                                                                         |
|  Contract type : CALL (Rise) or PUT (Fall)                             |
|  Settlement    : final tick above / below entry spot                   |
|                                                                         |
|  Signal engine (v3 upgrades):                                          |
|    1. GBM with EWMA drift (µ)  — captures structural index bias        |
|    2. Merton Jump-Diffusion    — fat-tail + jump direction bias (µ_J)  |
|    3. AR(1) mean-reversion correction on T-step distribution           |
|    4. Regime gate: σ-expansion suppresses marginal signals             |
|    5. Asymmetric CALL scrutiny for bear-biased symbols (RDBEAR)        |
|                                                                         |
|  Duration selection:                                                    |
|    Runs MC for every candidate duration, picks the one with            |
|    the highest EV above the floor.  Falls back to "no trade"           |
|    when no duration clears the EV threshold.                           |
|                                                                         |
|  Kelly sizing — $1 account:                                            |
|    Fractional Kelly (25%) with tiered proportional floor:              |
|      $0.35–$1.99  →  35%  (guarantees Deriv $0.35 minimum)            |
|      $2.00–$4.99  →  12%                                               |
|      $5.00–$14.99 →   7%                                               |
|      $15.00+      →   5%                                               |
|    Hard cap: 10% of balance, $5.00 absolute max, $0.35 absolute min.   |
|                                                                         |
|  Connection layer:                                                      |
|    DerivWSManager — self-healing WS, exponential back-off,             |
|    per-object heartbeat, thread-safe safe_send.                        |
|    (Direct port from RDBEAR v10 — zero changes.)                       |
|                                                                         |
|  Requirements: pip install numpy websocket-client pandas               |
|  Env:          DERIV_API_TOKEN                                          |
+=========================================================================+
"""

import csv, enum, json, logging, math, os, random, sys, threading, time
import io
from collections import deque
from datetime import datetime

import numpy as np
import websocket
import pandas as pd


# ---------------------------------------------------------------------------
# Logging
# ---------------------------------------------------------------------------
_sh = logging.StreamHandler(
    io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8", errors="replace"))
_sh.setFormatter(logging.Formatter("%(asctime)s [%(levelname)s] %(message)s"))
_fh = logging.FileHandler("deriv_rise_fall_mc.log", encoding="utf-8")
_fh.setFormatter(logging.Formatter("%(asctime)s [%(levelname)s] %(message)s"))
logging.basicConfig(level=logging.INFO, handlers=[_sh, _fh])
log = logging.getLogger("DerivRF_MC")

DATA_DIR = os.environ.get("DATA_DIR", "rf_bot_data")
os.makedirs(DATA_DIR, exist_ok=True)


# ===========================================================================
# CONFIGURATION
# ===========================================================================
CONFIG = {
    # -- Deriv -----------------------------------------------------------------
    "app_id"    : 1089,
    "api_token" : os.environ.get("DERIV_API_TOKEN", ""),

    # -- Symbol ----------------------------------------------------------------
    # Any Deriv symbol that supports CALL/PUT, e.g.:
    #   "R_100"  (Volatility 100 Index)
    #   "R_75"   (Volatility 75 Index)
    #   "R_50"   (Volatility 50 Index)
    #   "1HZ100V" (Volatility 100 (1s) Index)
    "symbol"    : "RDBEAR",

    # -- Tick collection -------------------------------------------------------
    "collect_hours" : 0.6,          # ~36 min history
    "data_dir"      : os.path.join(DATA_DIR, "tick_data"),
    "min_ticks"     : 300,          # warmup ticks before trading

    # -- Candidate durations (ticks) -------------------------------------------
    # Rise/Fall contracts on synthetic indices typically use tick durations.
    # The bot will run MC for each and select the one with the highest EV.
    # Deriv supports 1–10 ticks for CALL/PUT on synthetic indices.
    "hold_durations" : [1, 2, 3, 4, 5, 6, 7, 8, 9, 10],

    # -- Currency --------------------------------------------------------------
    "currency"  : "USD",

    # -- Monte Carlo -----------------------------------------------------------
    "mc_n_paths"      : 3000,    # paths per simulation (std error ≈ ±0.009)
    "mc_vol_window"   : 60,      # ticks for EWMA σ
    "mc_ewma_alpha"   : 0.06,    # EWMA decay (RiskMetrics λ = 0.94)
    "mc_ticks_per_min": None,    # None = auto-detect

    # Direction probability floor before considering a trade.
    # Any duration where p_rise or p_fall exceeds this passes to
    # the EV ranking step.  Pre-screen uses a neutral 95% payout
    # estimate so ranking is meaningful; the proposal gate uses
    # the real Deriv payout for the final EV decision.
    "mc_p_floor"      : 0.51,

    # -- Jump-Diffusion (Merton) -----------------------------------------------
    "jd_jump_threshold" : 3.0,   # σ-multiples for jump detection
    "jd_fit_window"     : 300,   # ticks of history for jump fitting
    "jd_min_jumps"      : 5,     # minimum jumps needed to enable JD
    "jd_weight"         : 0.1,   # 0 = pure GBM until jumps are observed

    # -- Drift estimation ------------------------------------------------------
    # Rolling EWMA of signed log-returns, used as the µ term in MC paths.
    # This is the single most important fix for biased symbols like RDBEAR.
    # Window should be long enough to smooth noise but short enough to
    # track intraday drift changes. 120 ticks ≈ 2 min on RDBEAR.
    "drift_window"       : 120,   # ticks for EWMA µ estimation
    "drift_ewma_alpha"   : 0.03,  # slower decay than σ (λ=0.97) — drift is slower
    "drift_scale"        : 1.0,   # multiplier on estimated drift (1.0 = use as-is)

    # -- Asymmetric CALL scrutiny (bear-biased symbols) ----------------------
    # For symbols with known downward bias (RDBEAR, RDBULL on inverse side),
    # CALL signals require a higher p_win floor than PUT signals.
    # Set call_scrutiny_mult > 1.0 to apply the additional hurdle.
    # 1.0 = symmetric (no extra scrutiny). 1.08 = CALL needs 8% more edge.
    "call_scrutiny_mult" : 1.08,  # CALL p_floor = mc_p_floor * call_scrutiny_mult

    # -- AR(1) mean-reversion correction -------------------------------------
    # Synthetic indices exhibit short-term mean-reversion (negative lag-1
    # autocorrelation). Incorporating this tightens the T-step distribution,
    # reducing p_win overestimation at short durations.
    # Set to 0.0 to disable (pure iid GBM). Typical range: -0.15 to -0.05.
    "ar1_correction"     : True,  # estimate and apply AR(1) rho from tick buffer
    "ar1_window"         : 60,    # ticks used to estimate rho

    # -- Regime gate -----------------------------------------------------------
    # If the current EWMA σ is expanding faster than this threshold relative
    # to a slower baseline σ, the signal floor is raised automatically.
    # Prevents trading into spike/regime-change events.
    "regime_vol_ratio"   : 1.4,   # fast_σ / slow_σ > this → floor raised by regime_floor_bump
    "regime_floor_bump"  : 0.02,  # additive bump to mc_p_floor in expanding-vol regime
    "regime_fast_alpha"  : 0.06,  # EWMA α for fast σ (same as main)
    "regime_slow_alpha"  : 0.01,  # EWMA α for slow σ (λ=0.99 baseline)

    # -- Intelligence layer ------------------------------------------------
    # "Is this signal good?" gate. A duration only qualifies if p_win
    # clears mc_p_floor AND every active model (GBM, and JD when enough
    # jumps have been observed) agrees on direction.
    # No EV / payout-ratio gating — purely signal-quality based.

    # -- Signal persistence ----------------------------------------------------
    # 1 = trade on first qualifying tick (fastest response).
    # Raise to 2-3 once the bot is confirmed trading correctly.
    "signal_persistence_ticks" : 2,

    # -- Post-trade cooldown ---------------------------------------------------
    "min_ticks_between_trades" : 50,

    # -- Kelly staking — $1 account -------------------------------------------
    "kelly_fraction"  : 0.25,   # fractional Kelly (25%)
    "kelly_max_pct"   : 0.10,   # hard cap: 10% of balance per trade
    "kelly_min_stake" : 0.35,   # Deriv minimum
    "kelly_max_stake" : 5.00,   # absolute max per trade

    # -- Risk limits -----------------------------------------------------------
    "max_daily_loss_pct"         : 0.80,   # stop if session P&L ≤ -80% start bal
    "take_profit_pct"            : 9999.0, # disabled by default
    "max_drawdown_from_peak_pct" : 0.80,

    # -- Consecutive-loss cooldown ---------------------------------------------
    "max_consec_losses"          : 5,
    "consec_loss_cooldown_ticks" : 60,

    # -- SPRT ------------------------------------------------------------------
    "sprt_p0"    : 0.50,
    "sprt_p1"    : 0.54,
    "sprt_alpha" : 0.10,
    "sprt_beta"  : 0.20,

    # -- Trade log -------------------------------------------------------------
    "trade_log" : os.path.join(DATA_DIR, "trade_log_rf.csv"),
}

os.makedirs(CONFIG["data_dir"], exist_ok=True)


# ===========================================================================
# UTILITIES
# ===========================================================================

def wilson_ci(wins, n, z=1.96):
    if n == 0: return 0.0, 1.0
    p      = wins / n
    denom  = 1 + z**2 / n
    centre = (p + z**2 / (2*n)) / denom
    margin = z * math.sqrt(p*(1-p)/n + z**2/(4*n**2)) / denom
    return max(0.0, centre - margin), min(1.0, centre + margin)


# ===========================================================================
# SPRT MONITOR
# ===========================================================================

class SPRTMonitor:
    """Sequential Probability Ratio Test — passive edge tracker."""
    def __init__(self, p0=0.50, p1=0.54, alpha=0.10, beta=0.20):
        self.A      = math.log((1 - beta) / alpha)
        self.B      = math.log(beta / (1 - alpha))
        self.p0     = p0; self.p1 = p1
        self.llr    = 0.0; self.n = 0; self.wins = 0
        self.status = "CONTINUE"

    def update(self, win: bool) -> str:
        self.n += 1
        if win:
            self.wins += 1
            self.llr  += math.log(self.p1 / self.p0)
        else:
            self.llr  += math.log((1 - self.p1) / (1 - self.p0))
        if   self.llr >= self.A: self.status = "ACCEPT_H1"; self.llr = 0.0
        elif self.llr <= self.B: self.status = "ACCEPT_H0"; self.llr = 0.0
        else:                    self.status = "CONTINUE"
        return self.status

    def summary(self):
        wr     = self.wins / self.n if self.n else 0.0
        lo, hi = wilson_ci(self.wins, self.n)
        return (f"{self.status}  n={self.n}  WR={wr:.3f}  "
                f"CI=[{lo:.3f},{hi:.3f}]")


# ===========================================================================
# JUMP PARAMS CONTAINER
# ===========================================================================

class JumpParams:
    __slots__ = ("lam", "mu_j", "sigma_j", "n_jumps", "n_obs")
    def __init__(self, lam=0.0, mu_j=0.0, sigma_j=0.001, n_jumps=0, n_obs=0):
        self.lam     = lam
        self.mu_j    = mu_j
        self.sigma_j = sigma_j
        self.n_jumps = n_jumps
        self.n_obs   = n_obs

    def __repr__(self):
        return (f"JumpParams(λ={self.lam:.5f}/tick "
                f"μ_J={self.mu_j:+.5f} σ_J={self.sigma_j:.5f} "
                f"n_jumps={self.n_jumps}/{self.n_obs})")


# ===========================================================================
# MONTE CARLO PRICER  (Direction — Rise or Fall)
# ===========================================================================

class MonteCarloPricer:
    """
    Estimates the probability that the final tick of a Rise/Fall contract
    ends strictly above (Rise) or at/below (Fall) the entry spot.

    v3 models (all blended):
      GBM+µ : dlog S = µ·dt + σ·Z  — with EWMA drift term
      Merton: dlog S = µ·dt + σ_d·Z + Σ Y_k  — JD with drift + µ_J bias
      AR(1) correction applied to T-step variance: Var(S_T) adjusted for
        autocorrelation structure (mean-reversion tightens distribution).

    Returns (p_rise, p_fall, sigma, drift) where:
        p_rise + p_fall ≈ 1.0  (small MC noise)
    """

    def __init__(self, cfg):
        self.n_paths        = cfg.get("mc_n_paths",        3000)
        self.vol_win        = cfg.get("mc_vol_window",      60)
        self.alpha          = cfg.get("mc_ewma_alpha",      0.06)
        self.p_floor        = cfg.get("mc_p_floor",         0.51)
        self._tpm_cfg       = cfg.get("mc_ticks_per_min",   None)
        self.jump_threshold = cfg.get("jd_jump_threshold",  3.0)
        self.jd_fit_window  = cfg.get("jd_fit_window",      300)
        self.jd_min_jumps   = cfg.get("jd_min_jumps",       5)
        self.jd_weight      = cfg.get("jd_weight",          0.5)

        # v3: drift
        self.drift_win      = cfg.get("drift_window",       120)
        self.drift_alpha    = cfg.get("drift_ewma_alpha",   0.03)
        self.drift_scale    = cfg.get("drift_scale",        1.0)

        # v3: AR(1) correction
        self.ar1_enabled    = cfg.get("ar1_correction",     True)
        self.ar1_window     = cfg.get("ar1_window",         60)

        # v3: regime gate
        self.regime_ratio   = cfg.get("regime_vol_ratio",   1.4)
        self.regime_bump    = cfg.get("regime_floor_bump",  0.02)
        self.regime_fast_a  = cfg.get("regime_fast_alpha",  0.06)
        self.regime_slow_a  = cfg.get("regime_slow_alpha",  0.01)

        # v3: asymmetric CALL scrutiny
        self.call_scrutiny  = cfg.get("call_scrutiny_mult", 1.0)

        self._rng           = np.random.default_rng()
        self.last_jump_params: JumpParams = JumpParams()
        self.last_components: dict = {}
        self.last_drift: float = 0.0
        self.last_rho:   float = 0.0
        self.last_regime_bump: float = 0.0

    # ── σ estimation ──────────────────────────────────────────────────────

    def ewma_sigma(self, tick_buf) -> float:
        buf    = list(tick_buf)[-(self.vol_win + 1):]
        if len(buf) < 3: return 0.001
        prices = np.array([t["price"] for t in buf], dtype=float)
        lr     = np.diff(np.log(np.maximum(prices, 1e-8)))
        if len(lr) < 2: return max(float(np.std(lr)), 1e-8)
        var = float(lr[0] ** 2)
        for r in lr[1:]:
            var = self.alpha * float(r**2) + (1.0 - self.alpha) * var
        return max(math.sqrt(var), 1e-8)

    # ── Drift (µ) estimation ──────────────────────────────────────────────

    def ewma_drift(self, tick_buf) -> float:
        """
        EWMA of signed log-returns — estimates the per-tick drift µ.

        On a bear-biased symbol like RDBEAR the rolling mean log-return
        is negative, giving the MC paths a downward tilt that correctly
        suppresses CALL signals.

        Uses a slower decay (drift_ewma_alpha < mc_ewma_alpha) because
        drift is a lower-frequency phenomenon than volatility.
        """
        buf = list(tick_buf)[-(self.drift_win + 1):]
        if len(buf) < 5:
            return 0.0
        prices = np.array([t["price"] for t in buf], dtype=float)
        lr     = np.diff(np.log(np.maximum(prices, 1e-8)))
        mu = float(lr[0])
        for r in lr[1:]:
            mu = self.drift_alpha * float(r) + (1.0 - self.drift_alpha) * mu
        return float(mu) * self.drift_scale

    # ── AR(1) autocorrelation estimation ─────────────────────────────────

    def estimate_ar1(self, tick_buf) -> float:
        """
        Estimates lag-1 autocorrelation (rho) of log-returns.

        Synthetic Deriv indices exhibit short-term mean-reversion
        (rho typically -0.15 to -0.05). A negative rho means the
        T-step variance is LESS than T * σ² (returns partially cancel).

        The corrected T-step std for iid-corrected GBM is:
            σ_T = σ * sqrt(T * (1 + 2*rho*(1 - rho^T)/(1 - rho) / T))
        For small T and negative rho this is meaningfully smaller than
        σ*sqrt(T), reducing overconfident p_win estimates at 1-5 ticks.

        Returns rho clamped to [-0.5, 0.0] (we only apply the
        mean-reversion side; positive autocorrelation is left to GBM).
        """
        if not self.ar1_enabled:
            return 0.0
        buf = list(tick_buf)[-(self.ar1_window + 1):]
        if len(buf) < 10:
            return 0.0
        prices = np.array([t["price"] for t in buf], dtype=float)
        lr     = np.diff(np.log(np.maximum(prices, 1e-8)))
        if len(lr) < 4:
            return 0.0
        lr_dm  = lr - lr.mean()
        cov0   = float(np.dot(lr_dm, lr_dm)) / len(lr_dm)
        cov1   = float(np.dot(lr_dm[:-1], lr_dm[1:])) / (len(lr_dm) - 1)
        rho    = cov1 / max(cov0, 1e-12)
        # Only apply mean-reversion correction (rho < 0); ignore momentum
        return float(np.clip(rho, -0.5, 0.0))

    @staticmethod
    def ar1_sigma_correction(sigma: float, T: int, rho: float) -> float:
        """
        Returns the AR(1)-corrected T-step standard deviation.
        For rho=0 this reduces exactly to sigma*sqrt(T).
        """
        if T <= 1 or rho == 0.0:
            return sigma * math.sqrt(T)
        # Variance of sum of T AR(1) variables:
        # Var(S_T) = T*σ² * [1 + 2*rho/(1-rho) * (1 - rho^T/T) / (1 - rho^T ... )]
        # Simplified exact formula:
        #   V = T + 2 * rho * (T - 1) + 2 * rho^2 * (T - 2) + ...
        #     = sum_{k=0}^{T-1} (T - k) * rho^k * 2  (for k>0)
        V = float(T)
        rho_k = rho
        for k in range(1, T):
            V += 2.0 * (T - k) * rho_k
            rho_k *= rho
        V = max(V, 0.01)
        return sigma * math.sqrt(V / T) * math.sqrt(T)
        # = sigma * sqrt(V)

    # ── Regime detection ──────────────────────────────────────────────────

    def regime_floor_bump(self, tick_buf) -> float:
        """
        Returns an additive p_floor bump when volatility is expanding.

        Fast EWMA σ / slow EWMA σ > regime_ratio → regime is expanding.
        In expanding-vol regimes, MC path distributions are less reliable
        (σ is changing during the forecast horizon), so we raise the bar.

        Returns 0.0 in normal regime, regime_bump otherwise.
        """
        buf = list(tick_buf)[-(self.vol_win + 1):]
        if len(buf) < 5:
            return 0.0
        prices = np.array([t["price"] for t in buf], dtype=float)
        lr     = np.diff(np.log(np.maximum(prices, 1e-8)))
        if len(lr) < 3:
            return 0.0
        # Fast variance
        var_f = float(lr[0]**2)
        # Slow variance
        var_s = float(lr[0]**2)
        for r in lr[1:]:
            r2    = float(r**2)
            var_f = self.regime_fast_a * r2 + (1.0 - self.regime_fast_a) * var_f
            var_s = self.regime_slow_a * r2 + (1.0 - self.regime_slow_a) * var_s
        sigma_f = math.sqrt(max(var_f, 1e-12))
        sigma_s = math.sqrt(max(var_s, 1e-12))
        ratio   = sigma_f / max(sigma_s, 1e-12)
        bump    = self.regime_bump if ratio > self.regime_ratio else 0.0
        self.last_regime_bump = bump
        return bump

    # ── ticks-per-minute detection ────────────────────────────────────────

    @staticmethod
    def detect_tpm(tick_buf, window=30) -> float:
        buf = list(tick_buf)[-window:]
        if len(buf) < 2: return 60.0
        dt = buf[-1]["timestamp"] - buf[0]["timestamp"]
        return (len(buf) - 1) / dt * 60.0 if dt > 0 else 60.0

    # ── Jump parameter estimation ─────────────────────────────────────────

    def fit_jumps(self, tick_buf, sigma_ewma: float) -> JumpParams:
        buf = list(tick_buf)[-(self.jd_fit_window + 1):]
        if len(buf) < 10: return JumpParams()
        prices     = np.array([t["price"] for t in buf], dtype=float)
        lr         = np.diff(np.log(np.maximum(prices, 1e-8)))
        n_obs      = len(lr)
        threshold  = self.jump_threshold * sigma_ewma
        jump_mask  = np.abs(lr) > threshold
        n_jumps    = int(jump_mask.sum())
        if n_jumps == 0:
            return JumpParams(lam=0.0, mu_j=0.0,
                              sigma_j=sigma_ewma * self.jump_threshold,
                              n_jumps=0, n_obs=n_obs)
        jump_returns = lr[jump_mask]
        lam          = n_jumps / n_obs
        mu_j         = float(np.mean(jump_returns))
        sigma_j      = (float(np.std(jump_returns, ddof=1))
                        if n_jumps > 1 else abs(mu_j) * 0.5 + 1e-8)
        return JumpParams(lam=lam, mu_j=mu_j, sigma_j=sigma_j,
                          n_jumps=n_jumps, n_obs=n_obs)

    # ── Core simulation: direction probability ────────────────────────────

    def simulate_direction(self, tick_buf, T: int,
                           sigma: float) -> tuple[float, float]:
        """
        Simulate T ticks forward. Returns (p_rise, p_fall).

        v3 improvements vs v2:
          - Drift term µ from EWMA of signed log-returns (critical for
            biased symbols; previously omitted entirely).
          - AR(1) mean-reversion correction on T-step std (tightens
            distribution at short durations, reduces overconfident p_win).
          - Jump direction bias: µ_J contributes to the expected final
            log-return of the JD model, so asymmetric jumps shift p_rise/fall.
          - All of the above feed into the same GBM/JD blend as before.
        """
        if not tick_buf:
            return 0.5, 0.5

        N = self.n_paths

        # ── Estimate drift and AR(1) rho ──────────────────────────────────
        mu  = self.ewma_drift(tick_buf)          # per-tick drift (signed)
        rho = self.estimate_ar1(tick_buf)        # lag-1 autocorr (≤ 0)
        self.last_drift = mu
        self.last_rho   = rho

        # AR(1)-corrected T-step std (smaller than σ√T when rho < 0)
        sigma_T = self.ar1_sigma_correction(sigma, T, rho)
        # T-step drift contribution (deterministic shift to all paths)
        mu_T    = mu * T

        # ── GBM + drift ───────────────────────────────────────────────────
        # dlog S_t = µ + σ·Z  (per tick)
        # Final log-return: N(µ·T, σ_T²)  where σ_T has AR(1) correction
        gbm_final   = mu_T + self._rng.standard_normal(N) * sigma_T
        p_rise_gbm  = float((gbm_final > 0).mean())
        p_fall_gbm  = float((gbm_final < 0).mean())

        # ── Jump-Diffusion + drift ─────────────────────────────────────────
        jp = self.fit_jumps(tick_buf, sigma)
        self.last_jump_params = jp
        sufficient = jp.n_jumps >= self.jd_min_jumps
        ew = self.jd_weight if sufficient else 0.0

        if ew > 0.0:
            # Diffusion-only σ (EWMA recomputed excluding jump ticks)
            buf     = list(tick_buf)[-(self.jd_fit_window + 1):]
            prices  = np.array([t["price"] for t in buf], dtype=float)
            lr_all  = np.diff(np.log(np.maximum(prices, 1e-8)))
            thr     = self.jump_threshold * sigma
            diff_lr = lr_all[np.abs(lr_all) <= thr]
            if len(diff_lr) >= 2:
                var_d = float(diff_lr[0]**2)
                for r in diff_lr[1:]:
                    var_d = self.alpha * float(r**2) + (1.0 - self.alpha) * var_d
                sigma_d = max(math.sqrt(var_d), 1e-8)
            else:
                sigma_d = sigma

            # AR(1)-corrected diffusion component
            sigma_d_T = self.ar1_sigma_correction(sigma_d, T, rho)

            # Diffusion final log-return: N(µ·T, σ_d_T²)
            diff_final = mu_T + self._rng.standard_normal(N) * sigma_d_T

            # Jump component: Poisson(λ·T) jumps per path.
            # Total jump = N(n_jumps·µ_J, n_jumps·σ_J²).
            # µ_J is the KEY addition for v3: if jumps on RDBEAR tend to be
            # negative (µ_J < 0), that directional bias is now captured —
            # it shifts the JD final distribution downward, correctly
            # suppressing CALL and amplifying PUT signals.
            lam_total        = jp.lam * T
            n_jumps_per_path = self._rng.poisson(lam_total, size=N)
            jump_final = np.where(
                n_jumps_per_path > 0,
                self._rng.normal(
                    n_jumps_per_path * jp.mu_j,       # signed bias preserved
                    np.sqrt(np.maximum(n_jumps_per_path, 0)) * jp.sigma_j,
                ),
                0.0,
            )
            jd_final  = diff_final + jump_final
            p_rise_jd = float((jd_final > 0).mean())
            p_fall_jd = float((jd_final < 0).mean())
        else:
            p_rise_jd = p_rise_gbm
            p_fall_jd = p_fall_gbm

        # ── Blend ─────────────────────────────────────────────────────────
        p_rise = (1.0 - ew) * p_rise_gbm + ew * p_rise_jd
        p_fall = (1.0 - ew) * p_fall_gbm + ew * p_fall_jd

        p_rise = float(np.clip(p_rise, 0.01, 0.99))
        p_fall = float(np.clip(p_fall, 0.01, 0.99))

        # Stash for intelligence layer and logging
        self.last_components = {
            "p_rise_gbm": p_rise_gbm, "p_fall_gbm": p_fall_gbm,
            "p_rise_jd" : p_rise_jd,  "p_fall_jd" : p_fall_jd,
            "jd_active" : ew > 0.0,
            "mu"        : mu,
            "rho"       : rho,
            "regime_bump": self.last_regime_bump,
        }

        return p_rise, p_fall

    # ── Convenience entry-point ───────────────────────────────────────────

    def simulate(self, tick_buf, T: int) -> tuple[float, float, float]:
        """Returns (p_rise, p_fall, sigma)."""
        sigma = self.ewma_sigma(tick_buf)
        p_rise, p_fall = self.simulate_direction(tick_buf, T, sigma)
        return p_rise, p_fall, sigma

    # ── Intelligence layer ──────────────────────────────────────────────
    # "Is this signal good?" — every available model must agree on
    # direction AND clear the probability floor. No EV / payout math.

    def signal_quality(self, direction: str, p_win: float,
                       p_floor: float) -> tuple[bool, str]:
        """
        Returns (is_good, reason).

        v3 checks (in order):
          1. Regime gate: if vol is expanding, p_floor is raised.
          2. Asymmetric CALL scrutiny: CALL requires a higher floor than PUT,
             calibrated via call_scrutiny_mult. This is the primary guard
             against betting against bear-biased symbols like RDBEAR.
          3. p_win clears the (possibly elevated) floor.
          4. GBM model agrees with the blended direction.
          5. Drift sign agreement: if µ is meaningfully negative, a CALL
             signal is flagged as drift-opposed and rejected.
          6. If JD is active, JD model also agrees with the direction.
        """
        comp = getattr(self, "last_components", None)
        if comp is None:
            return False, "no component data"

        # ── 1. Regime gate ────────────────────────────────────────────────
        effective_floor = p_floor + comp.get("regime_bump", 0.0)

        # ── 2. Asymmetric CALL scrutiny ───────────────────────────────────
        if direction == "CALL" and self.call_scrutiny > 1.0:
            effective_floor = effective_floor * self.call_scrutiny

        # ── 3. Probability floor ──────────────────────────────────────────
        if p_win < effective_floor:
            return False, (
                f"p_win {p_win:.4f} < floor {effective_floor:.4f}"
                f"{'(CALL scrutiny)' if direction=='CALL' and self.call_scrutiny>1 else ''}"
                f"{'(regime)' if comp.get('regime_bump', 0) > 0 else ''}"
            )

        # ── 4. GBM model agreement ────────────────────────────────────────
        gbm_dir = "CALL" if comp["p_rise_gbm"] >= comp["p_fall_gbm"] else "PUT"
        if gbm_dir != direction:
            return False, f"GBM disagrees ({gbm_dir} vs {direction})"

        # ── 5. Drift sign agreement ───────────────────────────────────────
        # If the rolling drift is strongly negative, reject CALL signals.
        # Threshold: drift more negative than -0.5σ per tick is meaningful.
        mu  = comp.get("mu", 0.0)
        sig = self.ewma_sigma.__func__(self, []) if False else 0.0  # placeholder
        # Use last_jump_params sigma as proxy (already computed)
        # A simpler heuristic: if mu < -2 * |mu_floor| suppress CALL.
        # We set mu_floor as a small multiple of typical drift noise.
        # In practice: suppress CALL if mu < -0.3 * sigma_ewma
        # We don't have sigma here directly, but we can use the GBM p_fall
        # as a proxy: if GBM already gives p_fall_gbm > 0.54, drift is negative.
        if direction == "CALL" and comp.get("p_fall_gbm", 0.5) > 0.54:
            return False, (
                f"Drift opposed: p_fall_gbm={comp['p_fall_gbm']:.4f} "
                f"µ={mu:+.7f}"
            )

        # ── 6. JD model agreement ─────────────────────────────────────────
        if comp["jd_active"]:
            jd_dir = "CALL" if comp["p_rise_jd"] >= comp["p_fall_jd"] else "PUT"
            if jd_dir != direction:
                return False, f"JD disagrees ({jd_dir} vs {direction})"

        regime_note = f" [regime+{comp.get('regime_bump',0):.2f}]" if comp.get("regime_bump", 0) > 0 else ""
        call_note   = f" [call_scrutiny x{self.call_scrutiny}]" if direction=="CALL" and self.call_scrutiny>1 else ""
        return True, f"agree{regime_note}{call_note}"


# ===========================================================================
# KELLY STAKER  — calibrated for $1 account, $0.35 minimum
# ===========================================================================

class KellyStaker:
    """
    Tiered proportional floor combined with fractional Kelly:

      Balance tier  |  Proportional floor
      $0.35–$1.99   |  35%  ← ensures $0.35 minimum is reachable
      $2.00–$4.99   |  12%
      $5.00–$14.99  |   7%
      $15.00+       |   5%

    Stake = max(proportional_floor, fractional_kelly)
    Then clamped:  [kelly_min_stake, min(kelly_max_stake, kelly_max_pct × balance)]

    Kelly formula:  f* = (p × b − q) / b,  b = payout_ratio
    Fractional:     stake = f* × kelly_fraction × balance
    """

    def __init__(self, cfg):
        self.fraction  = cfg["kelly_fraction"]   # 0.25
        self.max_pct   = cfg["kelly_max_pct"]    # 0.10
        self.min_stake = cfg["kelly_min_stake"]  # 0.35
        self.max_stake = cfg["kelly_max_stake"]  # 5.00
        self.wins = self.n = 0

    def _tiered_pct(self, balance: float) -> float:
        if balance < 2.00:  return 0.35
        if balance < 5.00:  return 0.12
        if balance < 15.00: return 0.07
        return 0.05

    def next_stake(self, p_win: float, balance: float,
                   payout_ratio: float = 0.85) -> float:
        if balance <= 0:
            return self.min_stake

        # Proportional floor (guarantees we can meet Deriv minimum)
        prop_stake = balance * self._tiered_pct(balance)

        # Fractional Kelly
        b       = payout_ratio
        q       = 1.0 - p_win
        f_star  = (p_win * b - q) / b
        if f_star > 0:
            kelly_stake = min(f_star * self.fraction, self.max_pct) * balance
        else:
            kelly_stake = 0.0

        stake = max(prop_stake, kelly_stake)
        stake = min(stake, self.max_stake, balance * self.max_pct)
        stake = max(stake, self.min_stake)
        stake = min(stake, balance)          # never bet more than we have
        return round(stake, 2)

    def record(self, win: bool):
        if win: self.wins += 1
        self.n += 1
        log.info("[Kelly] %s  WR=%.1f%%  wins=%d/%d",
                 "WIN" if win else "LOSS",
                 self.wins / self.n * 100 if self.n else 0,
                 self.wins, self.n)


# ===========================================================================
# TRADE LOGGER  (CSV)
# ===========================================================================

class TradeLogger:
    FIELDS = [
        "timestamp", "symbol", "direction", "duration_ticks",
        "price_at_entry",
        "p_rise", "p_fall", "p_win", "payout_ratio",
        "sigma_ewma", "mc_paths",
        "jd_lambda", "jd_mu_j", "jd_sigma_j", "jd_n_jumps", "jd_weight_used",
        "stake", "balance_before", "outcome", "profit", "balance_after",
        "sprt_status", "session_wr",
    ]

    def __init__(self, path: str):
        self.path    = path
        self._exists = os.path.isfile(path)

    def log(self, row: dict):
        with open(self.path, "a", newline="", encoding="utf-8") as f:
            w = csv.DictWriter(f, fieldnames=self.FIELDS, extrasaction="ignore")
            if not self._exists:
                w.writeheader()
                self._exists = True
            w.writerow(row)


# ===========================================================================
# CONNECTION LAYER  — DerivWSManager  (direct port from RDBEAR v10)
# ===========================================================================

class ConnState(enum.IntEnum):
    DISCONNECTED  = 0
    CONNECTING    = 1
    CONNECTED     = 2
    AUTHENTICATED = 3
    SUBSCRIBED    = 4


class DerivWSManager:
    """
    Persistent, self-healing WebSocket connection to Deriv.
    Ported verbatim from RDBEAR v10 — no functional changes.

    Key properties:
      • Fresh WebSocketApp every reconnect cycle (never reuse dead objects).
      • Exponential back-off with ±1s jitter, capped at 120s.
      • Single heartbeat daemon per WS object (exits with its socket).
      • Thread-safe safe_send() — fire-and-forget, never raises.
      • on_disconnect_cb fires before every reconnect (lets the trader
        reset stuck flags before the sleep begins).
      • stop() is idempotent, safe from any thread.
    """

    WS_URL             = "wss://ws.binaryws.com/websockets/v3"
    HEARTBEAT_INTERVAL = 20
    PING_INTERVAL      = 25
    PING_TIMEOUT       = 15
    RECONNECT_BASE     = 2.0
    RECONNECT_CAP      = 120.0

    def __init__(self, app_id: int,
                 on_open_cb,
                 on_message_cb,
                 on_disconnect_cb=None,
                 name: str = "DerivWS"):
        self.app_id            = app_id
        self._on_open_cb       = on_open_cb
        self._on_message_cb    = on_message_cb
        self._on_disconnect_cb = on_disconnect_cb
        self.name              = name
        self._lock    = threading.Lock()
        self.state    = ConnState.DISCONNECTED
        self._running = False
        self._ws      = None
        self._attempt = 0

    def safe_send(self, payload: dict) -> bool:
        with self._lock:
            ws   = self._ws
            live = (self.state >= ConnState.CONNECTED and ws is not None)
        if not live:
            return False
        try:
            ws.send(json.dumps(payload))
            return True
        except Exception as e:
            log.warning("[%s] safe_send failed: %s", self.name, e)
            return False

    def start(self):
        self._running = True
        self._loop()

    def stop(self):
        self._running = False
        self.state    = ConnState.DISCONNECTED
        with self._lock:
            ws = self._ws
        if ws:
            try: ws.close()
            except Exception: pass

    def _loop(self):
        while self._running:
            if self._attempt > 0:
                delay = min(
                    self.RECONNECT_BASE * (2 ** (self._attempt - 1)),
                    self.RECONNECT_CAP,
                ) + random.uniform(-1.0, 1.0)
                delay = max(1.0, delay)
                log.info("[%s] Reconnect #%d in %.1fs ...",
                         self.name, self._attempt, delay)
                time.sleep(delay)

            if not self._running:
                break

            self.state = ConnState.CONNECTING
            ws = websocket.WebSocketApp(
                f"{self.WS_URL}?app_id={self.app_id}",
                on_open    = self._cb_open,
                on_message = self._cb_message,
                on_error   = self._cb_error,
                on_close   = self._cb_close,
            )
            with self._lock:
                self._ws = ws

            try:
                ws.run_forever(
                    ping_interval = self.PING_INTERVAL,
                    ping_timeout  = self.PING_TIMEOUT,
                    sslopt        = {"check_hostname": True},
                )
            except Exception as e:
                log.error("[%s] run_forever raised: %s", self.name, e)

            self.state = ConnState.DISCONNECTED
            with self._lock:
                self._ws = None

            if not self._running:
                break

            if self._on_disconnect_cb:
                try: self._on_disconnect_cb()
                except Exception as e:
                    log.error("[%s] on_disconnect_cb raised: %s", self.name, e)

            self._attempt += 1

        log.info("[%s] Connection loop exited cleanly.", self.name)

    def _cb_open(self, ws):
        self._attempt = 0
        self.state    = ConnState.CONNECTED
        log.info("[%s] Connected.", self.name)
        self._spawn_heartbeat(ws)
        try:
            self._on_open_cb(ws)
        except Exception as e:
            log.error("[%s] on_open_cb raised: %s", self.name, e, exc_info=True)

    def _cb_message(self, ws, raw):
        try:
            self._on_message_cb(ws, raw)
        except Exception as e:
            log.error("[%s] on_message_cb raised: %s", self.name, e, exc_info=True)

    def _cb_error(self, ws, error):
        log.warning("[%s] WS error: %s", self.name, error)

    def _cb_close(self, ws, code, msg):
        log.info("[%s] WS closed  code=%s  msg=%s", self.name, code, msg)

    def _spawn_heartbeat(self, ws):
        def _beat():
            while self._running and self.state >= ConnState.CONNECTED:
                try:
                    ws.send(json.dumps({"ping": 1}))
                except Exception:
                    break
                time.sleep(self.HEARTBEAT_INTERVAL)
            log.debug("[%s] Heartbeat exiting.", self.name)

        threading.Thread(
            target=_beat, daemon=True,
            name=f"{self.name}-HB",
        ).start()


# ===========================================================================
# HISTORICAL COLLECTOR  (port from RDBEAR v10)
# ===========================================================================

class HistoricalCollector:
    WS_URL       = "wss://ws.binaryws.com/websockets/v3"
    MAX_PER_CALL = 5000

    def __init__(self, symbol, cfg, done, existing_df=None):
        self.symbol    = symbol
        self.cfg       = cfg
        self.done      = done
        self._existing = existing_df

    def _fetch_page_once(self, end_epoch):
        import queue as _q
        q = _q.Queue()

        def _on_open(ws):
            ws.send(json.dumps({
                "ticks_history": self.symbol,
                "end"          : end_epoch,
                "count"        : self.MAX_PER_CALL,
                "style"        : "ticks",
                "adjust_start_time": 1,
            }))

        def _on_msg(ws, raw):
            try:
                msg = json.loads(raw)
                if msg.get("msg_type") == "history":
                    h = msg.get("history", {})
                    q.put(sorted(
                        [{"timestamp": float(t), "price": float(p)}
                         for t, p in zip(h.get("times", []), h.get("prices", []))],
                        key=lambda x: x["timestamp"],
                    ))
                elif "error" in msg:
                    log.error("[Collector] Error: %s", msg["error"])
                    q.put([])
                else:
                    return
                ws.close()
            except Exception as e:
                log.warning("[Collector] Parse error: %s", e)
                q.put([]); ws.close()

        def _on_err(ws, e):
            log.warning("[Collector] WS error: %s", e)
            q.put([])

        ws = websocket.WebSocketApp(
            f"{self.WS_URL}?app_id={self.cfg['app_id']}",
            on_open=_on_open, on_message=_on_msg,
            on_error=_on_err, on_close=lambda *_: None,
        )
        t = threading.Thread(target=ws.run_forever, daemon=True)
        t.start()
        try:
            return q.get(timeout=25)
        except Exception:
            return []
        finally:
            try: ws.close()
            except Exception: pass
            t.join(timeout=3)

    def _fetch_page(self, end_epoch, max_retries=3):
        for attempt in range(max_retries):
            result = self._fetch_page_once(end_epoch)
            if result:
                return result
            if attempt < max_retries - 1:
                delay = (2 ** attempt) + random.uniform(0.0, 1.0)
                log.warning("[Collector/%s] Empty page (attempt %d/%d) "
                            "retrying in %.1fs ...",
                            self.symbol, attempt + 1, max_retries, delay)
                time.sleep(delay)
        return []

    def _collect(self):
        cfg          = self.cfg
        collect_secs = int(cfg["collect_hours"] * 3600)
        now_epoch    = int(time.time())
        cutoff_epoch = now_epoch - collect_secs

        log.info("[Collector/%s] Fetching %.1fh of ticks ...",
                 self.symbol, cfg["collect_hours"])

        all_ticks = []
        if self._existing is not None and not self._existing.empty:
            in_win = self._existing[self._existing["timestamp"] >= cutoff_epoch]
            if not in_win.empty:
                all_ticks = in_win.to_dict("records")
                if min(t["timestamp"] for t in all_ticks) <= cutoff_epoch:
                    log.info("[Collector/%s] Existing CSV covers window — reusing.",
                             self.symbol)
                    self._save(all_ticks)
                    return

        fetch_end = now_epoch
        for page_num in range(1, 20):
            page = self._fetch_page(fetch_end)
            if not page:
                log.warning("[Collector/%s] Empty page %d.", self.symbol, page_num)
                break
            all_ticks.extend(
                [tk for tk in page if tk["timestamp"] >= cutoff_epoch])
            if page[0]["timestamp"] <= cutoff_epoch:
                break
            fetch_end = int(page[0]["timestamp"]) - 1
            time.sleep(0.3)

        log.info("[Collector/%s] Collected %d ticks.", self.symbol, len(all_ticks))
        self._save(all_ticks)

    def _save(self, ticks):
        os.makedirs(self.cfg["data_dir"], exist_ok=True)
        path = os.path.join(self.cfg["data_dir"], f"ticks_{self.symbol}.csv")
        if ticks:
            df = (pd.DataFrame(ticks)
                  .drop_duplicates(subset=["timestamp"])
                  .sort_values("timestamp")
                  .reset_index(drop=True))
            df.to_csv(path, index=False)
            log.info("[Collector/%s] Saved %d ticks → %s",
                     self.symbol, len(df), path)
        self.done.set()

    def start(self):
        def _run():
            try:
                self._collect()
            except Exception as e:
                log.error("[Collector/%s] Fatal: %s", self.symbol, e, exc_info=True)
                self.done.set()
        threading.Thread(target=_run, daemon=True,
                         name=f"HistCol-{self.symbol}").start()


# ===========================================================================
# LIVE TRADER  — Rise / Fall
# ===========================================================================

class LiveTrader:
    """
    Connects to Deriv, accumulates ticks, runs Monte Carlo direction
    estimation across all candidate durations, and fires the CALL/PUT
    contract for the best signal that the intelligence layer agrees is good.

    Duration selection algorithm
    ----------------------------
    For each tick (after cooldowns):
      1. Compute σ once from the rolling tick buffer.
      2. For each candidate duration T ∈ hold_durations:
           (p_rise, p_fall) = MC_simulate(tick_buf, T, σ)
           direction   = "CALL" if p_rise > p_fall else "PUT"
           p_win       = max(p_rise, p_fall)
           Intelligence layer asks "is this signal good?":
             - p_win > mc_p_floor
             - GBM model agrees with the blended direction
             - JD model (if active) agrees with the blended direction
      3. Pick T* = argmax p_win among durations where the intelligence
         layer agrees the signal is good.
      4. If no T* found → no trade (wait for the next tick).
      5. Require signal_persistence_ticks consecutive ticks agreeing
         on the same (direction, duration) before placing the trade.

    Payout discovery
    ----------------
    A live proposal is sent to Deriv at the chosen stake to discover
    the real payout ratio, used for Kelly sizing and logging only.
    The trade is placed once the intelligence layer and persistence
    checks pass — there is no EV or payout-ratio gate.
    """

    def __init__(self, cfg, initial_ticks, staker, sprt, trade_logger):
        self.cfg     = cfg
        self.staker  = staker
        self.sprt    = sprt
        self.logger  = trade_logger
        self.pricer  = MonteCarloPricer(cfg)

        self.tick_buf = deque(initial_ticks, maxlen=5000)

        # State
        self.balance        = None
        self.start_balance  = None
        self.peak_balance   = None
        self.running        = False

        # Trade state
        self.waiting_result   = False
        self.waiting_proposal = False
        self.pending_pid      = None
        self.pending_stake    = 0.0
        self.pending_dur      = None
        self.pending_direction= None   # "CALL" or "PUT"
        self.pending_p_win    = 0.0
        self.pending_p_rise   = 0.0
        self.pending_p_fall   = 0.0
        self.pending_payout   = 0.85   # updated from live proposal
        self.pending_sigma    = 0.0

        # Counters
        self.live_tick_count = 0
        self.post_trade_tick = 0
        self.consec_losses   = 0
        self.cooldown_until  = 0
        self.wins = self.total = 0
        self.session_pnl     = 0.0

        # Signal persistence
        self._persist_count    = 0
        self._persist_best_p   = 0.0
        self._persist_dur      = None
        self._persist_dir      = None

        # Logging cache
        # (no EV cache — intelligence-layer gating only)

        # Connection manager (verbatim from RDBEAR v10)
        self._conn = DerivWSManager(
            app_id           = cfg["app_id"],
            on_open_cb       = self._on_ws_open,
            on_message_cb    = self._on_message,
            on_disconnect_cb = self._on_disconnect,
            name             = "LiveTrader-RF",
        )

    # ── Lifecycle ────────────────────────────────────────────────────────

    def run(self):
        self.running = True
        log.info("[Trader] Starting connection manager ...")
        self._conn.start()

    def stop(self):
        self.running = False
        self._conn.stop()

    def _on_ws_open(self, ws):
        log.info("[Trader] (Re)connected — authorising ...")
        self._conn.safe_send({"authorize": self.cfg["api_token"]})

    def _on_disconnect(self):
        if self.waiting_proposal:
            log.warning("[Trader] Connection lost mid-proposal — resetting.")
            self.waiting_proposal = False
        if self.waiting_result:
            log.warning("[Trader] Connection lost awaiting settlement — resetting.")
            self.waiting_result = False

    def _on_message(self, ws, raw):
        try:
            msg = json.loads(raw)
        except Exception:
            return
        mt = msg.get("msg_type", "")
        try:
            if   mt == "authorize":               self._on_auth(msg)
            elif mt == "balance":                 self._on_balance(msg)
            elif mt == "tick":                    self._on_tick(msg)
            elif mt == "proposal":                self._on_proposal(msg)
            elif mt == "buy":                     self._on_buy(msg)
            elif mt == "proposal_open_contract":  self._on_poc(msg)
            elif "error" in msg:
                log.warning("[Trader] API error: %s",
                            msg["error"].get("message", str(msg["error"])))
        except Exception as e:
            log.error("[Trader] Handler error: %s", e, exc_info=True)

    # ── Auth ─────────────────────────────────────────────────────────────

    def _on_auth(self, msg):
        info               = msg.get("authorize", {})
        self.balance       = float(info.get("balance", 0))
        self.start_balance = self.balance
        self.peak_balance  = self.balance
        log.info("[Trader] Authorised: %s | %s $%.2f",
                 info.get("loginid", "?"),
                 info.get("currency", "USD"),
                 self.balance)
        self._conn.state = ConnState.AUTHENTICATED
        self._conn.safe_send({"balance": 1, "subscribe": 1})
        self._conn.safe_send({"ticks": self.cfg["symbol"], "subscribe": 1})
        self._conn.state = ConnState.SUBSCRIBED
        log.info("[Trader] Subscribed to ticks. Waiting for buffer "
                 "(%d ticks) ...", self.cfg["min_ticks"])

    # ── Balance ───────────────────────────────────────────────────────────

    def _on_balance(self, msg):
        b = msg.get("balance", {})
        if "balance" in b:
            self.balance = float(b["balance"])
            if self.peak_balance is None or self.balance > self.peak_balance:
                self.peak_balance = self.balance

    # ── Tick ──────────────────────────────────────────────────────────────

    def _on_tick(self, msg):
        t = msg.get("tick", {})
        self.tick_buf.append({
            "timestamp": float(t.get("epoch", time.time())),
            "price"    : float(t.get("quote", 0)),
        })
        self.live_tick_count += 1

        if not self.running or self.waiting_result or self.waiting_proposal:
            return
        if len(self.tick_buf) < self.cfg["min_ticks"]:
            if self.live_tick_count % 50 == 0:
                log.info("[Trader] Buffer %d/%d ticks ...",
                         len(self.tick_buf), self.cfg["min_ticks"])
            return
        if self.live_tick_count < self.cooldown_until:
            return
        min_gap = self.cfg.get("min_ticks_between_trades", 30)
        if self.live_tick_count - self.post_trade_tick < min_gap:
            return

        self._evaluate_signal()

    # ── Duration-selection signal evaluator ───────────────────────────────

    def _evaluate_signal(self):
        cfg       = self.cfg
        p_floor   = cfg.get("mc_p_floor", 0.51)
        durations = cfg.get("hold_durations", [1, 2, 3, 4, 5])

        # Shared σ, drift, AR(1) rho, and regime bump — computed once
        shared_sigma = self.pricer.ewma_sigma(self.tick_buf)
        # Trigger regime check (result stored in pricer.last_regime_bump
        # and injected into last_components by simulate_direction)
        self.pricer.regime_floor_bump(self.tick_buf)

        best_p   = -1.0
        best_dur = None
        best_dir = None
        best_rise = 0.0
        best_fall = 0.0
        best_reason = ""

        # Probe every candidate duration. A duration only qualifies if
        # the intelligence layer agrees the signal is good (all models
        # agree on direction and p_win clears the floor).
        for dur in durations:
            p_rise, p_fall = self.pricer.simulate_direction(
                self.tick_buf, dur, shared_sigma)
            direction = "CALL" if p_rise >= p_fall else "PUT"
            p_win     = p_rise if direction == "CALL" else p_fall

            is_good, reason = self.pricer.signal_quality(
                direction, p_win, p_floor)
            if not is_good:
                continue

            # Among good signals, prefer the strongest directional edge.
            if p_win > best_p:
                best_p    = p_win
                best_dur  = dur
                best_dir  = direction
                best_rise = p_rise
                best_fall = p_fall
                best_reason = reason

        if best_dur is None:
            self._persist_count = max(0, self._persist_count - 1)
            if self.live_tick_count % 30 == 0:
                log.info("[MC] No good signal  best_p=%.4f σ=%.6f "
                         "µ=%+.7f rho=%.3f λ=%.5f  persist_decay=%d",
                         best_p, shared_sigma,
                         self.pricer.last_drift,
                         self.pricer.last_rho,
                         self.pricer.last_jump_params.lam,
                         self._persist_count)
            return

        # ── Signal persistence ─────────────────────────────────────────────
        required = cfg.get("signal_persistence_ticks", 2)
        same = (best_dur == self._persist_dur and best_dir == self._persist_dir)
        if same:
            self._persist_count += 1
            if best_p > self._persist_best_p:
                self._persist_best_p = best_p
        else:
            self._persist_count  = 1
            self._persist_best_p = best_p
            self._persist_dur    = best_dur
            self._persist_dir    = best_dir

        if self._persist_count < required:
            if self.live_tick_count % 10 == 0:
                log.info("[MC] Persistence %d/%d  %s  dur=%dt  p=%.4f  (%s)",
                         self._persist_count, required,
                         best_dir, best_dur, best_p, best_reason)
            return

        # ── All checks passed — intelligence layer agrees ──────────────────
        jp = self.pricer.last_jump_params
        log.info("[MC] *** SIGNAL *** %s  dur=%dt  p_win=%.4f  "
                 "p_rise=%.4f  p_fall=%.4f  σ=%.6f  µ=%+.7f  rho=%.3f  "
                 "regime_bump=%.2f  persist=%d  intel=%s | "
                 "JD: λ=%.5f μ_J=%+.5f σ_J=%.5f n_jumps=%d/%d  ew=%.2f",
                 best_dir, best_dur, best_p,
                 best_rise, best_fall, shared_sigma,
                 self.pricer.last_drift, self.pricer.last_rho,
                 self.pricer.last_regime_bump,
                 self._persist_count, best_reason,
                 jp.lam, jp.mu_j, jp.sigma_j, jp.n_jumps, jp.n_obs,
                 self.cfg.get("jd_weight", 0.5))

        self._persist_count = 0

        if not self.balance or self.balance <= 0:
            log.warning("[Trader] Balance not confirmed — skipping.")
            return

        # Stake computed with estimated payout; refined once the real
        # payout comes back in the proposal (used for sizing only).
        stake = self.staker.next_stake(best_p, self.balance, 0.80)

        self.pending_stake     = stake
        self.pending_dur       = best_dur
        self.pending_direction = best_dir
        self.pending_p_win     = best_p
        self.pending_p_rise    = best_rise
        self.pending_p_fall    = best_fall
        self.pending_sigma     = shared_sigma
        self.waiting_proposal  = True

        log.info("[Trader] Sending proposal  %s  dur=%dt  stake=$%.2f",
                 best_dir, best_dur, stake)

        self._conn.safe_send({
            "proposal"      : 1,
            "amount"        : stake,
            "basis"         : "stake",
            "contract_type" : best_dir,          # "CALL" or "PUT"
            "currency"      : cfg.get("currency", "USD"),
            "duration"      : best_dur,
            "duration_unit" : "t",               # ticks
            "symbol"        : cfg["symbol"],
        })

    # ── Proposal response ─────────────────────────────────────────────────

    def _on_proposal(self, msg):
        if "error" in msg:
            log.warning("[Trader] Proposal error: %s",
                        msg["error"].get("message", str(msg["error"])))
            self.waiting_proposal = False
            return

        prop          = msg.get("proposal", {})
        pid           = msg.get("id") or prop.get("id")
        ask_price     = float(prop.get("ask_price",
                              prop.get("cost", self.pending_stake)))
        payout_amount = float(prop.get("payout", ask_price * 1.85))
        # Net payout ratio: profit / stake (kept for Kelly sizing / logging
        # only — no longer used as a trade-acceptance gate).
        payout_ratio  = round((payout_amount - ask_price) / max(ask_price, 1e-8), 4)

        self.pending_pid      = pid
        self.pending_payout   = payout_ratio
        self.waiting_proposal = False
        self.waiting_result   = True

        log.info("[Trader] Proposal OK  %s  dur=%dt  payout=%.1f%%  "
                 "p_win=%.4f  stake=$%.2f",
                 self.pending_direction, self.pending_dur,
                 payout_ratio * 100, self.pending_p_win, self.pending_stake)

        self._conn.safe_send({
            "buy"  : pid,
            "price": self.pending_stake,
        })

    # ── Buy confirmation ──────────────────────────────────────────────────

    def _on_buy(self, msg):
        if "error" in msg:
            log.warning("[Trader] Buy error: %s",
                        msg["error"].get("message", str(msg["error"])))
            self.waiting_result = False
            return
        buy = msg.get("buy", {})
        cid = str(buy.get("contract_id", "?"))
        log.info("[Trader] Contract opened  cid=%s  paid=$%.2f  %s %dt",
                 cid,
                 float(buy.get("buy_price", self.pending_stake)),
                 self.pending_direction,
                 self.pending_dur)
        self._conn.safe_send({
            "proposal_open_contract": 1,
            "contract_id"           : int(cid),
            "subscribe"             : 1,
        })

    # ── Settlement ────────────────────────────────────────────────────────

    def _on_poc(self, msg):
        poc = msg.get("proposal_open_contract", {})
        if not poc.get("is_sold", 0):
            return

        profit  = float(poc.get("profit", 0))
        win     = profit > 0
        new_bal = float(poc.get("balance_after", self.balance or 0))
        old_bal = self.balance or new_bal

        self.balance = new_bal
        if self.peak_balance is None or new_bal > self.peak_balance:
            self.peak_balance = new_bal

        self.session_pnl += profit
        self.total += 1
        if win:
            self.wins += 1
            self.consec_losses = 0
        else:
            self.consec_losses += 1

        wr = self.wins / self.total * 100
        log.info("[Trade] %s #%d | %s %dt | P&L=%+.2f | bal=$%.2f | "
                 "W/L=%d/%d (%.1f%%) | SPRT:%s",
                 "WIN " if win else "LOSS",
                 self.total,
                 self.pending_direction, self.pending_dur,
                 profit, new_bal,
                 self.wins, self.total - self.wins, wr,
                 self.sprt.update(win))

        self.staker.record(win)
        self.post_trade_tick = self.live_tick_count
        self.waiting_result  = False

        # Consecutive-loss cooldown
        if not win and self.consec_losses >= self.cfg.get("max_consec_losses", 5):
            cd = self.cfg.get("consec_loss_cooldown_ticks", 60)
            self.cooldown_until = self.live_tick_count + cd
            log.warning("[Risk] %d consecutive losses — cooling down %d ticks.",
                        self.consec_losses, cd)
            self.consec_losses = 0

        # CSV log
        jp   = self.pricer.last_jump_params
        eff_w = (self.cfg.get("jd_weight", 0.5)
                 if jp.n_jumps >= self.cfg.get("jd_min_jumps", 5) else 0.0)
        self.logger.log({
            "timestamp"       : datetime.utcnow().isoformat(),
            "symbol"          : self.cfg["symbol"],
            "direction"       : self.pending_direction,
            "duration_ticks"  : self.pending_dur,
            "price_at_entry"  : round(list(self.tick_buf)[-1]["price"], 5)
                                if self.tick_buf else "",
            "p_rise"          : round(self.pending_p_rise, 5),
            "p_fall"          : round(self.pending_p_fall, 5),
            "p_win"           : round(self.pending_p_win, 5),
            "payout_ratio"    : round(self.pending_payout, 4),
            "sigma_ewma"      : round(self.pending_sigma, 8),
            "mc_paths"        : self.cfg.get("mc_n_paths", 3000),
            "jd_lambda"       : round(jp.lam, 6),
            "jd_mu_j"         : round(jp.mu_j, 6),
            "jd_sigma_j"      : round(jp.sigma_j, 6),
            "jd_n_jumps"      : jp.n_jumps,
            "jd_weight_used"  : round(eff_w, 2),
            "stake"           : round(self.pending_stake, 2),
            "balance_before"  : round(old_bal, 2),
            "outcome"         : "WIN" if win else "LOSS",
            "profit"          : round(profit, 2),
            "balance_after"   : round(new_bal, 2),
            "sprt_status"     : self.sprt.status,
            "session_wr"      : round(wr, 2),
        })

        if not self._risk_ok():
            log.warning("[Risk] Risk limit hit — stopping.")
            self.stop()

    # ── Risk ──────────────────────────────────────────────────────────────

    def _risk_ok(self) -> bool:
        cfg = self.cfg
        sb  = self.start_balance or 1.0
        pb  = self.peak_balance  or self.balance or 1.0
        bl  = self.balance       or 0.0

        if self.session_pnl <= -(sb * cfg["max_daily_loss_pct"]):
            log.warning("[Risk] Daily loss limit: P&L=%.2f", self.session_pnl)
            return False
        if bl > 0 and (pb - bl) >= sb * cfg["max_drawdown_from_peak_pct"]:
            log.warning("[Risk] Drawdown limit: peak=%.2f current=%.2f", pb, bl)
            return False
        if self.session_pnl >= sb * cfg.get("take_profit_pct", 9999):
            log.info("[Risk] Take-profit reached: P&L=%.2f", self.session_pnl)
            return False
        return True


# ===========================================================================
# MAIN
# ===========================================================================

def main():
    cfg    = CONFIG
    symbol = cfg["symbol"]

    log.info("=" * 65)
    log.info("  DERIV RISE/FALL BOT  v3  —  Monte Carlo + Jump-Diffusion")
    log.info("  Symbol      : %s", symbol)
    log.info("  MC paths    : %d   vol_window=%d  α=%.2f",
             cfg["mc_n_paths"], cfg["mc_vol_window"], cfg["mc_ewma_alpha"])
    log.info("  JD          : threshold=%.1fσ  window=%d  min_jumps=%d  w=%.2f",
             cfg["jd_jump_threshold"], cfg["jd_fit_window"],
             cfg["jd_min_jumps"],      cfg["jd_weight"])
    log.info("  p_floor     : %.3f  (intelligence layer: GBM+JD agreement)",
             cfg["mc_p_floor"])
    log.info("  Durations   : %s ticks", cfg["hold_durations"])
    log.info("  Kelly       : fraction=%.2f  max_pct=%.0f%%  "
             "min=$%.2f  max=$%.2f",
             cfg["kelly_fraction"], cfg["kelly_max_pct"] * 100,
             cfg["kelly_min_stake"], cfg["kelly_max_stake"])
    log.info("  Persist     : %d consecutive ticks",
             cfg["signal_persistence_ticks"])
    log.info("=" * 65)

    if not cfg.get("api_token"):
        log.error("DERIV_API_TOKEN not set.  "
                  "Export it:  export DERIV_API_TOKEN=your_token")
        sys.exit(1)

    # ── Phase 1: Collect historical ticks ─────────────────────────────────
    log.info("\n>> PHASE 1 — Historical tick collection")
    data_path = os.path.join(cfg["data_dir"], f"ticks_{symbol}.csv")
    existing  = (pd.read_csv(data_path)
                 if os.path.isfile(data_path) else pd.DataFrame())

    done = threading.Event()
    HistoricalCollector(symbol, cfg, done, existing).start()
    done.wait()

    df = pd.read_csv(data_path) if os.path.isfile(data_path) else pd.DataFrame()
    if len(df) < cfg["min_ticks"]:
        log.warning("[Main] Only %d ticks (need %d). "
                    "Will trade once buffer fills live.", len(df), cfg["min_ticks"])

    initial_ticks = (df.tail(5000).to_dict("records") if not df.empty else [])
    log.info("[Main] Seeding tick buffer with %d historical ticks.",
             len(initial_ticks))

    # ── Phase 2: Live trading ──────────────────────────────────────────────
    log.info("\n>> PHASE 2 — Starting live trading on %s", symbol)

    staker = KellyStaker(cfg)
    sprt   = SPRTMonitor(
        p0=cfg["sprt_p0"], p1=cfg["sprt_p1"],
        alpha=cfg["sprt_alpha"], beta=cfg["sprt_beta"],
    )
    tlog   = TradeLogger(cfg["trade_log"])
    trader = LiveTrader(cfg, initial_ticks, staker, sprt, tlog)

    import signal as _sig
    def _shutdown(s, f):
        log.info("\n[Main] Shutting down ...")
        trader.stop()
    _sig.signal(_sig.SIGINT, _shutdown)
    try:
        _sig.signal(_sig.SIGTERM, _shutdown)
    except (OSError, ValueError):
        pass

    trader.run()

    log.info("\n[Main] Session complete.")
    log.info("  Trades    : %d",  trader.total)
    log.info("  Win rate  : %.1f%%",
             trader.wins / trader.total * 100 if trader.total else 0)
    log.info("  P&L       : %+.2f", trader.session_pnl)
    log.info("  SPRT      : %s",   sprt.summary())
    log.info("  Trade log : %s",   cfg["trade_log"])


if __name__ == "__main__":
    main()
