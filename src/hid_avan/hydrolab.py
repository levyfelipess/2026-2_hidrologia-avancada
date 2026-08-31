"""HydroLab 1.0 - modelos hidrológicos conceituais em Python.
(C) Eduardo Martins

Filosofia
---------
Cada modelo hidrológico é uma função independente e autocontida. O objetivo é
manter o algoritmo legível e próximo das equações apresentadas no livro,
sem exigir do usuário conhecimento de programação orientada a objetos.

Convenções
----------
* Precipitação, evapotranspiração e vazão simulada são expressas como lâmina
  d'água por passo de tempo.
* Modelos diários trabalham em mm/dia.
* Modelos mensais trabalham em mm/mês.
* A área da bacia NÃO entra nas funções dos modelos.
* Conversões para m³/s devem ser feitas separadamente.
* Os parâmetros são passados diretamente na chamada da função.
* states é opcional e permite reiniciar uma simulação a partir de estados prévios.
* return_state=True retorna (Q, estados_finais).

Exemplos
--------
Q = gr2m(P, ETP, X1=500.0, X2=0.8)

Q = gr4j(
    P, ETP,
    X1=350.0,
    X2=0.0,
    X3=90.0,
    X4=1.7
)

Q = smap(
    P, ETP,
    Str=500.0,
    K2t=2.0,
    Crec=5.0,
    Ai=2.0,
    Capc=40.0,
    Kkt=30.0
)
"""

from __future__ import annotations

import math
import numpy as np


# =============================================================================
# UTILITÁRIOS INTERNOS
# =============================================================================

def _forcing_from_series(P, ETP):
    """Valida e retorna P e ETP como vetores numpy de mesma dimensão."""
    P = np.asarray(P, dtype=float)
    ETP = np.asarray(ETP, dtype=float)

    if P.ndim != 1 or ETP.ndim != 1:
        raise ValueError("P e ETP devem ser vetores unidimensionais.")
    if P.shape != ETP.shape:
        raise ValueError("P e ETP devem possuir o mesmo tamanho.")
    if np.any(~np.isfinite(P)) or np.any(~np.isfinite(ETP)):
        raise ValueError("P e ETP não podem conter NaN ou infinito.")
    if np.any(P < 0) or np.any(ETP < 0):
        raise ValueError("P e ETP devem ser não negativos.")

    return P, ETP


def _require_positive(value, name):
    if value <= 0:
        raise ValueError(f"{name} deve ser maior que zero.")


def _bounded_efficiency(value):
    """Transformação C2M de Mathevet et al. (2006)."""
    return value / (2.0 - value)

# =============================================================================
# FAMÍLIA GR
# =============================================================================

def gr1a(P, ETP, X, previous_precip=None):
    """Modelo GR1A na escala anual.

    Parameters
    ----------
    P : array-like
        Precipitação anual [mm/ano].
    ETP : array-like
        Evapotranspiração potencial anual [mm/ano].
    X : float
        Parâmetro adimensional do GR1A.
    previous_precip : float, optional
        Precipitação do período anterior ao primeiro ano.
        Se omitido, usa-se a primeira precipitação da série.

    Returns
    -------
    numpy.ndarray
        Vazão simulada em lâmina [mm/ano].
    """
    P, ETP = _forcing_from_series(P, ETP)

    X = float(X)
    _require_positive(X, "X")

    if previous_precip is None:
        previous_precip = float(P[0]) if len(P) else 0.0

    Q = np.zeros_like(P, dtype=float)
    p_previous = float(previous_precip)

    for i, (p, ep) in enumerate(zip(P, ETP)):
        if ep <= 0.0:
            Q[i] = max(p, 0.0)
        else:
            t = (0.7 * p + 0.3 * p_previous) / (X * ep)
            Q[i] = p * (1.0 - 1.0 / math.sqrt(1.0 + t * t))
        p_previous = p

    return Q


def gr2m(P, ETP, X1, X2, states=None, return_state=False, return_fluxes=False):
    """Modelo GR2M na escala mensal.

    Parameters
    ----------
    P : array-like
        Precipitação [mm/mês].
    ETP : array-like
        Evapotranspiração potencial [mm/mês].
    X1 : float
        Capacidade do reservatório de produção [mm].
    X2 : float
        Coeficiente de troca do reservatório de propagação [-].
    states : dict, optional
        Estados iniciais: production_store e routing_store [mm].
    return_state : bool, optional
        Se True, retorna também os estados finais.
    return_fluxes : bool, optional
        Se True, retorna também as séries internas passo a passo
        (production_store, routing_store, P1, P2, P3, R2).

    Returns
    -------
    Q : numpy.ndarray
        Vazão simulada em lâmina [mm/mês].

    Opcionalmente:
        ``Q, estados_finais`` se ``return_state=True``;
        ``Q, fluxos`` se ``return_fluxes=True``;
        ``Q, fluxos, estados_finais`` se ambos forem True.
    """
    P, ETP = _forcing_from_series(P, ETP)

    X1 = float(X1)
    X2 = float(X2)
    _require_positive(X1, "X1")

    states = {} if states is None else dict(states)
    production_store = float(states.get("production_store", 0.0))
    routing_store = float(states.get("routing_store", 0.0))

    n = len(P)
    Q = np.zeros(n, dtype=float)

    if return_fluxes:
        PROD = np.zeros(n, dtype=float)
        ROUT = np.zeros(n, dtype=float)
        P1_ = np.zeros(n, dtype=float)
        P2_ = np.zeros(n, dtype=float)
        P3_ = np.zeros(n, dtype=float)
        R2_ = np.zeros(n, dtype=float)

    for i, (p, ep) in enumerate(zip(P, ETP)):
        phi = math.tanh(p / X1)
        psi = math.tanh(ep / X1)

        S1 = (production_store + X1 * phi) / (
            1.0 + phi * production_store / X1
        )
        P1 = p + production_store - S1

        S2 = S1 * (1.0 - psi) / (
            1.0 + psi * (1.0 - S1 / X1)
        )

        production_store = S2 / (
            1.0 + (S2 / X1) ** 3.0
        ) ** (1.0 / 3.0)

        P2 = S2 - production_store
        P3 = P1 + P2

        R1 = routing_store + P3
        R2 = X2 * R1

        q = R2 * R2 / (R2 + 60.0) if R2 > 0.0 else 0.0
        routing_store = R2 - q
        Q[i] = q

        if return_fluxes:
            PROD[i] = production_store
            ROUT[i] = routing_store
            P1_[i] = P1
            P2_[i] = P2
            P3_[i] = P3
            R2_[i] = R2

    final_state = {
        "production_store": production_store,
        "routing_store": routing_store,
    }

    fluxes = {
        "production_store": PROD,
        "routing_store": ROUT,
        "P1": P1_,
        "P2": P2_,
        "P3": P3_,
        "R2": R2_,
    } if return_fluxes else None

    if return_fluxes and return_state:
        return Q, fluxes, final_state
    if return_fluxes:
        return Q, fluxes
    if return_state:
        return Q, final_state
    return Q


def gr4j(P, ETP, X1, X2, X3, X4, states=None, return_state=False, return_fluxes=False):
    """Modelo GR4J na escala diária.

    Parameters
    ----------
    P : array-like
        Precipitação [mm/dia].
    ETP : array-like
        Evapotranspiração potencial [mm/dia].
    X1 : float
        Capacidade do reservatório de produção [mm].
    X2 : float
        Coeficiente de troca subterrânea [mm/dia].
    X3 : float
        Capacidade do reservatório de propagação [mm].
    X4 : float
        Tempo-base de UH1 [dia].
    return_state : bool, optional
        Se True, retorna também os estados finais.
    return_fluxes : bool, optional
        Se True, retorna também as séries internas passo a passo
        (S, R, Pn, En, Ps, Es, Perc, Pr, Q9, Q1, F, Qr, Qd).

    Returns
    -------
    Q : numpy.ndarray
        Vazão simulada em lâmina [mm/dia].

    Opcionalmente:
        ``Q, estados_finais`` se ``return_state=True``;
        ``Q, fluxos`` se ``return_fluxes=True``;
        ``Q, fluxos, estados_finais`` se ambos forem True.
    """
    P, ETP = _forcing_from_series(P, ETP)

    X1 = float(X1)
    X2 = float(X2)
    X3 = float(X3)
    X4 = float(X4)

    _require_positive(X1, "X1")
    _require_positive(X3, "X3")
    _require_positive(X4, "X4")

    # ------------------------------------------------------------------
    # 0. Hidrogramas unitários UH1 e UH2
    # ------------------------------------------------------------------
    def SH1(t):
        if t <= 0.0:
            return 0.0
        if t < X4:
            return (t / X4) ** 2.5
        return 1.0

    def SH2(t):
        if t <= 0.0:
            return 0.0
        if t < X4:
            return 0.5 * (t / X4) ** 2.5
        if t < 2.0 * X4:
            return 1.0 - 0.5 * (2.0 - t / X4) ** 2.5
        return 1.0

    NH1 = max(1, int(math.ceil(X4)))
    NH2 = max(1, int(math.ceil(2.0 * X4)))

    OrdUH1 = np.array([SH1(j + 1) - SH1(j) for j in range(NH1)])
    OrdUH2 = np.array([SH2(j + 1) - SH2(j) for j in range(NH2)])

    # ------------------------------------------------------------------
    # 1. Condições iniciais
    # ------------------------------------------------------------------
    states = {} if states is None else dict(states)

    S = float(states.get("production_store", 0.30 * X1))
    R = float(states.get("routing_store", 0.50 * X3))

    UH1 = np.asarray(
        states.get("uh1", np.zeros(NH1)),
        dtype=float
    ).copy()

    UH2 = np.asarray(
        states.get("uh2", np.zeros(NH2)),
        dtype=float
    ).copy()

    if len(UH1) != NH1 or len(UH2) != NH2:
        raise ValueError("Os estados UH1/UH2 são incompatíveis com X4.")

    Q = np.zeros(len(P), dtype=float)

    if return_fluxes:
        n = len(P)
        S_ = np.zeros(n, dtype=float)
        R_ = np.zeros(n, dtype=float)
        Pn_ = np.zeros(n, dtype=float)
        En_ = np.zeros(n, dtype=float)
        Ps_ = np.zeros(n, dtype=float)
        Es_ = np.zeros(n, dtype=float)
        Perc_ = np.zeros(n, dtype=float)
        Pr_ = np.zeros(n, dtype=float)
        Q9_ = np.zeros(n, dtype=float)
        Q1_ = np.zeros(n, dtype=float)
        F_ = np.zeros(n, dtype=float)
        Qr_ = np.zeros(n, dtype=float)
        Qd_ = np.zeros(n, dtype=float)

    # ==================================================================
    # LOOP DIÁRIO DO GR4J
    # ==================================================================
    for i in range(len(P)):
        Pi = P[i]
        Ei = ETP[i]

        # 2. Neutralização P - E
        if Pi >= Ei:
            Pn = Pi - Ei
            En = 0.0
        else:
            Pn = 0.0
            En = Ei - Pi

        # 3. Reservatório de produção
        if Pn > 0.0:
            T = math.tanh(Pn / X1)
            Ps = X1 * (1.0 - (S / X1) ** 2) * T
            Ps = Ps / (1.0 + (S / X1) * T)

            S = S + Ps
            Pr = Pn - Ps
            Es = 0.0

        else:
            T = math.tanh(En / X1)
            Es = S * (2.0 - S / X1) * T
            Es = Es / (
                1.0 + (1.0 - S / X1) * T
            )

            S = S - Es
            Pr = 0.0
            Ps = 0.0

        # 4. Percolação
        Perc = S * (
            1.0
            - (
                1.0
                + ((4.0 / 9.0) * S / X1) ** 4.0
            ) ** (-0.25)
        )

        S = S - Perc
        Pr = Pr + Perc

        # 5. UH1 e UH2
        UH1 = UH1 + 0.90 * Pr * OrdUH1
        Q9 = UH1[0]

        if NH1 > 1:
            UH1[:-1] = UH1[1:]
        UH1[-1] = 0.0

        UH2 = UH2 + 0.10 * Pr * OrdUH2
        Q1 = UH2[0]

        if NH2 > 1:
            UH2[:-1] = UH2[1:]
        UH2[-1] = 0.0

        # 6. Troca subterrânea
        F = X2 * (max(R, 0.0) / X3) ** 3.5

        # 7. Reservatório de propagação
        R = max(0.0, R + Q9 + F)

        Qr = R * (
            1.0
            - (1.0 + (R / X3) ** 4.0) ** (-0.25)
        )

        R = R - Qr

        # 8. Vazão direta e vazão total
        Qd = max(0.0, Q1 + F)
        Q[i] = Qr + Qd

        if return_fluxes:
            S_[i], R_[i] = S, R
            Pn_[i], En_[i] = Pn, En
            Ps_[i], Es_[i] = Ps, Es
            Perc_[i], Pr_[i] = Perc, Pr
            Q9_[i], Q1_[i] = Q9, Q1
            F_[i] = F
            Qr_[i], Qd_[i] = Qr, Qd

    final_state = {
        "production_store": S,
        "routing_store": R,
        "uh1": UH1.copy(),
        "uh2": UH2.copy(),
    }

    fluxes = {
        "S": S_, "R": R_, "Pn": Pn_, "En": En_, "Ps": Ps_, "Es": Es_,
        "Perc": Perc_, "Pr": Pr_, "Q9": Q9_, "Q1": Q1_, "F": F_,
        "Qr": Qr_, "Qd": Qd_,
    } if return_fluxes else None

    if return_fluxes and return_state:
        return Q, fluxes, final_state
    if return_fluxes:
        return Q, fluxes
    if return_state:
        return Q, final_state
    return Q


# =============================================================================
# HYMOD - DIÁRIO
# =============================================================================

def hymod(
    P,
    ETP,
    Cmax,
    Bexp,
    Alpha,
    Kq,
    Ks,
    states=None,
    return_state=False,
    return_fluxes=False,
):
    """Modelo HYMOD diário.

    Parameters
    ----------
    P, ETP : array-like
        Entradas em mm/dia.
    Cmax : float
        Capacidade máxima local do solo [mm].
    Bexp : float
        Expoente da distribuição de capacidades [-].
    Alpha : float
        Fração destinada ao ramo rápido [0-1].
    Kq : float
        Coeficiente dos reservatórios rápidos [0-1].
    Ks : float
        Coeficiente do reservatório lento [0-1].
    return_state : bool, optional
        Se True, retorna também os estados finais.
    return_fluxes : bool, optional
        Se True, retorna também as séries internas passo a passo
        (H, Xs, ER, Qq, Qs).

    Returns
    -------
    Q : numpy.ndarray
        Vazão simulada em lâmina [mm/dia].

    Opcionalmente:
        ``Q, estados_finais`` se ``return_state=True``;
        ``Q, fluxos`` se ``return_fluxes=True``;
        ``Q, fluxos, estados_finais`` se ambos forem True.
    """
    P, ETP = _forcing_from_series(P, ETP)

    Cmax = float(Cmax)
    Bexp = float(Bexp)
    Alpha = float(Alpha)
    Kq = float(Kq)
    Ks = float(Ks)

    _require_positive(Cmax, "Cmax")

    if Bexp < 0.0:
        raise ValueError("Bexp deve ser maior ou igual a zero.")

    if not 0.0 <= Alpha <= 1.0:
        raise ValueError("Alpha deve estar no intervalo [0, 1].")

    if not 0.0 <= Kq <= 1.0:
        raise ValueError("Kq deve estar no intervalo [0, 1].")

    if not 0.0 <= Ks <= 1.0:
        raise ValueError("Ks deve estar no intervalo [0, 1].")

    Cpar = Cmax / (1.0 + Bexp)

    states = {} if states is None else dict(states)

    H = float(states.get("soil_height", 0.0))

    quick = np.asarray(
        states.get("quick_stores", [0.0, 0.0, 0.0]),
        dtype=float,
    ).copy()

    if len(quick) != 3:
        raise ValueError(
            "quick_stores deve conter exatamente 3 valores."
        )

    Xs = float(states.get("slow_store", 0.0))

    Q = np.zeros(len(P), dtype=float)

    if return_fluxes:
        n = len(P)
        H_ = np.zeros(n, dtype=float)
        Xs_ = np.zeros(n, dtype=float)
        ER_ = np.zeros(n, dtype=float)
        Qq_ = np.zeros(n, dtype=float)
        Qs_ = np.zeros(n, dtype=float)

    for i in range(len(P)):
        Pi = P[i]
        EPi = ETP[i]

        # 1. Armazenamento inicial do solo
        H = min(max(H, 0.0), Cmax)

        Cbeg = Cpar * (
            1.0
            - (1.0 - H / Cmax) ** (1.0 + Bexp)
        )

        # 2. Excesso de precipitação
        ER1 = max(0.0, Pi + H - Cmax)
        Pnet = Pi - ER1

        Hint = min(Cmax, H + Pnet)

        Cint = Cpar * (
            1.0
            - (1.0 - Hint / Cmax) ** (1.0 + Bexp)
        )

        ER2 = max(
            0.0,
            Pnet + Cbeg - Cint,
        )

        ER = ER1 + ER2

        # 3. Evapotranspiração
        ET = min(
            Cint,
            EPi * Cint / Cpar,
        )

        Cend = max(
            0.0,
            Cint - ET,
        )

        if Cend >= Cpar:
            H = Cmax
        else:
            H = Cmax * (
                1.0
                - (
                    1.0 - Cend / Cpar
                ) ** (
                    1.0 / (1.0 + Bexp)
                )
            )

        # 4. Separação rápida/lenta
        UQ = Alpha * ER
        US = (1.0 - Alpha) * ER

        # 5. Três reservatórios rápidos
        inflow = UQ

        for j in range(3):
            outflow = Kq * quick[j]
            quick[j] = (
                quick[j]
                + inflow
                - outflow
            )
            inflow = outflow

        Qq = inflow

        # 6. Reservatório lento
        Qs = Ks * Xs
        Xs = Xs + US - Qs

        # 7. Vazão total
        Q[i] = Qq + Qs

        if return_fluxes:
            H_[i], Xs_[i] = H, Xs
            ER_[i] = ER
            Qq_[i], Qs_[i] = Qq, Qs

    final_state = {
        "soil_height": H,
        "quick_stores": quick.copy(),
        "slow_store": Xs,
    }

    fluxes = {
        "H": H_, "Xs": Xs_, "ER": ER_, "Qq": Qq_, "Qs": Qs_,
    } if return_fluxes else None

    if return_fluxes and return_state:
        return Q, fluxes, final_state
    if return_fluxes:
        return Q, fluxes
    if return_state:
        return Q, final_state
    return Q


# =============================================================================
# SMAP - DIÁRIO E MENSAL
# =============================================================================

def smap(
    P,
    ETP,
    Str,
    K2t,
    Crec,
    Ai,
    Capc,
    Kkt,
    Pcof=1.0,
    states=None,
    return_state=False,
    return_fluxes=False,
):
    """Modelo SMAP diário.

    Parameters
    ----------
    P, ETP : array-like
        Entradas em mm/dia.
    Str : float
        Capacidade de saturação do solo [mm].
    K2t : float
        Tempo de meia-vida do reservatório superficial [dias].
    Crec : float
        Coeficiente de recarga [%].
    Ai : float
        Abstração inicial [mm].
    Capc : float
        Capacidade de campo [%].
    Kkt : float
        Tempo de meia-vida do reservatório subterrâneo [dias].
    Pcof : float, optional
        Fator multiplicativo de correção da precipitação.
    return_state : bool, optional
        Se True, retorna também os estados finais.
    return_fluxes : bool, optional
        Se True, retorna também as séries internas passo a passo
        (Rsolo, Rsup, Rsub, Es, Er, Rec, Ed, Eb).

    Returns
    -------
    Q : numpy.ndarray
        Vazão simulada em lâmina [mm/dia].

    Opcionalmente:
        ``Q, estados_finais`` se ``return_state=True``;
        ``Q, fluxos`` se ``return_fluxes=True``;
        ``Q, fluxos, estados_finais`` se ambos forem True.
    """
    P, ETP = _forcing_from_series(P, ETP)

    Str = float(Str)
    K2t = float(K2t)
    Crec = float(Crec) / 100.0
    Ai = float(Ai)
    Capc = float(Capc) / 100.0
    Kkt = float(Kkt)
    Pcof = float(Pcof)

    _require_positive(Str, "Str")
    _require_positive(K2t, "K2t")
    _require_positive(Kkt, "Kkt")

    if not 0.0 <= Capc <= 1.0:
        raise ValueError(
            "Capc deve ser informado entre 0 e 100%."
        )

    if Crec < 0.0:
        raise ValueError(
            "Crec deve ser não negativo."
        )

    # Coeficientes de recessão
    K2 = 0.5 ** (1.0 / K2t)
    Kk = 0.5 ** (1.0 / Kkt)

    states = {} if states is None else dict(states)

    Tuin = float(
        states.get("Tuin", 0.50)
    )

    Rsolo = float(
        states.get(
            "soil_store",
            Tuin * Str,
        )
    )

    Rsup = float(
        states.get(
            "surface_store",
            0.0,
        )
    )

    if "groundwater_store" in states:
        Rsub = float(
            states["groundwater_store"]
        )
    else:
        Ebin = float(
            states.get("Ebin", 0.0)
        )

        Rsub = (
            Ebin / (1.0 - Kk)
            if (1.0 - Kk) > 0.0
            else 0.0
        )

    Q = np.zeros(len(P), dtype=float)

    if return_fluxes:
        n = len(P)
        Rsolo_ = np.zeros(n, dtype=float)
        Rsup_ = np.zeros(n, dtype=float)
        Rsub_ = np.zeros(n, dtype=float)
        Es_ = np.zeros(n, dtype=float)
        Er_ = np.zeros(n, dtype=float)
        Rec_ = np.zeros(n, dtype=float)
        Ed_ = np.zeros(n, dtype=float)
        Eb_ = np.zeros(n, dtype=float)

    for i in range(len(P)):
        Pi = P[i] * Pcof
        ETPi = ETP[i]

        Rsolo = min(
            max(Rsolo, 0.0),
            Str,
        )

        Tu = Rsolo / Str

        # 1. Escoamento superficial
        if Pi > Ai:
            deficit = Str - Rsolo

            Es = (
                (Pi - Ai) ** 2
                / (
                    Pi
                    - Ai
                    + deficit
                )
            )
        else:
            Es = 0.0

        # 2. Evapotranspiração real
        Psolo = Pi - Es

        if Psolo >= ETPi:
            Er = ETPi
        else:
            Er = (
                Psolo
                + (ETPi - Psolo)
                * Tu
            )

        # 3. Recarga subterrânea
        Rcapc = Capc * Str

        if Rsolo > Rcapc:
            Rec = (
                Crec
                * Tu
                * (Rsolo - Rcapc)
            )
        else:
            Rec = 0.0

        # 4. Balanço do solo
        Rsolo = (
            Rsolo
            + Pi
            - Es
            - Er
            - Rec
        )

        if Rsolo > Str:
            Es = Es + (
                Rsolo - Str
            )
            Rsolo = Str

        Rsolo = max(
            Rsolo,
            0.0,
        )

        # 5. Reservatório superficial
        Rsup = Rsup + Es

        Ed = Rsup * (
            1.0 - K2
        )

        Rsup = Rsup - Ed

        # 6. Reservatório subterrâneo
        Eb = Rsub * (
            1.0 - Kk
        )

        Rsub = (
            Rsub
            + Rec
            - Eb
        )

        # 7. Vazão total
        Q[i] = Ed + Eb

        if return_fluxes:
            Rsolo_[i], Rsup_[i], Rsub_[i] = Rsolo, Rsup, Rsub
            Es_[i], Er_[i], Rec_[i] = Es, Er, Rec
            Ed_[i], Eb_[i] = Ed, Eb

    final_state = {
        "soil_store": Rsolo,
        "surface_store": Rsup,
        "groundwater_store": Rsub,
    }

    fluxes = {
        "Rsolo": Rsolo_, "Rsup": Rsup_, "Rsub": Rsub_,
        "Es": Es_, "Er": Er_, "Rec": Rec_, "Ed": Ed_, "Eb": Eb_,
    } if return_fluxes else None

    if return_fluxes and return_state:
        return Q, fluxes, final_state
    if return_fluxes:
        return Q, fluxes
    if return_state:
        return Q, final_state
    return Q


# Alias didático explícito.
smap_daily = smap


def smap_monthly(
    P,
    ETP,
    Sat,
    Pes,
    Crec,
    Kkt,
    Pcof=1.0,
    Ecof=1.0,
    states=None,
    return_state=False,
    return_fluxes=False,
):
    """Modelo SMAP mensal.

    Parameters
    ----------
    P, ETP : array-like
        Entradas em mm/mês.
    Sat : float
        Capacidade de saturação do solo [mm].
    Pes : float
        Expoente do escoamento superficial [-].
    Crec : float
        Coeficiente de recarga [%].
    Kkt : float
        Tempo de meia-vida do reservatório subterrâneo [meses].
    Pcof, Ecof : float, optional
        Fatores multiplicativos de correção de P e ETP.
    return_state : bool, optional
        Se True, retorna também os estados finais.
    return_fluxes : bool, optional
        Se True, retorna também as séries internas passo a passo
        (Rsolo, Rsub, Es, Er, Rec, Eb).

    Returns
    -------
    Q : numpy.ndarray
        Vazão simulada em lâmina [mm/mês].

    Opcionalmente:
        ``Q, estados_finais`` se ``return_state=True``;
        ``Q, fluxos`` se ``return_fluxes=True``;
        ``Q, fluxos, estados_finais`` se ambos forem True.
    """
    P, ETP = _forcing_from_series(P, ETP)

    Sat = float(Sat)
    Pes = float(Pes)
    Crec = float(Crec) / 100.0
    Kkt = float(Kkt)
    Pcof = float(Pcof)
    Ecof = float(Ecof)

    _require_positive(Sat, "Sat")
    _require_positive(Kkt, "Kkt")

    if Pes < 0.0:
        raise ValueError(
            "Pes deve ser não negativo."
        )

    if Crec < 0.0:
        raise ValueError(
            "Crec deve ser não negativo."
        )

    Kk = 0.5 ** (1.0 / Kkt)

    states = {} if states is None else dict(states)

    Tuin = float(
        states.get("Tuin", 0.50)
    )

    Rsolo = float(
        states.get(
            "soil_store",
            Tuin * Sat,
        )
    )

    if "groundwater_store" in states:
        Rsub = float(
            states["groundwater_store"]
        )
    else:
        Ebin = float(
            states.get("Ebin", 0.0)
        )

        Rsub = (
            Ebin / (1.0 - Kk)
            if (1.0 - Kk) > 0.0
            else 0.0
        )

    Q = np.zeros(len(P), dtype=float)

    if return_fluxes:
        n = len(P)
        Rsolo_ = np.zeros(n, dtype=float)
        Rsub_ = np.zeros(n, dtype=float)
        Es_ = np.zeros(n, dtype=float)
        Er_ = np.zeros(n, dtype=float)
        Rec_ = np.zeros(n, dtype=float)
        Eb_ = np.zeros(n, dtype=float)

    for i in range(len(P)):
        Pi = P[i] * Pcof
        ETPi = ETP[i] * Ecof

        Rsolo = min(
            max(Rsolo, 0.0),
            Sat,
        )

        Tu = Rsolo / Sat

        # 1. Pré-atualização do teor de umidade
        dRsolo = 0.5 * (
            Pi
            - Pi * Tu ** Pes
            - ETPi * Tu
            - Rsolo * Crec * Tu ** 4.0
        )

        Tu = min(
            max(
                (Rsolo + dRsolo) / Sat,
                0.0,
            ),
            1.0,
        )

        # 2. Funções de transferência
        Es = Pi * Tu ** Pes
        Er = ETPi * Tu
        Rec = Rsolo * Crec * Tu ** 4.0
        Eb = Rsub * (1.0 - Kk)

        # 3. Balanço dos reservatórios
        Rsolo = (
            Rsolo
            + Pi
            - Es
            - Er
            - Rec
        )

        if Rsolo > Sat:
            Es = Es + (
                Rsolo - Sat
            )
            Rsolo = Sat

        Rsolo = max(
            Rsolo,
            0.0,
        )

        Rsub = max(
            0.0,
            Rsub + Rec - Eb,
        )

        # 4. Vazão total
        Q[i] = Es + Eb

        if return_fluxes:
            Rsolo_[i], Rsub_[i] = Rsolo, Rsub
            Es_[i], Er_[i], Rec_[i] = Es, Er, Rec
            Eb_[i] = Eb

    final_state = {
        "soil_store": Rsolo,
        "groundwater_store": Rsub,
    }

    fluxes = {
        "Rsolo": Rsolo_, "Rsub": Rsub_,
        "Es": Es_, "Er": Er_, "Rec": Rec_, "Eb": Eb_,
    } if return_fluxes else None

    if return_fluxes and return_state:
        return Q, fluxes, final_state
    if return_fluxes:
        return Q, fluxes
    if return_state:
        return Q, final_state
    return Q

# =============================================================================
# MODHAC
# =============================================================================
def modhac(
    P,
    ETP,
    RSPX,
    RSSX,
    RSBX,
    RSBF,
    IMAX,
    IMIN,
    IDEC,
    ASP,
    ASS,
    ASB,
    PRED,
    CEVA,
    states=None,
    return_state=False,
    return_fluxes=False,
):
    """Modelo MODHAC em simulação contínua.

    O modelo é executado sequencialmente para toda a série temporal. Os estados
    finais de cada passo tornam-se automaticamente os estados iniciais do passo
    seguinte.

    Parameters
    ----------
    P : array-like
        Precipitação por passo de tempo [mm].
    ETP : array-like
        Evapotranspiração potencial por passo de tempo [mm].
    RSPX : float
        Capacidade do reservatório superficial [mm].
    RSSX : float
        Capacidade do reservatório subsuperficial [mm].
    RSBX : float
        Capacidade total do reservatório subterrâneo [mm].
    RSBF : float
        Nível mínimo do reservatório subterrâneo para contribuição ao
        escoamento de base, expresso em % de RSBX.
    IMAX : float
        Infiltração máxima [mm/passo].
    IMIN : float
        Infiltração mínima [mm/passo].
    IDEC : float
        Coeficiente de infiltração intermediária.
    ASP : float
        Coeficiente de percolação do reservatório superficial.
    ASS : float
        Coeficiente de recessão do reservatório subsuperficial.
    ASB : float
        Coeficiente de recessão/percolação do reservatório subterrâneo.
    PRED : float
        Coeficiente de alteração da precipitação. Use 999 para desativar.
    CEVA : float
        Coeficiente de evaporação do reservatório subsuperficial.
    states : dict, optional
        Estados iniciais:
        ``{'rsp': ..., 'rss': ..., 'rsb': ...}``.
        Se omitido, os três reservatórios iniciam em zero.
    return_state : bool, optional
        Se True, retorna também os estados finais.
    return_fluxes : bool, optional
        Se True, retorna também as séries dos fluxos internos.

    Returns
    -------
    Q : numpy.ndarray
        Escoamento total simulado em lâmina [mm/passo].

    Opcionalmente:
        ``Q, estados_finais`` se ``return_state=True``;
        ``Q, fluxos`` se ``return_fluxes=True``;
        ``Q, fluxos, estados_finais`` se ambos forem True.
    """
    P, ETP = _forcing_from_series(P, ETP)

    RSPX = float(RSPX)
    RSSX = float(RSSX)
    RSBX = float(RSBX)
    RSBF_abs = float(RSBF) * RSBX / 100.0
    IMAX = float(IMAX)
    IMIN = float(IMIN)
    IDEC = float(IDEC)
    ASP = float(ASP)
    ASS = float(ASS)
    ASB = float(ASB)
    PRED = float(PRED)
    CEVA = float(CEVA)

    states = {} if states is None else dict(states)
    rsp = float(states.get("rsp", 0.0))
    rss = float(states.get("rss", 0.0))
    rsb = float(states.get("rsb", 0.0))

    n = len(P)

    Q = np.zeros(n, dtype=float)

    ES = np.zeros(n, dtype=float)
    EB = np.zeros(n, dtype=float)
    ESP = np.zeros(n, dtype=float)
    ESS = np.zeros(n, dtype=float)
    ETR = np.zeros(n, dtype=float)
    VBF = np.zeros(n, dtype=float)

    RSP = np.zeros(n, dtype=float)
    RSS = np.zeros(n, dtype=float)
    RSB = np.zeros(n, dtype=float)

    for i in range(n):
        p_obs = float(P[i])
        etp = float(ETP[i])

        # --------------------------------------------------------------
        # 1. Correção opcional da precipitação
        # --------------------------------------------------------------
        if PRED != 999.0 and p_obs != 0.0:
            if PRED <= 0.0:
                correction = 1.0 + math.exp((PRED / 100.0) * p_obs)
            else:
                correction = 1.0 - math.exp((-PRED / 100.0) * p_obs)
            p = p_obs * correction
        else:
            p = p_obs

        # --------------------------------------------------------------
        # 2. Fase de umedecimento
        # --------------------------------------------------------------
        if p > etp:
            remaining_p = p - etp
            actual_et = etp
            evap_surface = 0.0
            evap_subsurface = 0.0

            surface_deficit = RSPX - rsp

            if remaining_p <= surface_deficit:
                rsp += remaining_p
                remaining_p = 0.0
                quick_runoff = 0.0
            else:
                remaining_p -= surface_deficit
                rsp = RSPX
                quick_runoff = 0.0

            # Percolação do reservatório superficial.
            infiltration_from_surface = rsp * (1.0 - math.exp(-ASP))
            rsp -= infiltration_from_surface

            # Partição da precipitação remanescente.
            if remaining_p <= IMIN:
                total_infiltration = remaining_p + infiltration_from_surface
                quick_runoff = 0.0

            elif IDEC >= 1.0:
                quick_runoff = (remaining_p - IMIN) * IDEC
                total_infiltration = (
                    remaining_p
                    - quick_runoff
                    + infiltration_from_surface
                )

            else:
                pmax = (IMAX - IMIN) / (1.0 - IDEC) + IMIN

                if remaining_p <= pmax:
                    quick_runoff = (remaining_p - IMIN) * IDEC
                    total_infiltration = (
                        remaining_p
                        - quick_runoff
                        + infiltration_from_surface
                    )
                else:
                    quick_runoff = remaining_p - IMAX
                    total_infiltration = IMAX + infiltration_from_surface

            # Recarga do reservatório subterrâneo.
            if RSBX <= 0.0:
                groundwater_recharge = 0.0
            else:
                denominator = total_infiltration + RSBX - rsb

                groundwater_recharge = (
                    total_infiltration
                    * (RSBX - rsb)
                    / denominator
                    if denominator != 0.0
                    else 0.0
                )

                rsb += groundwater_recharge

            # Reservatório subsuperficial.
            subsurface_inflow = total_infiltration - groundwater_recharge
            rss += subsurface_inflow

            if rss > RSSX:
                overflow = rss - RSSX
                rss = RSSX
                quick_runoff += overflow

        # --------------------------------------------------------------
        # 3. Fase de ressecamento
        # --------------------------------------------------------------
        else:
            actual_et = p
            quick_runoff = 0.0

            et_deficit = etp - actual_et

            if et_deficit > rsp:
                evap_surface = rsp
                actual_et += evap_surface
                rsp = 0.0
                et_deficit = etp - actual_et

            else:
                rsp -= et_deficit
                evap_surface = et_deficit
                actual_et = etp
                et_deficit = 0.0

                infiltration_from_surface = rsp * (
                    1.0 - math.exp(-ASP)
                )

                rsp -= infiltration_from_surface
                rss += infiltration_from_surface

                if rss > RSSX:
                    excess = rss - RSSX
                    rss = RSSX
                    rsp += excess

            if rss > 0.0:
                evap_subsurface = (
                    et_deficit
                    * (
                        CEVA
                        + (1.0 - CEVA)
                        * (rss / RSSX)
                    )
                    if RSSX > 0.0
                    else 0.0
                )

                evap_subsurface = min(
                    max(evap_subsurface, 0.0),
                    rss,
                )

                rss -= evap_subsurface
                actual_et += evap_subsurface

            else:
                evap_subsurface = 0.0

        # --------------------------------------------------------------
        # 4. Escoamento superficial
        # --------------------------------------------------------------
        surface_flow = quick_runoff

        # --------------------------------------------------------------
        # 5. Escoamento de base subsuperficial
        # --------------------------------------------------------------
        subsurface_baseflow = rss * (
            1.0 - math.exp(-ASS)
        )

        # --------------------------------------------------------------
        # 6. Reservatório subterrâneo
        # --------------------------------------------------------------
        if RSBX != 0.0:
            groundwater_fraction = (
                1.0 - math.exp(-ASB)
            )

            if rsb > RSBF_abs:
                groundwater_baseflow = (
                    rsb - RSBF_abs
                ) * groundwater_fraction

                deep_infiltration = (
                    RSBF_abs
                    * groundwater_fraction
                )
            else:
                groundwater_baseflow = 0.0

                deep_infiltration = (
                    rsb
                    * groundwater_fraction
                )
        else:
            groundwater_baseflow = 0.0
            deep_infiltration = 0.0

        base_flow = (
            subsurface_baseflow
            + groundwater_baseflow
        )

        rss -= subsurface_baseflow
        rsb -= (
            groundwater_baseflow
            + deep_infiltration
        )

        # --------------------------------------------------------------
        # 7. Vazão total e armazenamento dos resultados
        # --------------------------------------------------------------
        Q[i] = surface_flow + base_flow

        ES[i] = surface_flow
        EB[i] = base_flow
        ESP[i] = evap_surface
        ESS[i] = evap_subsurface
        ETR[i] = actual_et
        VBF[i] = deep_infiltration

        RSP[i] = rsp
        RSS[i] = rss
        RSB[i] = rsb

    final_state = {
        "rsp": rsp,
        "rss": rss,
        "rsb": rsb,
    }

    fluxes = {
        "ES": ES,
        "EB": EB,
        "ESP": ESP,
        "ESS": ESS,
        "ETR": ETR,
        "VBF": VBF,
        "RSP": RSP,
        "RSS": RSS,
        "RSB": RSB,
    }

    if return_fluxes and return_state:
        return Q, fluxes, final_state

    if return_fluxes:
        return Q, fluxes

    if return_state:
        return Q, final_state

    return Q


# =============================================================================
# MÉTRICAS / FUNÇÕES-OBJETIVO
# =============================================================================
def _paired_series(simulation, evaluation):
    sim = np.asarray(simulation, dtype=float)
    obs = np.asarray(evaluation, dtype=float)
    if sim.shape != obs.shape:
        raise ValueError("simulation e evaluation devem possuir a mesma forma.")
    mask = np.isfinite(sim) & np.isfinite(obs)
    if not np.any(mask):
        raise ValueError("Não há pares válidos para calcular a métrica.")
    return sim[mask], obs[mask]


def nse(simulation, evaluation):
    """Nash-Sutcliffe Efficiency."""
    sim, obs = _paired_series(simulation, evaluation)
    denominator = np.sum((obs - np.mean(obs)) ** 2)
    if denominator == 0.0:
        return np.nan
    return 1.0 - np.sum((obs - sim) ** 2) / denominator


def kge(simulation, evaluation, return_components=False):
    """Kling-Gupta Efficiency original (Gupta et al., 2009)."""
    sim, obs = _paired_series(simulation, evaluation)
    if len(sim) < 2:
        return np.nan

    r = np.corrcoef(sim, obs)[0, 1]
    obs_std = np.std(obs)
    obs_mean = np.mean(obs)
    if obs_std == 0.0 or obs_mean == 0.0:
        return np.nan

    alpha = np.std(sim) / obs_std
    beta = np.mean(sim) / obs_mean
    value = 1.0 - math.sqrt((r - 1.0) ** 2 + (alpha - 1.0) ** 2 + (beta - 1.0) ** 2)

    return (value, r, alpha, beta) if return_components else value


def kgeprime(simulation, evaluation, return_components=False):
    """KGE modificado (Kling et al., 2012)."""
    sim, obs = _paired_series(simulation, evaluation)
    if len(sim) < 2:
        return np.nan

    r = np.corrcoef(sim, obs)[0, 1]
    sim_mean, obs_mean = np.mean(sim), np.mean(obs)
    if sim_mean == 0.0 or obs_mean == 0.0:
        return np.nan

    cv_sim = np.std(sim) / sim_mean
    cv_obs = np.std(obs) / obs_mean
    if cv_obs == 0.0:
        return np.nan

    gamma = cv_sim / cv_obs
    beta = sim_mean / obs_mean
    value = 1.0 - math.sqrt((r - 1.0) ** 2 + (gamma - 1.0) ** 2 + (beta - 1.0) ** 2)

    return (value, r, gamma, beta) if return_components else value


def kgenp(simulation, evaluation, return_components=False):
    """KGE não paramétrico (Pool et al., 2018)."""
    sim, obs = _paired_series(simulation, evaluation)
    n = len(sim)
    if n < 2 or np.mean(sim) == 0.0 or np.mean(obs) == 0.0:
        return np.nan

    sim_rank = np.argsort(np.argsort(sim)).astype(float)
    obs_rank = np.argsort(np.argsort(obs)).astype(float)
    r = np.corrcoef(sim_rank, obs_rank)[0, 1]

    sim_fdc = np.sort(sim / (n * np.mean(sim)))
    obs_fdc = np.sort(obs / (n * np.mean(obs)))
    alpha = 1.0 - 0.5 * np.sum(np.abs(sim_fdc - obs_fdc))
    beta = np.mean(sim) / np.mean(obs)

    value = 1.0 - math.sqrt((r - 1.0) ** 2 + (alpha - 1.0) ** 2 + (beta - 1.0) ** 2)
    return (value, r, alpha, beta) if return_components else value


def rmse(simulation, evaluation):
    sim, obs = _paired_series(simulation, evaluation)
    return math.sqrt(np.mean((obs - sim) ** 2))


def mare(simulation, evaluation):
    """Mean Absolute Relative Error usando o volume observado como denominador."""
    sim, obs = _paired_series(simulation, evaluation)
    denominator = np.sum(obs)
    return np.sum(np.abs(obs - sim)) / denominator if denominator != 0.0 else np.nan


def pbias(simulation, evaluation):
    """Percent Bias; positivo indica subestimação pela convenção original do arquivo."""
    sim, obs = _paired_series(simulation, evaluation)
    denominator = np.sum(obs)
    return 100.0 * np.sum(obs - sim) / denominator if denominator != 0.0 else np.nan


def nse_c2m(simulation, evaluation):
    return _bounded_efficiency(nse(simulation, evaluation))


def kge_c2m(simulation, evaluation):
    return _bounded_efficiency(kge(simulation, evaluation))


def kgeprime_c2m(simulation, evaluation):
    return _bounded_efficiency(kgeprime(simulation, evaluation))


def kgenp_c2m(simulation, evaluation):
    return _bounded_efficiency(kgenp(simulation, evaluation))

# =============================================================================
# CONVERSÃO DE UNIDADES
# =============================================================================

def mmday_to_m3s(Q_mm_day, area_km2):
    """Converte lâmina diária [mm/dia] para vazão [m³/s]."""
    Q = np.asarray(Q_mm_day, dtype=float)
    return Q * float(area_km2) * 1000.0 / 86400.0


def m3s_to_mmday(Q_m3s, area_km2):
    """Converte vazão [m³/s] para lâmina diária [mm/dia]."""
    Q = np.asarray(Q_m3s, dtype=float)
    return Q * 86400.0 / (float(area_km2) * 1000.0)


def mmmonth_to_m3s(Q_mm_month, area_km2, days_in_month=30.0):
    """Converte lâmina mensal [mm/mês] para vazão média [m³/s]."""
    Q = np.asarray(Q_mm_month, dtype=float)
    return (
        Q
        * float(area_km2)
        * 1000.0
        / (float(days_in_month) * 86400.0)
    )


def m3s_to_mmmonth(Q_m3s, area_km2, days_in_month=30.0):
    """Converte vazão média [m³/s] para lâmina mensal [mm/mês]."""
    Q = np.asarray(Q_m3s, dtype=float)
    return (
        Q
        * float(days_in_month)
        * 86400.0
        / (float(area_km2) * 1000.0)
    )

__all__ = [
    "gr1a",
    "gr2m",
    "gr4j",
    "hymod",
    "smap",
    "smap_daily",
    "smap_monthly",
    "modhac",
    "nse",
    "kge",
    "kgeprime",
    "kgenp",
    "rmse",
    "mare",
    "pbias",
    "nse_c2m",
    "kge_c2m",
    "kgeprime_c2m",
    "kgenp_c2m",
    "mmday_to_m3s",
    "m3s_to_mmday",
    "mmmonth_to_m3s",
    "m3s_to_mmmonth",
]
