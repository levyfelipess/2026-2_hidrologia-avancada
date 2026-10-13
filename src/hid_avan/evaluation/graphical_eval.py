import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

from ..preprocessing.data_splitting import masks_temporal_splitting

def evaluate_with_data_splitting(df_obs, Q_pred, warmup_frac, calibration_frac, gap_frac, validation2_frac, validation_with_oldest_data = False):
    """
    Plota toda a série e destaca os períodos experimentais.
    'periodos' é uma lista de tuplas:
    (nome, mascara, cor)
    """
    datas = df_obs.loc[:, 'DATE']
    Pserie = df_obs.loc[:, 'P'].values
    obs = df_obs.loc[:, 'Q'].values
    sim = np.asarray(Q_pred, dtype=float)

    periodos_dict = masks_temporal_splitting(df=df_obs,
                                             warmup_frac=warmup_frac,
                                             calibration_frac=calibration_frac,
                                             gap_frac=gap_frac,
                                             validation2_frac=validation2_frac,
                                             validation_with_oldest_data=validation_with_oldest_data)
    periodos = list(periodos_dict.items())

    fig, (ax_p, ax_q) = plt.subplots(
        2, 1,
        figsize=(10, 5),
        sharex=True,
        gridspec_kw={"height_ratios": [1.0, 2.0]},
        constrained_layout=True
    )

    # Precipitação
    if len(datas) > 1:
        dt_dias = np.diff(datas.values).astype("timedelta64[D]").astype(int)
        passo_dias = np.nanmedian(dt_dias[dt_dias > 0]) if np.any(dt_dias > 0) else 1
    else:
        passo_dias = 1

    largura_barra = max(0.8, 0.75 * passo_dias)

    ax_p.bar(
        datas, Pserie,
        width=largura_barra,
        alpha=1., color='orange', lw=0
    )
    ax_p.invert_yaxis()
    ax_p.set_ylabel("P")
    ax_p.set_title("Precipitação", loc="left", fontweight="bold")
    ax_p.xaxis.set_visible(False)

    # Vazões
    sns.lineplot(ax=ax_q, x=datas, y=obs, label="Observada", lw=2., ls='-', color='blue')
    sns.lineplot(ax=ax_q, x=datas, y=sim, label="Predita", lw=1., ls='-', color='red')
    ax_q.set_ylabel("Q")
    ax_q.set_xlabel("Tempo")
    ax_q.set_title(
        "Hidrograma e períodos experimentais",
        loc="left", fontweight="bold"
    )
    ax_q.legend(frameon=False, ncol=2, loc="upper left")

    # Faixas dos períodos
    handles_periodos = []

    for nome, mascara in periodos:
        mascara = np.asarray(mascara, dtype=bool)

        if mascara.sum() == 0:
            continue

        idx = np.where(mascara)[0]
        inicio = datas.iloc[idx[0]]
        fim = datas.iloc[idx[-1]]

        # extensão até aproximadamente o próximo passo para preencher
        if len(datas) > 1:
            delta = pd.to_timedelta(passo_dias, unit="D")
        else:
            delta = pd.Timedelta(days=1)

        fim_span = fim + delta

        #ax_p.axvspan(inicio, fim_span,
        #           alpha=0.10,
        #           #color=cor,
        #           zorder=0)

        # linha vertical no início de cada bloco
        ax_p.axvline(inicio, alpha=0.75, linewidth=1.0, color='black', zorder=1)
        ax_q.axvline(inicio, alpha=0.75, linewidth=1.0, color='black', zorder=1)
        ax_q.set_ylim(-10, None)

        # rótulo dos períodos — abaixo do título "Precipitação"
        meio = inicio + (fim_span - inicio) / 2
        
        ax_p.text(
            meio,
            0.94,  # posição vertical dentro do painel
            nome,
            transform=ax_p.get_xaxis_transform(),
            ha="center",
            va="top",
            fontsize=8,
            fontweight="bold",
            #color=cor,
            zorder=10
        )

        #handles_periodos.append(
        #    plt.Line2D([0], [0], lw=6, alpha=0.5, label=nome)#, color=cor)
        #)

    #if handles_periodos:
    #    ax_q.legend(
    #        handles=ax_q.get_legend_handles_labels()[0] + handles_periodos,
    #        labels=ax_q.get_legend_handles_labels()[1]
    #               + [h.get_label() for h in handles_periodos],
    #        frameon=False,
    #        ncol=min(4, 2 + len(handles_periodos)),
    #        loc="upper center",
    #        bbox_to_anchor=(0.5, -0.16)
    #    )

    sns.despine(ax=ax_q)
    sns.despine(ax=ax_p, top=False, bottom=True)
    ax_p.grid(lw=.5)
    ax_q.grid(lw=.5)

    return fig, ax_p, ax_q