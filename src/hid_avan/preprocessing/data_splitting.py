import numpy as np

def masks_temporal_splitting(df, warmup_frac, calibration_frac, gap_frac, validation2_frac, validation_with_oldest_data = False):

    n = len(df)

    if n < 2:
        raise ValueError("A série possui dados insuficientes.")

    fracoes = {
        "Aquecimento": warmup_frac,
        "Calibração": calibration_frac,
        "Gap": gap_frac,
        "Validação adicional": validation2_frac,
    }

    for nome, valor in fracoes.items():
        if valor < 0 or valor > 1:
            raise ValueError(f"{nome} deve estar entre 0 e 1.")

    soma_fixa = warmup_frac + calibration_frac + gap_frac + validation2_frac

    if soma_fixa > 1.0 + 1e-12:
        raise ValueError("As soma das frações não pode exceder 1.")

    FRACAO_VAL = max(0.0, 1.0 - soma_fixa)

    # Caso especial: calibração com toda a série
    if np.isclose(calibration_frac, 1.0):
        if not (
            np.isclose(warmup_frac, 0.0)
            and np.isclose(gap_frac, 0.0)
            and np.isclose(validation2_frac, 0.0)
        ):
            raise ValueError(
                "Para calibrar com toda a série (calibration_frac = 1.), use 'warmup_frac' = 'gap_frac' = 'validation2_frac' = 0."
            )

    n_aquec = int(np.floor(warmup_frac * n))
    n_cal   = int(np.floor(calibration_frac   * n))
    n_gap   = int(np.floor(gap_frac   * n))
    n_val2  = int(np.floor(validation2_frac  * n))

    # A validação principal recebe todo o restante, evitando perda
    # de observações por arredondamento.
    n_val = n - n_aquec - n_cal - n_gap - n_val2

    if n_cal < 2:
        raise ValueError("A calibração precisa conter pelo menos 2 observações.")

    if n_val < 0:
        raise ValueError("As frações especificadas excedem o tamanho da série.")

    mascara_aquec = np.zeros(n, dtype=bool)
    mascara_cal   = np.zeros(n, dtype=bool)
    mascara_gap   = np.zeros(n, dtype=bool)
    mascara_val   = np.zeros(n, dtype=bool)
    mascara_val2  = np.zeros(n, dtype=bool)

    if not validation_with_oldest_data:
        # Aquecimento | Calibração | Gap | Validação | Validação 2
        i0 = 0
        i1 = i0 + n_aquec
        i2 = i1 + n_cal
        i3 = i2 + n_gap
        i4 = i3 + n_val
        i5 = i4 + n_val2

        mascara_aquec[i0:i1] = True
        mascara_cal[i1:i2]   = True
        mascara_gap[i2:i3]   = True
        mascara_val[i3:i4]   = True
        mascara_val2[i4:i5]  = True

    else:
        # Validação 2 | Validação | Gap | Aquecimento | Calibração
        i0 = 0
        i1 = i0 + n_val2
        i2 = i1 + n_val
        i3 = i2 + n_gap
        i4 = i3 + n_aquec
        i5 = i4 + n_cal

        mascara_val2[i0:i1]  = True
        mascara_val[i1:i2]   = True
        mascara_gap[i2:i3]   = True
        mascara_aquec[i3:i4] = True
        mascara_cal[i4:i5]   = True

    dict_masks = {
        'Warm-up':mascara_aquec,
        'Calibration':mascara_cal,
        'Gap':mascara_gap,
        'Validation':mascara_val,
        'Validation 2':mascara_val2
    }

    return dict_masks