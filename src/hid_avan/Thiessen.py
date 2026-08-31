"""Calcula Thiessen
Adaptado do cálculo do Thiessen versão Matlab (Autor: Dr. Eduardo Sávio)
"""

import numpy as np
import matplotlib.pyplot as plt
from functools import partial
from multiprocessing import Pool

import ThiessenUtils as tu

__author__ = 'Paulo Jarbas Camurca'
__credits__ = ['Fco Vasconcelos', 'Robson Franklin',
               'Marcelo Rodrigues', 'Alfredo Miranda',
               'Sullyandro Oliveira', 'Eduardo Sávio']
__license__ = 'GPL'
__version__ = '1.0'
__email__ = 'pjarbas312@gmail.com'


def thiessen(dados, lat: np.ndarray, lon: np.ndarray, pathshp: str,
             pf: float = -1, sep: str = ',', usenc: bool=False,
             min_est: int = 5, **kwargs):
    """
    Verifica a quantidade de postos dentro de uma região e calcula a média
    de uma variável ponderada pela zona de influência de cada posto.

    Exemplos de postos: Plataformas de Coleta de Dados (PCDs), Postos
                        Pluviométricos, grados em grades regulares, etc.

    Exemplos de regiões: bacias, estados, macro regiões do Ceará, etc.

    Exemplos de variáveis: precipitação, temperatura, etc.

    Os postos também podem ser interpretados como pontos de uma grade regular
    oriundos de modelos numéricos ou dados observados em grade. Muitas vezes, ao
    usar dados em grade, a maioria dos pontos da grade possuem a mesma
    influência dentro da região. Isso vai estar relacionado com tamanho da
    região e a quantidade de pontos em grade encontrados dentro da região.

    Relação entre pf, pf_max, min_est, pf_step:

    O paramentro 'pf' (pontos fora) é usado quando existe a necessidade de
    usar postos fora da região. Geralmente, isso acontece quando a região é
    muito pequena e não encontramos postos dentro da região. O 'pf' indica o
    início da zona (buffer) de procura  de postos fora da região. Na prática,
    um "retângulo" (na maioria das vezes) é usado para determinar a zona de
    procura. 'pf_max' indica o limite máximo do crecismento da zona de procura.
    'pf_step' indica o passo em que a zona de procura vai crescer.

    Exemplos:
        pf=1., pf_max=3., pf_step=0.5 :: A zona de procura iniciará 1 grau
        a partir dos limiares da região até 3 graus. Esse crescimento se dará
        a cada 0.5 graus.
        pf=0., pf_max=5., pf_step=1. :: A zona de procura iniciará 0 grau
        a partir dos limiares da região até 5 graus. Esse crescimento se dará
        a cada 1 grau.

    NOTA: 1 grau é aproximadamente 110km nas regiões próximo ao equardor.

    O min_est (mínimo de postos ou estações) indica o número mínimo de postos
    que, se encontrados durante o crescimento da zona de procura,
    irão interromper a procura de postos fora da região. É usado para
    determinar o número mínimo de postos que representa a região.

    Exemplos:
        pf=0., pf_max=5., pf_step=1., min_est=3 :: O crescimento da zona de
        procura será interrompido se o algoritimo encontrar 2 (dois) postos.

    :param dados:    - Matriz 3d com os dados que serão calculados as médias de thiessen. Formato dado[time, lat, lon]
    :param lat:      - Numpy array com as latitudes do arquivo nc
    :param lon:      - Numpy array com as longitudes do arquivo nc.
                       [obs: longitudes serão alteradas para o range de -180 a 180]
    :param pathshp:  - String com o caminho do shape do poligono. (ex: arquivo.dat)
    :param pf:       - Valor em graus para inserir pontos próximos que estão fora da bacia ( -1 não inclui ).
                       (ex: pf = 1, busca pontos fora a partir de 1 grau))
    :param sep:      - Separador do arquivo de pontos do shape. Delimitador default " , "
    :param usenc:    - Dados do tipo netcdf [True, False].
    :param figname:  - Opcional   Nome da figura (ex: thiessen.png)
    :param num_proc: - numero de processadores
    :return thimed:  - média de thiessen para a variável de entrada
    :return thimed, thimax, thimin, numpostos:  - média, max, min de thiessen para a variável de entrada e num de postos
    """

    figname = kwargs.get('figname', None)

    allvarreturn = kwargs.get('allvarreturn', False)

    # incremento para a busca de pontos fora
    pf_step: float = kwargs.get('pf_step', 0.5)

    # grau máximo de busca para pontos fora
    pf_max: float = kwargs.get('pf_max', 5.)

    num_proc: int = kwargs.get('num_proc', 1)

    if usenc:

        # Verifica fill value do arquivo nc
        dados[np.where(np.ma.getmask(dados))] = -999
        dados[np.where(np.isnan(dados))] = -999

        # Formatar a matriz dos dados para a forma: lon, lat, valores
        if dados.ndim > 2:

            nt, ny, nx = dados.shape
            dados = np.reshape(dados, [nt, nx*ny], order='F').T

        else:

            ny, nx = dados.shape
            dados = np.reshape(dados, [1, nx*ny], order='F').T

        px, py = np.meshgrid(lon, lat)

        px = np.reshape(px, [dados.shape[0], 1], order='F')
        py = np.reshape(py, [dados.shape[0], 1], order='F')

        #TODO: verificar isso
        px = np.where(px > 180, px - 360, px)

    else:

        # Converte valores nan para -999
        dados[np.where(np.isnan(dados))] = -999
        px = lon
        py = lat

    # Carregar dados da bacia
    bac = np.loadtxt(pathshp, delimiter=sep)
    bacx = bac[:, 0]
    bacy = bac[:, 1]

    Ndias = dados.shape[1]   # Número de passos de tempo

    numpostos = np.empty([0])

    # Verifica se os pontos estão dentro do polígono
    isin = tu.isinpoly(px, py, np.c_[bacx, bacy])

    if pf >= 0:

        # Usar pontos fora da bacia, a busca é feita aumentando-se de 0.5 grau
        # do grau indicado em pf
        tam_lim = np.arange(pf, pf_max, pf_step)

        # para debug
        # tam_lim = np.array([0., 0.5, 1.0, 1.5, 2., 5., 10])
        # tam_lim = np.array([1.0, 1.5, 2., 5., 10])

        # min_est = 5
        # vai parar quando atingir pf_max ou o minimo de pontos (min_est)

        # for lim_qtd in range(len(tam_lim)):

        for limite in tam_lim:

            # limite = tam_lim[lim_qtd]

            min_lon = np.min(bacx) - limite
            max_lon = np.max(bacx) + limite
            min_lat = np.min(bacy) - limite
            max_lat = np.max(bacy) + limite

            cc = np.array([max_lon, max_lon, min_lon, min_lon, max_lon])
            dd = np.array([max_lat, min_lat, min_lat, max_lat, max_lat])

            isin = tu.isinpoly(px, py, np.c_[cc, dd])

            # plota area de busca que está sendo criada.
            # manter para debug
            # plt.plot(px, py, 'o')
            # plt.plot(cc, dd, '-')
            # plt.plot(bacx, bacy, '-')
            # plt.show()
            # plt.close()

            if np.sum(isin) >= min_est:
                break

    # verifica se encontrou pontos
    if sum(isin) > 0:

        # Armazenamento da coordenadas dos postos que estão dentro do poligono
        px = px[np.where(isin == 1)]
        py = py[np.where(isin == 1)]

        # plota os pontos que foram encontrados.
        # manter para debug
        # plt.plot(px, py, 'o')
        # plt.plot(cc, dd, '-')
        # plt.plot(bacx, bacy, '-')
        # plt.show()
        # plt.close()

        # Armazenamento da chuva dos postos dentro da bacia
        dados = dados[np.where(isin == 1)]

        # Eliminar posto repetido
        [px, py, dados] = tu.corrposto(np.reshape(px, -1), np.reshape(py, -1), dados)

        dados = np.transpose(dados[:, 0:Ndias])

        # Chamada da rotina para o calculo da precipitacao media
        # Avalicao da disponibilidade dos dados
        [thi, ithi] = tu.dispthi(dados)

        [mthi, nthi] = thi.shape

        # Cria array vazio para ser concatenado nas linhas
        coef = np.array([]).reshape(0, nthi)

        pool = Pool(num_proc)

        try:
            alphas = pool.map(partial(tu.wrap_voronoi, px=px, py=py, bacx=bacx,
                              bacy=bacy, thi=thi, figname=figname),
                              list(range(mthi)))
        except Exception:
            pool.close()
            pool.join()
            raise
        else:
            pool.close()
            pool.join()

        for i in range(mthi):
            I = np.nonzero(thi[i, :])
            aux = np.zeros((nthi)) * 0.
            aux[I] = alphas[i]
            coef = np.vstack((coef, aux))

        if coef.size == 0:
            thimed = -999 * np.ones((1, Ndias))
            thimax = thimed
            thimin = thimed
            numpostos = -999 * np.ones((1, Ndias))
        else:
            thimed = tu.precimd(coef, ithi, dados)
            thimax = np.full((thimed.shape[0], 1), 0.)
            thimin = np.full((thimed.shape[0], 1), 0.)

            for dd in range(dados.shape[0]):

                paux = dados[dd, :]

                if sum(paux >= 0) > 0:
                    paux2 = paux[thi[ithi[dd], :] == 1]
                    thimax[dd, 0] = max(paux2)
                    thimin[dd, 0] = min(paux2)
                else:
                    thimax[dd, 0] = -999
                    thimin[dd, 0] = -999

        # Calcula o número de postos
        npostos = np.sum(thi, 1)

        for i in ithi:

            numpostos = np.append(numpostos, npostos[i])

    else:

        thimed = -999 * np.ones((1, Ndias))
        thimax = thimed
        thimin = thimed
        numpostos = -999 * np.ones((1, Ndias))

    thimed = thimed.reshape(1, -1)
    thimax = thimax.reshape(1, -1)
    thimin = thimin.reshape(1, -1)
    numpostos = numpostos.reshape(1, -1)

    if allvarreturn:
        return thimed, thimax, thimin, numpostos
    else:
        return thimed
