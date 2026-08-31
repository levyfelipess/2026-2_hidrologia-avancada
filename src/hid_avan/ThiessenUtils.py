# -*- coding: utf-8 -*-

__author__ = ["Paulo Jarbas Camurca"]
__credits__ = ["Eduardo Sávio"]
__license__ = "GPL"
__version__ = "1.0"
__email__ = "pjarbas312@gmail.com"

import matplotlib.path as mpltpath
import matplotlib.pylab as plt
import numpy as np
from scipy.spatial import Voronoi
from shapely.geometry import Polygon, Point, LineString


def isinpoly(lon, lat, polig):
    """
    Encontra se os pontos com coordenadas lon e lat
    estão dentro ou fora de um poligono.

    :param lon: - array com a longitude dos pontos
    :param lat: - array com a latitude dos pontos
    :param polig: -  matriz com os pontos do polígono
    :return isin: - array booleano com o valor 0 para pontos fora do polígono, 1 para
                    pontos dentro do polígono. Pontos encima do polígono recebem 0.
    """

    coords = np.c_[lon, lat]

    pth = mpltpath.Path(polig)
    isin = pth.contains_points(coords)

    isin = np.array(isin)
    isin = np.where(isin == True, 1., 0.)

    return isin


def corrposto(x, y, z):
    """
    Elimina posto de coordenadas repetidas
    versão de melhor performance que corrposto

    :param x: array de longitudes
    :type x: numpy array
    :param y: array de latitudes
    :type y: numpy array
    :param z: array de valores
    :type z: numpy array
    :return xnew: array de longitudes sem repetição
    :return ynew: array de latitudes sem repetição
    :return new_values: array de valores
    """

    d = {}

    for i, j, k in zip(x, y, z):

        if tuple((i, j)) not in d:
            d[(i, j)] = k

    # extract lon, lat, values from list
    new_values = []
    xnew = []
    ynew = []
    for xy, value in d.items():
        xnew.append(xy[0])
        ynew.append(xy[1])
        new_values.append(value)

    xnew = np.array(xnew)
    ynew = np.array(ynew)
    new_values = np.array(new_values)

    return xnew, ynew, new_values


def dispthi(dados):
    """
    Script para avaliar a disponibilidade de dados dos postos
    mapeia os dias que possuem falhas em cada posto.

    :param dados: array com os valores
    :type dados: numpy array
    :return thi: array com o mapa de ocorrência [0-falha, 1-valor]
    :return ithi: indices no array de dados para cada ocorrência

    exemplo:

    dados  = np.array(
            # posto1  posto2    ...
            [[-999.,   64.,    2.,    0.,  100.,   10.],   # dia1
            [-999.,    0.,    4., -999.,   10.,   10.],    # dia2
            [-999.,    0.,    8., -999., -999.,   10.],    #  .
            [-999.,    0.,    1., -999.,   30.,   10.],    #  .
            [-999.,    1.,    1.,    2.,    6.,   10.],    #  .
            [-999.,    8.,    5.,    3.,    2.,   10.]])

    [thi, ithi] = dispthi2(dados)

    # mapa com as ocorrências de falhas para cada posto
    thi =
         [[ 0.,  1.,  1.,  1.,  1.,  1.],
          [ 0.,  1.,  1.,  0.,  1.,  1.],
          [ 0.,  1.,  1.,  0.,  0.,  1.]]

    # índice da linha da matriz de dados com a ocorrência na matriz thi
    ithi =
          [[0],
           [1],
           [2],
           [1],
           [0],
           [0]]
    """

    m, n = dados.shape  # m = numero de dias, n = numero de postos

    # Dbin = np.where(dados >= 0, 1, 0) # Dbin = Determina os dias que possuem falhas em cada posto

    # modificado por Robson na versao matlab para permitir que seja calculado o Thiessen de
    # valores negativos
    Dbin = np.logical_and((np.where(np.isnan(dados) == True, 0, 1)),  (np.where(dados >= -998, 1, 0)))

    Dbin = np.where(Dbin == True, 1, 0)

    dic = {}
    ithi = np.zeros(m).astype(int)
    j = 0
    for i, key in enumerate(Dbin):

        if tuple(key) not in dic:
            dic[tuple(key)] = j
            ithi[i] = j
            j += 1
        else:
            ithi[i] = dic[tuple(key)]

    # obtem a chave do dicionário pelo valor
    thi = []
    for k in sorted(dic.values()):
        thi.append(list(dic.keys())[list(dic.values()).index(k)])

    thi = np.array(thi)

    return thi, ithi


def areaxy(x, y):
    """
    Calcula  a área de um polígono bidimensional
    formado pelos vértices com vetores de coordenadas x e y.
    O resultado é sensível à direção: a área é
    positiva se o contorno delimitador é antihorário
    e negativa se horário.

    Adaptado da versão em matlab de:
    Copyright (c) 1995 by Kirill K. Pankratov,
    kirill@plume.mit.edu.
    04/20/94, 05/20/95
    """
    # Calcula a integral de contorno Int -y*dx  (mesmo como Int x*dy).
    lx = len(x)-1
    x = x[1:lx+1]-x[0:lx]
    y = y[0:lx]+y[1:lx+1]
    a = -np.dot(x, y/2.)
    return a


def wrap_voronoi(i, px, py, bacx, bacy, thi, figname):
    I = np.nonzero(thi[i, :])
    alpha = voronoi(px[I], py[I], bacx, bacy, fign=figname, cont=i)
    return alpha


def voronoi(px, py, bacx, bacy, fign=False, cont=False):
    """
    Calculo dos pesos de cada ponto.

    verifica os pontos que estão dentro da bacia
    e retorna os coeficientes de influencia de cada um deles.

    :param px: - longitude dos pontos ( ex: lon dos postos)
    :param py: - latitude dos pontos ( ex: lat dos postos)
    :param bacx: - longitude do polígono ( ex: lon das bacias)
    :param bacy: - latitude do polígono ( ex: lat das bacias)
    :param fign: - nome da figura, opcional
    :param cont: - número da figura, opcional
    :return alpha: - retorna os pesos de cada ponto
    """

    # Cálculo do Thiessen
    mnx = np.min(np.append(px, bacx))
    mny = np.min(np.append(py, bacy))

    mxx = np.max(np.append(px, bacx))
    mxy = np.max(np.append(py, bacy))

    lx = mxx - mnx
    ly = mxy - mny

    # Definindo os limites da area de contorno
    mnx = mnx - 10*lx
    mny = mny - 10*ly
    mxx = mxx + 10*lx
    mxy = mxy + 10*ly

    mdx = (mnx + mxx)/2.
    mdy = (mny + mxy)/2.

    if fign:
        plt.plot(px, py, 'r^', bacx, bacy, 'b-')

    px = np.append(px, np.array([[mnx], [mdx], [mxx], [mdx]]))
    py = np.append(py, np.array([[mdy], [mxy], [mdy], [mny]]))

    vorpt = np.column_stack([px,py])

    # Gerar voronoi
    vor = Voronoi(vorpt)

    # formatar vetor de polígonos
    bacv = np.column_stack([bacx, bacy])

    area = np.empty(0)

    for region_idx in vor.point_region:
        region = vor.regions[region_idx]

        # Se a região é finita
        if region and -1 not in region:
            coords = np.empty(0)

            # Obtém coordenadas dos vértices
            for vertex_idx in region:

                coords = np.append(coords, vor.vertices[vertex_idx])

            # Intersecção do posto com o polígono

            # coords = coords.reshape(int(len(coords))/2, 2)
            coords = coords.reshape(int(len(coords)/2), 2)

            intersc = Polygon(coords).intersection(Polygon(bacv))

            # print(intersc)
            # for geom in intersc.geoms:
            #     print(1)
            #     xs, ys = geom.exterior.xy
            #     axs.fill(xs, ys, alpha=0.5, fc='r', ec='none')
            #
            # plt.show()
            if intersc.is_empty or isinstance(intersc, Point) or isinstance(intersc, LineString):
            #if intersc.is_empty:
                area = np.append(area, 0)
                continue

            if isinstance(intersc, Polygon):

                # Extrair lat, lon
                lat, lon = np.array(intersc.exterior.xy)

                # Caso do posto selecionado não ter área de intersecção com a bacia
                if lat.shape[0] == 0:
                    area = np.append(area, 0)
                else:
                    area = np.append(area, areaxy(lat, lon))
                    if fign:
                        plt.plot(lat, lon, 'b-')
            else:

                aux = 0.

                # Identifica Multipolígono
                for inter in intersc.geoms:

                    if isinstance(inter, Point) or \
                            isinstance(inter, LineString):
                        continue

                    lat, lon = np.array(inter.exterior.xy)

                    if lat.shape[0] == 0:
                        aux = 0
                    else:
                        aux = aux + areaxy(lat, lon)
                        if fign:
                            plt.plot(lat, lon, 'b-')

                area = np.append(area, aux)

    if fign:
        fign = "{0}_{1}".format(cont, fign)
        plt.savefig(fign)
        plt.close()

    alpha = area/np.sum(area)

    return alpha


def precimd(coef, ithi, dados):
    """
     Calcula a precipitação média sobre uma bacia
    """

    lithi = len(ithi)
    pmed = np.array([])

    for i in range(lithi):

        if np.sum(coef[ithi[i], :]) ==0:

            pmed = np.append(pmed, -999)

        else:
            aux = np.sum(coef[ithi[i], :]*dados[i, :])
            pmed = np.append(pmed, aux)

    return pmed
