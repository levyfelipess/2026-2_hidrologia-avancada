import os
import argparse
import geopandas as gpd

def arguments():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--shp",
        required=True,
        help="Caminho do shapefile")
    parser.add_argument(
        "--municipio",
        required=False,
        default=None,
        help="Nome do município (opcional)")
    parser.add_argument(
        "--passo",
        default=1,
        type=int,
        help="Pular coordenadas, por exemplo: 2, 3... pro arquivo nao ficar muito grande quando tem muitos pontos")
    parser.add_argument(
        "--saida",
        required=True,
        help="Pasta de saída")
    parser.add_argument(
        "--formato",
        default="gpkg",
        choices=["gpkg", "asc"],
        help="Formato de saída")
    return parser.parse_args()

def pegar_coordenadas(geom):
    coordenadas = []
    if geom is None or geom.is_empty:
        return coordenadas
    if geom.geom_type == "Polygon":
        coordenadas.extend(
            list(geom.exterior.coords))
    elif geom.geom_type == "MultiPolygon":
        for poligono in geom.geoms:
            coordenadas.extend(
                list(poligono.exterior.coords))
    elif geom.geom_type == "LineString":
        coordenadas.extend(
            list(geom.coords))
    elif geom.geom_type == "MultiLineString":
        for linha in geom.geoms:
            coordenadas.extend(
                list(linha.coords))
    else:
        print(f"Geometria com erro/não suportada:" f"{geom.geom_type}")
    return coordenadas

def converter_shape(
    caminho_shp,
    pasta_saida,
    formato="gpkg",
    municipio=None,
    passo=1
):
    # Lê o arquivo espacial
    gdf = gpd.read_file(
        caminho_shp)
    if gdf.empty:
        print("Shapefile vazio.")
        return
    if municipio:
        if "NM_MUN" not in gdf.columns:
            print("ERRO: o shapefile não possui a coluna NM_MUN.")
            return
        gdf_selecionado = gdf[
            gdf["NM_MUN"]
            .str.lower()
            == municipio.lower()].copy()
        if gdf_selecionado.empty:
            print(f"Município não encontrado: " f"{municipio}")
            return
        nome_base = (municipio.replace(" ", "_").lower())
        if "SIGLA_UF" in gdf_selecionado.columns:
            uf = (gdf_selecionado.iloc[0]["SIGLA_UF"].lower())
            nome_base = (f"{nome_base}_{uf}")
    else:
        gdf_selecionado = gdf
        nome_base = os.path.splitext(os.path.basename(caminho_shp))[0]
    coords = []
    for geom in gdf_selecionado.geometry:
        coords.extend(pegar_coordenadas(geom))
    coords = coords[::passo]
    if not coords:
        print("coordenada não encontrada.")
        return
    os.makedirs(pasta_saida,exist_ok=True)
    if formato == "gpkg":
        caminho_saida = os.path.join(pasta_saida,f"{nome_base}.gpkg")
        if os.path.exists(caminho_saida):
            os.remove(caminho_saida)
        gdf_selecionado.to_file(caminho_saida, driver="GPKG")
        print(f"GPKG salvo: " f"{caminho_saida}")
    elif formato == "asc":
        caminho_saida = os.path.join(pasta_saida, f"{nome_base}.asc")
        with open(caminho_saida,"w", encoding="utf-8") as arquivo:
            for x, y, *resto in coords:
                arquivo.write(f"{x},{y}\n")
        print(f"ASC salvo: " f"{caminho_saida}")
    print(f"Arquivo: " f"{os.path.basename(caminho_shp)}")
    if municipio:
        print(f"Município: " f"{municipio}")
    print(f"Geometrias: " f"{len(gdf_selecionado)}")
    print(f"Total de coordenadas: " f"{len(coords)}")
    print(f"CRS: " f"{gdf.crs}")

def main():
    args = arguments()
    converter_shape(
        caminho_shp=args.shp,
        pasta_saida=args.saida,
        formato=args.formato,
        municipio=args.municipio,
        passo=args.passo
    )

if __name__ == "__main__":
    main()