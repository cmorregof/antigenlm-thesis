"""
preprocess_gisaid.py
====================
Preprocesamiento de datos GISAID para replicación de AntigenLM.

Diferencias clave respecto al preprocessor de NCBI:
    - Pareo perfecto por EPI_ISL (no por nombre de cepa)
    - Fechas completas YYYY-MM-DD
    - Segmento como texto ("HA"/"NA") o número ("4"/"6")
    - Subtipo puede ser "H3N2" o "A_/_H1N1" — se normaliza
    - Mucho más datos: ~137k cepas vs ~47k de NCBI

Outputs:
    data/processed_gisaid/dataset_H3N2.json
    data/processed_gisaid/dataset_H1N1.json
    data/processed_gisaid/stats.txt

Uso:
    python preprocess_gisaid.py

Autor: Carlos (tesis de maestría)
"""

import os
import re
import json
from collections import defaultdict
from typing import Dict, List, Optional, Tuple

# ---------------------------------------------------------------------------
# Configuración
# ---------------------------------------------------------------------------

GISAID_DIR    = "data/gisaid"
PROCESSED_DIR = "data/processed_gisaid"

# Archivos por subtipo
FILES = {
    "H3N2": [
        "GISAID_H3N2_2000_2014.fasta",
        "GISAID_H3N2_2015_2017.fasta",
        "GISAID_H3N2_2017_2019.fasta",
        "GISAID_H3N2_2019_2021.fasta",
        "GISAID_H3N2_2021_2022.fasta",
    ],
    "H1N1": [
        "GISAID_H1N1_2000_2015.fasta",
        "GISAID_H1N1_2016_2019.fasta",
        "GISAID_H1N1_2019_2022.fasta",
    ],
}

# Filtros de calidad
MAX_AMBIGUOUS_RATIO   = 0.01
MIN_HA_LENGTH         = 1650
MAX_HA_LENGTH         = 1800
MIN_NA_LENGTH         = 1200
MAX_NA_LENGTH         = 1600
CONTEXT_MONTHS        = 3
MIN_STRAINS_PER_MONTH = 5


# ---------------------------------------------------------------------------
# Parsing de FASTA
# ---------------------------------------------------------------------------

def parse_fasta_gisaid(filepaths: List[str]) -> Dict[str, dict]:
    """
    Parsea múltiples archivos FASTA de GISAID.

    Header esperado (dos formatos observados):
        >EPI_ISL_163846|A/Taiwan/362/2014|A_/_H1N1|2014-03-18|NA
        >EPI_ISL_20428741|A/New_Zealand/HB0534/2017|H3N2|2017-06-15|4

    Retorna dict: {EPI_ISL: {ha: seq, na: seq, year, month, day, strain_name, subtype}}
    """
    strains = {}  # EPI_ISL → info

    for filepath in filepaths:
        if not os.path.exists(filepath):
            print(f"  AVISO: no encontrado {filepath}")
            continue

        print(f"  Leyendo {os.path.basename(filepath)}...")
        current_header = None
        current_seq    = []
        count          = 0

        with open(filepath, "r", encoding="utf-8", errors="replace") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue

                if line.startswith(">"):
                    # Guardar secuencia anterior
                    if current_header is not None:
                        _store_sequence(strains, current_header,
                                        "".join(current_seq).upper())
                        count += 1

                    current_header = line[1:]
                    current_seq    = []
                else:
                    current_seq.append(line)

        # Última secuencia
        if current_header is not None:
            _store_sequence(strains, current_header, "".join(current_seq).upper())
            count += 1

        print(f"    {count:,} secuencias leídas")

    print(f"  Total cepas únicas (EPI_ISL): {len(strains):,}")
    return strains


def _store_sequence(strains: dict, header: str, sequence: str):
    """
    Parsea el header GISAID y almacena la secuencia en el dict de cepas.
    Maneja ambos formatos de segmento (texto y número).
    """
    info = parse_gisaid_header(header)
    if info is None:
        return

    epi_isl = info["epi_isl"]
    segment = info["segment"]  # "HA", "NA", "4", o "6"

    # Normalizar segmento
    is_ha = segment in ("HA", "4", "ha", "Seg4", "4_HA")
    is_na = segment in ("NA", "6", "na", "Seg6", "6_NA")

    if not (is_ha or is_na):
        return

    if epi_isl not in strains:
        strains[epi_isl] = {
            "epi_isl":    epi_isl,
            "strain_name": info["strain_name"],
            "subtype":    info["subtype"],
            "year":       info["year"],
            "month":      info["month"],
            "day":        info["day"],
            "ha_sequence": None,
            "na_sequence": None,
        }

    if is_ha:
        strains[epi_isl]["ha_sequence"] = sequence
    elif is_na:
        strains[epi_isl]["na_sequence"] = sequence


def parse_gisaid_header(header: str) -> Optional[dict]:
    """
    Parsea el header de GISAID.

    Formatos observados:
        EPI_ISL_163846|A/Taiwan/362/2014|A_/_H1N1|2014-03-18|NA
        EPI_ISL_20428741|A/New_Zealand/HB0534/2017|H3N2|2017-06-15|4

    Retorna dict con: epi_isl, strain_name, subtype, year, month, day, segment
    """
    parts = [p.strip() for p in header.split("|")]
    if len(parts) < 5:
        return None

    epi_isl     = parts[0]
    strain_name = parts[1].replace("_", " ")
    subtype_raw = parts[2]
    date_str    = parts[3]
    segment     = parts[4]

    # Normalizar subtipo
    # "A_/_H1N1" → "H1N1", "H3N2" → "H3N2"
    subtype_match = re.search(r'H(\d+)N(\d+)', subtype_raw, re.IGNORECASE)
    if subtype_match:
        subtype = f"H{subtype_match.group(1)}N{subtype_match.group(2)}"
    else:
        subtype = subtype_raw

    # Parsear fecha
    year, month, day = None, None, None
    date_match = re.match(r'(\d{4})-(\d{2})-(\d{2})', date_str)
    if date_match:
        year  = int(date_match.group(1))
        month = int(date_match.group(2))
        day   = int(date_match.group(3))
    else:
        # Solo año
        year_match = re.match(r'(\d{4})', date_str)
        if year_match:
            year  = int(year_match.group(1))
            month = 6
            day   = 15

    if year is None:
        return None

    return {
        "epi_isl":    epi_isl,
        "strain_name": strain_name,
        "subtype":    subtype,
        "year":       year,
        "month":      month,
        "day":        day,
        "segment":    segment,
    }


# ---------------------------------------------------------------------------
# Filtro de calidad
# ---------------------------------------------------------------------------

def quality_filter(sequence: str, min_len: int, max_len: int) -> bool:
    if sequence is None:
        return False
    if not (min_len <= len(sequence) <= max_len):
        return False
    if len(sequence) == 0:
        return False
    if sequence.count("N") / len(sequence) > MAX_AMBIGUOUS_RATIO:
        return False
    if not all(c in "ATCGN" for c in set(sequence)):
        return False
    return True


# ---------------------------------------------------------------------------
# Construcción del dataset
# ---------------------------------------------------------------------------

def build_dataset(subtype: str) -> dict:
    print(f"\n{'='*60}")
    print(f"Procesando: {subtype}")
    print(f"{'='*60}")

    token     = f"<{subtype}>"
    filepaths = [os.path.join(GISAID_DIR, f) for f in FILES[subtype]]

    # Leer todos los archivos
    all_strains = parse_fasta_gisaid(filepaths)

    # Filtrar por calidad y completitud
    print(f"\n  Aplicando filtros de calidad...")
    paired     = []
    no_ha      = 0
    no_na      = 0
    bad_ha     = 0
    bad_na     = 0
    no_date    = 0

    for epi_isl, info in all_strains.items():
        # Verificar que tiene ambos segmentos
        if info["ha_sequence"] is None:
            no_ha += 1
            continue
        if info["na_sequence"] is None:
            no_na += 1
            continue

        # Verificar fecha
        if info["year"] is None:
            no_date += 1
            continue

        # Filtros de calidad
        if not quality_filter(info["ha_sequence"], MIN_HA_LENGTH, MAX_HA_LENGTH):
            bad_ha += 1
            continue
        if not quality_filter(info["na_sequence"], MIN_NA_LENGTH, MAX_NA_LENGTH):
            bad_na += 1
            continue

        paired.append({
            "epi_isl":       epi_isl,
            "strain_name":   info["strain_name"],
            "subtype":       subtype,
            "subtype_token": token,
            "year":          info["year"],
            "month":         info["month"] or 6,
            "day":           info["day"] or 15,
            "ha_sequence":   info["ha_sequence"],
            "na_sequence":   info["na_sequence"],
        })

    print(f"    Cepas únicas totales:     {len(all_strains):,}")
    print(f"    Sin HA:                   {no_ha:,}")
    print(f"    Sin NA:                   {no_na:,}")
    print(f"    HA fuera de rango:        {bad_ha:,}")
    print(f"    NA fuera de rango:        {bad_na:,}")
    print(f"    Sin fecha:                {no_date:,}")
    print(f"    PAREADAS y válidas:       {len(paired):,}")

    # Agrupar por mes
    print(f"\n  Agrupando por mes (año+mes)...")
    monthly = defaultdict(list)
    for i, strain in enumerate(paired):
        key = (strain["year"], strain["month"])
        monthly[key].append(i)

    months_kept = {k: v for k, v in monthly.items()
                   if len(v) >= MIN_STRAINS_PER_MONTH}

    print(f"    Meses con datos:          {len(monthly)}")
    print(f"    Meses con >={MIN_STRAINS_PER_MONTH} cepas:       {len(months_kept)}")

    if months_kept:
        years  = sorted(set(k[0] for k in months_kept))
        months = sorted(months_kept.keys())
        print(f"    Rango:                    {min(years)}-{max(years)}")
        print(f"    Primer mes:               {months[0][0]}-{months[0][1]:02d}")
        print(f"    Último mes:               {months[-1][0]}-{months[-1][1]:02d}")

    # Ventanas temporales [t-2, t-1, t] → t+1
    sorted_months = sorted(months_kept.keys())
    windows       = []

    for i in range(CONTEXT_MONTHS, len(sorted_months)):
        ctx = sorted_months[i - CONTEXT_MONTHS:i]
        tgt = sorted_months[i]

        # Para cada mes del contexto, usar la cepa más frecuente
        # (la más frecuente = la dominante del mes)
        windows.append({
            "context": [
                {
                    "year":       m[0],
                    "month":      m[1],
                    "strain_idx": months_kept[m][0],
                    "n_strains":  len(months_kept[m]),
                }
                for m in ctx
            ],
            "target": {
                "year":       tgt[0],
                "month":      tgt[1],
                "strain_idx": months_kept[tgt][0],
                "n_strains":  len(months_kept[tgt]),
            }
        })

    print(f"    Ventanas temporales:      {len(windows)}")

    return {
        "subtype":        subtype,
        "paired_strains": paired,
        "monthly_groups": {
            f"{k[0]}-{k[1]:02d}": v
            for k, v in months_kept.items()
        },
        "windows": windows,
        "stats": {
            "total_unique":  len(all_strains),
            "paired_valid":  len(paired),
            "no_ha":         no_ha,
            "no_na":         no_na,
            "bad_ha":        bad_ha,
            "bad_na":        bad_na,
            "no_date":       no_date,
            "months":        len(months_kept),
            "windows":       len(windows),
        }
    }


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    os.makedirs(PROCESSED_DIR, exist_ok=True)

    print("=" * 60)
    print("AntigenLM — Preprocesamiento GISAID")
    print("=" * 60)

    # Diagnóstico de headers
    print("\n[PASO 0] Verificando formato de headers GISAID...")
    for subtype, files in FILES.items():
        fpath = os.path.join(GISAID_DIR, files[0])
        if not os.path.exists(fpath):
            continue
        print(f"\n  {subtype} — primeros 4 headers:")
        with open(fpath, "r", encoding="utf-8", errors="replace") as f:
            count = 0
            for line in f:
                if line.startswith(">") and count < 4:
                    h = line.strip()[1:]
                    p = parse_gisaid_header(h)
                    print(f"    RAW:    {h}")
                    if p:
                        print(f"    PARSED: epi={p['epi_isl']} "
                              f"subtype={p['subtype']} "
                              f"date={p['year']}-{p['month']:02d}-{p['day']:02d} "
                              f"seg={p['segment']}")
                    else:
                        print(f"    PARSED: ERROR - header no reconocido")
                    count += 1
                if count >= 4:
                    break

    # Procesar subtipos
    all_stats = {}
    for subtype in ["H3N2", "H1N1"]:
        dataset = build_dataset(subtype)
        all_stats[subtype] = dataset["stats"]

        out = os.path.join(PROCESSED_DIR, f"dataset_{subtype}.json")
        with open(out, "w", encoding="utf-8") as f:
            json.dump(dataset, f, indent=2, ensure_ascii=False)
        print(f"\n  ✅ Guardado: {out}")

    # Reporte final
    print("\n" + "=" * 60)
    print("REPORTE FINAL")
    print("=" * 60)

    stats_path = os.path.join(PROCESSED_DIR, "stats.txt")
    with open(stats_path, "w", encoding="utf-8") as f:
        for subtype, s in all_stats.items():
            line = (
                f"{subtype}: {s['paired_valid']:,} cepas válidas "
                f"| {s['months']} meses "
                f"| {s['windows']} ventanas temporales"
            )
            print(line)
            f.write(line + "\n")

        total = sum(s["paired_valid"] for s in all_stats.values())
        line  = f"TOTAL: {total:,} cepas"
        print(line)
        f.write(line + "\n")

    print(f"\nEstadísticas: {stats_path}")
    print("\n✅ Preprocesamiento GISAID completado.")
    print("\nSiguiente paso:")
    print("  python latent_geometry.py  ← con datos GISAID reales")
