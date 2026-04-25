"""
preprocess_data.py  (v2 — parser corregido para formato NCBI real)
==================================================================
Cambios respecto a v1:
    - Parser adaptado al formato real observado:
      ACCESSION|strain_name_o_descripcion|{subtype}|{collection_date}|seg
    - Extracción de año desde nombre de cepa (A/Ciudad/num/YYYY)
    - Pareo por nombre de cepa normalizado (KM, HM, KC, etc.)
    - Pareo por proximidad de accession para secuencias sin nombre (LC...)
"""

import os
import re
import json
from collections import defaultdict
from typing import Dict, Optional

# ---------------------------------------------------------------------------
# Configuración
# ---------------------------------------------------------------------------

RAW_DIR       = "data/raw"
PROCESSED_DIR = "data/processed"

FILES = {
    "H3N2": {
        "ha": "HA_nucleotide_H3N2.fa",
        "na": "NA_nucleotide_H3N2.fasta.fa",
    },
    "H1N1": {
        "ha": "HA_nucleotide_H1N1.fasta.fa",
        "na": "NA_nucleotide_H1N1.fasta.fa",
    },
}

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

def parse_fasta(filepath: str) -> Dict[str, str]:
    sequences      = {}
    current_header = None
    current_seq    = []
    print(f"  Leyendo {os.path.basename(filepath)}...")
    with open(filepath, "r", encoding="utf-8", errors="replace") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            if line.startswith(">"):
                if current_header is not None:
                    sequences[current_header] = "".join(current_seq).upper()
                current_header = line[1:]
                current_seq    = []
            else:
                current_seq.append(line)
    if current_header is not None:
        sequences[current_header] = "".join(current_seq).upper()
    print(f"    {len(sequences):,} secuencias leidas")
    return sequences


# ---------------------------------------------------------------------------
# Parser de headers
# ---------------------------------------------------------------------------

def parse_ncbi_header(header: str) -> dict:
    """
    Formato real observado:
        KM821323|A/AUCKLAND/20/2003|{subtype}|{collection_date}|4
        LC720189|Influenza A virus|{subtype}|{collection_date}|4

    Extrae: accession, strain_name, year, month, has_strain_name
    """
    parts = header.split("|")
    info  = {
        "accession":       parts[0].strip() if parts else header[:20],
        "strain_name":     None,
        "year":            None,
        "month":           None,
        "has_strain_name": False,
    }

    if len(parts) >= 2:
        second = parts[1].strip()
        if re.match(r'^A/', second, re.IGNORECASE):
            info["strain_name"]     = second
            info["has_strain_name"] = True
            # Año de 4 dígitos al final del nombre
            m = re.search(r'/(\d{4})$', second)
            if m:
                info["year"] = int(m.group(1))
            else:
                # Año de 2 dígitos
                m2 = re.search(r'/(\d{2})$', second)
                if m2:
                    y = int(m2.group(1))
                    info["year"] = 2000 + y if y <= 30 else 1900 + y
        else:
            info["strain_name"] = info["accession"]

    # Buscar año en cualquier parte del header si no se encontró
    if info["year"] is None:
        m = re.search(r'[|/\s]((?:19|20)\d{2})[|/\s]', header)
        if m:
            info["year"] = int(m.group(1))

    return info


# ---------------------------------------------------------------------------
# Filtro de calidad
# ---------------------------------------------------------------------------

def quality_filter(sequence: str, min_len: int, max_len: int) -> bool:
    if not (min_len <= len(sequence) <= max_len):
        return False
    if sequence.count("N") / len(sequence) > MAX_AMBIGUOUS_RATIO:
        return False
    if not all(c in "ATCGN" for c in set(sequence)):
        return False
    return True


# ---------------------------------------------------------------------------
# Construcción del dataset
# ---------------------------------------------------------------------------

def _make_pair(ha: dict, na: dict, subtype: str, token: str) -> dict:
    return {
        "strain_name":   ha["strain_name"],
        "subtype":       subtype,
        "subtype_token": token,
        "year":          ha["year"],
        "month":         ha.get("month") or 6,
        "ha_sequence":   ha["sequence"],
        "na_sequence":   na["sequence"],
        "ha_accession":  ha["accession"],
        "na_accession":  na["accession"],
    }


def build_dataset(subtype: str) -> dict:
    print(f"\n{'='*60}")
    print(f"Procesando: {subtype}")
    print(f"{'='*60}")

    token = f"<{subtype}>"
    files = FILES[subtype]

    ha_raw = parse_fasta(os.path.join(RAW_DIR, files["ha"]))
    na_raw = parse_fasta(os.path.join(RAW_DIR, files["na"]))

    # Filtrar y parsear
    print("\n  Filtros de calidad...")
    ha_parsed, ha_rej = {}, 0
    for h, s in ha_raw.items():
        if quality_filter(s, MIN_HA_LENGTH, MAX_HA_LENGTH):
            info = parse_ncbi_header(h)
            ha_parsed[h] = {**info, "sequence": s}
        else:
            ha_rej += 1

    na_parsed, na_rej = {}, 0
    for h, s in na_raw.items():
        if quality_filter(s, MIN_NA_LENGTH, MAX_NA_LENGTH):
            info = parse_ncbi_header(h)
            na_parsed[h] = {**info, "sequence": s}
        else:
            na_rej += 1

    print(f"    HA: {len(ha_parsed):,} ok, {ha_rej:,} rechazadas")
    print(f"    NA: {len(na_parsed):,} ok, {na_rej:,} rechazadas")
    print(f"    HA con nombre cepa: {sum(v['has_strain_name'] for v in ha_parsed.values()):,}")
    print(f"    HA con anio:        {sum(v['year'] is not None for v in ha_parsed.values()):,}")

    # Indices para pareo
    def normalize(name):
        name = name.upper().strip()
        name = re.sub(r'\s+', '', name)
        name = re.sub(r'[Hh]\d+[Nn]\d+$', '', name)
        name = re.sub(r'[()]', '', name)
        return name

    na_by_strain = defaultdict(list)
    for info in na_parsed.values():
        na_by_strain[normalize(info["strain_name"])].append(info)

    na_by_acc = {info["accession"]: info for info in na_parsed.values()}

    # Pareo
    print("\n  Pareando HA con NA...")
    paired       = []
    by_strain    = 0
    by_proximity = 0
    unmatched    = 0

    for h, ha_info in ha_parsed.items():
        matched = False

        # Estrategia 1: por nombre de cepa
        if ha_info["has_strain_name"]:
            key = normalize(ha_info["strain_name"])
            candidates = na_by_strain.get(key, [])
            if candidates:
                paired.append(_make_pair(ha_info, candidates[0], subtype, token))
                by_strain += 1
                matched = True

        # Estrategia 2: por proximidad de accession (LC720189 -> LC720191)
        if not matched:
            m = re.search(r'^([A-Za-z]+)(\d+)$', ha_info["accession"])
            if m:
                prefix, num = m.group(1), int(m.group(2))
                for delta in [2, 1, 3, 4]:
                    cand_acc = f"{prefix}{num + delta}"
                    if cand_acc in na_by_acc:
                        paired.append(_make_pair(ha_info, na_by_acc[cand_acc], subtype, token))
                        by_proximity += 1
                        matched = True
                        break

        if not matched:
            unmatched += 1

    print(f"    Por nombre de cepa: {by_strain:,}")
    print(f"    Por proximidad acc: {by_proximity:,}")
    print(f"    Sin pareja:         {unmatched:,}")
    print(f"    TOTAL pareadas:     {len(paired):,}")

    # Agrupar por mes
    print("\n  Agrupando por mes...")
    monthly = defaultdict(list)
    no_date = 0
    for i, s in enumerate(paired):
        if s["year"] is not None:
            monthly[(s["year"], s["month"] or 6)].append(i)
        else:
            no_date += 1

    months_kept = {k: v for k, v in monthly.items()
                   if len(v) >= MIN_STRAINS_PER_MONTH}

    print(f"    Sin fecha:          {no_date:,}")
    print(f"    Meses con datos:    {len(monthly)}")
    print(f"    Meses con >={MIN_STRAINS_PER_MONTH} cepas: {len(months_kept)}")
    if months_kept:
        years = sorted(set(k[0] for k in months_kept))
        print(f"    Rango:              {min(years)}-{max(years)}")

    # Ventanas temporales
    sorted_months = sorted(months_kept.keys())
    windows = []
    for i in range(CONTEXT_MONTHS, len(sorted_months)):
        ctx = sorted_months[i - CONTEXT_MONTHS:i]
        tgt = sorted_months[i]
        windows.append({
            "context": [{"year": m[0], "month": m[1],
                         "strain_idx": months_kept[m][0]} for m in ctx],
            "target":  {"year": tgt[0], "month": tgt[1],
                        "strain_idx": months_kept[tgt][0]},
        })
    print(f"    Ventanas:           {len(windows)}")

    return {
        "subtype":        subtype,
        "paired_strains": paired,
        "monthly_groups": {f"{k[0]}-{k[1]:02d}": v for k, v in months_kept.items()},
        "windows":        windows,
        "stats": {
            "ha_total":     len(ha_raw),
            "na_total":     len(na_raw),
            "ha_filtered":  len(ha_parsed),
            "na_filtered":  len(na_parsed),
            "paired":       len(paired),
            "by_strain":    by_strain,
            "by_proximity": by_proximity,
            "months":       len(months_kept),
            "windows":      len(windows),
        }
    }


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    os.makedirs(PROCESSED_DIR, exist_ok=True)

    print("=" * 60)
    print("AntigenLM - Preprocesamiento v2")
    print("=" * 60)

    # Diagnostico rapido
    print("\n[PASO 0] Diagnostico de headers")
    for subtype, files in FILES.items():
        fpath = os.path.join(RAW_DIR, files["ha"])
        print(f"\n  {subtype} - primeros 4 headers:")
        with open(fpath, "r", encoding="utf-8", errors="replace") as f:
            count = 0
            for line in f:
                if line.startswith(">") and count < 4:
                    h = line.strip()[1:]
                    p = parse_ncbi_header(h)
                    print(f"    {h}")
                    print(f"    -> strain={p['strain_name']}  year={p['year']}")
                    count += 1
                if count >= 4:
                    break

    # Procesar
    all_stats = {}
    for subtype in ["H3N2", "H1N1"]:
        dataset = build_dataset(subtype)
        all_stats[subtype] = dataset["stats"]
        out = os.path.join(PROCESSED_DIR, f"dataset_{subtype}.json")
        with open(out, "w", encoding="utf-8") as f:
            json.dump(dataset, f, indent=2, ensure_ascii=False)
        print(f"\n  Guardado: {out}")

    # Reporte
    print("\n" + "=" * 60)
    print("REPORTE FINAL")
    print("=" * 60)
    with open(os.path.join(PROCESSED_DIR, "stats.txt"), "w") as f:
        for subtype, s in all_stats.items():
            line = (f"{subtype}: {s['paired']:,} pareadas "
                    f"(nombre:{s['by_strain']:,} prox:{s['by_proximity']:,}) "
                    f"| {s['months']} meses | {s['windows']} ventanas")
            print(line)
            f.write(line + "\n")

    print("\nPreprocesamiento completado.")
