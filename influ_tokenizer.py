"""
influ_tokenizer.py
==================
Reimplementación de InfluTokenizer para AntigenLM.

Esta clase fue reconstruida a partir de los archivos de configuración
del repositorio (vocab.json, added_tokens.json, tokenizer_config.json)
ya que el código fuente original no fue publicado por los autores.

Referencia:
    Pei et al., "AntigenLM: Structure-Aware DNA Language Modeling
    for Influenza", ICLR 2026. arXiv:2602.09067
"""

import json
import os
from typing import List, Optional, Union, Dict


# ---------------------------------------------------------------------------
# Vocabulario base — derivado de vocab.json + added_tokens.json
# ---------------------------------------------------------------------------

# Núcleotidos + tokens estructurales (vocab.json, vocab_size base)
BASE_VOCAB = {
    "A":     0,   # Adenina
    "C":     1,   # Citosina
    "G":     2,   # Guanina
    "T":     3,   # Timina
    "N":     4,   # Nucleótido ambiguo (también es unk_token)
    "<pad>": 5,   # Padding
    "<sep>": 6,   # Separador entre segmentos (HA | NA)
    "<eos>": 7,   # Fin de secuencia / fin de cepa
    "<HA>":  8,   # Marcador de inicio del segmento Hemaglutinitna
    "<NA>":  9,   # Marcador de inicio del segmento Neuraminidasa
}

# Tokens de subtipo — derivados de added_tokens.json
# Solo presentes en prediction_sequence (vocab_size=25)
# Ausentes en subtype_classifier (vocab_size=10, los subtipos son LABELS allí)
SUBTYPE_TOKENS = {
    "<H1N1>":  10,
    "<H3N2>":  11,
    "<H5N6>":  12,
    "<H7N9>":  13,
    "<H9N2>":  14,
    "<H5N1>":  15,
    "<H10N3>": 16,
    "<H1N2>":  17,
    "<H3N8>":  18,
    "<H6N1>":  19,
    "<H2N2>":  20,
    "<H10N8>": 21,
    "<H10N5>": 22,
    "<H5N8>":  23,
    "<H7N4>":  24,
}

# Vocabulario completo para predicción de secuencia
FULL_VOCAB = {**BASE_VOCAB, **SUBTYPE_TOKENS}

# Vocabulario reducido para clasificador de subtipo
CLASSIFIER_VOCAB = BASE_VOCAB.copy()


# ---------------------------------------------------------------------------
# Clase principal
# ---------------------------------------------------------------------------

class InfluTokenizer:
    """
    Tokenizador character-level para secuencias de influenza.

    Diseño:
        - Cada nucleótido (A, C, G, T) es un token individual.
        - Tokens especiales marcan estructura biológica:
            <subtipo>  →  condicionante de subtipo al inicio
            <HA>       →  inicio del segmento Hemaglutinina
            <sep>      →  separador entre HA y NA
            <NA>       →  inicio del segmento Neuraminidasa
            <eos>      →  fin de la cepa
            <pad>      →  relleno hasta longitud máxima
        - 'N' actúa como token desconocido (unk) para bases ambiguas.

    Formato de secuencia de entrada para predicción:
        <subtipo> <HA> ATCG...GCTA <sep> <NA> GCTA...ATCG <eos>

    Formato para clasificación de subtipo:
        <HA> ATCG...GCTA <sep> <NA> GCTA...ATCG <eos>
        (sin token de subtipo — el subtipo es el label de salida)

    Args:
        mode: 'prediction' usa vocabulario de 25 tokens (con subtipos).
              'classification' usa vocabulario de 10 tokens (sin subtipos).
    """

    def __init__(self, mode: str = "prediction"):
        assert mode in ("prediction", "classification"), \
            "mode debe ser 'prediction' o 'classification'"

        self.mode = mode
        self.vocab = FULL_VOCAB.copy() if mode == "prediction" else CLASSIFIER_VOCAB.copy()
        self.id_to_token = {v: k for k, v in self.vocab.items()}

        # Tokens especiales
        self.pad_token = "<pad>"
        self.eos_token = "<eos>"
        self.sep_token = "<sep>"
        self.unk_token = "N"

        self.pad_token_id = self.vocab["<pad>"]
        self.eos_token_id = self.vocab["<eos>"]
        self.sep_token_id = self.vocab["<sep>"]
        self.unk_token_id = self.vocab["N"]

    # ------------------------------------------------------------------
    # Propiedades básicas
    # ------------------------------------------------------------------

    @property
    def vocab_size(self) -> int:
        return len(self.vocab)

    # ------------------------------------------------------------------
    # Tokenización de una sola cepa (HA + NA)
    # ------------------------------------------------------------------

    def encode_strain(
        self,
        ha_sequence: str,
        na_sequence: str,
        subtype: Optional[str] = None,
        add_eos: bool = True,
    ) -> List[int]:
        """
        Convierte una cepa (HA + NA) en una lista de IDs de tokens.

        Args:
            ha_sequence: Secuencia nucleotídica del segmento HA (string de A/C/G/T/N).
            na_sequence:  Secuencia nucleotídica del segmento NA.
            subtype:      Subtipo de influenza, e.g. "<H3N2>". Solo para mode='prediction'.
            add_eos:      Si True, agrega <eos> al final.

        Returns:
            Lista de enteros (IDs de tokens).

        Ejemplo:
            tokenizer.encode_strain("ATCG", "GCTA", subtype="<H3N2>")
            → [11, 8, 0, 3, 1, 2, 6, 9, 2, 1, 3, 0, 7]
               H3N2 HA  A  T  C  G sep NA  G  C  T  A eos
        """
        tokens = []

        # 1. Token de subtipo (solo en modo predicción)
        if self.mode == "prediction":
            if subtype is None:
                raise ValueError(
                    "En mode='prediction' debes proveer el subtipo, e.g. '<H3N2>'"
                )
            if subtype not in self.vocab:
                raise ValueError(
                    f"Subtipo '{subtype}' no está en el vocabulario. "
                    f"Disponibles: {list(SUBTYPE_TOKENS.keys())}"
                )
            tokens.append(self.vocab[subtype])

        # 2. Segmento HA
        tokens.append(self.vocab["<HA>"])
        tokens.extend(self._encode_sequence(ha_sequence))

        # 3. Separador
        tokens.append(self.vocab["<sep>"])

        # 4. Segmento NA
        tokens.append(self.vocab["<NA>"])
        tokens.extend(self._encode_sequence(na_sequence))

        # 5. Fin de cepa
        if add_eos:
            tokens.append(self.vocab["<eos>"])

        return tokens

    def _encode_sequence(self, sequence: str) -> List[int]:
        """
        Convierte una cadena de nucleótidos en IDs.
        Caracteres no reconocidos se mapean a N (unk_token_id).
        """
        sequence = sequence.upper()
        return [
            self.vocab.get(char, self.unk_token_id)
            for char in sequence
        ]

    # ------------------------------------------------------------------
    # Decodificación
    # ------------------------------------------------------------------

    def decode(self, token_ids: List[int], skip_special_tokens: bool = False) -> str:
        """
        Convierte una lista de IDs de tokens de vuelta a string.

        Args:
            token_ids:           Lista de enteros.
            skip_special_tokens: Si True, omite tokens especiales en el output.

        Returns:
            String reconstruido.
        """
        special_ids = {5, 6, 7, 8, 9} | set(SUBTYPE_TOKENS.values())
        result = []
        for tid in token_ids:
            token = self.id_to_token.get(tid, "N")
            if skip_special_tokens and tid in special_ids:
                continue
            result.append(token)
        return "".join(result)

    # ------------------------------------------------------------------
    # Padding y batch
    # ------------------------------------------------------------------

    def pad_sequence(
        self,
        token_ids: List[int],
        max_length: int,
        padding_side: str = "right",
    ) -> List[int]:
        """
        Rellena o trunca una secuencia hasta max_length.

        Args:
            token_ids:    Secuencia a ajustar.
            max_length:   Longitud objetivo.
            padding_side: 'right' (estándar para GPT) o 'left'.

        Returns:
            Secuencia de longitud exactamente max_length.
        """
        current_len = len(token_ids)

        if current_len >= max_length:
            return token_ids[:max_length]

        pad_length = max_length - current_len
        pad = [self.pad_token_id] * pad_length

        if padding_side == "right":
            return token_ids + pad
        else:
            return pad + token_ids

    def encode_batch(
        self,
        strains: List[Dict],
        max_length: Optional[int] = None,
        padding: bool = True,
    ) -> Dict[str, List[List[int]]]:
        """
        Tokeniza un batch de cepas y opcionalmente aplica padding.

        Args:
            strains:    Lista de dicts con claves 'ha', 'na', y opcionalmente 'subtype'.
            max_length: Longitud máxima. Si None, usa la más larga del batch.
            padding:    Si True, aplica padding hasta max_length.

        Returns:
            Dict con 'input_ids' y 'attention_mask'.

        Ejemplo de entrada:
            strains = [
                {"ha": "ATCG...", "na": "GCTA...", "subtype": "<H3N2>"},
                {"ha": "TTAG...", "na": "CCGA...", "subtype": "<H1N1>"},
            ]
        """
        encoded = []
        for strain in strains:
            ids = self.encode_strain(
                ha_sequence=strain["ha"],
                na_sequence=strain["na"],
                subtype=strain.get("subtype"),
            )
            encoded.append(ids)

        if max_length is None:
            max_length = max(len(e) for e in encoded)

        input_ids = []
        attention_masks = []

        for ids in encoded:
            if padding:
                padded = self.pad_sequence(ids, max_length)
                mask = [1 if tid != self.pad_token_id else 0 for tid in padded]
            else:
                padded = ids
                mask = [1] * len(ids)

            input_ids.append(padded)
            attention_masks.append(mask)

        return {
            "input_ids": input_ids,
            "attention_mask": attention_masks,
        }

    # ------------------------------------------------------------------
    # Utilidades de inspección
    # ------------------------------------------------------------------

    def get_vocab(self) -> Dict[str, int]:
        return self.vocab.copy()

    def token_to_id(self, token: str) -> int:
        return self.vocab.get(token, self.unk_token_id)

    def id_to_token_str(self, token_id: int) -> str:
        return self.id_to_token.get(token_id, "N")

    def __repr__(self) -> str:
        return (
            f"InfluTokenizer("
            f"mode='{self.mode}', "
            f"vocab_size={self.vocab_size})"
        )


# ---------------------------------------------------------------------------
# Bloque de prueba — ejecutar con: python influ_tokenizer.py
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    print("=" * 60)
    print("TEST: InfluTokenizer")
    print("=" * 60)

    # --- Test 1: modo predicción ---
    print("\n[1] Modo predicción (vocab_size=25)")
    tok = InfluTokenizer(mode="prediction")
    print(f"    {tok}")

    ha = "ATCGNNTAGC"
    na = "GCTATTCGAA"
    ids = tok.encode_strain(ha, na, subtype="<H3N2>")
    print(f"    HA input:    {ha}")
    print(f"    NA input:    {na}")
    print(f"    Token IDs:   {ids}")
    print(f"    Longitud:    {len(ids)} tokens")

    # Verificar estructura esperada
    assert ids[0] == 11,  "Error: primer token debe ser <H3N2>=11"
    assert ids[1] == 8,   "Error: segundo token debe ser <HA>=8"
    assert ids[12] == 6,  "Error: separador <sep>=6 esperado en posición 11"
    assert ids[13] == 9,  "Error: <NA>=9 esperado en posición 12"
    assert ids[-1] == 7,  "Error: último token debe ser <eos>=7"
    print("    ✅ Estructura de tokens correcta")

    decoded = tok.decode(ids, skip_special_tokens=True)
    print(f"    Decodificado: {decoded}")
    assert decoded == ha + na, "Error: decodificación no coincide con input"
    print("    ✅ Decodificación correcta")

    # --- Test 2: modo clasificación ---
    print("\n[2] Modo clasificación (vocab_size=10)")
    tok_cls = InfluTokenizer(mode="classification")
    print(f"    {tok_cls}")
    ids_cls = tok_cls.encode_strain(ha, na)  # sin subtype
    print(f"    Token IDs:   {ids_cls}")
    assert ids_cls[0] == 8, "Error: primer token debe ser <HA>=8"
    print("    ✅ Sin token de subtipo en la entrada — correcto para clasificación")

    # --- Test 3: batch con padding ---
    print("\n[3] Batch con padding")
    batch = tok.encode_batch([
        {"ha": "ATCG", "na": "GCTA", "subtype": "<H3N2>"},
        {"ha": "TTAAGGCC", "na": "CCGGAATT", "subtype": "<H1N1>"},
    ])
    lengths = [len(row) for row in batch["input_ids"]]
    print(f"    Longitudes antes de padding: {[11, 19]}")
    print(f"    Longitudes después:          {lengths}")
    assert len(set(lengths)) == 1, "Error: todas las secuencias deben tener la misma longitud"
    print("    ✅ Padding aplicado correctamente")

    # --- Test 4: token desconocido ---
    print("\n[4] Manejo de nucleótido ambiguo")
    ids_unk = tok._encode_sequence("ATXGN")
    print(f"    'ATXGN' → {ids_unk}")
    assert ids_unk[2] == 4, "Error: 'X' debe mapearse a N (id=4)"
    assert ids_unk[4] == 4, "Error: 'N' debe mapearse a id=4"
    print("    ✅ Ambigüedades mapeadas a N correctamente")

    print("\n" + "=" * 60)
    print("✅ Todos los tests pasaron. InfluTokenizer funciona correctamente.")
    print("=" * 60)
