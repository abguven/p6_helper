import pandas as pd
import numpy as np
import json
from pathlib import Path
import matplotlib.pyplot as plt
import os
from typing import Tuple, Literal, overload

_VERBOSE_ICONS = {
    "info": "ℹ️",
    "quartiles": "🍕",
    "bounds": "🚧",
    "outliers": "☢️",
    "examples": "🔍",
    "result": "✅",
    "update": "🔄️",
}


def _emit(verbose: bool, icon_key: str, message: str) -> None:
    """Emit a verbose message prefixed with an icon when enabled."""

    if not verbose:
        return

    icon = _VERBOSE_ICONS.get(icon_key, _VERBOSE_ICONS["info"])
    print(f"{icon} {message}")


def export_use_type_summary(df, columns, output_path, overwrite=False):
    """
    Create a CSV summary of unique building use types by column.
    For comma-separated columns, only the first value before the comma is counted.

    Parameters
    ----------
    df : pd.DataFrame
        The dataset containing building use columns.
    columns : list[str]
        Column names to analyze.
    output_path : str or Path
        CSV output path.
    """

    if not overwrite and os.path.exists(output_path):
        print(f"✅ Fichier déjà existant : {output_path}")
        return

    results = []

    for col in columns:
        if col not in df.columns:
            print(f"⚠️ Column '{col}' not found, skipping.")
            continue

        # Récupération de la colonne, suppression des NaN
        series = df[col].dropna().astype(str)

        # Si la colonne contient des listes séparées par virgule, ne garder que la première
        if series.str.contains(',').any():
            first_values = series.str.split(',', n=1).str[0].str.strip()
        else:
            first_values = series.str.strip()

        # Comptage des occurrences
        counts = first_values.value_counts().sort_index()

        # Construction du DataFrame temporaire
        temp_df = pd.DataFrame({
            "df_column": col,
            "use_type": counts.index,
            "count": counts.values
        })

        results.append(temp_df)

    # Fusionner tous les résultats
    summary_df = pd.concat(results, ignore_index=True)

    # Export
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    summary_df.to_csv(output_path, index=False)

    print(f"✅ Exported {len(summary_df)} rows to {output_path}")
    return summary_df

def analyse_cardinality(df, simple_features, multi_features=None, separator=',', plot=False, figsize=(8,4), color='skyblue'):
    """
    Calcule la cardinalité pour des colonnes single-label et multi-label,
    avec option d'afficher un barplot annoté.

    Parameters:
        df : pd.DataFrame
        simple_features : list -> colonnes à traiter comme single-label
        multi_features : list -> colonnes à traiter comme multi-label (split + explode)
        separator : str -> séparateur pour les colonnes multi-label
        plot : bool -> si True, affiche un barplot annoté
        figsize : tuple -> taille du plot
        color : str -> couleur des barres du plot

    Returns:
        pd.DataFrame avec les colonnes 'column' et 'cardinality'
    """
    multi_features = multi_features or []
    all_cols = simple_features + multi_features

    def cardinality(col_series, multi=False):
        s = col_series.dropna()
        if multi:
            return s.str.split(separator).explode().str.strip().nunique()
        else:
            return s.nunique()

    df_analyse = pd.DataFrame({
        "column": all_cols,
        "cardinality": [cardinality(df[col], multi=(col in multi_features)) for col in all_cols]
    })

    if plot:
        plt.figure(figsize=figsize)
        bars = plt.barh(df_analyse['column'], df_analyse['cardinality'], color=color)
        plt.xlabel("Cardinalité")
        plt.ylabel("Colonne")
        plt.title("Cardinalité des colonnes")
        plt.gca().invert_yaxis()

        # ➕ Ajout des annotations sur chaque barre
        for bar in bars:
            width = bar.get_width()
            plt.text(
                width + max(df_analyse['cardinality']) * 0.01,  # petit décalage à droite
                bar.get_y() + bar.get_height() / 2,
                f"{int(width)}",
                va='center',
                fontsize=9,
                color='black'
            )

        plt.tight_layout()
        plt.show()

    return df_analyse



@overload
def impute_other(df: pd.DataFrame, verbose: bool = True, return_changes: Literal[False] = False) -> pd.DataFrame: ...

@overload
def impute_other(df: pd.DataFrame, verbose: bool = True, return_changes: Literal[True] = True) -> Tuple[pd.DataFrame, pd.DataFrame]: ...

def impute_other(df, verbose=True, return_changes=False): 

    df = df.copy()

    mask_other = df["LargestPropertyUseType"].eq("Other")

    # Préparation des sources potentielles
    primary_ok = (
        df["PrimaryPropertyType"].notna()
        & ~df["PrimaryPropertyType"].eq("Mixed Use Property")
    )
    first_use = df["ListOfAllPropertyUseTypes"].astype(str).str.split(",").str[0].str.strip()
    list_ok = ~first_use.eq("Other") & first_use.ne("nan")  # évite les 'nan' string

    # Création d’une colonne d’imputation prioritaire
    new_values = df["LargestPropertyUseType"].copy()
    source_col = pd.Series([None] * len(df), index=df.index)

    # Imputation prioritaire : PrimaryPropertyType, sinon ListOfAllPropertyUseTypes
    new_values.loc[mask_other & primary_ok] = df.loc[mask_other & primary_ok, "PrimaryPropertyType"]
    source_col.loc[mask_other & primary_ok] = "PrimaryPropertyType"

    remaining_mask = mask_other & ~primary_ok & list_ok
    new_values.loc[remaining_mask] = first_use.loc[remaining_mask]
    source_col.loc[remaining_mask] = "ListOfAllPropertyUseTypes"

    # Détection des modifications
    changed = new_values != df["LargestPropertyUseType"]
    df["LargestPropertyUseType"] = new_values

    modified_rows = df.loc[changed, ["PropertyName", "LargestPropertyUseType"]].assign(
        OldValue="Other",
        NewValue=df.loc[changed, "LargestPropertyUseType"],
        Source=source_col.loc[changed].values
    ).reset_index(names="index")

    if verbose:
        print(f"\n🔍 {len(modified_rows)} lignes imputées sur {mask_other.sum()} 'Other'")
        if not modified_rows.empty:
            print(modified_rows.to_string(index=False))
        else:
            print("Aucune modification effectuée.")

    if return_changes:
        return df, modified_rows

    return df

def get_SiteEnergyUse(df):
    return (df["Electricity(kWh)"] * 3.412) + (df["NaturalGas(therms)"] *100) + df["SteamUse(kBtu)"]


def get_SiteEUI(df):
    return ( df["SiteEnergyUse(kBtu)(kWh)"] / (df["PropertyGFABuilding"]) )

def get_sparsity_report(df, round_digits=2):
    """
    Retourne un DataFrame indiquant le pourcentage de zéros par feature.
    """
    return (
        (1 - df.mean())                     # Ratio de zéros
        .mul(100)                           # En pourcentage
        .round(round_digits)                # Arrondi
        .sort_values(ascending=False)       # Tri décroissant
        .reset_index(name="zero_rate")      # Transforme en DataFrame
        .rename(columns={"index": "features"})  # Renomme la colonne d'index
    )

def filter_by_malus_level(
    df: pd.DataFrame,
    malus_col: str = "malus_score",
    level: int = 0,
    verbose: bool = True
) -> pd.DataFrame:
    """
    Filtre le DataFrame selon un niveau de tolérance de malus.
    Plus le niveau augmente, plus le filtrage est souple.

    Parameters
    ----------
    df : pd.DataFrame
        Jeu de données à filtrer.
    malus_col : str, default='malus_score'
        Colonne contenant les scores de malus cumulés.
    level : int, default=0
        Niveau de tolérance (chaque incrément relâche le seuil).
    verbose : bool, default=True
        Si True, affiche le nombre de lignes conservées.

    Returns
    -------
    pd.DataFrame
        Sous-ensemble filtré du DataFrame original.
    """
    
    if malus_col not in df.columns:
        raise KeyError(f"La colonne '{malus_col}' est absente du DataFrame.")
    
    if not np.issubdtype(df[malus_col].dtype, np.number):
        raise TypeError(f"La colonne '{malus_col}' doit être numérique.")

    bonus_max = df[malus_col].max()
    threshold = bonus_max - level

    mask = df[malus_col] < threshold
    filtered = df.loc[mask, :]

    if verbose:
        retained = mask.sum()
        total = len(df)
        pct = retained / total * 100
        print(f"✅ Level {level} → seuil < {threshold} | {retained}/{total} lignes conservées ({pct:.1f}%) — {total - retained} supprimées")


    return filtered
