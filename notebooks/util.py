import sys
import os
import json
from os import path
from zipfile import ZipFile

# Add the parent directory to the sys.path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
script_path = os.path.abspath(__file__)
script_dir = os.path.dirname(script_path)
base_dir = os.path.dirname(os.path.dirname(script_dir))
folder_name = os.path.basename(script_dir)

print(base_dir+"/KBUtilLib/src")
sys.path = [base_dir+"/KBUtilLib/src",base_dir+"/cobrakbase",base_dir+"/ModelSEEDpy/"] + sys.path

# Import utilities with error handling
from kbutillib import ModelStandardizationUtils, MSFBAUtils, AICurationUtils, NotebookUtils, EscherUtils, KBPLMUtils

import hashlib
import pandas as pd
from pandas import DataFrame, read_csv, concat, set_option
from cobrakbase.core.kbasefba import FBAModel
import cobra
from cobra import Reaction, Metabolite
from cobra.flux_analysis import pfba
from cobra.io import save_json_model, load_json_model
from modelseedpy import AnnotationOntology, MSPackageManager, MSMedia, MSModelUtil, MSBuilder, MSATPCorrection, MSGapfill, MSGrowthPhenotype, MSGrowthPhenotypes, ModelSEEDBiochem, MSExpression
import re
import copy
import numpy as np

# Define the base classes based on what's available
# Note: KBPLMUtils inherits from KBGenomeUtils, so we use KBPLMUtils instead of KBGenomeUtils
class NotebookUtil(ModelStandardizationUtils,MSFBAUtils, AICurationUtils, NotebookUtils, KBPLMUtils, EscherUtils):
    def __init__(self,**kwargs):
        super().__init__(
            notebook_folder=script_dir,
            name="ANMENotebookUtils",
            user="chenry",
            retries=5,
            proxy_port=None,
            **kwargs
        )

    def parse_reaction_formula(self, formula_str):
        """Parse a reaction formula string into substrates and products with stoichiometry."""
        if pd.isna(formula_str) or not formula_str:
            return None, None

        # Fix common typos
        formula_str = formula_str.replace('2h[c]', '2 h[c]')
        formula_str = formula_str.replace('tpicox [m]', 'tpicox[m]')

        # Split by reaction arrow
        if '<=>' in formula_str:
            parts = formula_str.split('<=>')
        elif '->' in formula_str:
            parts = formula_str.split('->')
        elif '=' in formula_str:
            parts = formula_str.split('=')
        else:
            return None, None

        if len(parts) != 2:
            return None, None

        substrates_str, products_str = parts[0].strip(), parts[1].strip()

        def parse_metabolites(met_str):
            metabolites = {}
            for item in met_str.split('+'):
                item = item.strip()
                if not item:
                    continue
                parts_match = re.match(r'^([\d.]+)\s+(.+)$', item)
                if parts_match:
                    coeff = float(parts_match.group(1))
                    met_id = parts_match.group(2).strip()
                else:
                    coeff = 1.0
                    met_id = item.strip()
                metabolites[met_id] = coeff
            return metabolites

        return parse_metabolites(substrates_str), parse_metabolites(products_str)

    def extract_cpd_rxn_translations(self, translation_data):
        """Extract compound and reaction translations from translation_results structure.

        The translation_results.json has this structure:
        [
            cpd_translations_dict,   # {cpd_id: [ms_id, metadata]}
            rxn_translations_dict,   # {rxn_id: [ms_id, score]}
            cpd_matches_dict,        # detailed compound match info
            rxn_matches_dict         # detailed reaction match info
        ]

        This function converts them to simple dictionaries for apply_translation_to_model:
        - cpd_translations: {cpd_id: ms_id}
        - rxn_translations: {rxn_id: [ms_id, score]}

        Args:
            translation_data: List containing translation results from translate_model_to_ms_namespace

        Returns:
            Tuple of (cpd_translations, rxn_translations)
        """
        cpd_trans_raw = translation_data[0]
        rxn_trans_raw = translation_data[1]

        # Convert compound translations from [ms_id, metadata] to just ms_id
        cpd_translations = {}
        for cpd_id, trans_data in cpd_trans_raw.items():
            if isinstance(trans_data, list) and len(trans_data) > 0:
                cpd_translations[cpd_id] = trans_data[0]  # Get the ModelSEED ID
            elif isinstance(trans_data, str):
                cpd_translations[cpd_id] = trans_data

        # Reaction translations are already in the right format: {rxn_id: [ms_id, score]}
        rxn_translations = rxn_trans_raw

        return (cpd_translations, rxn_translations)

# Initialize the NotebookUtil instance
util = NotebookUtil() 