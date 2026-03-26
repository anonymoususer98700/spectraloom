"""
Helper functions for loading ZuCo v2.0 HDF5 (.mat) files.
Provides utilities to extract string data and word-level EEG features
from the HDF5 structure used by MATLAB v7.3+ .mat files.
"""

import numpy as np


def load_matlab_string(matlab_extracted_object):
    """
    Load a MATLAB string stored in an HDF5 file.
    MATLAB stores strings as arrays of uint16 values (Unicode code points).
    """
    extracted_string = u''
    for c in matlab_extracted_object:
        extracted_string += chr(c[0])
    return extracted_string


def extract_word_level_data(f, word_data_ref):
    """
    Extract word-level EEG data from HDF5 word data reference.
    
    Args:
        f: HDF5 file object
        word_data_ref: reference to the word data group in the HDF5 file
    
    Returns:
        word_data: list of dicts with word content, fixation count, and EEG features
        word_tokens_all: list of all word tokens
        word_tokens_has_fixation: list of word tokens that have fixation data
        word_tokens_with_mask: list of word tokens with [MASK] for missing fixations
    """
    word_data = []
    word_tokens_all = []
    word_tokens_has_fixation = []
    word_tokens_with_mask = []

    try:
        # Get references to word-level data fields
        content_refs = word_data_ref['content']
        nfix_refs = word_data_ref['nFixations']

        # EEG feature references
        eeg_fields = {
            'FFD': ['FFD_t1', 'FFD_t2', 'FFD_a1', 'FFD_a2', 'FFD_b1', 'FFD_b2', 'FFD_g1', 'FFD_g2'],
            'GD': ['GD_t1', 'GD_t2', 'GD_a1', 'GD_a2', 'GD_b1', 'GD_b2', 'GD_g1', 'GD_g2'],
            'TRT': ['TRT_t1', 'TRT_t2', 'TRT_a1', 'TRT_a2', 'TRT_b1', 'TRT_b2', 'TRT_g1', 'TRT_g2'],
        }

        # Check which EEG fields are available
        available_eeg = {}
        for eeg_type, bands in eeg_fields.items():
            available_bands = {}
            for band in bands:
                if band in word_data_ref:
                    available_bands[band] = word_data_ref[band]
            if available_bands:
                available_eeg[eeg_type] = available_bands

        num_words = len(content_refs)

        for widx in range(num_words):
            # Get word content string
            word_string = load_matlab_string(f[content_refs[widx][0]])
            word_tokens_all.append(word_string)

            # Get number of fixations
            nfix_raw = f[nfix_refs[widx][0]][()]
            nfix_squeezed = np.squeeze(nfix_raw)
            # Handle both scalar and array cases
            if nfix_squeezed.ndim == 0:
                nfix_val = int(nfix_squeezed)
            else:
                nfix_val = int(nfix_squeezed.flat[0]) if nfix_squeezed.size > 0 else 0

            data_dict = {
                'content': word_string,
                'nFix': nfix_val,
            }

            if nfix_val > 0:
                word_tokens_has_fixation.append(word_string)
                word_tokens_with_mask.append(word_string)

                # Extract EEG features for each type (GD, FFD, TRT)
                for eeg_type, bands_dict in available_eeg.items():
                    eeg_features = []
                    for band_name in eeg_fields[eeg_type]:
                        if band_name in bands_dict:
                            raw_data = f[bands_dict[band_name][widx][0]][()]
                            band_data = np.squeeze(raw_data)
                            # Ensure it's a 1D array
                            if band_data.ndim == 0:
                                band_data = np.array([float(band_data)])
                            eeg_features.append(band_data)
                    if eeg_features:
                        data_dict[f'{eeg_type}_EEG'] = eeg_features
            else:
                word_tokens_with_mask.append('[MASK]')

            word_data.append(data_dict)

    except Exception as e:
        print(f'Error extracting word level data: {e}')
        return {}, [], [], []

    return word_data, word_tokens_all, word_tokens_has_fixation, word_tokens_with_mask
