"""
Implementation of evaluation metrics for OCR correction.
This module provides functions to calculate various metrics for OCR correction evaluation.
"""

from typing import Dict, List, Tuple, Union
import numpy as np
import re
from difflib import SequenceMatcher

def character_error_rate(reference: str, hypothesis: str) -> float:
    """
    Calculate the Character Error Rate (CER) between reference and hypothesis texts.
    
    CER = (S + D + I) / N
    where:
    S is the number of substitutions
    D is the number of deletions
    I is the number of insertions
    N is the number of characters in the reference
    
    Args:
        reference: Reference (ground truth) text.
        hypothesis: Hypothesis (corrected) text.
        
    Returns:
        Character Error Rate as a float between 0 and 1.
    """
    # Use SequenceMatcher to find the differences
    matcher = SequenceMatcher(None, reference, hypothesis)
    
    # Initialize counters
    substitutions = 0
    deletions = 0
    insertions = 0
    
    for tag, i1, i2, j1, j2 in matcher.get_opcodes():
        if tag == 'replace':
            # Substitution
            substitutions += max(i2 - i1, j2 - j1)
        elif tag == 'delete':
            # Deletion
            deletions += i2 - i1
        elif tag == 'insert':
            # Insertion
            insertions += j2 - j1
    
    # Calculate CER
    total_errors = substitutions + deletions + insertions
    reference_length = len(reference)
    
    if reference_length == 0:
        return 1.0 if total_errors > 0 else 0.0
    
    cer = total_errors / reference_length
    
    return cer

def word_error_rate(reference: str, hypothesis: str) -> float:
    """
    Calculate the Word Error Rate (WER) between reference and hypothesis texts.
    
    WER = (S + D + I) / N
    where:
    S is the number of substituted words
    D is the number of deleted words
    I is the number of inserted words
    N is the number of words in the reference
    
    Args:
        reference: Reference (ground truth) text.
        hypothesis: Hypothesis (corrected) text.
        
    Returns:
        Word Error Rate as a float between 0 and 1.
    """
    # Tokenize into words
    reference_words = reference.split()
    hypothesis_words = hypothesis.split()
    
    # Use SequenceMatcher to find the differences
    matcher = SequenceMatcher(None, reference_words, hypothesis_words)
    
    # Initialize counters
    substitutions = 0
    deletions = 0
    insertions = 0
    
    for tag, i1, i2, j1, j2 in matcher.get_opcodes():
        if tag == 'replace':
            # Substitution
            substitutions += max(i2 - i1, j2 - j1)
        elif tag == 'delete':
            # Deletion
            deletions += i2 - i1
        elif tag == 'insert':
            # Insertion
            insertions += j2 - j1
    
    # Calculate WER
    total_errors = substitutions + deletions + insertions
    reference_length = len(reference_words)
    
    if reference_length == 0:
        return 1.0 if total_errors > 0 else 0.0
    
    wer = total_errors / reference_length
    
    return wer

def error_distribution(reference: str, hypothesis: str) -> Dict[str, int]:
    """
    Calculate the distribution of error types between reference and hypothesis texts.
    
    Args:
        reference: Reference (ground truth) text.
        hypothesis: Hypothesis (corrected) text.
        
    Returns:
        Dictionary with counts of substitutions, deletions, and insertions.
    """
    # Use SequenceMatcher to find the differences
    matcher = SequenceMatcher(None, reference, hypothesis)
    
    # Initialize counters
    error_counts = {
        "substitutions": 0,
        "deletions": 0,
        "insertions": 0
    }
    
    for tag, i1, i2, j1, j2 in matcher.get_opcodes():
        if tag == 'replace':
            # Substitution
            error_counts["substitutions"] += max(i2 - i1, j2 - j1)
        elif tag == 'delete':
            # Deletion
            error_counts["deletions"] += i2 - i1
        elif tag == 'insert':
            # Insertion
            error_counts["insertions"] += j2 - j1
    
    return error_counts

def character_accuracy(reference: str, hypothesis: str) -> float:
    """
    Calculate the Character Accuracy between reference and hypothesis texts.
    
    Character Accuracy = 1 - CER
    
    Args:
        reference: Reference (ground truth) text.
        hypothesis: Hypothesis (corrected) text.
        
    Returns:
        Character Accuracy as a float between 0 and 1.
    """
    cer = character_error_rate(reference, hypothesis)
    return 1.0 - cer

def word_accuracy(reference: str, hypothesis: str) -> float:
    """
    Calculate the Word Accuracy between reference and hypothesis texts.
    
    Word Accuracy = 1 - WER
    
    Args:
        reference: Reference (ground truth) text.
        hypothesis: Hypothesis (corrected) text.
        
    Returns:
        Word Accuracy as a float between 0 and 1.
    """
    wer = word_error_rate(reference, hypothesis)
    return 1.0 - wer

def error_improvement(original_ocr: str, corrected: str, ground_truth: str) -> Dict[str, float]:
    """
    Calculate the improvement in error rates from original OCR to corrected text.
    
    Args:
        original_ocr: Original OCR text.
        corrected: Corrected text.
        ground_truth: Ground truth text.
        
    Returns:
        Dictionary with improvement metrics.
    """
    # Calculate error rates for original OCR
    original_cer = character_error_rate(ground_truth, original_ocr)
    original_wer = word_error_rate(ground_truth, original_ocr)
    
    # Calculate error rates for corrected text
    corrected_cer = character_error_rate(ground_truth, corrected)
    corrected_wer = word_error_rate(ground_truth, corrected)
    
    # Calculate relative improvement
    cer_improvement = (original_cer - corrected_cer) / original_cer if original_cer > 0 else 0.0
    wer_improvement = (original_wer - corrected_wer) / original_wer if original_wer > 0 else 0.0
    
    # Calculate absolute improvement
    cer_absolute_improvement = original_cer - corrected_cer
    wer_absolute_improvement = original_wer - corrected_wer
    
    return {
        "original_cer": original_cer,
        "corrected_cer": corrected_cer,
        "cer_relative_improvement": cer_improvement,
        "cer_absolute_improvement": cer_absolute_improvement,
        "original_wer": original_wer,
        "corrected_wer": corrected_wer,
        "wer_relative_improvement": wer_improvement,
        "wer_absolute_improvement": wer_absolute_improvement
    }

def language_specific_metrics(reference: str, hypothesis: str, language: str) -> Dict[str, float]:
    """
    Calculate language-specific metrics for OCR correction evaluation.
    
    Args:
        reference: Reference (ground truth) text.
        hypothesis: Hypothesis (corrected) text.
        language: Language code.
        
    Returns:
        Dictionary with language-specific metrics.
    """
    metrics = {}
    
    # Basic metrics for all languages
    metrics["cer"] = character_error_rate(reference, hypothesis)
    metrics["wer"] = word_error_rate(reference, hypothesis)
    
    # Language-specific metrics
    if language == "sa":  # Sanskrit
        # Example: Sanskrit-specific metrics
        # This could include diacritic accuracy, compound word accuracy, etc.
        pass
    elif language == "he":  # Hebrew
        # Example: Hebrew-specific metrics
        # This could include right-to-left text handling, niqqud accuracy, etc.
        pass
    elif language in ["ain", "grk", "ybh"]:  # Endangered languages
        # Example: Metrics for endangered languages
        # This could include special character handling, etc.
        pass
    
    return metrics

def detailed_error_analysis(reference: str, hypothesis: str) -> Dict:
    """
    Perform a detailed analysis of the errors between reference and hypothesis texts.
    
    Args:
        reference: Reference (ground truth) text.
        hypothesis: Hypothesis (corrected) text.
        
    Returns:
        Dictionary with detailed error analysis.
    """
    # Use SequenceMatcher to find the differences
    matcher = SequenceMatcher(None, reference, hypothesis)
    
    # Initialize analysis dictionary
    analysis = {
        "error_positions": [],
        "error_contexts": [],
        "error_types": {
            "substitutions": [],
            "deletions": [],
            "insertions": []
        },
        "common_errors": {}
    }
    
    # Analyze each difference
    for tag, i1, i2, j1, j2 in matcher.get_opcodes():
        if tag == 'replace':
            # Substitution
            ref_segment = reference[i1:i2]
            hyp_segment = hypothesis[j1:j2]
            
            # Record position and context
            position = i1
            context_start = max(0, i1 - 10)
            context_end = min(len(reference), i2 + 10)
            context = reference[context_start:context_end]
            
            analysis["error_positions"].append(position)
            analysis["error_contexts"].append(context)
            analysis["error_types"]["substitutions"].append((ref_segment, hyp_segment))
            
            # Record common error
            error_key = f"{ref_segment} -> {hyp_segment}"
            if error_key in analysis["common_errors"]:
                analysis["common_errors"][error_key] += 1
            else:
                analysis["common_errors"][error_key] = 1
                
        elif tag == 'delete':
            # Deletion
            ref_segment = reference[i1:i2]
            
            # Record position and context
            position = i1
            context_start = max(0, i1 - 10)
            context_end = min(len(reference), i2 + 10)
            context = reference[context_start:context_end]
            
            analysis["error_positions"].append(position)
            analysis["error_contexts"].append(context)
            analysis["error_types"]["deletions"].append(ref_segment)
            
            # Record common error
            error_key = f"{ref_segment} -> [DEL]"
            if error_key in analysis["common_errors"]:
                analysis["common_errors"][error_key] += 1
            else:
                analysis["common_errors"][error_key] = 1
                
        elif tag == 'insert':
            # Insertion
            hyp_segment = hypothesis[j1:j2]
            
            # Record position and context
            position = i1
            context_start = max(0, i1 - 10)
            context_end = min(len(reference), i1 + 10)
            context = reference[context_start:context_end]
            
            analysis["error_positions"].append(position)
            analysis["error_contexts"].append(context)
            analysis["error_types"]["insertions"].append(hyp_segment)
            
            # Record common error
            error_key = f"[INS] -> {hyp_segment}"
            if error_key in analysis["common_errors"]:
                analysis["common_errors"][error_key] += 1
            else:
                analysis["common_errors"][error_key] = 1
    
    # Sort common errors by frequency
    analysis["common_errors"] = dict(sorted(
        analysis["common_errors"].items(), 
        key=lambda item: item[1], 
        reverse=True
    ))
    
    return analysis


def main():
    """
    Example usage of the evaluation metrics.
    """
    # Example texts
    reference = "This is a sample text for OCR evaluation."
    hypothesis = "Th1s is a sampl text fr OCR evaluaton."
    original_ocr = "Th1s iz a sampl txt fr OCR evaluaton."
    
    # Calculate metrics
    cer = character_error_rate(reference, hypothesis)
    wer = word_error_rate(reference, hypothesis)
    error_dist = error_distribution(reference, hypothesis)
    improvement = error_improvement(original_ocr, hypothesis, reference)
    
    # Print results
    print(f"Character Error Rate (CER): {cer:.4f}")
    print(f"Word Error Rate (WER): {wer:.4f}")
    print(f"Error Distribution: {error_dist}")
    print(f"Improvement from original OCR: {improvement}")


if __name__ == "__main__":
    main() 