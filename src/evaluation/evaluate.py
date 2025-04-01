"""
Main evaluation script for OCR correction research.
This script evaluates and compares different OCR correction approaches.
"""

import os
import json
import argparse
import logging
from typing import Dict, List, Optional, Tuple, Union
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from tqdm import tqdm

from src.evaluation.metrics import (
    character_error_rate,
    word_error_rate,
    error_distribution,
    error_improvement,
    language_specific_metrics,
    detailed_error_analysis
)

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

class OCRCorrectionEvaluator:
    """
    Evaluator for OCR correction approaches.
    
    This class provides methods to evaluate and compare different OCR correction approaches
    across multiple languages and datasets.
    """
    
    def __init__(
        self,
        ground_truth_dir: str,
        original_ocr_dir: str,
        results_dirs: Dict[str, str],
        output_dir: str,
        languages: Optional[List[str]] = None
    ):
        """
        Initialize the OCR correction evaluator.
        
        Args:
            ground_truth_dir: Directory containing ground truth texts.
            original_ocr_dir: Directory containing original OCR texts.
            results_dirs: Dictionary mapping approach names to result directories.
            output_dir: Directory to save evaluation results.
            languages: List of language codes to evaluate. If None, evaluate all available languages.
        """
        self.ground_truth_dir = ground_truth_dir
        self.original_ocr_dir = original_ocr_dir
        self.results_dirs = results_dirs
        self.output_dir = output_dir
        self.languages = languages
        
        # Create output directory if it doesn't exist
        os.makedirs(output_dir, exist_ok=True)
        
        # Initialize results storage
        self.results = {
            "by_language": {},
            "by_approach": {},
            "overall": {}
        }
        
        logger.info(f"Initialized evaluator with {len(results_dirs)} approaches")
    
    def load_texts(self, directory: str, language: str) -> Dict[str, str]:
        """
        Load texts from a directory for a specific language.
        
        Args:
            directory: Directory containing text files.
            language: Language code.
            
        Returns:
            Dictionary mapping file names to text content.
        """
        texts = {}
        language_dir = os.path.join(directory, language)
        
        if not os.path.exists(language_dir):
            logger.warning(f"Directory not found: {language_dir}")
            return texts
        
        for filename in os.listdir(language_dir):
            if filename.endswith(".txt"):
                file_path = os.path.join(language_dir, filename)
                try:
                    with open(file_path, "r", encoding="utf-8") as f:
                        text = f.read()
                    
                    # Use base filename without extension as key
                    base_name = os.path.splitext(filename)[0]
                    texts[base_name] = text
                except Exception as e:
                    logger.error(f"Error loading {file_path}: {str(e)}")
        
        logger.info(f"Loaded {len(texts)} texts from {language_dir}")
        return texts
    
    def evaluate_language(self, language: str) -> Dict:
        """
        Evaluate all approaches for a specific language.
        
        Args:
            language: Language code.
            
        Returns:
            Dictionary with evaluation results for the language.
        """
        logger.info(f"Evaluating language: {language}")
        
        # Load ground truth texts
        ground_truth_texts = self.load_texts(self.ground_truth_dir, language)
        
        if not ground_truth_texts:
            logger.warning(f"No ground truth texts found for language: {language}")
            return {}
        
        # Load original OCR texts
        original_ocr_texts = self.load_texts(self.original_ocr_dir, language)
        
        # Initialize results for this language
        language_results = {
            "document_count": len(ground_truth_texts),
            "approaches": {}
        }
        
        # Evaluate each approach
        for approach_name, results_dir in self.results_dirs.items():
            # Load corrected texts for this approach
            corrected_texts = self.load_texts(results_dir, language)
            
            # Initialize metrics for this approach
            approach_metrics = {
                "overall": {
                    "cer": 0.0,
                    "wer": 0.0,
                    "cer_improvement": 0.0,
                    "wer_improvement": 0.0
                },
                "documents": {}
            }
            
            # Evaluate each document
            valid_docs = 0
            for doc_name, ground_truth in ground_truth_texts.items():
                # Skip if original OCR or corrected text is missing
                if doc_name not in original_ocr_texts or doc_name not in corrected_texts:
                    continue
                
                original_ocr = original_ocr_texts[doc_name]
                corrected = corrected_texts[doc_name]
                
                # Calculate metrics
                original_cer = character_error_rate(ground_truth, original_ocr)
                original_wer = word_error_rate(ground_truth, original_ocr)
                
                corrected_cer = character_error_rate(ground_truth, corrected)
                corrected_wer = word_error_rate(ground_truth, corrected)
                
                improvement = error_improvement(original_ocr, corrected, ground_truth)
                
                # Store document-level metrics
                approach_metrics["documents"][doc_name] = {
                    "original_cer": original_cer,
                    "original_wer": original_wer,
                    "corrected_cer": corrected_cer,
                    "corrected_wer": corrected_wer,
                    "cer_improvement": improvement["cer_relative_improvement"],
                    "wer_improvement": improvement["wer_relative_improvement"]
                }
                
                # Accumulate for overall metrics
                approach_metrics["overall"]["cer"] += corrected_cer
                approach_metrics["overall"]["wer"] += corrected_wer
                approach_metrics["overall"]["cer_improvement"] += improvement["cer_relative_improvement"]
                approach_metrics["overall"]["wer_improvement"] += improvement["wer_relative_improvement"]
                
                valid_docs += 1
            
            # Calculate averages for overall metrics
            if valid_docs > 0:
                approach_metrics["overall"]["cer"] /= valid_docs
                approach_metrics["overall"]["wer"] /= valid_docs
                approach_metrics["overall"]["cer_improvement"] /= valid_docs
                approach_metrics["overall"]["wer_improvement"] /= valid_docs
            
            # Store approach results
            language_results["approaches"][approach_name] = approach_metrics
        
        return language_results
    
    def evaluate_all(self) -> Dict:
        """
        Evaluate all approaches for all languages.
        
        Returns:
            Dictionary with complete evaluation results.
        """
        # Determine languages to evaluate
        if self.languages is None:
            # Auto-detect languages from ground truth directory
            self.languages = [d for d in os.listdir(self.ground_truth_dir) 
                             if os.path.isdir(os.path.join(self.ground_truth_dir, d))]
        
        logger.info(f"Evaluating languages: {', '.join(self.languages)}")
        
        # Evaluate each language
        for language in self.languages:
            language_results = self.evaluate_language(language)
            if language_results:
                self.results["by_language"][language] = language_results
        
        # Aggregate results by approach
        for approach_name in self.results_dirs.keys():
            approach_results = {
                "overall": {
                    "cer": 0.0,
                    "wer": 0.0,
                    "cer_improvement": 0.0,
                    "wer_improvement": 0.0
                },
                "by_language": {}
            }
            
            # Collect results for this approach across all languages
            language_count = 0
            for language, language_results in self.results["by_language"].items():
                if approach_name in language_results["approaches"]:
                    approach_lang_metrics = language_results["approaches"][approach_name]["overall"]
                    
                    # Store language-specific metrics for this approach
                    approach_results["by_language"][language] = approach_lang_metrics
                    
                    # Accumulate for overall metrics
                    approach_results["overall"]["cer"] += approach_lang_metrics["cer"]
                    approach_results["overall"]["wer"] += approach_lang_metrics["wer"]
                    approach_results["overall"]["cer_improvement"] += approach_lang_metrics["cer_improvement"]
                    approach_results["overall"]["wer_improvement"] += approach_lang_metrics["wer_improvement"]
                    
                    language_count += 1
            
            # Calculate averages for overall metrics
            if language_count > 0:
                approach_results["overall"]["cer"] /= language_count
                approach_results["overall"]["wer"] /= language_count
                approach_results["overall"]["cer_improvement"] /= language_count
                approach_results["overall"]["wer_improvement"] /= language_count
            
            # Store approach results
            self.results["by_approach"][approach_name] = approach_results
        
        # Calculate overall results across all approaches
        overall_metrics = {
            "cer": 0.0,
            "wer": 0.0,
            "cer_improvement": 0.0,
            "wer_improvement": 0.0
        }
        
        approach_count = len(self.results["by_approach"])
        if approach_count > 0:
            for approach_name, approach_results in self.results["by_approach"].items():
                overall_metrics["cer"] += approach_results["overall"]["cer"]
                overall_metrics["wer"] += approach_results["overall"]["wer"]
                overall_metrics["cer_improvement"] += approach_results["overall"]["cer_improvement"]
                overall_metrics["wer_improvement"] += approach_results["overall"]["wer_improvement"]
            
            # Calculate averages
            overall_metrics["cer"] /= approach_count
            overall_metrics["wer"] /= approach_count
            overall_metrics["cer_improvement"] /= approach_count
            overall_metrics["wer_improvement"] /= approach_count
        
        self.results["overall"] = overall_metrics
        
        return self.results
    
    def save_results(self) -> None:
        """
        Save evaluation results to files.
        """
        # Save complete results as JSON
        results_path = os.path.join(self.output_dir, "evaluation_results.json")
        with open(results_path, "w", encoding="utf-8") as f:
            json.dump(self.results, f, indent=2)
        
        logger.info(f"Saved complete results to {results_path}")
        
        # Save summary as CSV
        summary_data = []
        
        for approach_name, approach_results in self.results["by_approach"].items():
            for language, language_metrics in approach_results["by_language"].items():
                summary_data.append({
                    "Approach": approach_name,
                    "Language": language,
                    "CER": language_metrics["cer"],
                    "WER": language_metrics["wer"],
                    "CER Improvement": language_metrics["cer_improvement"],
                    "WER Improvement": language_metrics["wer_improvement"]
                })
        
        summary_df = pd.DataFrame(summary_data)
        summary_path = os.path.join(self.output_dir, "evaluation_summary.csv")
        summary_df.to_csv(summary_path, index=False)
        
        logger.info(f"Saved summary to {summary_path}")
    
    def generate_plots(self) -> None:
        """
        Generate plots from evaluation results.
        """
        # Create plots directory
        plots_dir = os.path.join(self.output_dir, "plots")
        os.makedirs(plots_dir, exist_ok=True)
        
        # Prepare data for plotting
        plot_data = []
        
        for approach_name, approach_results in self.results["by_approach"].items():
            for language, language_metrics in approach_results["by_language"].items():
                plot_data.append({
                    "Approach": approach_name,
                    "Language": language,
                    "CER": language_metrics["cer"],
                    "WER": language_metrics["wer"],
                    "CER Improvement": language_metrics["cer_improvement"],
                    "WER Improvement": language_metrics["wer_improvement"]
                })
        
        plot_df = pd.DataFrame(plot_data)
        
        # Set plot style
        sns.set(style="whitegrid")
        
        # Plot 1: CER by approach and language
        plt.figure(figsize=(12, 8))
        cer_plot = sns.barplot(x="Language", y="CER", hue="Approach", data=plot_df)
        plt.title("Character Error Rate by Approach and Language")
        plt.xlabel("Language")
        plt.ylabel("Character Error Rate (CER)")
        plt.xticks(rotation=45)
        plt.tight_layout()
        plt.savefig(os.path.join(plots_dir, "cer_by_language.png"))
        plt.close()
        
        # Plot 2: WER by approach and language
        plt.figure(figsize=(12, 8))
        wer_plot = sns.barplot(x="Language", y="WER", hue="Approach", data=plot_df)
        plt.title("Word Error Rate by Approach and Language")
        plt.xlabel("Language")
        plt.ylabel("Word Error Rate (WER)")
        plt.xticks(rotation=45)
        plt.tight_layout()
        plt.savefig(os.path.join(plots_dir, "wer_by_language.png"))
        plt.close()
        
        # Plot 3: CER Improvement by approach and language
        plt.figure(figsize=(12, 8))
        cer_imp_plot = sns.barplot(x="Language", y="CER Improvement", hue="Approach", data=plot_df)
        plt.title("CER Improvement by Approach and Language")
        plt.xlabel("Language")
        plt.ylabel("CER Improvement (%)")
        plt.xticks(rotation=45)
        plt.tight_layout()
        plt.savefig(os.path.join(plots_dir, "cer_improvement_by_language.png"))
        plt.close()
        
        # Plot 4: WER Improvement by approach and language
        plt.figure(figsize=(12, 8))
        wer_imp_plot = sns.barplot(x="Language", y="WER Improvement", hue="Approach", data=plot_df)
        plt.title("WER Improvement by Approach and Language")
        plt.xlabel("Language")
        plt.ylabel("WER Improvement (%)")
        plt.xticks(rotation=45)
        plt.tight_layout()
        plt.savefig(os.path.join(plots_dir, "wer_improvement_by_language.png"))
        plt.close()
        
        # Plot 5: Overall comparison of approaches
        plt.figure(figsize=(10, 6))
        overall_df = pd.DataFrame([
            {
                "Approach": approach_name,
                "CER": approach_results["overall"]["cer"],
                "WER": approach_results["overall"]["wer"],
                "CER Improvement": approach_results["overall"]["cer_improvement"],
                "WER Improvement": approach_results["overall"]["wer_improvement"]
            }
            for approach_name, approach_results in self.results["by_approach"].items()
        ])
        
        overall_plot = sns.barplot(x="Approach", y="CER", data=overall_df, color="blue", alpha=0.5, label="CER")
        overall_plot = sns.barplot(x="Approach", y="WER", data=overall_df, color="red", alpha=0.5, label="WER")
        plt.title("Overall Error Rates by Approach")
        plt.xlabel("Approach")
        plt.ylabel("Error Rate")
        plt.legend()
        plt.tight_layout()
        plt.savefig(os.path.join(plots_dir, "overall_error_rates.png"))
        plt.close()
        
        logger.info(f"Generated plots in {plots_dir}")


def parse_args():
    """
    Parse command line arguments.
    """
    parser = argparse.ArgumentParser(description="Evaluate OCR correction approaches")
    
    parser.add_argument("--ground-truth-dir", type=str, required=True,
                        help="Directory containing ground truth texts")
    
    parser.add_argument("--original-ocr-dir", type=str, required=True,
                        help="Directory containing original OCR texts")
    
    parser.add_argument("--results-dirs", type=str, required=True, nargs="+",
                        help="Space-separated list of 'name:directory' pairs for results")
    
    parser.add_argument("--output-dir", type=str, default="experiments/analysis",
                        help="Directory to save evaluation results")
    
    parser.add_argument("--languages", type=str, nargs="+",
                        help="Space-separated list of language codes to evaluate")
    
    return parser.parse_args()


def main():
    """
    Main function to run the evaluation.
    """
    args = parse_args()
    
    # Parse results directories
    results_dirs = {}
    for result_dir_pair in args.results_dirs:
        name, directory = result_dir_pair.split(":", 1)
        results_dirs[name] = directory
    
    # Initialize evaluator
    evaluator = OCRCorrectionEvaluator(
        ground_truth_dir=args.ground_truth_dir,
        original_ocr_dir=args.original_ocr_dir,
        results_dirs=results_dirs,
        output_dir=args.output_dir,
        languages=args.languages
    )
    
    # Run evaluation
    results = evaluator.evaluate_all()
    
    # Save results
    evaluator.save_results()
    
    # Generate plots
    evaluator.generate_plots()
    
    logger.info("Evaluation completed successfully")


if __name__ == "__main__":
    main() 