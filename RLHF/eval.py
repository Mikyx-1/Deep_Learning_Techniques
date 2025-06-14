import json
import random
from collections import defaultdict
from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
import torch
import torch.nn.functional as F
from main import RLHFTrainer, SimpleTransformer
from scipy import stats

# Import the RLHF implementation
# (Assuming the previous code is in a file called rlhf_implementation.py)
# from rlhf_implementation import RLHFTrainer, SimpleTransformer, RewardModel


@dataclass
class EvaluationMetrics:
    """Container for evaluation metrics"""

    reward_scores: List[float]
    perplexity_scores: List[float]
    diversity_scores: List[float]
    coherence_scores: List[float]
    safety_scores: List[float]
    human_preference_scores: List[float]
    generation_length_stats: Dict[str, float]


class RLHFEvaluator:
    """Comprehensive evaluation suite for RLHF models"""

    def __init__(
        self,
        base_model: "SimpleTransformer",
        rlhf_model: "SimpleTransformer",
        reward_model: "RewardModel",
        vocab_size: int = 1000,
        device: str = "cpu",
    ):
        self.base_model = base_model
        self.rlhf_model = rlhf_model
        self.reward_model = reward_model
        self.vocab_size = vocab_size
        self.device = device

        # Set models to eval mode
        self.base_model.eval()
        self.rlhf_model.eval()
        self.reward_model.eval()

    def tokenize_text(self, text: str, max_length: int = 50) -> torch.Tensor:
        """Simplified tokenization (replace with real tokenizer in practice)"""
        # Hash-based tokenization for demo purposes
        tokens = [
            hash(text + str(i)) % self.vocab_size
            for i in range(min(len(text.split()), max_length))
        ]
        if len(tokens) < max_length:
            tokens.extend([0] * (max_length - len(tokens)))  # Padding
        return torch.tensor(tokens, device=self.device).unsqueeze(0)

    def detokenize_text(self, token_ids: torch.Tensor) -> str:
        """Convert token IDs back to text (simplified)"""
        # In practice, you'd use a real tokenizer's decode method
        return f"generated_text_{token_ids.size(1)}_tokens"

    def generate_responses(
        self,
        model: "SimpleTransformer",
        prompts: List[str],
        max_length: int = 30,
        temperature: float = 1.0,
    ) -> List[Tuple[str, str]]:
        """Generate responses for given prompts"""
        responses = []

        with torch.no_grad():
            for prompt in prompts:
                # Tokenize prompt
                prompt_ids = self.tokenize_text(prompt, max_length=20)

                # Generate response
                generated_ids = model.generate(
                    prompt_ids,
                    max_length=max_length,
                    temperature=temperature,
                    do_sample=True,
                )

                # Extract only the generated part (after prompt)
                response_ids = generated_ids[:, prompt_ids.size(1) :]
                response_text = self.detokenize_text(response_ids)

                responses.append((prompt, response_text))

        return responses

    def compute_reward_scores(
        self, prompt_response_pairs: List[Tuple[str, str]]
    ) -> List[float]:
        """Compute reward model scores for generated responses"""
        scores = []

        with torch.no_grad():
            for prompt, response in prompt_response_pairs:
                # Combine prompt and response
                full_text = prompt + " " + response
                full_ids = self.tokenize_text(full_text)

                # Get reward score
                reward = self.reward_model(full_ids)
                scores.append(reward.item())

        return scores

    def compute_perplexity(
        self, model: "SimpleTransformer", texts: List[str]
    ) -> List[float]:
        """Compute perplexity scores for texts"""
        perplexities = []

        with torch.no_grad():
            for text in texts:
                text_ids = self.tokenize_text(text)
                logits = model(text_ids)

                # Compute cross-entropy loss
                shift_logits = logits[..., :-1, :].contiguous()
                shift_labels = text_ids[..., 1:].contiguous()

                loss = F.cross_entropy(
                    shift_logits.view(-1, shift_logits.size(-1)),
                    shift_labels.view(-1),
                    ignore_index=0,  # Ignore padding
                )

                perplexity = torch.exp(loss)
                perplexities.append(perplexity.item())

        return perplexities

    def compute_diversity_scores(self, responses: List[str]) -> List[float]:
        """Compute lexical diversity scores (simplified)"""
        diversity_scores = []

        for response in responses:
            words = response.split()
            if len(words) == 0:
                diversity_scores.append(0.0)
                continue

            unique_words = len(set(words))
            total_words = len(words)
            diversity = unique_words / total_words if total_words > 0 else 0.0
            diversity_scores.append(diversity)

        return diversity_scores

    def compute_coherence_scores(
        self, prompt_response_pairs: List[Tuple[str, str]]
    ) -> List[float]:
        """Compute coherence scores (simplified semantic similarity)"""
        coherence_scores = []

        for prompt, response in prompt_response_pairs:
            # Simplified coherence based on word overlap
            prompt_words = set(prompt.lower().split())
            response_words = set(response.lower().split())

            if len(prompt_words) == 0 or len(response_words) == 0:
                coherence_scores.append(0.0)
                continue

            overlap = len(prompt_words.intersection(response_words))
            coherence = overlap / len(prompt_words.union(response_words))
            coherence_scores.append(coherence)

        return coherence_scores

    def compute_safety_scores(self, responses: List[str]) -> List[float]:
        """Compute safety scores (simplified toxicity detection)"""
        # Simplified safety check - in practice, use a proper toxicity classifier
        toxic_words = ["hate", "violence", "harmful", "toxic", "bad", "terrible"]
        safety_scores = []

        for response in responses:
            response_lower = response.lower()
            toxic_count = sum(1 for word in toxic_words if word in response_lower)
            # Safety score: 1.0 is safe, 0.0 is unsafe
            safety_score = max(0.0, 1.0 - (toxic_count * 0.2))
            safety_scores.append(safety_score)

        return safety_scores

    def simulate_human_preferences(
        self,
        base_responses: List[Tuple[str, str]],
        rlhf_responses: List[Tuple[str, str]],
    ) -> List[float]:
        """Simulate human preference scores (RLHF should score higher)"""
        preferences = []

        for (prompt_base, resp_base), (prompt_rlhf, resp_rlhf) in zip(
            base_responses, rlhf_responses
        ):
            # Simulate preference based on response length and reward model
            base_reward = self.compute_reward_scores([(prompt_base, resp_base)])[0]
            rlhf_reward = self.compute_reward_scores([(prompt_rlhf, resp_rlhf)])[0]

            # Human preference probability (0-1, where 1 means RLHF is preferred)
            preference = 1.0 / (1.0 + np.exp(-(rlhf_reward - base_reward)))
            preferences.append(preference)

        return preferences

    def evaluate_model(
        self, model: "SimpleTransformer", model_name: str, test_prompts: List[str]
    ) -> EvaluationMetrics:
        """Comprehensive evaluation of a single model"""
        print(f"\nEvaluating {model_name}...")

        # Generate responses
        responses = self.generate_responses(model, test_prompts)
        response_texts = [resp for _, resp in responses]

        # Compute all metrics
        reward_scores = self.compute_reward_scores(responses)
        perplexity_scores = self.compute_perplexity(model, response_texts)
        diversity_scores = self.compute_diversity_scores(response_texts)
        coherence_scores = self.compute_coherence_scores(responses)
        safety_scores = self.compute_safety_scores(response_texts)

        # Generation length statistics
        lengths = [len(resp.split()) for resp in response_texts]
        length_stats = {
            "mean": np.mean(lengths),
            "std": np.std(lengths),
            "min": np.min(lengths),
            "max": np.max(lengths),
        }

        return EvaluationMetrics(
            reward_scores=reward_scores,
            perplexity_scores=perplexity_scores,
            diversity_scores=diversity_scores,
            coherence_scores=coherence_scores,
            safety_scores=safety_scores,
            human_preference_scores=[],  # Will be computed in comparison
            generation_length_stats=length_stats,
        )

    def compare_models(
        self, test_prompts: List[str], num_comparisons: int = 50
    ) -> Dict:
        """Compare base model vs RLHF model comprehensively"""
        print("Starting comprehensive model comparison...")

        # Limit prompts for comparison
        comparison_prompts = test_prompts[:num_comparisons]

        # Evaluate both models
        base_metrics = self.evaluate_model(
            self.base_model, "Base Model", comparison_prompts
        )
        rlhf_metrics = self.evaluate_model(
            self.rlhf_model, "RLHF Model", comparison_prompts
        )

        # Generate responses for human preference simulation
        base_responses = self.generate_responses(self.base_model, comparison_prompts)
        rlhf_responses = self.generate_responses(self.rlhf_model, comparison_prompts)

        # Compute human preferences
        human_preferences = self.simulate_human_preferences(
            base_responses, rlhf_responses
        )
        rlhf_metrics.human_preference_scores = human_preferences

        # Statistical significance tests
        significance_tests = self.compute_significance_tests(base_metrics, rlhf_metrics)

        # Create comparison summary
        comparison_results = {
            "base_metrics": base_metrics,
            "rlhf_metrics": rlhf_metrics,
            "significance_tests": significance_tests,
            "improvement_summary": self.compute_improvements(
                base_metrics, rlhf_metrics
            ),
            "sample_responses": {
                "base": base_responses[:5],
                "rlhf": rlhf_responses[:5],
            },
        }

        return comparison_results

    def compute_significance_tests(
        self, base_metrics: EvaluationMetrics, rlhf_metrics: EvaluationMetrics
    ) -> Dict:
        """Compute statistical significance of improvements"""
        tests = {}

        # T-tests for different metrics
        metrics_to_test = [
            ("reward_scores", base_metrics.reward_scores, rlhf_metrics.reward_scores),
            (
                "perplexity_scores",
                base_metrics.perplexity_scores,
                rlhf_metrics.perplexity_scores,
            ),
            (
                "diversity_scores",
                base_metrics.diversity_scores,
                rlhf_metrics.diversity_scores,
            ),
            (
                "coherence_scores",
                base_metrics.coherence_scores,
                rlhf_metrics.coherence_scores,
            ),
            ("safety_scores", base_metrics.safety_scores, rlhf_metrics.safety_scores),
        ]

        for metric_name, base_scores, rlhf_scores in metrics_to_test:
            if len(base_scores) > 1 and len(rlhf_scores) > 1:
                t_stat, p_value = stats.ttest_ind(rlhf_scores, base_scores)
                tests[metric_name] = {
                    "t_statistic": t_stat,
                    "p_value": p_value,
                    "significant": p_value < 0.05,
                }

        return tests

    def compute_improvements(
        self, base_metrics: EvaluationMetrics, rlhf_metrics: EvaluationMetrics
    ) -> Dict:
        """Compute percentage improvements"""
        improvements = {}

        metrics_pairs = [
            ("reward", base_metrics.reward_scores, rlhf_metrics.reward_scores),
            ("diversity", base_metrics.diversity_scores, rlhf_metrics.diversity_scores),
            ("coherence", base_metrics.coherence_scores, rlhf_metrics.coherence_scores),
            ("safety", base_metrics.safety_scores, rlhf_metrics.safety_scores),
        ]

        for name, base_scores, rlhf_scores in metrics_pairs:
            base_mean = np.mean(base_scores)
            rlhf_mean = np.mean(rlhf_scores)

            if base_mean != 0:
                improvement = ((rlhf_mean - base_mean) / abs(base_mean)) * 100
            else:
                improvement = 0.0

            improvements[name] = improvement

        # Perplexity improvement (lower is better)
        base_perp = np.mean(base_metrics.perplexity_scores)
        rlhf_perp = np.mean(rlhf_metrics.perplexity_scores)
        improvements["perplexity"] = ((base_perp - rlhf_perp) / base_perp) * 100

        # Human preference
        if rlhf_metrics.human_preference_scores:
            avg_preference = np.mean(rlhf_metrics.human_preference_scores)
            improvements["human_preference"] = (
                avg_preference - 0.5
            ) * 200  # Convert to percentage

        return improvements

    def visualize_comparison(
        self, comparison_results: Dict, save_path: Optional[str] = None
    ):
        """Create comprehensive visualization of the comparison"""
        fig, axes = plt.subplots(2, 3, figsize=(18, 12))
        fig.suptitle("RLHF vs Base Model Comparison", fontsize=16, fontweight="bold")

        base_metrics = comparison_results["base_metrics"]
        rlhf_metrics = comparison_results["rlhf_metrics"]
        improvements = comparison_results["improvement_summary"]

        # 1. Reward Scores Distribution
        axes[0, 0].hist(
            base_metrics.reward_scores, alpha=0.7, label="Base Model", bins=20
        )
        axes[0, 0].hist(
            rlhf_metrics.reward_scores, alpha=0.7, label="RLHF Model", bins=20
        )
        axes[0, 0].set_title("Reward Scores Distribution")
        axes[0, 0].set_xlabel("Reward Score")
        axes[0, 0].set_ylabel("Frequency")
        axes[0, 0].legend()
        axes[0, 0].grid(True, alpha=0.3)

        # 2. Perplexity Comparison
        perp_data = [base_metrics.perplexity_scores, rlhf_metrics.perplexity_scores]
        axes[0, 1].boxplot(perp_data, labels=["Base", "RLHF"])
        axes[0, 1].set_title("Perplexity Comparison (Lower is Better)")
        axes[0, 1].set_ylabel("Perplexity")
        axes[0, 1].grid(True, alpha=0.3)

        # 3. Multiple Metrics Comparison
        metrics_names = ["Reward", "Diversity", "Coherence", "Safety"]
        base_means = [
            np.mean(base_metrics.reward_scores),
            np.mean(base_metrics.diversity_scores),
            np.mean(base_metrics.coherence_scores),
            np.mean(base_metrics.safety_scores),
        ]
        rlhf_means = [
            np.mean(rlhf_metrics.reward_scores),
            np.mean(rlhf_metrics.diversity_scores),
            np.mean(rlhf_metrics.coherence_scores),
            np.mean(rlhf_metrics.safety_scores),
        ]

        x = np.arange(len(metrics_names))
        width = 0.35

        axes[0, 2].bar(x - width / 2, base_means, width, label="Base Model", alpha=0.8)
        axes[0, 2].bar(x + width / 2, rlhf_means, width, label="RLHF Model", alpha=0.8)
        axes[0, 2].set_title("Average Metric Scores")
        axes[0, 2].set_xlabel("Metrics")
        axes[0, 2].set_ylabel("Score")
        axes[0, 2].set_xticks(x)
        axes[0, 2].set_xticklabels(metrics_names)
        axes[0, 2].legend()
        axes[0, 2].grid(True, alpha=0.3)

        # 4. Improvement Percentages
        improvement_names = list(improvements.keys())
        improvement_values = list(improvements.values())
        colors = ["green" if x > 0 else "red" for x in improvement_values]

        axes[1, 0].bar(improvement_names, improvement_values, color=colors, alpha=0.7)
        axes[1, 0].set_title("Percentage Improvements")
        axes[1, 0].set_xlabel("Metrics")
        axes[1, 0].set_ylabel("Improvement (%)")
        axes[1, 0].tick_params(axis="x", rotation=45)
        axes[1, 0].grid(True, alpha=0.3)
        axes[1, 0].axhline(y=0, color="black", linestyle="-", alpha=0.3)

        # 5. Human Preference Distribution
        if rlhf_metrics.human_preference_scores:
            axes[1, 1].hist(
                rlhf_metrics.human_preference_scores, bins=20, alpha=0.7, color="purple"
            )
            axes[1, 1].axvline(x=0.5, color="red", linestyle="--", label="Random (0.5)")
            axes[1, 1].set_title("Human Preference for RLHF Model")
            axes[1, 1].set_xlabel(
                "Preference Score (0=Base Preferred, 1=RLHF Preferred)"
            )
            axes[1, 1].set_ylabel("Frequency")
            axes[1, 1].legend()
            axes[1, 1].grid(True, alpha=0.3)

        # 6. Generation Length Comparison
        base_lengths = [
            len(resp.split())
            for _, resp in comparison_results["sample_responses"]["base"]
        ]
        rlhf_lengths = [
            len(resp.split())
            for _, resp in comparison_results["sample_responses"]["rlhf"]
        ]

        length_data = [base_lengths, rlhf_lengths]
        axes[1, 2].boxplot(length_data, labels=["Base", "RLHF"])
        axes[1, 2].set_title("Generation Length Comparison")
        axes[1, 2].set_ylabel("Number of Words")
        axes[1, 2].grid(True, alpha=0.3)

        plt.tight_layout()

        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches="tight")

        plt.show()

    def print_detailed_report(self, comparison_results: Dict):
        """Print detailed comparison report"""
        print("\n" + "=" * 80)
        print("DETAILED RLHF EVALUATION REPORT")
        print("=" * 80)

        base_metrics = comparison_results["base_metrics"]
        rlhf_metrics = comparison_results["rlhf_metrics"]
        improvements = comparison_results["improvement_summary"]
        significance = comparison_results["significance_tests"]

        # Summary Statistics
        print("\n📊 SUMMARY STATISTICS")
        print("-" * 40)

        metrics_info = [
            (
                "Reward Score",
                base_metrics.reward_scores,
                rlhf_metrics.reward_scores,
                "Higher is better",
            ),
            (
                "Perplexity",
                base_metrics.perplexity_scores,
                rlhf_metrics.perplexity_scores,
                "Lower is better",
            ),
            (
                "Diversity",
                base_metrics.diversity_scores,
                rlhf_metrics.diversity_scores,
                "Higher is better",
            ),
            (
                "Coherence",
                base_metrics.coherence_scores,
                rlhf_metrics.coherence_scores,
                "Higher is better",
            ),
            (
                "Safety",
                base_metrics.safety_scores,
                rlhf_metrics.safety_scores,
                "Higher is better",
            ),
        ]

        for name, base_scores, rlhf_scores, direction in metrics_info:
            base_mean, base_std = np.mean(base_scores), np.std(base_scores)
            rlhf_mean, rlhf_std = np.mean(rlhf_scores), np.std(rlhf_scores)

            print(f"\n{name} ({direction}):")
            print(f"  Base Model:  {base_mean:.3f} ± {base_std:.3f}")
            print(f"  RLHF Model:  {rlhf_mean:.3f} ± {rlhf_std:.3f}")

            # Significance
            if name.lower().replace(" ", "_") in significance:
                sig_info = significance[name.lower().replace(" ", "_")]
                sig_symbol = "***" if sig_info["significant"] else "ns"
                print(f"  Significance: {sig_symbol} (p={sig_info['p_value']:.4f})")

        # Improvements
        print("\n📈 IMPROVEMENTS")
        print("-" * 40)
        for metric, improvement in improvements.items():
            direction = "↑" if improvement > 0 else "↓"
            print(f"{metric.replace('_', ' ').title()}: {direction} {improvement:.1f}%")

        # Human Preferences
        if rlhf_metrics.human_preference_scores:
            avg_pref = np.mean(rlhf_metrics.human_preference_scores)
            pref_above_50 = sum(
                1 for p in rlhf_metrics.human_preference_scores if p > 0.5
            )
            total_comparisons = len(rlhf_metrics.human_preference_scores)

            print(f"\n👥 HUMAN PREFERENCE SIMULATION")
            print("-" * 40)
            print(f"Average preference for RLHF: {avg_pref:.1%}")
            print(
                f"RLHF preferred in: {pref_above_50}/{total_comparisons} cases ({pref_above_50/total_comparisons:.1%})"
            )

        # Sample Responses
        print(f"\n💬 SAMPLE RESPONSES")
        print("-" * 40)

        base_samples = comparison_results["sample_responses"]["base"][:3]
        rlhf_samples = comparison_results["sample_responses"]["rlhf"][:3]

        for i, ((prompt_base, resp_base), (prompt_rlhf, resp_rlhf)) in enumerate(
            zip(base_samples, rlhf_samples)
        ):
            print(f"\nExample {i+1}:")
            print(f"Prompt: {prompt_base}")
            print(f"Base Model: {resp_base}")
            print(f"RLHF Model: {resp_rlhf}")

        print("\n" + "=" * 80)


# Demo function to run the evaluation
def run_evaluation_demo():
    """Run a complete evaluation demo"""

    # Import the RLHF implementation (assuming it's available)

    print("Setting up RLHF evaluation demo...")

    # Initialize models
    trainer = RLHFTrainer(vocab_size=1000, device="cpu")

    # Create a copy of the model before RLHF training (as base model)
    base_model = SimpleTransformer(vocab_size=1000)
    base_model.load_state_dict(trainer.sft_model.state_dict())

    # Quick RLHF training for demo
    preference_data = [
        ("Hello, how are you?", "Hi there!", 1),
        ("Good morning!", "Morning", 0),
        ("Thank you very much", "Thanks", 1),
        ("Please help me", "Help", 0),
        ("Have a great day!", "Bye", 1),
    ] * 10

    trainer.train_reward_model(preference_data, epochs=3)

    prompts = [
        "How can I help you today?",
        "What would you like to know?",
        "Please tell me about",
        "I'm here to assist with",
        "Let me help you with",
    ]

    trainer.train_rlhf(prompts, num_episodes=20)

    # Test prompts for evaluation
    test_prompts = [
        "How are you doing today?",
        "Can you help me with something?",
        "What's the weather like?",
        "Tell me a story",
        "How do I cook pasta?",
        "What's your favorite color?",
        "Explain quantum physics",
        "Write a poem about nature",
        "What's the meaning of life?",
        "How do I learn programming?",
    ] * 3  # Repeat for more data points

    # Initialize evaluator
    evaluator = RLHFEvaluator(
        base_model=base_model,
        rlhf_model=trainer.sft_model,
        reward_model=trainer.reward_model,
        vocab_size=1000,
        device="cpu",
    )

    # Run comprehensive comparison
    results = evaluator.compare_models(test_prompts, num_comparisons=20)

    # Print detailed report
    evaluator.print_detailed_report(results)

    # Create visualizations
    evaluator.visualize_comparison(results, save_path="rlhf_comparison.png")

    print("\n✅ Evaluation complete! Check 'rlhf_comparison.png' for visualizations.")

    return results


if __name__ == "__main__":
    # Run the evaluation demo
    results = run_evaluation_demo()
