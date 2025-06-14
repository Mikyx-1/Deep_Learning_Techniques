import math
import random
from collections import deque
from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.distributions import Categorical

# =====================================================
# STEP 1: BASE LANGUAGE MODEL (SFT Model)
# =====================================================


class SimpleTransformer(nn.Module):
    """Simplified transformer model for demonstration"""

    def __init__(
        self,
        vocab_size: int,
        d_model: int = 256,
        nhead: int = 8,
        num_layers: int = 4,
        max_seq_len: int = 512,
    ):
        super().__init__()
        self.d_model = d_model
        self.max_seq_len = max_seq_len

        # Embeddings
        self.token_embedding = nn.Embedding(vocab_size, d_model)
        self.pos_embedding = nn.Embedding(max_seq_len, d_model)

        # Transformer layers
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model, nhead=nhead, batch_first=True
        )
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers)

        # Output head
        self.ln_f = nn.LayerNorm(d_model)
        self.head = nn.Linear(d_model, vocab_size)

    def forward(
        self, input_ids: torch.Tensor, attention_mask: Optional[torch.Tensor] = None
    ):
        seq_len = input_ids.size(1)
        pos_ids = torch.arange(seq_len, device=input_ids.device).unsqueeze(0)

        # Embeddings
        token_embeds = self.token_embedding(input_ids)
        pos_embeds = self.pos_embedding(pos_ids)
        x = token_embeds + pos_embeds

        # Create causal mask
        causal_mask = torch.triu(torch.ones(seq_len, seq_len), diagonal=1).bool()
        causal_mask = causal_mask.to(input_ids.device)

        # Transformer
        x = self.transformer(
            x,
            mask=causal_mask,
            src_key_padding_mask=(
                ~attention_mask if attention_mask is not None else None
            ),
        )
        x = self.ln_f(x)
        logits = self.head(x)

        return logits

    def generate(
        self,
        input_ids: torch.Tensor,
        max_length: int = 50,
        temperature: float = 1.0,
        do_sample: bool = True,
    ):
        """Generate text continuation"""
        self.eval()
        with torch.no_grad():
            for _ in range(max_length):
                logits = self.forward(input_ids)
                next_token_logits = logits[:, -1, :] / temperature

                if do_sample:
                    probs = F.softmax(next_token_logits, dim=-1)
                    next_token = torch.multinomial(probs, 1)
                else:
                    next_token = torch.argmax(next_token_logits, dim=-1, keepdim=True)

                input_ids = torch.cat([input_ids, next_token], dim=1)

                # Stop if we hit max sequence length
                if input_ids.size(1) >= self.max_seq_len:
                    break

        return input_ids


# =====================================================
# STEP 2: REWARD MODEL
# =====================================================


class RewardModel(nn.Module):
    """Reward model that scores text quality based on human preferences"""

    def __init__(self, base_model: SimpleTransformer):
        super().__init__()
        self.base_model = base_model

        # Replace the language modeling head with a reward head
        self.reward_head = nn.Linear(base_model.d_model, 1)

        # Freeze base model initially (optional)
        # for param in self.base_model.parameters():
        #     param.requires_grad = False

    def forward(
        self, input_ids: torch.Tensor, attention_mask: Optional[torch.Tensor] = None
    ):
        # Get hidden states from base model (without the LM head)
        seq_len = input_ids.size(1)
        pos_ids = torch.arange(seq_len, device=input_ids.device).unsqueeze(0)

        token_embeds = self.base_model.token_embedding(input_ids)
        pos_embeds = self.base_model.pos_embedding(pos_ids)
        x = token_embeds + pos_embeds

        causal_mask = torch.triu(torch.ones(seq_len, seq_len), diagonal=1).bool()
        causal_mask = causal_mask.to(input_ids.device)

        x = self.base_model.transformer(
            x,
            mask=causal_mask,
            src_key_padding_mask=(
                ~attention_mask if attention_mask is not None else None
            ),
        )
        x = self.base_model.ln_f(x)

        # Use the last token's hidden state for reward prediction
        if attention_mask is not None:
            # Get the last non-padded token for each sequence
            last_non_padded = attention_mask.sum(dim=1) - 1
            rewards = self.reward_head(x[range(x.size(0)), last_non_padded])
        else:
            rewards = self.reward_head(x[:, -1, :])

        return rewards.squeeze(-1)


# =====================================================
# STEP 3: PPO TRAINER FOR RLHF
# =====================================================


@dataclass
class PPOConfig:
    """Configuration for PPO training"""

    learning_rate: float = 1e-5
    batch_size: int = 8
    ppo_epochs: int = 4
    clip_epsilon: float = 0.2
    value_loss_coef: float = 0.5
    entropy_coef: float = 0.01
    max_grad_norm: float = 1.0
    gamma: float = 0.99
    gae_lambda: float = 0.95
    kl_penalty: float = 0.1


class PPOTrainer:
    """PPO trainer for RLHF"""

    def __init__(
        self,
        policy_model: SimpleTransformer,
        reward_model: RewardModel,
        config: PPOConfig,
    ):
        self.policy_model = policy_model
        self.reward_model = reward_model
        self.config = config

        # Create reference model (frozen copy of initial policy)
        self.ref_model = SimpleTransformer(
            vocab_size=policy_model.head.out_features,
            d_model=policy_model.d_model,
            max_seq_len=policy_model.max_seq_len,
        )
        self.ref_model.load_state_dict(policy_model.state_dict())
        for param in self.ref_model.parameters():
            param.requires_grad = False

        # Optimizers
        self.optimizer = optim.Adam(policy_model.parameters(), lr=config.learning_rate)

    def compute_advantages(
        self, rewards: torch.Tensor, values: torch.Tensor, dones: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Compute GAE advantages"""
        advantages = torch.zeros_like(rewards)
        returns = torch.zeros_like(rewards)

        gae = 0
        for t in reversed(range(len(rewards))):
            if t == len(rewards) - 1:
                next_value = 0
                next_non_terminal = 1 - dones[t]
            else:
                next_value = values[t + 1]
                next_non_terminal = 1 - dones[t + 1]

            delta = (
                rewards[t]
                + self.config.gamma * next_value * next_non_terminal
                - values[t]
            )
            gae = (
                delta
                + self.config.gamma * self.config.gae_lambda * next_non_terminal * gae
            )
            advantages[t] = gae
            returns[t] = advantages[t] + values[t]

        return advantages, returns

    def compute_policy_loss(
        self,
        logits: torch.Tensor,
        old_logits: torch.Tensor,
        actions: torch.Tensor,
        advantages: torch.Tensor,
        old_log_probs: torch.Tensor,
    ) -> torch.Tensor:
        """Compute PPO policy loss"""
        # Current policy log probabilities
        log_probs = F.log_softmax(logits, dim=-1)
        log_probs = log_probs.gather(-1, actions.unsqueeze(-1)).squeeze(-1)

        # Ratio between new and old policy
        ratio = torch.exp(log_probs - old_log_probs)

        # Clipped surrogate objective
        surr1 = ratio * advantages
        surr2 = (
            torch.clamp(
                ratio, 1 - self.config.clip_epsilon, 1 + self.config.clip_epsilon
            )
            * advantages
        )
        policy_loss = -torch.min(surr1, surr2).mean()

        return policy_loss

    def compute_kl_penalty(
        self, logits: torch.Tensor, ref_logits: torch.Tensor
    ) -> torch.Tensor:
        """Compute KL divergence penalty"""
        log_probs = F.log_softmax(logits, dim=-1)
        ref_log_probs = F.log_softmax(ref_logits, dim=-1)

        kl_div = F.kl_div(log_probs, ref_log_probs.exp(), reduction="batchmean")
        return self.config.kl_penalty * kl_div

    def train_step(
        self, queries: torch.Tensor, responses: torch.Tensor, rewards: torch.Tensor
    ) -> Dict[str, float]:
        """Single PPO training step"""
        batch_size, seq_len = responses.shape

        # Combine queries and responses
        full_sequences = torch.cat([queries, responses], dim=1)

        # Get initial policy outputs (old policy)
        with torch.no_grad():
            old_logits = self.policy_model(full_sequences)
            old_log_probs = F.log_softmax(old_logits, dim=-1)
            old_log_probs = old_log_probs[:, queries.size(1) :, :]  # Only response part

            ref_logits = self.ref_model(full_sequences)
            ref_log_probs = F.log_softmax(ref_logits, dim=-1)
            ref_log_probs = ref_log_probs[:, queries.size(1) :, :]

        # Dummy values for this simplified implementation
        values = torch.zeros_like(rewards)
        dones = torch.ones_like(rewards)

        advantages, returns = self.compute_advantages(rewards, values, dones)
        advantages = (advantages - advantages.mean()) / (advantages.std() + 1e-8)

        total_loss = 0
        for epoch in range(self.config.ppo_epochs):
            # Forward pass
            logits = self.policy_model(full_sequences)
            response_logits = logits[:, queries.size(1) :, :]

            # Flatten for loss computation
            flat_logits = response_logits.reshape(-1, response_logits.size(-1))
            flat_responses = responses.reshape(-1)
            flat_old_log_probs = (
                old_log_probs.gather(-1, responses.unsqueeze(-1))
                .squeeze(-1)
                .reshape(-1)
            )
            flat_advantages = advantages.repeat_interleave(seq_len)

            # Policy loss
            policy_loss = self.compute_policy_loss(
                flat_logits, None, flat_responses, flat_advantages, flat_old_log_probs
            )

            # KL penalty
            kl_penalty = self.compute_kl_penalty(response_logits, ref_log_probs)

            # Entropy bonus
            entropy = Categorical(logits=flat_logits).entropy().mean()

            # Total loss
            loss = policy_loss + kl_penalty - self.config.entropy_coef * entropy

            # Backward pass
            self.optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(
                self.policy_model.parameters(), self.config.max_grad_norm
            )
            self.optimizer.step()

            total_loss += loss.item()

        return {
            "policy_loss": policy_loss.item(),
            "kl_penalty": kl_penalty.item(),
            "entropy": entropy.item(),
            "total_loss": total_loss / self.config.ppo_epochs,
        }


# =====================================================
# STEP 4: TRAINING PIPELINE
# =====================================================


class RLHFTrainer:
    """Complete RLHF training pipeline"""

    def __init__(self, vocab_size: int = 1000, device: str = "cpu"):
        self.device = device
        self.vocab_size = vocab_size

        # Initialize models
        self.sft_model = SimpleTransformer(vocab_size).to(device)
        self.reward_model = RewardModel(self.sft_model).to(device)

        # PPO trainer
        self.ppo_config = PPOConfig()
        self.ppo_trainer = PPOTrainer(
            self.sft_model, self.reward_model, self.ppo_config
        )

    def train_reward_model(
        self, preference_data: List[Tuple[str, str, int]], epochs: int = 10
    ):
        """
        Train reward model on preference data
        preference_data: List of (text1, text2, preference) where preference is 0 or 1
        """
        print("Training Reward Model...")
        optimizer = optim.Adam(self.reward_model.parameters(), lr=1e-4)

        for epoch in range(epochs):
            total_loss = 0
            random.shuffle(preference_data)

            for i in range(0, len(preference_data), self.ppo_config.batch_size):
                batch = preference_data[i : i + self.ppo_config.batch_size]

                # Convert text to token ids (simplified - you'd use a real tokenizer)
                text1_ids = []
                text2_ids = []
                preferences = []

                for text1, text2, pref in batch:
                    # Simplified tokenization (hash-based)
                    ids1 = torch.tensor(
                        [hash(text1) % self.vocab_size for _ in range(10)],
                        device=self.device,
                    ).unsqueeze(0)
                    ids2 = torch.tensor(
                        [hash(text2) % self.vocab_size for _ in range(10)],
                        device=self.device,
                    ).unsqueeze(0)

                    text1_ids.append(ids1)
                    text2_ids.append(ids2)
                    preferences.append(pref)

                if not text1_ids:
                    continue

                text1_batch = torch.cat(text1_ids, dim=0)
                text2_batch = torch.cat(text2_ids, dim=0)
                pref_batch = torch.tensor(
                    preferences, device=self.device, dtype=torch.float
                )

                # Get reward scores
                rewards1 = self.reward_model(text1_batch)
                rewards2 = self.reward_model(text2_batch)

                # Bradley-Terry loss
                logits = rewards1 - rewards2
                loss = F.binary_cross_entropy_with_logits(logits, pref_batch)

                optimizer.zero_grad()
                loss.backward()
                optimizer.step()

                total_loss += loss.item()

            avg_loss = total_loss / max(
                1, len(preference_data) // self.ppo_config.batch_size
            )
            print(f"Reward Model Epoch {epoch+1}/{epochs}, Loss: {avg_loss:.4f}")

    def train_rlhf(self, prompts: List[str], num_episodes: int = 100):
        """Train policy using RLHF"""
        print("Starting RLHF Training...")

        for episode in range(num_episodes):
            # Sample batch of prompts
            batch_prompts = random.sample(
                prompts, min(len(prompts), self.ppo_config.batch_size)
            )

            # Convert prompts to token ids (simplified)
            query_ids = []
            for prompt in batch_prompts:
                ids = torch.tensor(
                    [hash(prompt + str(i)) % self.vocab_size for i in range(10)],
                    device=self.device,
                ).unsqueeze(0)
                query_ids.append(ids)

            queries = torch.cat(query_ids, dim=0)

            # Generate responses
            with torch.no_grad():
                full_sequences = self.sft_model.generate(queries, max_length=20)
                responses = full_sequences[:, queries.size(1) :]

            # Get rewards from reward model
            with torch.no_grad():
                rewards = self.reward_model(full_sequences)

            # PPO training step
            metrics = self.ppo_trainer.train_step(queries, responses, rewards)

            if episode % 10 == 0:
                print(f"Episode {episode}/{num_episodes}")
                print(f"  Policy Loss: {metrics['policy_loss']:.4f}")
                print(f"  KL Penalty: {metrics['kl_penalty']:.4f}")
                print(f"  Entropy: {metrics['entropy']:.4f}")
                print(f"  Average Reward: {rewards.mean().item():.4f}")


# =====================================================
# EXAMPLE USAGE
# =====================================================


def demo_rlhf():
    """Demonstrate RLHF training"""

    # Initialize trainer
    trainer = RLHFTrainer(vocab_size=1000, device="cpu")

    # Step 1: Train reward model with preference data
    preference_data = [
        ("Hello, how are you?", "Hi there!", 1),  # Prefer second
        ("Good morning!", "Morning", 0),  # Prefer first
        ("Thank you very much", "Thanks", 1),  # Prefer second
        ("Please help me", "Help", 0),  # Prefer first
        ("Have a great day!", "Bye", 1),  # Prefer second
    ] * 20  # Repeat for more training data

    trainer.train_reward_model(preference_data, epochs=5)

    # Step 2: RLHF training with prompts
    prompts = [
        "How can I help you today?",
        "What would you like to know?",
        "Please tell me about",
        "I'm here to assist with",
        "Let me help you with",
    ]

    trainer.train_rlhf(prompts, num_episodes=50)

    print("RLHF Training Complete!")

    # Test generation
    test_prompt = torch.tensor([[hash("Test prompt") % 1000 for _ in range(10)]])
    with torch.no_grad():
        generated = trainer.sft_model.generate(test_prompt, max_length=15)
        print(f"Generated sequence shape: {generated.shape}")


if __name__ == "__main__":
    demo_rlhf()
