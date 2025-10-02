"""
World Model implementation for BuilderBrain.

Provides a compact latent simulator for domains where actions have external consequences.
Used for planning, EVSI estimation, and safer training.
"""

from typing import Dict, List, Any, Optional, Tuple
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np


class RSSM(nn.Module):
    """
    Recurrent State-Space Model for world modeling.

    Models the world as a latent state that evolves through actions and observations.
    """

    def __init__(
        self,
        obs_dim: int,
        action_dim: int,
        state_dim: int = 64,
        hidden_dim: int = 128,
        num_layers: int = 2
    ):
        super().__init__()

        self.obs_dim = obs_dim
        self.action_dim = action_dim
        self.state_dim = state_dim
        self.hidden_dim = hidden_dim

        # Encoder: observation -> latent state
        self.encoder = nn.Sequential(
            nn.Linear(obs_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, state_dim * 2)  # Mean and logvar for stochastic state
        )

        # Transition model: state + action -> next state
        self.transition = nn.GRUCell(state_dim + action_dim, state_dim)

        # Decoder: state -> observation prediction
        self.decoder = nn.Sequential(
            nn.Linear(state_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, obs_dim)
        )

        # Reward predictor
        self.reward_head = nn.Sequential(
            nn.Linear(state_dim, hidden_dim // 2),
            nn.ReLU(),
            nn.Linear(hidden_dim // 2, 1)
        )

        # Safety predictor (risk energy)
        self.safety_head = nn.Sequential(
            nn.Linear(state_dim, hidden_dim // 2),
            nn.ReLU(),
            nn.Linear(hidden_dim // 2, 1)
        )

    def encode(self, observation: torch.Tensor) -> torch.Tensor:
        """
        Encode observation into latent state.

        Args:
            observation: Observation tensor (batch, obs_dim)

        Returns:
            Latent state tensor (batch, state_dim)
        """
        # Get mean and logvar for stochastic state
        state_params = self.encoder(observation)  # (batch, state_dim * 2)
        mean, logvar = state_params.chunk(2, dim=-1)

        # Sample from Gaussian (reparameterization trick)
        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std)
        state = mean + eps * std

        return state

    def transition_step(self, state: torch.Tensor, action: torch.Tensor) -> torch.Tensor:
        """
        Predict next state given current state and action.

        Args:
            state: Current latent state (batch, state_dim)
            action: Action tensor (batch, action_dim)

        Returns:
            Next latent state (batch, state_dim)
        """
        # Concatenate state and action
        state_action = torch.cat([state, action], dim=-1)

        # GRU transition
        next_state = self.transition(state_action, state)

        return next_state

    def predict_observation(self, state: torch.Tensor) -> torch.Tensor:
        """
        Predict observation from latent state.

        Args:
            state: Latent state (batch, state_dim)

        Returns:
            Predicted observation (batch, obs_dim)
        """
        return self.decoder(state)

    def predict_reward(self, state: torch.Tensor) -> torch.Tensor:
        """
        Predict reward from latent state.

        Args:
            state: Latent state (batch, state_dim)

        Returns:
            Predicted reward (batch, 1)
        """
        return self.reward_head(state)

    def predict_safety(self, state: torch.Tensor) -> torch.Tensor:
        """
        Predict safety/risk energy from latent state.

        Args:
            state: Latent state (batch, state_dim)

        Returns:
            Predicted risk energy (batch, 1)
        """
        return self.safety_head(state)

    def rollout(
        self,
        initial_state: torch.Tensor,
        actions: torch.Tensor,
        horizon: int
    ) -> Dict[str, torch.Tensor]:
        """
        Generate trajectory rollout.

        Args:
            initial_state: Initial latent state (batch, state_dim)
            actions: Action sequence (batch, horizon, action_dim)
            horizon: Number of steps to rollout

        Returns:
            Dictionary with rollout results
        """
        batch_size = initial_state.size(0)
        state_dim = initial_state.size(1)

        # Initialize trajectory
        states = [initial_state]
        observations = []
        rewards = []
        safeties = []

        current_state = initial_state

        for t in range(horizon):
            # Get action for this timestep
            action = actions[:, t, :] if t < actions.size(1) else torch.zeros_like(actions[:, 0, :])

            # Transition to next state
            current_state = self.transition_step(current_state, action)
            states.append(current_state)

            # Predict observation, reward, and safety
            obs_pred = self.predict_observation(current_state)
            reward_pred = self.predict_reward(current_state)
            safety_pred = self.predict_safety(current_state)

            observations.append(obs_pred)
            rewards.append(reward_pred)
            safeties.append(safety_pred)

        # Stack results
        states_tensor = torch.stack(states, dim=1)  # (batch, horizon+1, state_dim)
        observations_tensor = torch.stack(observations, dim=1)  # (batch, horizon, obs_dim)
        rewards_tensor = torch.stack(rewards, dim=1)  # (batch, horizon, 1)
        safeties_tensor = torch.stack(safeties, dim=1)  # (batch, horizon, 1)

        return {
            'states': states_tensor,
            'observations': observations_tensor,
            'rewards': rewards_tensor,
            'safeties': safeties_tensor
        }


class WorldModel(nn.Module):
    """
    Complete world model with RSSM and training utilities.
    """

    def __init__(
        self,
        obs_dim: int,
        action_dim: int,
        state_dim: int = 64,
        hidden_dim: int = 128,
        kl_weight: float = 1.0,
        reward_weight: float = 1.0,
        safety_weight: float = 1.0
    ):
        super().__init__()

        self.rssm = RSSM(obs_dim, action_dim, state_dim, hidden_dim)

        # Loss weights
        self.kl_weight = kl_weight
        self.reward_weight = reward_weight
        self.safety_weight = safety_weight

        # Reconstruction loss
        self.reconstruction_loss = nn.MSELoss()

    def compute_loss(
        self,
        observations: torch.Tensor,  # (batch, seq, obs_dim)
        actions: torch.Tensor,       # (batch, seq, action_dim)
        rewards: torch.Tensor,       # (batch, seq, 1)
        safeties: torch.Tensor       # (batch, seq, 1)
    ) -> Dict[str, torch.Tensor]:
        """
        Compute world model training losses.

        Args:
            observations: Real observations (batch, seq, obs_dim)
            actions: Actions taken (batch, seq, action_dim)
            rewards: Real rewards (batch, seq, 1)
            safeties: Real safety values (batch, seq, 1)

        Returns:
            Dictionary of loss components
        """
        batch_size, seq_len, _ = observations.shape

        # Initialize state from first observation
        initial_state = self.rssm.encode(observations[:, 0, :])

        total_reconstruction_loss = 0.0
        total_kl_loss = 0.0
        total_reward_loss = 0.0
        total_safety_loss = 0.0

        current_state = initial_state

        for t in range(seq_len - 1):
            # Get current observation and action
            obs_t = observations[:, t, :]
            action_t = actions[:, t, :]

            # Encode current observation to get target state for KL
            target_state = self.rssm.encode(obs_t)

            # Transition to next state
            next_state = self.rssm.transition_step(current_state, action_t)

            # Predict next observation
            obs_pred = self.rssm.predict_observation(next_state)

            # Compute reconstruction loss
            reconstruction_loss = self.reconstruction_loss(obs_pred, observations[:, t + 1, :])
            total_reconstruction_loss += reconstruction_loss

            # Predict reward and safety
            reward_pred = self.rssm.predict_reward(next_state)
            safety_pred = self.rssm.predict_safety(next_state)

            # Compute reward and safety losses
            reward_loss = F.mse_loss(reward_pred, rewards[:, t, :])
            safety_loss = F.mse_loss(safety_pred, safeties[:, t, :])

            total_reward_loss += reward_loss
            total_safety_loss += safety_loss

            # Update current state
            current_state = next_state

        # Average losses over sequence
        losses = {
            'reconstruction': total_reconstruction_loss / (seq_len - 1),
            'reward': total_reward_loss / (seq_len - 1),
            'safety': total_safety_loss / (seq_len - 1)
        }

        return losses

    def compute_evsi(
        self,
        state: torch.Tensor,
        tool_action: torch.Tensor,
        cost: float
    ) -> torch.Tensor:
        """
        Compute Expected Value of Sample Information for tool calls.

        Args:
            state: Current latent state (batch, state_dim)
            tool_action: Tool action to evaluate (batch, action_dim)
            cost: Cost of taking the tool action

        Returns:
            EVSI values (batch,)
        """
        # Rollout with tool action
        with_tool = self.rollout(state, tool_action.unsqueeze(1), horizon=1)

        # Rollout without tool action (assume zero action)
        no_tool_action = torch.zeros_like(tool_action)
        without_tool = self.rollout(state, no_tool_action.unsqueeze(1), horizon=1)

        # Compute value difference
        reward_with = with_tool['rewards'][:, 0, 0]  # (batch,)
        reward_without = without_tool['rewards'][:, 0, 0]  # (batch,)

        # EVSI = E[max_a U(s,o,a)] - max_a E[U(s,o,a)] - cost
        # For simplicity, approximate as reward difference minus cost
        evsi = reward_with - reward_without - cost

        return evsi

    def rollout(
        self,
        initial_state: torch.Tensor,
        actions: torch.Tensor,
        horizon: int
    ) -> Dict[str, torch.Tensor]:
        """
        Generate trajectory rollout for planning.

        Args:
            initial_state: Initial state (batch, state_dim)
            actions: Action sequence (batch, horizon, action_dim)
            horizon: Planning horizon

        Returns:
            Rollout results
        """
        return self.rssm.rollout(initial_state, actions, horizon)

    def encode(self, observation: torch.Tensor) -> torch.Tensor:
        """Encode observation to latent state."""
        return self.rssm.encode(observation)

    def predict_next(self, state: torch.Tensor, action: torch.Tensor) -> Tuple[torch.Tensor, Dict[str, float]]:
        """
        Predict next state and associated metrics.

        Args:
            state: Current state (batch, state_dim)
            action: Action to take (batch, action_dim)

        Returns:
            Next state and prediction metrics
        """
        next_state = self.rssm.transition_step(state, action)

        # Get predictions
        obs_pred = self.rssm.predict_observation(next_state)
        reward_pred = self.rssm.predict_reward(next_state)
        safety_pred = self.rssm.predict_safety(next_state)

        return next_state, {
            'observation': obs_pred,
            'reward': reward_pred,
            'safety': safety_pred
        }


class SimpleWorldModel(WorldModel):
    """
    Simplified world model for demonstration and testing.

    Uses smaller dimensions and simpler architecture.
    """

    def __init__(self):
        # Simple dimensions for testing
        super().__init__(
            obs_dim=10,      # Example observation dimension
            action_dim=5,    # Example action dimension
            state_dim=32,    # Smaller state dimension
            hidden_dim=64,   # Smaller hidden dimension
            kl_weight=0.1,
            reward_weight=1.0,
            safety_weight=1.0
        )

    def demo_rollout(self) -> Dict[str, torch.Tensor]:
        """Generate a demo rollout for testing."""
        batch_size = 2
        horizon = 5

        # Random initial state
        initial_state = torch.randn(batch_size, self.rssm.state_dim)

        # Random actions
        actions = torch.randn(batch_size, horizon, self.rssm.action_dim)

        return self.rollout(initial_state, actions, horizon)

