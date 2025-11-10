"""
Reinforcement learning for trading strategies.
This module provides RL-based components for trading strategies.
"""

import numpy as np
import pandas as pd
from typing import Dict, Any, List, Optional, Tuple, Union
import os
import pickle
import logging
import datetime
import random
from collections import deque

# Set up logging
logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO)

class TradingEnvironment:
    """Environment for reinforcement learning-based trading."""
    
    def __init__(self, data: pd.DataFrame = None, 
                 initial_balance: float = 10000.0,
                 commission: float = 0.001,
                 slippage: float = 0.0005,
                 window_size: int = 20,
                 reward_function: str = 'returns'):
        """
        Initialize the trading environment.
        
        Args:
            data: DataFrame with OHLCV data
            initial_balance: Initial account balance
            commission: Commission rate for trades
            slippage: Slippage for trades
            window_size: Number of past periods to include in state
            reward_function: Type of reward function to use
        """
        self.data = data
        self.initial_balance = initial_balance
        self.commission = commission
        self.slippage = slippage
        self.window_size = window_size
        self.reward_function = reward_function
        
        # State variables
        self.reset()
    
    def reset(self) -> np.ndarray:
        """
        Reset the environment to initial state.
        
        Returns:
            Initial state observation
        """
        self.current_step = self.window_size
        self.balance = self.initial_balance
        self.position = 0.0  # No position
        self.position_history = np.zeros(len(self.data))
        self.balance_history = np.ones(len(self.data)) * self.initial_balance
        self.nav_history = np.ones(len(self.data)) * self.initial_balance  # Net Asset Value
        self.done = False
        
        return self._get_observation()
    
    def _get_observation(self) -> np.ndarray:
        """
        Get the current state observation.
        
        Returns:
            Observation array
        """
        # Get price data for the window
        prices = self.data['close'].values[self.current_step-self.window_size:self.current_step]
        
        # Normalize prices
        normalized_prices = prices / prices[0] - 1.0
        
        # Add other features if desired
        # For example, returns, volatility, moving averages, etc.
        
        # Add position information
        observation = np.append(normalized_prices, [self.position])
        
        return observation
    
    def step(self, action: float) -> Tuple[np.ndarray, float, bool, Dict[str, Any]]:
        """
        Take a step in the environment.
        
        Args:
            action: Action to take (-1 to 1, representing position size)
            
        Returns:
            observation: New state observation
            reward: Reward for the action
            done: Whether the episode is done
            info: Additional information
        """
        if self.done:
            return self._get_observation(), 0.0, self.done, {}
        
        # Clip action to valid range
        action = np.clip(action, -1.0, 1.0)
        
        # Get current and next price
        current_price = self.data['close'].values[self.current_step]
        
        # Calculate the trade to make (difference in position)
        trade = action - self.position
        
        # Apply trading costs
        trading_cost = abs(trade) * current_price * self.commission
        
        # Update balance
        self.balance -= trading_cost
        
        # Update position
        old_position = self.position
        self.position = action
        
        # Record history
        self.position_history[self.current_step] = self.position
        self.balance_history[self.current_step] = self.balance
        
        # Calculate NAV (Net Asset Value)
        self.nav_history[self.current_step] = self.balance + self.position * current_price
        
        # Move to next step
        self.current_step += 1
        
        # Check if done
        if self.current_step >= len(self.data) - 1:
            self.done = True
        
        # Calculate reward
        reward = self._calculate_reward(old_position, current_price)
        
        # Get new observation
        observation = self._get_observation()
        
        # Additional info
        info = {
            'step': self.current_step,
            'balance': self.balance,
            'position': self.position,
            'nav': self.nav_history[self.current_step-1],
            'trade': trade,
            'trading_cost': trading_cost
        }
        
        return observation, reward, self.done, info
    
    def _calculate_reward(self, old_position: float, current_price: float) -> float:
        """
        Calculate reward for the current step.
        
        Args:
            old_position: Position before the action
            current_price: Current price
            
        Returns:
            Reward value
        """
        if self.current_step >= len(self.data):
            return 0.0
        
        next_price = self.data['close'].values[self.current_step]
        price_change = next_price / current_price - 1.0
        
        if self.reward_function == 'returns':
            # Reward is the return on the position
            reward = old_position * price_change
        elif self.reward_function == 'log_returns':
            # Reward is the log return on the position
            reward = old_position * np.log(next_price / current_price)
        elif self.reward_function == 'sharpe':
            # Reward is the Sharpe ratio (approximation)
            returns = np.diff(self.nav_history[max(0, self.current_step-30):self.current_step]) / \
                      self.nav_history[max(0, self.current_step-30):self.current_step-1]
            if len(returns) > 1:
                reward = (np.mean(returns) / (np.std(returns) + 1e-6)) * old_position * price_change
            else:
                reward = old_position * price_change
        else:
            # Default to returns
            reward = old_position * price_change
        
        return reward
    
    def render(self) -> None:
        """Render the environment (for debugging)."""
        print(f"Step: {self.current_step}, Balance: {self.balance:.2f}, Position: {self.position:.2f}, NAV: {self.nav_history[self.current_step-1]:.2f}")


class ReplayBuffer:
    """Experience replay buffer for reinforcement learning."""
    
    def __init__(self, capacity: int = 10000):
        """
        Initialize the replay buffer.
        
        Args:
            capacity: Maximum number of experiences to store
        """
        self.buffer = deque(maxlen=capacity)
    
    def add(self, state, action, reward, next_state, done):
        """Add experience to buffer."""
        self.buffer.append((state, action, reward, next_state, done))
    
    def sample(self, batch_size: int) -> List[Tuple]:
        """Sample a batch of experiences."""
        return random.sample(self.buffer, min(batch_size, len(self.buffer)))
    
    def __len__(self) -> int:
        """Get buffer length."""
        return len(self.buffer)


class DQNAgent:
    """Deep Q-Network agent for reinforcement learning trading."""
    
    def __init__(self, state_size: int, action_size: int, model_params: Dict[str, Any] = None):
        """
        Initialize the DQN agent.
        
        Args:
            state_size: Dimension of state space
            action_size: Dimension of action space
            model_params: Parameters for the model
        """
        self.state_size = state_size
        self.action_size = action_size
        self.memory = ReplayBuffer(100000)
        self.gamma = 0.95  # Discount factor
        self.epsilon = 1.0  # Exploration rate
        self.epsilon_min = 0.01
        self.epsilon_decay = 0.995
        self.model_params = model_params or {}
        
        # Load keras if available
        try:
            from keras.models import Sequential
            from keras.layers import Dense
            from keras.optimizers import Adam
            
            # Initialize Q-network
            self.model = Sequential()
            self.model.add(Dense(64, input_dim=state_size, activation='relu'))
            self.model.add(Dense(64, activation='relu'))
            self.model.add(Dense(action_size, activation='linear'))
            self.model.compile(loss='mse', optimizer=Adam(learning_rate=0.001))
            
            # Initialize target Q-network
            self.target_model = Sequential()
            self.target_model.add(Dense(64, input_dim=state_size, activation='relu'))
            self.target_model.add(Dense(64, activation='relu'))
            self.target_model.add(Dense(action_size, activation='linear'))
            self.target_model.compile(loss='mse', optimizer=Adam(learning_rate=0.001))
            
            self._update_target_model()
            self.keras_available = True
            
        except ImportError:
            logger.warning("Keras not available. DQNAgent will not function properly.")
            self.model = None
            self.target_model = None
            self.keras_available = False
    
    def _update_target_model(self):
        """Update target model with weights from main model."""
        if self.keras_available:
            self.target_model.set_weights(self.model.get_weights())
    
    def act(self, state: np.ndarray) -> int:
        """
        Select an action using epsilon-greedy policy.
        
        Args:
            state: Current state observation
            
        Returns:
            Selected action index
        """
        if not self.keras_available:
            return random.randrange(self.action_size)
        
        if np.random.rand() <= self.epsilon:
            return random.randrange(self.action_size)
        
        state = np.reshape(state, [1, self.state_size])
        act_values = self.model.predict(state)
        return np.argmax(act_values[0])
    
    def remember(self, state, action, reward, next_state, done):
        """Add experience to memory."""
        self.memory.add(state, action, reward, next_state, done)
    
    def replay(self, batch_size: int):
        """
        Train the model using experiences from memory.
        
        Args:
            batch_size: Number of experiences to sample
        """
        if not self.keras_available or len(self.memory) < batch_size:
            return
        
        minibatch = self.memory.sample(batch_size)
        
        for state, action, reward, next_state, done in minibatch:
            state = np.reshape(state, [1, self.state_size])
            next_state = np.reshape(next_state, [1, self.state_size])
            
            if done:
                target = reward
            else:
                target = reward + self.gamma * np.amax(self.target_model.predict(next_state)[0])
            
            target_f = self.model.predict(state)
            target_f[0][action] = target
            
            self.model.fit(state, target_f, epochs=1, verbose=0)
        
        if self.epsilon > self.epsilon_min:
            self.epsilon *= self.epsilon_decay
    
    def load(self, filepath: str):
        """
        Load model from disk.
        
        Args:
            filepath: Path to load model from
        """
        if not self.keras_available:
            logger.warning("Keras not available. Cannot load model.")
            return
        
        try:
            self.model.load_weights(filepath)
            self.target_model.load_weights(filepath)
            logger.info(f"Model loaded successfully from {filepath}")
        except Exception as e:
            logger.error(f"Error loading model: {str(e)}")
    
    def save(self, filepath: str):
        """
        Save model to disk.
        
        Args:
            filepath: Path to save model to
        """
        if not self.keras_available:
            logger.warning("Keras not available. Cannot save model.")
            return
        
        try:
            # Create directory if it doesn't exist
            os.makedirs(os.path.dirname(os.path.abspath(filepath)), exist_ok=True)
            
            self.model.save_weights(filepath)
            logger.info(f"Model saved successfully to {filepath}")
        except Exception as e:
            logger.error(f"Error saving model: {str(e)}")


def discrete_action_to_position(action: int, action_space: int = 3) -> float:
    """
    Convert discrete action index to continuous position.
    
    Args:
        action: Action index
        action_space: Size of action space
        
    Returns:
        Position value
    """
    # Map action to position in [-1, 1]
    if action_space == 3:
        # 0 = short, 1 = neutral, 2 = long
        return action - 1
    else:
        # Map evenly across [-1, 1]
        return 2 * (action / (action_space - 1)) - 1


def train_rl_agent(data: pd.DataFrame, window_size: int = 20, 
                   episodes: int = 100, batch_size: int = 32, action_space: int = 3) -> DQNAgent:
    """
    Train a reinforcement learning agent for trading.
    
    Args:
        data: DataFrame with OHLCV data
        window_size: Number of past periods to include in state
        episodes: Number of episodes to train
        batch_size: Batch size for replay
        action_space: Number of discrete actions
        
    Returns:
        Trained DQNAgent
    """
    # Initialize environment
    env = TradingEnvironment(data=data, window_size=window_size)
    
    # Calculate state and action dimensions
    state_size = window_size + 1  # price history + position
    action_size = action_space
    
    # Initialize agent
    agent = DQNAgent(state_size, action_size)
    
    # Training loop
    for episode in range(episodes):
        # Reset environment
        state = env.reset()
        
        total_reward = 0
        
        while not env.done:
            # Select action
            action = agent.act(state)
            
            # Convert discrete action to continuous position
            position = discrete_action_to_position(action, action_space)
            
            # Take action
            next_state, reward, done, info = env.step(position)
            
            # Remember experience
            agent.remember(state, action, reward, next_state, done)
            
            # Update state
            state = next_state
            
            # Update total reward
            total_reward += reward
            
            # Train agent
            if len(agent.memory) > batch_size:
                agent.replay(batch_size)
        
        # Update target model periodically
        if episode % 10 == 0:
            agent._update_target_model()
        
        # Log progress
        logger.info(f"Episode: {episode+1}/{episodes}, Total Reward: {total_reward:.4f}, Final NAV: {env.nav_history[env.current_step-1]:.2f}, Epsilon: {agent.epsilon:.4f}")
    
    return agent


def rl_signal_generator(data: pd.DataFrame, context: Dict[str, Any], 
                        window_size: int = 20, action_space: int = 3,
                        model_path: Optional[str] = None) -> np.ndarray:
    """
    Generate trading signals using reinforcement learning.
    
    Args:
        data: DataFrame with OHLCV data
        context: Strategy context dictionary
        window_size: Number of past periods to include in state
        action_space: Number of discrete actions
        model_path: Path to pre-trained model (if None, will train a new one)
        
    Returns:
        Array of trading signals
    """
    # Initialize environment
    env = TradingEnvironment(data=data, window_size=window_size)
    
    # Calculate state and action dimensions
    state_size = window_size + 1  # price history + position
    action_size = action_space
    
    # Initialize or retrieve agent
    if 'rl_agent' not in context:
        agent = DQNAgent(state_size, action_size)
        
        # Load pre-trained model if available
        if model_path is not None and os.path.exists(model_path):
            agent.load(model_path)
        
        context['rl_agent'] = agent
    else:
        agent = context['rl_agent']
    
    # Generate signals
    env.reset()
    signals = np.zeros(len(data))
    
    # Generate signals only if we have enough data
    if len(data) > window_size:
        while not env.done:
            state = env._get_observation()
            action = agent.act(state)
            
            # Convert discrete action to continuous position
            position = discrete_action_to_position(action, action_space)
            
            # Store signal
            current_step = env.current_step
            signals[current_step] = position
            
            # Take action
            _, _, done, _ = env.step(position)
    
    return signals
