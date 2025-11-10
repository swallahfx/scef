"""
Online learning system for adapting trading strategies.

This module provides bandit algorithms and online learning techniques
for adapting strategies based on recent performance.
"""
import numpy as np
import pandas as pd
from typing import Dict, List, Optional, Union, Any, Callable, Tuple
from dataclasses import dataclass
from datetime import datetime

from src.strategy_dsl import Strategy, StrategyComponent


class EpsilonGreedy:
    """Epsilon-greedy bandit algorithm for strategy selection."""
    
    def __init__(self, epsilon: float = 0.1, decay: float = 0.0):
        """
        Initialize epsilon-greedy bandit.
        
        Args:
            epsilon: Exploration probability (0-1)
            decay: Epsilon decay rate per step (0-1)
        """
        self.epsilon = epsilon
        self.decay = decay
        self.counts = {}  # Arm selection counts
        self.rewards = {}  # Accumulated rewards
        self.values = {}  # Estimated values
    
    def select_arm(self, arms: List[str]) -> str:
        """Select an arm based on epsilon-greedy policy."""
        # Initialize arms if not seen before
        for arm in arms:
            if arm not in self.counts:
                self.counts[arm] = 0
                self.rewards[arm] = 0.0
                self.values[arm] = 0.0
        
        # Exploration
        if np.random.random() < self.epsilon:
            return np.random.choice(arms)
        
        # Exploitation (with random tie-breaking)
        max_value = max(self.values[arm] for arm in arms)
        best_arms = [arm for arm in arms if self.values[arm] == max_value]
        return np.random.choice(best_arms)
    
    def update(self, arm: str, reward: float):
        """Update estimates based on observed reward."""
        self.counts[arm] += 1
        self.rewards[arm] += reward
        
        # Update estimated value using incremental average
        self.values[arm] += (reward - self.values[arm]) / self.counts[arm]
        
        # Update epsilon with decay
        self.epsilon *= (1.0 - self.decay)
    
    def get_stats(self) -> Dict[str, Dict[str, float]]:
        """Get current statistics for all arms."""
        return {
            arm: {
                'count': self.counts[arm],
                'value': self.values[arm],
                'total_reward': self.rewards[arm]
            }
            for arm in self.counts
        }


class UCB:
    """Upper Confidence Bound bandit algorithm for strategy selection."""
    
    def __init__(self, c: float = 1.0):
        """
        Initialize UCB bandit.
        
        Args:
            c: Exploration parameter (higher = more exploration)
        """
        self.c = c
        self.counts = {}  # Arm selection counts
        self.rewards = {}  # Accumulated rewards
        self.values = {}  # Estimated values
        self.total_count = 0  # Total number of selections
    
    def select_arm(self, arms: List[str]) -> str:
        """Select an arm based on UCB policy."""
        # Initialize arms if not seen before
        for arm in arms:
            if arm not in self.counts:
                self.counts[arm] = 0
                self.rewards[arm] = 0.0
                self.values[arm] = 0.0
        
        # If any arm has zero count, select it
        zero_count_arms = [arm for arm in arms if self.counts[arm] == 0]
        if zero_count_arms:
            return np.random.choice(zero_count_arms)
        
        # Calculate UCB for each arm
        ucb_values = {}
        for arm in arms:
            exploration = self.c * np.sqrt(2 * np.log(self.total_count) / self.counts[arm])
            ucb_values[arm] = self.values[arm] + exploration
        
        # Select arm with highest UCB (with random tie-breaking)
        max_ucb = max(ucb_values.values())
        best_arms = [arm for arm in arms if ucb_values[arm] == max_ucb]
        return np.random.choice(best_arms)
    
    def update(self, arm: str, reward: float):
        """Update estimates based on observed reward."""
        self.total_count += 1
        self.counts[arm] += 1
        self.rewards[arm] += reward
        
        # Update estimated value using incremental average
        self.values[arm] += (reward - self.values[arm]) / self.counts[arm]
    
    def get_stats(self) -> Dict[str, Dict[str, float]]:
        """Get current statistics for all arms."""
        stats = {}
        for arm in self.counts:
            exploration = 0.0
            if self.counts[arm] > 0:
                exploration = self.c * np.sqrt(2 * np.log(self.total_count) / self.counts[arm])
            
            stats[arm] = {
                'count': self.counts[arm],
                'value': self.values[arm],
                'total_reward': self.rewards[arm],
                'ucb': self.values[arm] + exploration
            }
        
        return stats


class ThompsonSampling:
    """Thompson Sampling bandit algorithm for strategy selection."""
    
    def __init__(self, alpha: float = 1.0, beta: float = 1.0):
        """
        Initialize Thompson Sampling bandit with Beta prior.
        
        Args:
            alpha: Initial alpha parameter for Beta distribution
            beta: Initial beta parameter for Beta distribution
        """
        self.alpha_init = alpha
        self.beta_init = beta
        self.alphas = {}  # Alpha parameters for each arm
        self.betas = {}  # Beta parameters for each arm
    
    def select_arm(self, arms: List[str]) -> str:
        """Select an arm based on Thompson Sampling."""
        # Initialize arms if not seen before
        for arm in arms:
            if arm not in self.alphas:
                self.alphas[arm] = self.alpha_init
                self.betas[arm] = self.beta_init
        
        # Sample from Beta distribution for each arm
        samples = {arm: np.random.beta(self.alphas[arm], self.betas[arm]) for arm in arms}
        
        # Select arm with highest sample (with random tie-breaking)
        max_sample = max(samples.values())
        best_arms = [arm for arm in arms if samples[arm] == max_sample]
        return np.random.choice(best_arms)
    
    def update(self, arm: str, reward: float):
        """Update parameters based on observed reward."""
        # Ensure reward is between 0 and 1
        reward = min(max(reward, 0.0), 1.0)
        
        # Update Beta distribution parameters
        self.alphas[arm] += reward
        self.betas[arm] += (1.0 - reward)
    
    def get_stats(self) -> Dict[str, Dict[str, float]]:
        """Get current statistics for all arms."""
        return {
            arm: {
                'alpha': self.alphas[arm],
                'beta': self.betas[arm],
                'mean': self.alphas[arm] / (self.alphas[arm] + self.betas[arm]),
                'variance': (self.alphas[arm] * self.betas[arm]) / 
                            ((self.alphas[arm] + self.betas[arm])**2 * 
                             (self.alphas[arm] + self.betas[arm] + 1))
            }
            for arm in self.alphas
        }


class StrategyBandit:
    """Multi-armed bandit for selecting between multiple strategies."""
    
    def __init__(self, strategies: Dict[str, Strategy], bandit_algo: str = 'ucb',
                reward_function: Optional[Callable] = None, **bandit_params):
        """
        Initialize strategy bandit.
        
        Args:
            strategies: Dictionary of strategies to choose from
            bandit_algo: Bandit algorithm to use ('eps_greedy', 'ucb', or 'thompson')
            reward_function: Custom reward function, if None uses Sharpe ratio
            **bandit_params: Parameters for the bandit algorithm
        """
        self.strategies = strategies
        
        # Initialize bandit algorithm
        if bandit_algo == 'eps_greedy':
            self.bandit = EpsilonGreedy(**bandit_params)
        elif bandit_algo == 'ucb':
            self.bandit = UCB(**bandit_params)
        elif bandit_algo == 'thompson':
            self.bandit = ThompsonSampling(**bandit_params)
        else:
            raise ValueError(f"Unknown bandit algorithm: {bandit_algo}")
        
        # Set reward function
        self.reward_function = reward_function or self._default_reward_function
        
        # Initialize performance tracking
        self.performance_history = {}
        for name in strategies.keys():
            self.performance_history[name] = []
    
    def _default_reward_function(self, returns: np.ndarray) -> float:
        """Default reward function using Sharpe ratio."""
        if len(returns) < 2:
            return 0.0
        
        mean_return = np.mean(returns)
        std_return = np.std(returns)
        
        if std_return == 0:
            return 0.0
            
        # Calculate Sharpe ratio (annualized, assuming daily returns)
        sharpe = mean_return / std_return * np.sqrt(252)
        
        # Scale to 0-1 range for Thompson sampling
        scaled_sharpe = 1 / (1 + np.exp(-sharpe))  # Sigmoid function
        
        return scaled_sharpe
    
    def select_strategy(self) -> Tuple[str, Strategy]:
        """Select a strategy using the bandit algorithm."""
        strategy_names = list(self.strategies.keys())
        selected_name = self.bandit.select_arm(strategy_names)
        return selected_name, self.strategies[selected_name]
    
    def update(self, strategy_name: str, returns: np.ndarray):
        """Update bandit based on strategy performance."""
        # Calculate reward
        reward = self.reward_function(returns)
        
        # Update bandit
        self.bandit.update(strategy_name, reward)
        
        # Track performance
        self.performance_history[strategy_name].append({
            'time': datetime.now(),
            'returns': returns,
            'reward': reward
        })
    
    def get_stats(self) -> Dict[str, Any]:
        """Get current statistics for all strategies."""
        return self.bandit.get_stats()


class OnlineLearner:
    """Online learning system for adapting strategies."""
    
    def __init__(self, base_strategy: Strategy, 
                learning_rate: float = 0.01,
                window_size: int = 20):
        """
        Initialize online learner.
        
        Args:
            base_strategy: Base strategy to adapt
            learning_rate: Learning rate for parameter updates
            window_size: Window size for recent performance evaluation
        """
        self.base_strategy = base_strategy
        self.learning_rate = learning_rate
        self.window_size = window_size
        
        # Track component parameters and performance
        self.component_params = {}
        self.performance_history = []
        
        # Initialize component parameters
        for component_list in [base_strategy.indicators, base_strategy.signals, 
                             base_strategy.allocations, base_strategy.executions,
                             base_strategy.risk_controls, base_strategy.post_trades]:
            for component in component_list:
                self.component_params[component.name] = component.parameters.copy()
    
    def _evaluate_gradient(self, component_name: str, param_name: str, 
                         data: pd.DataFrame, current_returns: np.ndarray) -> float:
        """Estimate gradient for a parameter using finite differences."""
        # Get current parameter value
        component = self._find_component(component_name)
        if component is None:
            return 0.0
            
        current_value = component.parameters.get(param_name, 0.0)
        
        # Skip non-numeric parameters
        if not isinstance(current_value, (int, float)):
            return 0.0
            
        # Small perturbation for finite difference
        epsilon = max(0.01, abs(current_value * 0.01))
        
        # Create modified strategy with increased parameter
        modified_strategy_up = self._create_modified_strategy(
            component_name, param_name, current_value + epsilon
        )
        
        # Create modified strategy with decreased parameter
        modified_strategy_down = self._create_modified_strategy(
            component_name, param_name, current_value - epsilon
        )
        
        # Run both modified strategies
        result_up = modified_strategy_up.run(data)
        result_down = modified_strategy_down.run(data)
        
        # Extract returns
        if 'returns' in result_up and 'returns' in result_down:
            returns_up = result_up['returns'][-self.window_size:]
            returns_down = result_down['returns'][-self.window_size:]
            
            # Calculate performance (Sharpe ratio)
            performance_up = np.mean(returns_up) / (np.std(returns_up) + 1e-10)
            performance_down = np.mean(returns_down) / (np.std(returns_down) + 1e-10)
            performance_current = np.mean(current_returns) / (np.std(current_returns) + 1e-10)
            
            # Calculate gradient
            gradient_up = (performance_up - performance_current) / epsilon
            gradient_down = (performance_current - performance_down) / epsilon
            
            # Average the gradients
            gradient = (gradient_up + gradient_down) / 2.0
            
            return gradient
        
        return 0.0
    
    def _find_component(self, component_name: str) -> Optional[StrategyComponent]:
        """Find a component by name in the strategy."""
        for component_list in [self.base_strategy.indicators, self.base_strategy.signals, 
                             self.base_strategy.allocations, self.base_strategy.executions,
                             self.base_strategy.risk_controls, self.base_strategy.post_trades]:
            for component in component_list:
                if component.name == component_name:
                    return component
        
        return None
    
    def _create_modified_strategy(self, component_name: str, param_name: str, 
                                param_value: Any) -> Strategy:
        """Create a modified strategy with updated parameter."""
        # Create a new strategy
        new_strategy = Strategy(self.base_strategy.name + '_modified')
        
        # Copy all components with modifications
        for component_list_name in ['indicators', 'signals', 'allocations', 
                                   'executions', 'risk_controls', 'post_trades']:
            component_list = getattr(self.base_strategy, component_list_name)
            
            for component in component_list:
                # Clone component
                new_params = component.parameters.copy()
                
                # Update parameter if this is the target component
                if component.name == component_name and param_name in new_params:
                    new_params[param_name] = param_value
                
                new_component = StrategyComponent(
                    name=component.name,
                    function=component.function,
                    category=component.category,
                    parameters=new_params
                )
                
                # Add to appropriate component list
                add_method = getattr(new_strategy, f"add_{component.category}")
                add_method(new_component)
        
        return new_strategy
    
    def update(self, data: pd.DataFrame, returns: np.ndarray):
        """Update strategy parameters based on recent performance."""
        # Use only recent returns for evaluation
        recent_returns = returns[-min(len(returns), self.window_size):]
        
        # Calculate current performance
        current_performance = np.mean(recent_returns) / (np.std(recent_returns) + 1e-10)
        
        # Track performance
        self.performance_history.append(current_performance)
        
        # Update parameters for each component
        updates = {}
        
        for component_name, params in self.component_params.items():
            component = self._find_component(component_name)
            if component is None:
                continue
                
            component_updates = {}
            
            # Evaluate gradient for each numeric parameter
            for param_name, param_value in params.items():
                if not isinstance(param_value, (int, float)):
                    continue
                    
                # Estimate gradient
                gradient = self._evaluate_gradient(
                    component_name, param_name, data, recent_returns
                )
                
                # Update parameter
                new_value = param_value + self.learning_rate * gradient
                
                # Store update
                component_updates[param_name] = new_value
            
            if component_updates:
                updates[component_name] = component_updates
        
        # Apply updates
        for component_name, param_updates in updates.items():
            component = self._find_component(component_name)
            if component is None:
                continue
                
            # Create updated parameters
            new_params = component.parameters.copy()
            new_params.update(param_updates)
            
            # Update component
            new_component = StrategyComponent(
                name=component.name,
                function=component.function,
                category=component.category,
                parameters=new_params
            )
            
            # Replace component in strategy
            component_list_name = component.category + 's'
            component_list = getattr(self.base_strategy, component_list_name)
            
            # Find index of component to replace
            for i, c in enumerate(component_list):
                if c.name == component.name:
                    component_list[i] = new_component
                    break
            
            # Update tracked parameters
            self.component_params[component_name].update(param_updates)
        
        return updates
    
    def get_optimized_strategy(self) -> Strategy:
        """Get the current optimized strategy."""
        return self.base_strategy
