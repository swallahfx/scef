# """
# Strategy DSL module for composing trading strategies.

# This module provides a functional programming approach to create composable
# trading strategy components that can be combined to form complete strategies.
# """
# from typing import Callable, Dict, List, Optional, Union, Any, TypeVar, Generic
# import numpy as np
# import pandas as pd
# from dataclasses import dataclass
# from functools import partial, reduce

# # Type definitions
# Data = TypeVar('Data')  # Usually pd.DataFrame
# Signal = TypeVar('Signal')  # Usually float or np.ndarray
# Context = Dict[str, Any]

# @dataclass(frozen=True)
# class StrategyComponent:
#     """A pure function component of a trading strategy."""
#     name: str
#     function: Callable
#     category: str  # e.g., "indicator", "signal", "allocation", "execution"
#     parameters: Dict[str, Any]
    
#     def __call__(self, *args, **kwargs):
#         """Allow component to be called directly like a function."""
#         return self.function(*args, **kwargs)
    
#     def with_params(self, **kwargs):
#         """Return a new component with updated parameters."""
#         new_params = {**self.parameters, **kwargs}
#         return StrategyComponent(
#             name=self.name,
#             function=self.function,
#             category=self.category,
#             parameters=new_params
#         )


# class Strategy:
#     """A complete trading strategy composed of multiple components."""
    
#     def __init__(self, name: str):
#         self.name = name
#         self.indicators: List[StrategyComponent] = []
#         self.signals: List[StrategyComponent] = []
#         self.allocations: List[StrategyComponent] = []
#         self.executions: List[StrategyComponent] = []
#         self.risk_controls: List[StrategyComponent] = []
#         self.post_trades: List[StrategyComponent] = []
        
#     def add_indicator(self, component: StrategyComponent) -> 'Strategy':
#         """Add an indicator component to the strategy."""
#         assert component.category == "indicator", "Component must be an indicator"
#         self.indicators.append(component)
#         return self
        
#     def add_signal(self, component: StrategyComponent) -> 'Strategy':
#         """Add a signal generation component to the strategy."""
#         assert component.category == "signal", "Component must be a signal"
#         self.signals.append(component)
#         return self
    
#     def add_allocation(self, component: StrategyComponent) -> 'Strategy':
#         """Add a position allocation component to the strategy."""
#         assert component.category == "allocation", "Component must be an allocation"
#         self.allocations.append(component)
#         return self
    
#     def add_execution(self, component: StrategyComponent) -> 'Strategy':
#         """Add an execution component to the strategy."""
#         assert component.category == "execution", "Component must be an execution"
#         self.executions.append(component)
#         return self
    
#     def add_risk_control(self, component: StrategyComponent) -> 'Strategy':
#         """Add a risk control component to the strategy."""
#         assert component.category == "risk_control", "Component must be a risk control"
#         self.risk_controls.append(component)
#         return self
    
#     def add_post_trade(self, component: StrategyComponent) -> 'Strategy':
#         """Add a post-trade analysis component to the strategy."""
#         assert component.category == "post_trade", "Component must be a post-trade"
#         self.post_trades.append(component)
#         return self
    
#     def combine_signals(self, data: pd.DataFrame, context: Context) -> Signal:
#         """Combine all signals into a composite signal."""
#         if not self.signals:
#             return np.zeros(len(data))
            
#         # Calculate all signals
#         signals = [component(data, context) for component in self.signals]
        
#         # Simple weighted average combining approach (can be customized)
#         weights = np.ones(len(signals)) / len(signals)
#         combined = np.zeros_like(signals[0], dtype=float)
        
#         for i, signal in enumerate(signals):
#             combined += signal * weights[i]
            
#         return combined
    
#     def run_indicators(self, data: pd.DataFrame, context: Context) -> Dict[str, np.ndarray]:
#         """Run all indicator functions and return results."""
#         results = {}
#         for indicator in self.indicators:
#             results[indicator.name] = indicator(data, context, **indicator.parameters)
#         return results
    
#     def allocate(self, signal: Signal, data: pd.DataFrame, context: Context) -> np.ndarray:
#         """Determine position sizes based on signal and allocation components."""
#         if not self.allocations:
#             # Default to signal-proportional allocation if no allocators specified
#             return signal
            
#         # Allow each allocator to transform the positions
#         positions = signal
#         for allocator in self.allocations:
#             positions = allocator(positions, data, context, **allocator.parameters)
            
#         # Apply risk controls
#         for risk_control in self.risk_controls:
#             positions = risk_control(positions, data, context, **risk_control.parameters)
            
#         return positions
    
#     def execute(self, positions: np.ndarray, data: pd.DataFrame, context: Context) -> Dict[str, Any]:
#         """Simulate or execute trades to achieve target positions."""
#         results = {"target_positions": positions}
        
#         if not self.executions:
#             # With no execution components, just return target positions
#             results["actual_positions"] = positions
#             results["trades"] = np.zeros_like(positions)
#             return results
            
#         # Allow execution components to transform positions into actual trades
#         current_positions = context.get("current_positions", np.zeros_like(positions))
#         trades = positions - current_positions
        
#         # Execute each execution component
#         for execution in self.executions:
#             trades = execution(trades, data, context, **execution.parameters)
        
#         # Calculate final positions
#         final_positions = current_positions + trades
        
#         results["actual_positions"] = final_positions
#         results["trades"] = trades
        
#         # Run post-trade analysis
#         for post_trade in self.post_trades:
#             post_result = post_trade(trades, data, context, **post_trade.parameters)
#             results[post_trade.name] = post_result
            
#         return results
    
#     def run(self, data: pd.DataFrame, context: Optional[Context] = None) -> Dict[str, Any]:
#         """Run the complete strategy on the provided data."""
#         if context is None:
#             context = {}
            
#         # Compute all indicators
#         indicator_results = self.run_indicators(data, context)
#         context["indicators"] = indicator_results
        
#         # Generate signals
#         signal = self.combine_signals(data, context)
#         context["signal"] = signal
        
#         # Determine position sizes
#         positions = self.allocate(signal, data, context)
#         context["positions"] = positions
        
#         # Execute trades
#         execution_results = self.execute(positions, data, context)
#         context.update(execution_results)
        
#         return context
    
#     def __repr__(self):
#         components = {
#             "indicators": len(self.indicators),
#             "signals": len(self.signals),
#             "allocations": len(self.allocations),
#             "executions": len(self.executions),
#             "risk_controls": len(self.risk_controls),
#             "post_trades": len(self.post_trades)
#         }
#         return f"Strategy(name='{self.name}', components={components})"


# # Factory functions to create strategy components
# def create_indicator(name: str, function: Callable, **parameters) -> StrategyComponent:
#     """Create an indicator component."""
#     return StrategyComponent(name=name, function=function, category="indicator", parameters=parameters)

# def create_signal(name: str, function: Callable, **parameters) -> StrategyComponent:
#     """Create a signal component."""
#     return StrategyComponent(name=name, function=function, category="signal", parameters=parameters)

# def create_allocation(name: str, function: Callable, **parameters) -> StrategyComponent:
#     """Create an allocation component."""
#     return StrategyComponent(name=name, function=function, category="allocation", parameters=parameters)

# def create_execution(name: str, function: Callable, **parameters) -> StrategyComponent:
#     """Create an execution component."""
#     return StrategyComponent(name=name, function=function, category="execution", parameters=parameters)

# def create_risk_control(name: str, function: Callable, **parameters) -> StrategyComponent:
#     """Create a risk control component."""
#     return StrategyComponent(name=name, function=function, category="risk_control", parameters=parameters)

# def create_post_trade(name: str, function: Callable, **parameters) -> StrategyComponent:
#     """Create a post-trade component."""
#     return StrategyComponent(name=name, function=function, category="post_trade", parameters=parameters)


# # Composition utilities
# def compose(*functions):
#     """Compose functions: compose(f, g, h)(x) = f(g(h(x)))"""
#     return reduce(lambda f, g: lambda x: f(g(x)), functions)

# def pipe(value, *functions):
#     """Pipe a value through a sequence of functions."""
#     return reduce(lambda x, f: f(x), functions, value)



"""
Strategy DSL module for composing trading strategies.

This module provides a functional programming approach to create composable
trading strategy components that can be combined to form complete strategies.
"""
from typing import Callable, Dict, List, Optional, Union, Any, TypeVar, Generic
import numpy as np
import pandas as pd
from dataclasses import dataclass
from functools import partial, reduce

# Type definitions
Data = TypeVar('Data')  # Usually pd.DataFrame
Signal = TypeVar('Signal')  # Usually float or np.ndarray
Context = Dict[str, Any]

@dataclass(frozen=True)
class StrategyComponent:
    """A pure function component of a trading strategy."""
    name: str
    function: Callable
    category: str  # e.g., "indicator", "signal", "allocation", "execution"
    parameters: Dict[str, Any]
    
    def __call__(self, *args, **kwargs):
        """Allow component to be called directly like a function."""
        return self.function(*args, **kwargs)
    
    def with_params(self, **kwargs):
        """Return a new component with updated parameters."""
        new_params = {**self.parameters, **kwargs}
        return StrategyComponent(
            name=self.name,
            function=self.function,
            category=self.category,
            parameters=new_params
        )


class Strategy:
    """A complete trading strategy composed of multiple components."""
    
    def __init__(self, name: str):
        self.name = name
        self.indicators: List[StrategyComponent] = []
        self.signals: List[StrategyComponent] = []
        self.allocations: List[StrategyComponent] = []
        self.executions: List[StrategyComponent] = []
        self.risk_controls: List[StrategyComponent] = []
        self.post_trades: List[StrategyComponent] = []
        
    def add_indicator(self, component: StrategyComponent) -> 'Strategy':
        """Add an indicator component to the strategy."""
        assert component.category == "indicator", "Component must be an indicator"
        self.indicators.append(component)
        return self
        
    def add_signal(self, component: StrategyComponent) -> 'Strategy':
        """Add a signal generation component to the strategy."""
        assert component.category == "signal", "Component must be a signal"
        self.signals.append(component)
        return self
    
    def add_allocation(self, component: StrategyComponent) -> 'Strategy':
        """Add a position allocation component to the strategy."""
        assert component.category == "allocation", "Component must be an allocation"
        self.allocations.append(component)
        return self
    
    def add_execution(self, component: StrategyComponent) -> 'Strategy':
        """Add an execution component to the strategy."""
        assert component.category == "execution", "Component must be an execution"
        self.executions.append(component)
        return self
    
    def add_risk_control(self, component: StrategyComponent) -> 'Strategy':
        """Add a risk control component to the strategy."""
        assert component.category == "risk_control", "Component must be a risk control"
        self.risk_controls.append(component)
        return self
    
    def add_post_trade(self, component: StrategyComponent) -> 'Strategy':
        """Add a post-trade analysis component to the strategy."""
        assert component.category == "post_trade", "Component must be a post-trade"
        self.post_trades.append(component)
        return self
    
    def combine_signals(self, data: pd.DataFrame, context: Context) -> Signal:
        """Combine all signals into a composite signal."""
        if not self.signals:
            return np.zeros(len(data))
            
        # Calculate all signals
        signals = [component(data, context) for component in self.signals]
        
        # Simple weighted average combining approach (can be customized)
        weights = np.ones(len(signals)) / len(signals)
        combined = np.zeros_like(signals[0], dtype=float)
        
        for i, signal in enumerate(signals):
            combined += signal * weights[i]
            
        return combined
    
    def run_indicators(self, data: pd.DataFrame, context: Context) -> Dict[str, np.ndarray]:
        """Run all indicator functions and return results."""
        results = {}
        for indicator in self.indicators:
            results[indicator.name] = indicator(data, context, **indicator.parameters)
        return results
    
    def allocate(self, signal: Signal, data: pd.DataFrame, context: Context) -> np.ndarray:
        """Determine position sizes based on signal and allocation components."""
        if not self.allocations:
            # Default to signal-proportional allocation if no allocators specified
            return signal
            
        # Allow each allocator to transform the positions
        positions = signal
        for allocator in self.allocations:
            positions = allocator(positions, data, context, **allocator.parameters)
            
        # Apply risk controls
        for risk_control in self.risk_controls:
            positions = risk_control(positions, data, context, **risk_control.parameters)
            
        return positions
    
    def execute(self, positions: np.ndarray, data: pd.DataFrame, context: Context) -> Dict[str, Any]:
        """Simulate or execute trades to achieve target positions."""
        results = {"target_positions": positions}
        
        if not self.executions:
            # With no execution components, just return target positions
            results["actual_positions"] = positions
            results["trades"] = np.zeros_like(positions)
            return results
            
        # Allow execution components to transform positions into actual trades
        current_positions = context.get("current_positions", np.zeros_like(positions))
        trades = positions - current_positions
        
        # Execute each execution component
        for execution in self.executions:
            trades = execution(trades, data, context, **execution.parameters)
        
        # Calculate final positions
        final_positions = current_positions + trades
        
        results["actual_positions"] = final_positions
        results["trades"] = trades
        
        # Run post-trade analysis
        for post_trade in self.post_trades:
            post_result = post_trade(trades, data, context, **post_trade.parameters)
            results[post_trade.name] = post_result
            
        return results
    
    def run(self, data: pd.DataFrame, context: Optional[Context] = None) -> Dict[str, Any]:
        """Run the complete strategy on the provided data."""
        if context is None:
            context = {}
            
        # Compute all indicators
        indicator_results = self.run_indicators(data, context)
        context["indicators"] = indicator_results
        
        # Generate signals
        signal = self.combine_signals(data, context)
        context["signal"] = signal
        
        # Determine position sizes
        positions = self.allocate(signal, data, context)
        context["positions"] = positions
        
        # Execute trades
        execution_results = self.execute(positions, data, context)
        context.update(execution_results)
        
        return context
    
    def __repr__(self):
        components = {
            "indicators": len(self.indicators),
            "signals": len(self.signals),
            "allocations": len(self.allocations),
            "executions": len(self.executions),
            "risk_controls": len(self.risk_controls),
            "post_trades": len(self.post_trades)
        }
        return f"Strategy(name='{self.name}', components={components})"


# Factory functions to create strategy components
def create_indicator(name: str, function: Callable, **parameters) -> StrategyComponent:
    """Create an indicator component."""
    return StrategyComponent(name=name, function=function, category="indicator", parameters=parameters)

def create_signal(name: str, function: Callable, **parameters) -> StrategyComponent:
    """Create a signal component."""
    return StrategyComponent(name=name, function=function, category="signal", parameters=parameters)

def create_allocation(name: str, function: Callable, **parameters) -> StrategyComponent:
    """Create an allocation component."""
    return StrategyComponent(name=name, function=function, category="allocation", parameters=parameters)

def create_execution(name: str, function: Callable, **parameters) -> StrategyComponent:
    """Create an execution component."""
    return StrategyComponent(name=name, function=function, category="execution", parameters=parameters)

def create_risk_control(name: str, function: Callable, **parameters) -> StrategyComponent:
    """Create a risk control component."""
    return StrategyComponent(name=name, function=function, category="risk_control", parameters=parameters)

def create_post_trade(name: str, function: Callable, **parameters) -> StrategyComponent:
    """Create a post-trade component."""
    return StrategyComponent(name=name, function=function, category="post_trade", parameters=parameters)


# Composition utilities
def compose(*functions):
    """Compose functions: compose(f, g, h)(x) = f(g(h(x)))"""
    return reduce(lambda f, g: lambda x: f(g(x)), functions)

def pipe(value, *functions):
    """Pipe a value through a sequence of functions."""
    return reduce(lambda x, f: f(x), functions, value)


# Dictionary to store registered components by category
_registered_components = {
    "indicator": {},
    "signal": {},
    "allocation": {},
    "risk_control": {},
    "execution": {},
    "post_trade": {}
}

def register_indicator_component(name: str, function: Callable) -> None:
    """Register an indicator component function."""
    _registered_components["indicator"][name] = function

def register_signal_component(name: str, function: Callable) -> None:
    """Register a signal component function."""
    _registered_components["signal"][name] = function

def register_allocation_component(name: str, function: Callable) -> None:
    """Register an allocation component function."""
    _registered_components["allocation"][name] = function

def register_risk_control_component(name: str, function: Callable) -> None:
    """Register a risk control component function."""
    _registered_components["risk_control"][name] = function

def register_execution_component(name: str, function: Callable) -> None:
    """Register an execution component function."""
    _registered_components["execution"][name] = function

def register_post_trade_component(name: str, function: Callable) -> None:
    """Register a post-trade component function."""
    _registered_components["post_trade"][name] = function

def get_registered_component(category: str, name: str) -> Optional[Callable]:
    """Get a registered component by category and name."""
    return _registered_components.get(category, {}).get(name)

def list_registered_components(category: str = None) -> Dict[str, List[str]]:
    """List all registered components, optionally filtered by category."""
    if category:
        return {category: list(_registered_components.get(category, {}).keys())}
    return {cat: list(components.keys()) for cat, components in _registered_components.items()}