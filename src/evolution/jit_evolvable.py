"""
Evolvable JiT wrapper with selective layer evolution capabilities.

This module provides a wrapper around the JiT model that:
1. Identifies and categorizes parameters for evolution
2. Provides layer-wise statistics and visualization
3. Supports different evolution strategies per layer group
"""

import torch
import torch.nn as nn
from typing import Dict, List, Tuple, Optional, Set
from dataclasses import dataclass, field
import json

from .config import SelectiveLayerConfig


@dataclass
class LayerGroup:
    """A group of related parameters for evolution."""
    name: str
    param_names: List[str]
    total_params: int = 0
    evolution_priority: int = 1  # Higher = more important to evolve


class EvolvableJiT:
    """
    Wrapper that adds evolution capabilities to JiT model.

    Provides:
    - Automatic layer grouping and categorization
    - Parameter statistics and analysis
    - Selective freezing/unfreezing
    - Evolution progress tracking per layer group
    """

    # Default layer groups in order of evolution priority
    LAYER_GROUPS = [
        ('final_layer', ['final_layer.']),
        ('in_context', ['in_context_posemb']),
        ('late_attention', ['blocks.{LATE}.attn.']),
        ('late_adaln', ['blocks.{LATE}.adaLN_modulation.']),
        ('late_mlp', ['blocks.{LATE}.mlp.']),
        ('embedders', ['t_embedder.', 'y_embedder.']),
        ('patch_embed', ['x_embedder.']),
        ('early_blocks', ['blocks.{EARLY}.']),
    ]

    def __init__(
        self,
        model: nn.Module,
        config: Optional[SelectiveLayerConfig] = None,
    ):
        """
        Initialize evolvable wrapper.

        Args:
            model: JiT model instance
            config: Optional SelectiveLayerConfig
        """
        self.model = model
        self.config = config or SelectiveLayerConfig()

        # Get model depth
        self.depth = len(model.blocks) if hasattr(model, 'blocks') else 24
        self.late_start = self.depth - self.config.num_late_layers

        # Analyze and categorize parameters
        self.layer_groups: Dict[str, LayerGroup] = {}
        self.param_to_group: Dict[str, str] = {}
        self._analyze_parameters()

        # Track evolution statistics per group
        self.group_stats: Dict[str, Dict[str, float]] = {}
        self._initialize_stats()

    def _analyze_parameters(self):
        """Categorize all parameters into layer groups."""
        all_params = dict(self.model.named_parameters())

        for group_name, patterns in self.LAYER_GROUPS:
            expanded_patterns = self._expand_patterns(patterns)
            matching_params = []

            for name, param in all_params.items():
                if any(p in name for p in expanded_patterns):
                    matching_params.append(name)
                    self.param_to_group[name] = group_name

            self.layer_groups[group_name] = LayerGroup(
                name=group_name,
                param_names=matching_params,
                total_params=sum(all_params[n].numel() for n in matching_params),
            )

        # Check for uncategorized parameters
        uncategorized = [n for n in all_params if n not in self.param_to_group]
        if uncategorized:
            self.layer_groups['other'] = LayerGroup(
                name='other',
                param_names=uncategorized,
                total_params=sum(all_params[n].numel() for n in uncategorized),
            )
            for n in uncategorized:
                self.param_to_group[n] = 'other'

    def _expand_patterns(self, patterns: List[str]) -> List[str]:
        """Expand {LATE} and {EARLY} placeholders in patterns."""
        expanded = []
        for pattern in patterns:
            if '{LATE}' in pattern:
                for i in range(self.late_start, self.depth):
                    expanded.append(pattern.replace('{LATE}', str(i)))
            elif '{EARLY}' in pattern:
                for i in range(self.late_start):
                    expanded.append(pattern.replace('{EARLY}', str(i)))
            else:
                expanded.append(pattern)
        return expanded

    def _initialize_stats(self):
        """Initialize statistics tracking for each group."""
        for group_name in self.layer_groups:
            self.group_stats[group_name] = {
                'num_updates': 0,
                'cumulative_change': 0.0,
                'positive_votes': 0,
                'negative_votes': 0,
            }

    def get_evolvable_params(
        self,
        groups: Optional[List[str]] = None
    ) -> Dict[str, nn.Parameter]:
        """
        Get parameters to evolve.

        Args:
            groups: Optional list of group names to include

        Returns:
            Dict of parameter name to parameter
        """
        if groups is None:
            # Use config to determine which groups
            groups = self._get_active_groups()

        result = {}
        for group_name in groups:
            if group_name in self.layer_groups:
                for param_name in self.layer_groups[group_name].param_names:
                    result[param_name] = dict(self.model.named_parameters())[param_name]

        return result

    def _get_active_groups(self) -> List[str]:
        """Get list of active group names based on config."""
        active = []

        if self.config.evolve_final_layer:
            active.append('final_layer')
        if self.config.evolve_in_context_tokens:
            active.append('in_context')
        if self.config.evolve_late_attention:
            active.append('late_attention')
        if self.config.evolve_adaln_modulation:
            active.append('late_adaln')
        if self.config.evolve_mlp:
            active.append('late_mlp')
        if self.config.evolve_embedders:
            active.append('embedders')
        if self.config.evolve_patch_embed:
            active.append('patch_embed')

        return active

    def get_param_patterns(self) -> List[str]:
        """Get list of parameter patterns for SelectiveParameterPerturbation."""
        patterns = []
        for group_name in self._get_active_groups():
            if group_name in self.layer_groups:
                patterns.extend(self.layer_groups[group_name].param_names)
        return patterns

    def freeze_group(self, group_name: str):
        """Freeze all parameters in a group."""
        if group_name not in self.layer_groups:
            return

        for param_name in self.layer_groups[group_name].param_names:
            param = dict(self.model.named_parameters())[param_name]
            param.requires_grad = False

    def unfreeze_group(self, group_name: str):
        """Unfreeze all parameters in a group."""
        if group_name not in self.layer_groups:
            return

        for param_name in self.layer_groups[group_name].param_names:
            param = dict(self.model.named_parameters())[param_name]
            param.requires_grad = True

    def get_group_statistics(self) -> Dict[str, Dict[str, any]]:
        """Get detailed statistics for each layer group."""
        stats = {}
        all_params = dict(self.model.named_parameters())

        for group_name, group in self.layer_groups.items():
            group_params = [all_params[n] for n in group.param_names]

            if not group_params:
                continue

            # Flatten all parameters
            flat = torch.cat([p.data.flatten().float() for p in group_params])

            stats[group_name] = {
                'num_params': group.total_params,
                'num_tensors': len(group.param_names),
                'mean': flat.mean().item(),
                'std': flat.std().item(),
                'min': flat.min().item(),
                'max': flat.max().item(),
                'abs_mean': flat.abs().mean().item(),
                'sparsity': (flat.abs() < 1e-6).float().mean().item(),
                **self.group_stats[group_name],
            }

        return stats

    def print_summary(self):
        """Print summary of model parameters and evolution targets."""
        total_params = sum(g.total_params for g in self.layer_groups.values())
        active_groups = self._get_active_groups()
        evolvable_params = sum(
            self.layer_groups[g].total_params
            for g in active_groups
            if g in self.layer_groups
        )

        print("\n" + "=" * 70)
        print("EVOLVABLE JiT SUMMARY")
        print("=" * 70)
        print(f"Model depth: {self.depth} blocks")
        print(f"Late layer start: block {self.late_start}")
        print(f"Total parameters: {total_params:,}")
        print(f"Evolvable parameters: {evolvable_params:,} ({100*evolvable_params/total_params:.1f}%)")
        print("\nLayer Groups:")
        print("-" * 70)

        for group_name, group in self.layer_groups.items():
            status = "✓ EVOLVE" if group_name in active_groups else "✗ frozen"
            pct = 100 * group.total_params / total_params
            print(f"  {group_name:20s} | {group.total_params:12,} params ({pct:5.1f}%) | {status}")

        print("=" * 70 + "\n")

    def export_config(self) -> Dict:
        """Export configuration for reproducibility."""
        return {
            'depth': self.depth,
            'late_start': self.late_start,
            'layer_config': {
                'evolve_final_layer': self.config.evolve_final_layer,
                'evolve_in_context_tokens': self.config.evolve_in_context_tokens,
                'evolve_late_attention': self.config.evolve_late_attention,
                'evolve_adaln_modulation': self.config.evolve_adaln_modulation,
                'evolve_mlp': self.config.evolve_mlp,
                'evolve_embedders': self.config.evolve_embedders,
                'evolve_patch_embed': self.config.evolve_patch_embed,
                'num_late_layers': self.config.num_late_layers,
            },
            'layer_groups': {
                name: {
                    'num_params': group.total_params,
                    'num_tensors': len(group.param_names),
                }
                for name, group in self.layer_groups.items()
            },
            'active_groups': self._get_active_groups(),
        }


def analyze_jit_model(model: nn.Module) -> Dict:
    """
    Standalone function to analyze a JiT model's parameter distribution.

    Useful for understanding the model before configuring evolution.

    Args:
        model: JiT model instance

    Returns:
        Dict with analysis results
    """
    wrapper = EvolvableJiT(model)
    wrapper.print_summary()
    return wrapper.get_group_statistics()


def create_evolvable_jit(
    model: nn.Module,
    evolve_strategy: str = 'default',
) -> EvolvableJiT:
    """
    Factory function to create EvolvableJiT with predefined strategies.

    Args:
        model: JiT model
        evolve_strategy: One of 'minimal', 'default', 'aggressive', 'full'

    Returns:
        EvolvableJiT instance
    """
    strategies = {
        'minimal': SelectiveLayerConfig(
            evolve_final_layer=True,
            evolve_in_context_tokens=True,
            evolve_late_attention=False,
            evolve_adaln_modulation=False,
            evolve_mlp=False,
            evolve_embedders=False,
            evolve_patch_embed=False,
            num_late_layers=2,
        ),
        'default': SelectiveLayerConfig(
            evolve_final_layer=True,
            evolve_in_context_tokens=True,
            evolve_late_attention=True,
            evolve_adaln_modulation=True,
            evolve_mlp=False,
            evolve_embedders=False,
            evolve_patch_embed=False,
            num_late_layers=4,
        ),
        'aggressive': SelectiveLayerConfig(
            evolve_final_layer=True,
            evolve_in_context_tokens=True,
            evolve_late_attention=True,
            evolve_adaln_modulation=True,
            evolve_mlp=True,
            evolve_embedders=True,
            evolve_patch_embed=False,
            num_late_layers=6,
        ),
        'full': SelectiveLayerConfig(
            evolve_final_layer=True,
            evolve_in_context_tokens=True,
            evolve_late_attention=True,
            evolve_adaln_modulation=True,
            evolve_mlp=True,
            evolve_embedders=True,
            evolve_patch_embed=True,
            num_late_layers=8,
        ),
    }

    if evolve_strategy not in strategies:
        raise ValueError(f"Unknown strategy: {evolve_strategy}. "
                        f"Choose from: {list(strategies.keys())}")

    return EvolvableJiT(model, strategies[evolve_strategy])
