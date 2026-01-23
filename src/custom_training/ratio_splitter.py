"""
Simple ratio-based splitter for arbitrary datasets.
Splits scenarios randomly based on train/val/test ratios.
"""

from typing import List
from nuplan.planning.scenario_builder.abstract_scenario import AbstractScenario
from nuplan.planning.training.data_loader.splitter import AbstractSplitter
from nuplan.planning.utils.multithreading.worker_pool import WorkerPool
import random


class RatioSplitter(AbstractSplitter):
    """Random ratio-based splitter."""

    def __init__(self, train_ratio: float = 0.7, val_ratio: float = 0.2, test_ratio: float = 0.1, seed: int = 0):
        """
        Initialize the ratio splitter.
        
        :param train_ratio: Fraction of samples for training (default 0.7)
        :param val_ratio: Fraction of samples for validation (default 0.2)
        :param test_ratio: Fraction of samples for testing (default 0.1)
        :param seed: Random seed for reproducibility
        """
        assert abs(train_ratio + val_ratio + test_ratio - 1.0) < 1e-5, \
            "Train, val, test ratios must sum to 1.0"
        self.train_ratio = train_ratio
        self.val_ratio = val_ratio
        self.test_ratio = test_ratio
        self.seed = seed
        random.seed(seed)

    def get_train_samples(self, scenarios: List[AbstractScenario], worker: WorkerPool) -> List[AbstractScenario]:
        """Get training samples."""
        scenarios_shuffled = scenarios.copy()
        random.shuffle(scenarios_shuffled)
        n_train = int(len(scenarios_shuffled) * self.train_ratio)
        return scenarios_shuffled[:n_train]

    def get_val_samples(self, scenarios: List[AbstractScenario], worker: WorkerPool) -> List[AbstractScenario]:
        """Get validation samples."""
        scenarios_shuffled = scenarios.copy()
        random.shuffle(scenarios_shuffled)
        n_train = int(len(scenarios_shuffled) * self.train_ratio)
        n_val = int(len(scenarios_shuffled) * self.val_ratio)
        return scenarios_shuffled[n_train:n_train + n_val]

    def get_test_samples(self, scenarios: List[AbstractScenario], worker: WorkerPool) -> List[AbstractScenario]:
        """Get test samples."""
        scenarios_shuffled = scenarios.copy()
        random.shuffle(scenarios_shuffled)
        n_train = int(len(scenarios_shuffled) * self.train_ratio)
        n_val = int(len(scenarios_shuffled) * self.val_ratio)
        return scenarios_shuffled[n_train + n_val:]
