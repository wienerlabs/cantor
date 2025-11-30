"""Temporal pattern analysis for transaction sequences."""

from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime, timezone
from typing import Sequence
from collections import defaultdict

import numpy as np
from numpy.typing import NDArray
import structlog

from cantor.core.types import Transaction, BlockData

logger = structlog.get_logger()


@dataclass(frozen=True, slots=True)
class TemporalPattern:
    """A time-based recurring pattern."""
    
    pattern_type: str
    period_seconds: int
    strength: float
    peak_hour: int | None = None
    peak_day: int | None = None


class TemporalPatternAnalyzer:
    """Analyze temporal patterns in transaction activity."""

    def __init__(self, min_samples: int = 100) -> None:
        self.min_samples = min_samples
        self._hourly_distribution: NDArray[np.float32] | None = None
        self._daily_distribution: NDArray[np.float32] | None = None
        self._patterns: list[TemporalPattern] = []

    def analyze(self, blocks: Sequence[BlockData]) -> list[TemporalPattern]:
        """Analyze temporal patterns from block data."""
        if len(blocks) < self.min_samples:
            logger.warning("insufficient_blocks", count=len(blocks))
            return []

        timestamps = [block.timestamp for block in blocks]
        tx_counts = [len(block.transactions) for block in blocks]
        
        self._patterns = []
        
        # Hourly analysis
        hourly = self._compute_hourly_distribution(timestamps, tx_counts)
        self._hourly_distribution = hourly
        
        peak_hour = int(np.argmax(hourly))
        hourly_variance = float(np.var(hourly))
        
        if hourly_variance > 0.01:
            self._patterns.append(TemporalPattern(
                pattern_type="HOURLY_CYCLE",
                period_seconds=3600,
                strength=float(np.std(hourly) / np.mean(hourly)) if np.mean(hourly) > 0 else 0,
                peak_hour=peak_hour,
            ))
        
        # Daily analysis
        daily = self._compute_daily_distribution(timestamps, tx_counts)
        self._daily_distribution = daily
        
        peak_day = int(np.argmax(daily))
        daily_variance = float(np.var(daily))
        
        if daily_variance > 0.01:
            self._patterns.append(TemporalPattern(
                pattern_type="DAILY_CYCLE",
                period_seconds=86400,
                strength=float(np.std(daily) / np.mean(daily)) if np.mean(daily) > 0 else 0,
                peak_day=peak_day,
            ))
        
        # Detect burst patterns
        burst_pattern = self._detect_bursts(tx_counts)
        if burst_pattern:
            self._patterns.append(burst_pattern)
        
        logger.info("temporal_analysis_complete", patterns=len(self._patterns))
        return self._patterns

    def _compute_hourly_distribution(
        self, timestamps: Sequence[int], tx_counts: Sequence[int]
    ) -> NDArray[np.float32]:
        """Compute transaction distribution by hour of day."""
        hourly = np.zeros(24, dtype=np.float32)
        counts = np.zeros(24, dtype=np.int32)
        
        for ts, count in zip(timestamps, tx_counts):
            dt = datetime.fromtimestamp(ts, tz=timezone.utc)
            hourly[dt.hour] += count
            counts[dt.hour] += 1
        
        # Normalize by number of observations
        with np.errstate(divide='ignore', invalid='ignore'):
            hourly = np.where(counts > 0, hourly / counts, 0)
        
        # Normalize to probability distribution
        total = hourly.sum()
        if total > 0:
            hourly /= total
        
        return hourly

    def _compute_daily_distribution(
        self, timestamps: Sequence[int], tx_counts: Sequence[int]
    ) -> NDArray[np.float32]:
        """Compute transaction distribution by day of week."""
        daily = np.zeros(7, dtype=np.float32)
        counts = np.zeros(7, dtype=np.int32)
        
        for ts, count in zip(timestamps, tx_counts):
            dt = datetime.fromtimestamp(ts, tz=timezone.utc)
            daily[dt.weekday()] += count
            counts[dt.weekday()] += 1
        
        with np.errstate(divide='ignore', invalid='ignore'):
            daily = np.where(counts > 0, daily / counts, 0)
        
        total = daily.sum()
        if total > 0:
            daily /= total
        
        return daily

    def _detect_bursts(self, tx_counts: Sequence[int]) -> TemporalPattern | None:
        """Detect burst activity patterns."""
        if len(tx_counts) < 10:
            return None
        
        counts = np.array(tx_counts, dtype=np.float32)
        mean_count = np.mean(counts)
        std_count = np.std(counts)
        
        if std_count == 0:
            return None
        
        # Count blocks with activity > 2 std above mean
        threshold = mean_count + 2 * std_count
        burst_blocks = np.sum(counts > threshold)
        burst_ratio = burst_blocks / len(counts)
        
        if burst_ratio > 0.05:
            return TemporalPattern(
                pattern_type="BURST_ACTIVITY",
                period_seconds=0,
                strength=float(burst_ratio),
            )
        
        return None

    def get_hourly_distribution(self) -> NDArray[np.float32] | None:
        return self._hourly_distribution

    def get_daily_distribution(self) -> NDArray[np.float32] | None:
        return self._daily_distribution

    def get_patterns(self) -> list[TemporalPattern]:
        return self._patterns

