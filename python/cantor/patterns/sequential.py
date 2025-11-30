"""Sequential pattern mining using PrefixSpan algorithm."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Sequence
from collections import Counter

import structlog
from prefixspan import PrefixSpan

from cantor.core.types import Transaction

logger = structlog.get_logger()


@dataclass(frozen=True, slots=True)
class TransactionPattern:
    """A frequently occurring transaction sequence pattern."""
    
    sequence: tuple[str, ...]
    support: int
    frequency: float
    
    def __len__(self) -> int:
        return len(self.sequence)


class SequentialPatternMiner:
    """Mine sequential patterns from transaction sequences using PrefixSpan."""

    def __init__(
        self,
        min_support: int = 10,
        max_pattern_length: int = 10,
    ) -> None:
        self.min_support = min_support
        self.max_pattern_length = max_pattern_length
        self._patterns: list[TransactionPattern] = []

    def classify_transaction(self, tx: Transaction) -> str:
        """Classify a transaction into a symbolic category."""
        if tx.to_addr is None:
            return "CONTRACT_DEPLOY"
        
        if not tx.input_data or len(tx.input_data) < 4:
            return "ETH_TRANSFER"
        
        # Extract function selector
        selector = tx.input_data[:4].hex()
        
        # Common ERC-20 selectors
        selectors = {
            "a9059cbb": "ERC20_TRANSFER",
            "095ea7b3": "ERC20_APPROVE", 
            "23b872dd": "ERC20_TRANSFER_FROM",
            "40c10f19": "ERC20_MINT",
            "42966c68": "ERC20_BURN",
            # Uniswap V2
            "7ff36ab5": "SWAP_ETH_FOR_TOKENS",
            "38ed1739": "SWAP_TOKENS_FOR_TOKENS",
            "18cbafe5": "SWAP_TOKENS_FOR_ETH",
            "e8e33700": "ADD_LIQUIDITY",
            "baa2abde": "REMOVE_LIQUIDITY",
            # Uniswap V3
            "c04b8d59": "V3_EXACT_INPUT",
            "db3e2198": "V3_EXACT_OUTPUT",
            # NFT
            "23b872dd": "NFT_TRANSFER",
            "a22cb465": "NFT_SET_APPROVAL",
            "b88d4fde": "NFT_SAFE_TRANSFER",
        }
        
        return selectors.get(selector, f"FUNC_{selector[:8]}")

    def prepare_sequences(
        self, transactions: Sequence[Transaction], window_size: int = 50
    ) -> list[list[str]]:
        """Convert transactions into symbolic sequences for mining."""
        symbols = [self.classify_transaction(tx) for tx in transactions]
        
        sequences: list[list[str]] = []
        for i in range(0, len(symbols) - window_size + 1, window_size // 2):
            window = symbols[i:i + window_size]
            sequences.append(window)
        
        return sequences

    def mine_patterns(
        self, transactions: Sequence[Transaction], window_size: int = 50
    ) -> list[TransactionPattern]:
        """Mine frequent sequential patterns from transactions."""
        sequences = self.prepare_sequences(transactions, window_size)
        
        if not sequences:
            logger.warning("no_sequences_to_mine")
            return []
        
        logger.info("mining_patterns", num_sequences=len(sequences))
        
        ps = PrefixSpan(sequences)
        raw_patterns = ps.frequent(self.min_support, closed=True)
        
        total_sequences = len(sequences)
        patterns: list[TransactionPattern] = []
        
        for support, pattern in raw_patterns:
            if len(pattern) <= self.max_pattern_length:
                patterns.append(TransactionPattern(
                    sequence=tuple(pattern),
                    support=support,
                    frequency=support / total_sequences,
                ))
        
        patterns.sort(key=lambda p: p.support, reverse=True)
        self._patterns = patterns
        
        logger.info("patterns_mined", count=len(patterns))
        return patterns

    def get_pattern_coverage(self, transactions: Sequence[Transaction]) -> float:
        """Calculate what fraction of transactions match known patterns."""
        if not self._patterns:
            return 0.0
        
        symbols = [self.classify_transaction(tx) for tx in transactions]
        matched = 0
        
        pattern_set = {p.sequence for p in self._patterns}
        
        for i in range(len(symbols)):
            for length in range(2, min(self.max_pattern_length + 1, len(symbols) - i + 1)):
                subseq = tuple(symbols[i:i + length])
                if subseq in pattern_set:
                    matched += length
                    break
        
        return matched / len(symbols) if symbols else 0.0

    def get_top_patterns(self, n: int = 20) -> list[TransactionPattern]:
        """Return top N patterns by support."""
        return self._patterns[:n]

    def get_pattern_distribution(self) -> dict[int, int]:
        """Get distribution of pattern lengths."""
        return dict(Counter(len(p) for p in self._patterns))

