"""Main client for CANTOR compression operations."""

from __future__ import annotations

from pathlib import Path
from typing import AsyncIterator

import numpy as np
from numpy.typing import NDArray
import torch
import structlog

from cantor.core.config import CantorConfig
from cantor.core.types import CompressionResult, Bytes32
from cantor.models.transformer import StatePredictor
from cantor.compression.delta import DeltaCompressor
from cantor.verification.verifier import StateVerifier
from cantor.data.extractor import BlockchainExtractor
from cantor.data.features import FeatureEncoder

logger = structlog.get_logger()


class CantorClient:
    """High-level client for blockchain state compression."""

    def __init__(
        self,
        config: CantorConfig,
        model_path: Path | None = None,
        device: str = "cpu",
    ) -> None:
        self.config = config
        self.device = device
        
        self.model = StatePredictor(config.model)
        if model_path and model_path.exists():
            self._load_model(model_path)
        self.model.to(device)
        self.model.eval()
        
        self.compressor = DeltaCompressor(
            self.model,
            config.compression,
            device,
        )
        
        self.verifier = StateVerifier(
            self.model,
            model_version=config.model.version,
            device=device,
        )
        
        self.extractor = BlockchainExtractor(config.data.rpc_url)
        self.encoder = FeatureEncoder(config.model.state_dim)
        
        self._connected = False

    async def connect(self) -> None:
        """Connect to blockchain RPC."""
        await self.extractor.connect()
        self._connected = True
        logger.info("client_connected", rpc=self.config.data.rpc_url)

    async def disconnect(self) -> None:
        """Disconnect from blockchain RPC."""
        await self.extractor.disconnect()
        self._connected = False

    async def compress_block(self, block_number: int) -> CompressionResult:
        """Compress a single block's state transitions."""
        if not self._connected:
            await self.connect()
        
        block = await self.extractor.get_block(block_number)
        
        sequences: list[NDArray[np.float32]] = []
        actual_states: list[NDArray[np.float32]] = []
        tx_hashes: list[Bytes32] = []
        
        for tx in block.transactions:
            features = self.encoder.encode(tx)
            sequences.append(features.reshape(1, -1))
            actual_states.append(features)
            tx_hashes.append(bytes.fromhex(tx.hash[2:]))
        
        result = self.compressor.compress_block(
            sequences,
            actual_states,
            tx_hashes,
            block_number,
        )
        
        logger.info(
            "block_compressed",
            block=block_number,
            txs=len(block.transactions),
            ratio=result.original_size / max(result.compressed_size, 1),
        )
        
        return result

    async def compress_range(
        self,
        start_block: int,
        end_block: int,
    ) -> AsyncIterator[CompressionResult]:
        """Compress a range of blocks."""
        for block_num in range(start_block, end_block + 1):
            yield await self.compress_block(block_num)

    def verify_result(
        self,
        result: CompressionResult,
        sequences: list[NDArray[np.float32]],
    ) -> bool:
        """Verify a compression result."""
        verifications = self.verifier.verify_block(result, sequences)
        return all(v.status.value == "valid" for v in verifications)

    def _load_model(self, path: Path) -> None:
        checkpoint = torch.load(path, map_location=self.device)
        self.model.load_state_dict(checkpoint["model_state"])
        logger.info("model_loaded", path=str(path))

    def save_model(self, path: Path) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        torch.save({"model_state": self.model.state_dict()}, path)
        logger.info("model_saved", path=str(path))

    @property
    def compression_stats(self) -> dict[str, float]:
        return {
            "model_params": self.model.get_num_parameters(),
            "device": self.device,
        }

