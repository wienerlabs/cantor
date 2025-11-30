"""Blockchain data extraction from Ethereum RPC."""

from __future__ import annotations

import asyncio
from typing import AsyncIterator

import structlog
from web3 import AsyncWeb3, AsyncHTTPProvider
from web3.types import BlockData as Web3BlockData

from cantor.core.types import (
    BlockData,
    Transaction,
    TransactionType,
    Address,
    Bytes32,
)
from cantor.core.config import DataConfig

logger = structlog.get_logger()


class BlockchainExtractor:
    """Extracts block and transaction data from Ethereum RPC endpoint."""

    def __init__(self, config: DataConfig) -> None:
        self.config = config
        self._web3: AsyncWeb3 | None = None

    async def connect(self) -> None:
        provider = AsyncHTTPProvider(self.config.rpc_url)
        self._web3 = AsyncWeb3(provider)
        chain_id = await self._web3.eth.chain_id
        latest = await self._web3.eth.block_number
        logger.info("connected_to_rpc", chain_id=chain_id, latest_block=latest)

    async def close(self) -> None:
        if self._web3 and self._web3.provider:
            await self._web3.provider.disconnect()
        self._web3 = None

    @property
    def web3(self) -> AsyncWeb3:
        if self._web3 is None:
            raise RuntimeError("Not connected. Call connect() first.")
        return self._web3

    async def get_latest_block(self) -> int:
        return await self.web3.eth.block_number

    async def get_block(self, block_number: int) -> BlockData:
        raw = await self.web3.eth.get_block(block_number, full_transactions=True)
        return self._parse_block(raw)

    def _parse_block(self, raw: Web3BlockData) -> BlockData:
        txs = tuple(self._parse_transaction(tx, raw["number"]) for tx in raw.get("transactions", []))
        return BlockData(
            number=raw["number"],
            hash=bytes(raw["hash"]),
            parent_hash=bytes(raw["parentHash"]),
            timestamp=raw["timestamp"],
            gas_used=raw["gasUsed"],
            gas_limit=raw["gasLimit"],
            base_fee=raw.get("baseFeePerGas"),
            transactions=txs,
            state_root=bytes(raw["stateRoot"]),
        )

    def _parse_transaction(self, tx: dict, block_number: int) -> Transaction:
        tx_type = TransactionType(tx.get("type", 0))
        return Transaction(
            hash=bytes(tx["hash"]),
            block_number=block_number,
            tx_index=tx["transactionIndex"],
            tx_type=tx_type,
            from_addr=bytes.fromhex(tx["from"][2:]),
            to_addr=bytes.fromhex(tx["to"][2:]) if tx.get("to") else None,
            value=tx["value"],
            gas_limit=tx["gas"],
            gas_used=tx.get("gasUsed", tx["gas"]),
            gas_price=tx.get("gasPrice", 0),
            input_data=bytes.fromhex(tx["input"][2:]) if tx.get("input") else b"",
            nonce=tx["nonce"],
            success=True,
        )

    async def stream_blocks(
        self, start: int | None = None, end: int | None = None
    ) -> AsyncIterator[BlockData]:
        start_block = start or self.config.start_block
        end_block = end or self.config.end_block or await self.get_latest_block()

        logger.info("streaming_blocks", start=start_block, end=end_block)

        for block_num in range(start_block, end_block + 1):
            try:
                block = await self.get_block(block_num)
                yield block
            except Exception as e:
                logger.error("block_extraction_failed", block=block_num, error=str(e))
                raise

    async def stream_blocks_batch(
        self, start: int, end: int, batch_size: int | None = None
    ) -> AsyncIterator[list[BlockData]]:
        batch_size = batch_size or self.config.batch_size
        current = start

        while current <= end:
            batch_end = min(current + batch_size - 1, end)
            tasks = [self.get_block(n) for n in range(current, batch_end + 1)]
            blocks = await asyncio.gather(*tasks, return_exceptions=True)

            valid_blocks = [b for b in blocks if isinstance(b, BlockData)]
            if valid_blocks:
                yield valid_blocks

            current = batch_end + 1

