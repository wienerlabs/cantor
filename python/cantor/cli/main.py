"""CLI entry point for CANTOR."""

from __future__ import annotations

import asyncio
from pathlib import Path
from typing import Optional

import typer
import structlog

from cantor.core.config import CantorConfig

app = typer.Typer(
    name="cantor",
    help="Blockchain state compression using ML-driven prediction",
)

logger = structlog.get_logger()


@app.command()
def train(
    config_path: Path = typer.Option(
        Path("config.yaml"),
        "--config", "-c",
        help="Path to configuration file",
    ),
    data_dir: Path = typer.Option(
        Path("data"),
        "--data", "-d",
        help="Directory containing training data",
    ),
    output_dir: Path = typer.Option(
        Path("checkpoints"),
        "--output", "-o",
        help="Directory for model checkpoints",
    ),
    epochs: int = typer.Option(100, "--epochs", "-e"),
    batch_size: int = typer.Option(32, "--batch-size", "-b"),
    device: str = typer.Option("cpu", "--device"),
) -> None:
    """Train the state prediction model."""
    from cantor.training.trainer import Trainer
    from cantor.data.dataset import TransactionDataset
    from cantor.models.transformer import StatePredictor
    
    config = CantorConfig.from_yaml(config_path) if config_path.exists() else CantorConfig()
    config.training.epochs = epochs
    config.training.batch_size = batch_size
    
    model = StatePredictor(config.model)
    dataset = TransactionDataset(data_dir, config.model.state_dim)
    
    trainer = Trainer(model, config.training, device)
    trainer.train(dataset, output_dir)
    
    typer.echo(f"Training complete. Model saved to {output_dir}")


@app.command()
def compress(
    start_block: int = typer.Argument(..., help="Starting block number"),
    end_block: int = typer.Argument(..., help="Ending block number"),
    model_path: Path = typer.Option(
        Path("checkpoints/best.pt"),
        "--model", "-m",
        help="Path to trained model",
    ),
    output_dir: Path = typer.Option(
        Path("compressed"),
        "--output", "-o",
        help="Output directory for compressed data",
    ),
    rpc_url: Optional[str] = typer.Option(
        None,
        "--rpc",
        help="Ethereum RPC URL",
    ),
) -> None:
    """Compress blockchain state for a range of blocks."""
    from cantor.client.compressor import CantorClient
    from cantor.client.storage import CompressedStateStore
    
    config = CantorConfig()
    if rpc_url:
        config.data.rpc_url = rpc_url
    
    client = CantorClient(config, model_path)
    store = CompressedStateStore(output_dir)
    
    async def run() -> None:
        await client.connect()
        try:
            async for result in client.compress_range(start_block, end_block):
                store.store(result)
                typer.echo(
                    f"Block {result.block_number}: "
                    f"{result.original_size} -> {result.compressed_size} bytes "
                    f"({result.original_size / max(result.compressed_size, 1):.2f}x)"
                )
        finally:
            await client.disconnect()
    
    asyncio.run(run())
    
    stats = store.get_stats()
    typer.echo(f"\nCompression complete: {stats['compression_ratio']:.2f}x overall")


@app.command()
def verify(
    block_number: int = typer.Argument(..., help="Block number to verify"),
    data_dir: Path = typer.Option(
        Path("compressed"),
        "--data", "-d",
        help="Directory containing compressed data",
    ),
    model_path: Path = typer.Option(
        Path("checkpoints/best.pt"),
        "--model", "-m",
        help="Path to trained model",
    ),
) -> None:
    """Verify compressed state for a block."""
    from cantor.client.storage import CompressedStateStore
    from cantor.verification.verifier import StateVerifier
    from cantor.models.transformer import StatePredictor
    from cantor.core.config import CantorConfig
    import torch
    
    config = CantorConfig()
    model = StatePredictor(config.model)
    
    if model_path.exists():
        checkpoint = torch.load(model_path, map_location="cpu")
        model.load_state_dict(checkpoint["model_state"])
    
    store = CompressedStateStore(data_dir)
    result = store.load(block_number)
    
    if not result:
        typer.echo(f"Block {block_number} not found", err=True)
        raise typer.Exit(1)
    
    verifier = StateVerifier(model, config.model.version)
    verifications = verifier.verify_block(result, [])
    
    valid = sum(1 for v in verifications if v.status.value == "valid")
    typer.echo(f"Block {block_number}: {valid}/{len(verifications)} valid")


@app.command()
def stats(
    data_dir: Path = typer.Option(
        Path("compressed"),
        "--data", "-d",
        help="Directory containing compressed data",
    ),
) -> None:
    """Show compression statistics."""
    from cantor.client.storage import CompressedStateStore
    
    store = CompressedStateStore(data_dir)
    stats = store.get_stats()
    
    typer.echo(f"Blocks: {stats['block_count']}")
    typer.echo(f"Original: {stats['total_original_bytes']:,} bytes")
    typer.echo(f"Compressed: {stats['total_compressed_bytes']:,} bytes")
    typer.echo(f"Ratio: {stats['compression_ratio']:.2f}x")


def main() -> None:
    app()


if __name__ == "__main__":
    main()

