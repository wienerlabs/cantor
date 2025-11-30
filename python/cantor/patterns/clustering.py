"""Transaction clustering for state transition grouping."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Sequence

import numpy as np
from numpy.typing import NDArray
from sklearn.cluster import MiniBatchKMeans
from sklearn.preprocessing import StandardScaler
import structlog

from cantor.core.types import Transaction
from cantor.data.features import FeatureEncoder
from cantor.core.config import ModelConfig

logger = structlog.get_logger()


@dataclass(frozen=True, slots=True)
class TransactionCluster:
    """A cluster of similar transactions."""
    
    cluster_id: int
    centroid: NDArray[np.float32]
    size: int
    label: str


class TransactionClusterer:
    """Cluster transactions by feature similarity using MiniBatch K-Means."""

    def __init__(
        self,
        n_clusters: int = 50,
        batch_size: int = 1024,
        random_state: int = 42,
    ) -> None:
        self.n_clusters = n_clusters
        self._kmeans = MiniBatchKMeans(
            n_clusters=n_clusters,
            batch_size=batch_size,
            random_state=random_state,
            n_init=3,
        )
        self._scaler = StandardScaler()
        self._encoder: FeatureEncoder | None = None
        self._fitted = False
        self._clusters: list[TransactionCluster] = []

    def fit(
        self,
        transactions: Sequence[Transaction],
        encoder: FeatureEncoder,
    ) -> list[TransactionCluster]:
        """Fit clustering model on transaction features."""
        self._encoder = encoder
        
        features = np.stack([encoder.encode_transaction(tx) for tx in transactions])
        logger.info("clustering_transactions", n_samples=len(features))
        
        features_scaled = self._scaler.fit_transform(features)
        self._kmeans.fit(features_scaled)
        self._fitted = True
        
        labels = self._kmeans.labels_
        centroids = self._scaler.inverse_transform(self._kmeans.cluster_centers_)
        
        cluster_sizes = np.bincount(labels, minlength=self.n_clusters)
        
        self._clusters = []
        for i in range(self.n_clusters):
            cluster_txs = [tx for tx, l in zip(transactions, labels) if l == i]
            label = self._generate_cluster_label(cluster_txs)
            
            self._clusters.append(TransactionCluster(
                cluster_id=i,
                centroid=centroids[i].astype(np.float32),
                size=int(cluster_sizes[i]),
                label=label,
            ))
        
        self._clusters.sort(key=lambda c: c.size, reverse=True)
        logger.info("clustering_complete", n_clusters=len(self._clusters))
        
        return self._clusters

    def predict(self, transactions: Sequence[Transaction]) -> NDArray[np.int32]:
        """Predict cluster assignments for transactions."""
        if not self._fitted or self._encoder is None:
            raise RuntimeError("Model not fitted. Call fit() first.")
        
        features = np.stack([self._encoder.encode_transaction(tx) for tx in transactions])
        features_scaled = self._scaler.transform(features)
        return self._kmeans.predict(features_scaled).astype(np.int32)

    def partial_fit(self, transactions: Sequence[Transaction]) -> None:
        """Incrementally update clustering with new transactions."""
        if self._encoder is None:
            raise RuntimeError("Model not initialized. Call fit() first.")
        
        features = np.stack([self._encoder.encode_transaction(tx) for tx in transactions])
        features_scaled = self._scaler.transform(features)
        self._kmeans.partial_fit(features_scaled)

    def get_cluster_distribution(self) -> dict[str, int]:
        """Get distribution of cluster sizes by label."""
        return {c.label: c.size for c in self._clusters}

    def _generate_cluster_label(self, transactions: Sequence[Transaction]) -> str:
        """Generate a descriptive label for a cluster."""
        if not transactions:
            return "EMPTY"
        
        # Analyze cluster characteristics
        has_input = sum(1 for tx in transactions if tx.input_data)
        is_contract_call = has_input / len(transactions) > 0.5
        
        avg_value = np.mean([tx.value for tx in transactions])
        avg_gas = np.mean([tx.gas_used for tx in transactions])
        
        if not is_contract_call:
            if avg_value > 1e18:
                return "HIGH_VALUE_TRANSFER"
            return "ETH_TRANSFER"
        
        if avg_gas > 200000:
            return "COMPLEX_CONTRACT"
        elif avg_gas > 50000:
            return "DEFI_INTERACTION"
        else:
            return "SIMPLE_CONTRACT"

    def get_clusters(self) -> list[TransactionCluster]:
        """Return all clusters sorted by size."""
        return self._clusters

