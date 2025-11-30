use criterion::{black_box, criterion_group, criterion_main, Criterion};
use cantor_core::{Hash32, MerkleProof};
use cantor_merkle::MerkleDeltaTree;

fn bench_merkle_verification(c: &mut Criterion) {
    let deltas: Vec<Vec<u8>> = (0..1000)
        .map(|i| format!("delta_{}", i).into_bytes())
        .collect();
    let delta_refs: Vec<&[u8]> = deltas.iter().map(|d| d.as_slice()).collect();
    let tree = MerkleDeltaTree::build(&delta_refs);
    let proof = tree.generate_proof(500).unwrap();
    let root = tree.root();
    
    c.bench_function("merkle_verify", |b| {
        b.iter(|| MerkleDeltaTree::verify_proof(black_box(&proof), black_box(&root)));
    });
}

fn bench_batch_verification(c: &mut Criterion) {
    let deltas: Vec<Vec<u8>> = (0..100)
        .map(|i| format!("delta_{}", i).into_bytes())
        .collect();
    let delta_refs: Vec<&[u8]> = deltas.iter().map(|d| d.as_slice()).collect();
    let tree = MerkleDeltaTree::build(&delta_refs);
    let root = tree.root();
    
    let proofs: Vec<MerkleProof> = (0..100)
        .map(|i| tree.generate_proof(i).unwrap())
        .collect();
    
    c.bench_function("batch_merkle_verify_100", |b| {
        b.iter(|| {
            for proof in &proofs {
                MerkleDeltaTree::verify_proof(black_box(proof), black_box(&root));
            }
        });
    });
}

criterion_group!(benches, bench_merkle_verification, bench_batch_verification);
criterion_main!(benches);

