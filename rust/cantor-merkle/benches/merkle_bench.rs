use criterion::{black_box, criterion_group, criterion_main, Criterion, BenchmarkId};
use cantor_merkle::MerkleDeltaTree;

fn bench_tree_build(c: &mut Criterion) {
    let mut group = c.benchmark_group("merkle_build");
    
    for size in [100, 1000, 10000].iter() {
        let deltas: Vec<Vec<u8>> = (0..*size)
            .map(|i| format!("delta_{}", i).into_bytes())
            .collect();
        let delta_refs: Vec<&[u8]> = deltas.iter().map(|d| d.as_slice()).collect();
        
        group.bench_with_input(
            BenchmarkId::from_parameter(size),
            &delta_refs,
            |b, deltas| {
                b.iter(|| MerkleDeltaTree::build(black_box(deltas)));
            },
        );
    }
    
    group.finish();
}

fn bench_proof_generation(c: &mut Criterion) {
    let deltas: Vec<Vec<u8>> = (0..1000)
        .map(|i| format!("delta_{}", i).into_bytes())
        .collect();
    let delta_refs: Vec<&[u8]> = deltas.iter().map(|d| d.as_slice()).collect();
    let tree = MerkleDeltaTree::build(&delta_refs);
    
    c.bench_function("proof_generation", |b| {
        b.iter(|| tree.generate_proof(black_box(500)));
    });
}

fn bench_proof_verification(c: &mut Criterion) {
    let deltas: Vec<Vec<u8>> = (0..1000)
        .map(|i| format!("delta_{}", i).into_bytes())
        .collect();
    let delta_refs: Vec<&[u8]> = deltas.iter().map(|d| d.as_slice()).collect();
    let tree = MerkleDeltaTree::build(&delta_refs);
    let proof = tree.generate_proof(500).unwrap();
    let root = tree.root();
    
    c.bench_function("proof_verification", |b| {
        b.iter(|| MerkleDeltaTree::verify_proof(black_box(&proof), black_box(&root)));
    });
}

criterion_group!(benches, bench_tree_build, bench_proof_generation, bench_proof_verification);
criterion_main!(benches);

