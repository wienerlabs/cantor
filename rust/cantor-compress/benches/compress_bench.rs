use criterion::{black_box, criterion_group, criterion_main, Criterion, BenchmarkId};
use cantor_compress::{DeltaEncoder, CompressionMethod};

fn bench_compression(c: &mut Criterion) {
    let mut group = c.benchmark_group("compression");
    
    // Sparse delta (many zeros)
    let sparse: Vec<f32> = (0..4096)
        .map(|i| if i % 10 == 0 { 0.1 } else { 0.0 })
        .collect();
    
    // Dense delta
    let dense: Vec<f32> = (0..4096)
        .map(|i| (i as f32 * 0.001).sin())
        .collect();
    
    for method in [CompressionMethod::Lz4, CompressionMethod::Varint, CompressionMethod::RunLength] {
        let encoder = DeltaEncoder::new(method);
        
        group.bench_with_input(
            BenchmarkId::new(format!("{:?}_sparse", method), 4096),
            &sparse,
            |b, data| {
                b.iter(|| encoder.encode(black_box(data)));
            },
        );
        
        group.bench_with_input(
            BenchmarkId::new(format!("{:?}_dense", method), 4096),
            &dense,
            |b, data| {
                b.iter(|| encoder.encode(black_box(data)));
            },
        );
    }
    
    group.finish();
}

fn bench_decompression(c: &mut Criterion) {
    let encoder = DeltaEncoder::new(CompressionMethod::Lz4);
    let delta: Vec<f32> = (0..4096).map(|i| (i as f32 * 0.001).sin()).collect();
    let compressed = encoder.encode(&delta).unwrap();
    
    c.bench_function("lz4_decompress", |b| {
        b.iter(|| encoder.decode(black_box(&compressed)));
    });
}

criterion_group!(benches, bench_compression, bench_decompression);
criterion_main!(benches);

