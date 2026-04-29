use arrow_array::{ArrayRef, BinaryArray, Float32Array};
use std::sync::Arc;

pub trait ToEmbedInput {
    fn to_embed_input(self) -> ArrayRef;
}

impl ToEmbedInput for ArrayRef {
    fn to_embed_input(self) -> ArrayRef {
        self
    }
}

impl ToEmbedInput for BinaryArray {
    fn to_embed_input(self) -> ArrayRef {
        Arc::new(self)
    }
}

impl ToEmbedInput for Vec<Vec<u8>> {
    fn to_embed_input(self) -> ArrayRef {
        Arc::new(BinaryArray::from_iter_values(self))
    }
}

impl ToEmbedInput for Vec<&[u8]> {
    fn to_embed_input(self) -> ArrayRef {
        Arc::new(BinaryArray::from(self))
    }
}

pub trait Embedder {
    fn embed_array(&self, source: ArrayRef) -> eyre::Result<Float32Array>;
    fn dim(&self) -> usize;
}

pub trait EmbedderExt: Embedder {
    fn embed<I: ToEmbedInput>(&self, input: I) -> eyre::Result<Float32Array> {
        self.embed_array(input.to_embed_input())
    }
}

impl<T: Embedder + ?Sized> EmbedderExt for T {}

pub fn centroid(vectors: Vec<Vec<f32>>) -> eyre::Result<Vec<f32>> {
    if vectors.is_empty() {
        eyre::bail!("Cannot compute centroid of empty vector list");
    }
    let rows = vectors.len();
    let cols = vectors[0].len();

    let flat: Vec<f32> = vectors.into_iter().flatten().collect();
    let arr = ndarray::Array2::from_shape_vec((rows, cols), flat)
        .map_err(|e| eyre::eyre!("Failed to create ndarray: {e}"))?;

    let mean = arr.mean_axis(ndarray::Axis(0)).unwrap();
    let norm = mean.mapv(|x| x.powi(2)).sum().sqrt();
    let normalized = if norm > 1e-10 {
        (&mean / norm).to_vec()
    } else {
        mean.to_vec()
    };

    Ok(normalized)
}
