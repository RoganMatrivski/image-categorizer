pub fn centroid(vectors: Vec<Vec<f32>>) -> eyre::Result<Vec<f32>> {
    if vectors.is_empty() {
        eyre::bail!("Cannot compute centroid of empty vector list");
    }
    let rows = vectors.len();
    let cols = vectors[0].len();

    let flat: Vec<f32> = vectors.into_iter().flatten().collect();
    let arr = ndarray::Array2::from_shape_vec((rows, cols), flat)
        .map_err(|e| eyre::eyre!("Failed to create ndarray: {e}"))?;

    Ok(arr.mean_axis(ndarray::Axis(0)).unwrap().to_vec())
}
