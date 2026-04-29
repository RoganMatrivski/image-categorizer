use colored::Colorize;
use ndarray::Array2;
use std::collections::HashMap;
use turso::Connection;

pub async fn load_vectors(
    conn: &Connection,
    table_name: &str,
    dim: usize,
) -> eyre::Result<(Vec<String>, Array2<f32>)> {
    let mut rows = conn
        .query(
            &format!("SELECT filename, embedding FROM {}", table_name),
            turso::params![],
        )
        .await?;

    let mut filenames: Vec<String> = Vec::new();
    let mut flat: Vec<f32> = Vec::new();
    let mut n_rows = 0usize;

    while let Some(row) = rows.next().await? {
        let filename: String = row.get(0)?;
        let embedding_blob: Vec<u8> = row.get(1)?;

        // Convert BLOB back to Vec<f32>
        let embedding: Vec<f32> = embedding_blob
            .chunks_exact(4)
            .map(|chunk| f32::from_ne_bytes(chunk.try_into().unwrap()))
            .collect();

        if embedding.len() != dim {
            eyre::bail!(
                "Embedding dimension mismatch: expected {}, got {}",
                dim,
                embedding.len()
            );
        }

        filenames.push(filename);
        flat.extend(embedding);
        n_rows += 1;
    }

    let data = Array2::from_shape_vec((n_rows, dim), flat)?;
    Ok((filenames, data))
}

pub fn print_clusters(
    filenames: &[String],
    data: &Array2<f32>,
    clusters: &HashMap<usize, Vec<usize>>,
    outliers: &[usize],
) {
    for (cluster_id, indices) in clusters {
        let names: Vec<&str> = indices.iter().map(|&i| filenames[i].as_str()).collect();

        let centroid: Vec<f32> = (0..data.ncols())
            .map(|col| indices.iter().map(|&i| data[[i, col]]).sum::<f32>() / indices.len() as f32)
            .collect();
        let medoid_idx = indices
            .iter()
            .min_by(|&&a, &&b| {
                let da: f32 = data
                    .row(a)
                    .iter()
                    .zip(&centroid)
                    .map(|(x, c)| (x - c).powi(2))
                    .sum();
                let db: f32 = data
                    .row(b)
                    .iter()
                    .zip(&centroid)
                    .map(|(x, c)| (x - c).powi(2))
                    .sum();
                da.partial_cmp(&db).unwrap()
            })
            .unwrap();

        println!(
            "{} {} {}",
            format!("Cluster {cluster_id}").bold().cyan(),
            format!("(representative: {})", filenames[*medoid_idx]).green(),
            format!("[{} images]", indices.len()).dimmed(),
        );
        for name in &names {
            println!("  {} {}", "·".dimmed(), name);
        }
    }

    println!();
    println!("{}", "Outliers:".bold().red());
    for &i in outliers {
        println!("  {} {}", "·".dimmed(), filenames[i].dimmed());
    }
}

pub fn cluster_score(
    embeddings: &ndarray::Array<f32, ndarray::Ix2>,
    clusters: &HashMap<usize, Vec<usize>>,
    outliers: &[usize],
    noise_ratio: f32,
) -> eyre::Result<f32> {
    use scirs2_metrics::clustering::{
        calinski_harabasz_score, davies_bouldin_score, density_based_cluster_validity, dunn_index,
        silhouette_score,
    };

    let n_samples = embeddings.nrows();

    // Start with all labels as 0, then fill in the real cluster ids.
    // clusters is { cluster_id -> [sample_index, ...] }, so we flip it:
    // for each cluster, stamp its id onto every sample index it owns.
    let mut labels = vec![0usize; n_samples];
    for (cluster_id, indices) in clusters {
        for &idx in indices {
            labels[idx] = *cluster_id;
        }
    }

    // Put outlier indices in a HashSet so we can check membership in O(1).
    let outlier_set: std::collections::HashSet<usize> = outliers.iter().cloned().collect();

    // Walk every sample index, skip the outliers, and keep the rest.
    // We collect two parallel lists: the actual data rows, and their labels.
    let (clean_rows, clean_labels): (Vec<_>, Vec<_>) = (0..n_samples)
        .filter(|i| !outlier_set.contains(i)) // drop outlier samples
        .map(|i| (embeddings.row(i).to_owned(), labels[i]))
        .unzip(); // split the (row, label) pairs into two separate Vecs

    // Stack the kept rows back into a 2D matrix that silhouette_score expects.
    let clean_x = ndarray::stack(
        ndarray::Axis(0),
        &clean_rows.iter().map(|r| r.view()).collect::<Vec<_>>(),
    )?;

    // Convert the label Vec into an ndarray Array1.
    let clean_labels = ndarray::Array1::from(clean_labels);

    let dbcv = density_based_cluster_validity(&clean_x, &clean_labels, None).unwrap_or(-1.0);
    let sil = silhouette_score(&clean_x, &clean_labels, "euclidean").unwrap_or(-1.0);
    let db = davies_bouldin_score(&clean_x, &clean_labels).unwrap_or(f32::MAX);
    // let ch = calinski_harabasz_score(&clean_x, &clean_labels).unwrap_or(0.0);
    // let dunn = dunn_index(&clean_x, &clean_labels).unwrap_or(0.0);

    // --- normalize each to [0, 1], all "higher = better" ---

    // already in [-1, 1] → shift to [0, 1]
    let dbcv_n = (dbcv + 1.0) / 2.0;
    let sil_n = (sil + 1.0) / 2.0;

    // lower is better → invert. clamp so outliers don't blow up
    let db_n = 1.0 / (1.0 + db);

    // // CH and Dunn are unbounded → normalize across your config sweep
    // let ch_n   = if (ch_max - ch_min).abs() < 1e-9 { 0.5 }
    //              else { (ch - ch_min) / (ch_max - ch_min) };
    // let dunn_n = if (dunn_max - dunn_min).abs() < 1e-9 { 0.5 }
    //              else { (dunn - dunn_min) / (dunn_max - dunn_min) };

    // noise penalty: 20% noise → 0.8x multiplier
    let noise_penalty = 1.0 - noise_ratio;

    // --- weighted sum ---
    // weights reflect density-based algorithm priorities
    // let score = 0.30 * dbcv_n   // primary: density-aware
    //           + 0.25 * sil_n    // shape quality
    //           + 0.20 * db_n     // separation
    //           + 0.15 * ch_n     // compactness
    //           + 0.10 * dunn_n;  // min/max ratio

    let score = 0.50 * dbcv_n   // density-aware (dominant)
              + 0.30 * sil_n  // shape quality
              + 0.20 * db_n; // separation

    Ok(score * noise_penalty) // penalize configs that throw away too much data

    // Ok(score)
}
