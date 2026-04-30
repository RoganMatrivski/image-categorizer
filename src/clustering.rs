use colored::Colorize;
use ndarray::{stack, Array2, Axis};
use petal_clustering::{Fit, HDbscan};
use rand::RngExt;
use std::collections::HashMap;
use turso::Connection;
use tracing::{info, instrument, warn};

pub async fn load_vectors(
    conn: &Connection,
    table_name: &str,
    dim: usize,
) -> eyre::Result<(Vec<String>, Array2<f32>)> {
// ... (rest of file)

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
    embeddings: &ndarray::Array2<f32>,
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

pub fn lhs_subsample(x: &Array2<f32>, ratio: f32) -> Array2<f32> {
    use rand::seq::SliceRandom;

    let mut rng = rand::rng();
    let n_rows = x.nrows();
    let n_features = x.ncols();
    let n_samples = ((n_rows as f32) * ratio).ceil() as usize;

    // Step 1: one random permutation of bucket indices per feature
    // this guarantees each bucket is used exactly once per axis
    let perms: Vec<Vec<usize>> = (0..n_features)
        .map(|_| {
            let mut perm: Vec<usize> = (0..n_samples).collect();
            perm.shuffle(&mut rng);
            perm
        })
        .collect();

    // Step 2: compute min/max per feature for scaling
    let mins: Vec<f32> = (0..n_features)
        .map(|j| x.column(j).fold(f32::MAX, |acc, &xi| acc.min(xi)))
        .collect();
    let maxs: Vec<f32> = (0..n_features)
        .map(|j| x.column(j).fold(f32::MIN, |acc, &xi| acc.max(xi)))
        .collect();

    // Step 3: generate synthetic LHS coordinates
    // (bucket + random offset) / n_samples  →  scaled to [min, max]
    let lhs_points: Vec<Vec<f32>> = (0..n_samples)
        .map(|i| {
            (0..n_features)
                .map(|j| {
                    let bucket = perms[j][i];
                    let offset = rng.random::<f32>();
                    let unit = (bucket as f32 + offset) / n_samples as f32;
                    mins[j] + unit * (maxs[j] - mins[j]) // scale to actual data range
                })
                .collect()
        })
        .collect();

    // Step 4: nearest real row to each LHS point
    let mut indices: Vec<usize> = lhs_points
        .iter()
        .map(|point| {
            (0..n_rows)
                .min_by(|&a, &b| {
                    let da: f32 = x
                        .row(a)
                        .iter()
                        .zip(point)
                        .map(|(xi, pi)| (xi - pi).powi(2))
                        .sum();
                    let db: f32 = x
                        .row(b)
                        .iter()
                        .zip(point)
                        .map(|(xi, pi)| (xi - pi).powi(2))
                        .sum();
                    da.partial_cmp(&db).unwrap()
                })
                .unwrap()
        })
        .collect();

    // Step 5: deduplicate
    indices.sort_unstable();
    indices.dedup();

    // Extract from full array
    let rows: Vec<_> = indices.iter().map(|&i| x.row(i)).collect();
    let x_sampled = ndarray::stack(Axis(0), &rows).unwrap();

    x_sampled
}

pub fn cluster(embeddings: &ndarray::Array2<f32>) -> eyre::Result<()> {
    Ok(())
}

fn reservoir_sample(n_rows: usize, k: usize) -> Vec<usize> {
    let mut rng = rand::rng();
    let mut reservoir: Vec<usize> = (0..k).collect();

    for i in k..n_rows {
        let j = rng.random_range(0..=i);
        if j < k {
            reservoir[j] = i;
        }
    }
    reservoir
}

fn optimize_hdbscan(x_sampled: &Array2<f32>) -> HDbscan<f32, petal_neighbors::distance::Euclidean> {
    // Step 1: define candidate grid
    let min_samples_opts = [3, 5, 10, 20, 50];
    let min_cluster_opts = [3, 5, 10, 20, 50];

    let mut candidates: Vec<(usize, usize)> = min_samples_opts
        .iter()
        .flat_map(|&ms| min_cluster_opts.iter().map(move |&mc| (ms, mc)))
        .collect(); // 25 combos

    // Step 2: successive halving — score on increasing data fractions
    // each round: keep top half, double the data
    let rounds = [
        (0.25, candidates.len()),
        (0.50, candidates.len() / 2),
        (1.00, candidates.len() / 4),
    ];

    for (ratio, keep) in rounds {
        // subsample x_sampled further for cheap early rounds
        let sub_indices = reservoir_sample(
            x_sampled.nrows(),
            ((x_sampled.nrows() as f64) * ratio) as usize,
        );
        let x_round = ndarray::stack(
            Axis(0),
            &sub_indices
                .iter()
                .map(|&i| x_sampled.row(i))
                .collect::<Vec<_>>(),
        )
        .unwrap();

        let mut scored: Vec<((usize, usize), f32)> = candidates
            .iter()
            .map(|&(ms, mc)| {
                let score = score_hdbscan(&x_round, ms, mc);
                ((ms, mc), score)
            })
            .collect();

        // keep top half by score
        scored.sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap());
        candidates = scored[..keep].iter().map(|s| s.0).collect();
    }

    HDbscan {
        min_samples: candidates[0].0,
        min_cluster_size: candidates[0].1,
        ..Default::default()
    } // winner
}

fn score_hdbscan(x: &Array2<f32>, min_samples: usize, min_cluster_size: usize) -> f32 {
    let mut hdbscan = HDbscan {
        min_samples,
        min_cluster_size,
        ..Default::default()
    };

    let (clusters, outliers, _) = hdbscan.fit(&x, None);

    // need at least 2 clusters to score meaningfully
    if clusters.len() < 2 {
        return f32::NEG_INFINITY;
    }

    let mut labels = vec![-1i32; x.nrows()];
    for (id, members) in &clusters {
        for &i in members {
            labels[i] = *id as i32;
        }
    }

    let noise_ratio = outliers.len() as f32 / x.nrows() as f32;

    // too much noise = bad params
    if noise_ratio > 0.5 {
        return f32::NEG_INFINITY;
    }

    // let clean: Vec<usize> = (0..x.nrows()).filter(|&i| labels[i] != -1).collect();
    // let x_clean = stack(
    //     Axis(0),
    //     &clean.iter().map(|&i| x.row(i)).collect::<Vec<_>>(),
    // )
    // .unwrap();
    // let labels_clean: Vec<i32> = clean.iter().map(|&i| labels[i]).collect();

    cluster_score(&x, &clusters, &outliers, noise_ratio).unwrap_or(f32::NEG_INFINITY)
}

pub fn optimize_pca_dim(embeddings: &ndarray::Array2<f32>) -> usize {
    // coarse candidates — log-spaced
    let mut candidates: Vec<usize> = vec![2, 5, 10, 20, 50, 100, 200, 500];
    candidates.retain(|&d| d < embeddings.ncols());

    let rounds: &[(f32, usize)] = &[
        (0.2, candidates.len()),         // all candidates, 20% data
        (0.5, candidates.len() / 2),     // top half, 50% data
        (1.0, candidates.len() / 4 + 1), // top quarter, full data
    ];

    for &(ratio, keep) in rounds {
        let indices = lhs_subsample(embeddings, ratio);
        let rows: Vec<_> = indices
            .iter()
            .map(|&i| embeddings.row(i as usize))
            .collect();
        let x_sub = stack(Axis(0), &rows).unwrap();

        let mut scored: Vec<(usize, f32)> = candidates
            .iter()
            .map(|&dim| {
                let x_pca = petal_decomposition::PcaBuilder::new(dim)
                    .build()
                    .fit_transform(&x_sub)
                    .unwrap();
                // let x_pca = pca_reduce(&x_sub, dim); // your pca fn here
                let (clusters, outliers, _) = HDbscan {
                    min_samples: 5,
                    min_cluster_size: 5,
                    ..HDbscan::default()
                }
                .fit(&x_pca, None);
                let noise_ratio = outliers.len() as f32 / x_pca.nrows() as f32;
                let score = cluster_score(&x_pca, &clusters, &outliers, noise_ratio)
                    .unwrap_or(f32::NEG_INFINITY);
                (dim, score)
            })
            .collect();

        scored.sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap());
        candidates = scored[..keep].iter().map(|s| s.0).collect();
    }

    candidates[0]
}
