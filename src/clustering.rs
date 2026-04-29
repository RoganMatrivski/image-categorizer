use ndarray::Array2;
use colored::Colorize;
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
