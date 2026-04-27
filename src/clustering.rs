use arrow_array::{cast::AsArray, FixedSizeListArray, Float32Array, RecordBatch};
use lancedb::{table::Table, query::ExecutableQuery};
use ndarray::Array2;
use indicatif::ProgressBar;
use futures::TryStreamExt;
use colored::Colorize;
use std::collections::HashMap;

pub async fn load_vectors(table: &Table, dim: usize, pb: &ProgressBar) -> eyre::Result<(Vec<String>, Array2<f32>)> {
    let batches = table
        .query()
        .execute()
        .await?
        .try_collect::<Vec<RecordBatch>>()
        .await?;

    pb.set_length(batches.len() as u64);
    pb.set_message("Loading vectors");

    let mut filenames: Vec<String> = Vec::new();
    let mut flat: Vec<f32> = Vec::new();
    let mut n_rows = 0usize;

    for batch in &batches {
        filenames.extend(
            batch
                .column_by_name("filename")
                .unwrap()
                .as_string::<i32>()
                .iter()
                .flatten()
                .map(|s| s.to_string()),
        );
        let vecs = batch
            .column_by_name("vector")
            .unwrap()
            .as_any()
            .downcast_ref::<FixedSizeListArray>()
            .unwrap()
            .values()
            .as_any()
            .downcast_ref::<Float32Array>()
            .unwrap();
        flat.extend_from_slice(vecs.values());
        n_rows += batch.num_rows();
        pb.inc(1);
    }
    pb.finish_with_message("Vectors loaded");

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
