use std::sync::{Arc, LazyLock};

use arrow_array::{cast::AsArray, RecordBatch};
use arrow_schema::{DataType, Schema};
use color_eyre::Report;
use eyre::ContextCompat;
use futures::{StreamExt, TryStreamExt};
use indicatif::{MultiProgress, ProgressBar, ProgressStyle};
use lancedb::{
    arrow::{SendableRecordBatchStream, SimpleRecordBatchStream},
    query::{ExecutableQuery, QueryBase},
};
use open_clip_inference::VisionEmbedder;
use petal_clustering::{Fit, HDbscan};

mod clustering;
mod db;
mod embeddings;
mod init;

use embeddings::OpenClipInference;

use crate::embeddings::LlamaCppInference;

#[cfg(target_env = "musl")]
#[global_allocator]
static GLOBAL: mimalloc::MiMalloc = mimalloc::MiMalloc;

const MODEL: &str = "RuteNL/MobileCLIP2-S2-OpenCLIP-ONNX";

fn get_filename(p: impl AsRef<std::path::Path> + std::fmt::Debug) -> eyre::Result<String> {
    Ok(p.as_ref()
        .file_name()
        .wrap_err_with(|| format!("Invalid path: {p:?}"))?
        .to_string_lossy()
        .to_string())
}

static MPB: LazyLock<MultiProgress> = LazyLock::new(|| MultiProgress::new());

#[tracing::instrument]
#[tokio::main]
async fn main() -> Result<(), Report> {
    let args = init::initialize()?;

    tracing::info!(model = MODEL, "Loading vision embedder");
    let vis = {
        use ort::ep::{DirectML, WebGPU};

        VisionEmbedder::from_hf(MODEL)
            .with_execution_providers(&[DirectML::default().build(), WebGPU::default().build()])
            .build()
            .await?
    };

    tracing::info!("Vision embedder loaded");

    tracing::debug!(db_path = %args.db_path, "Connecting to LanceDB");
    let db = lancedb::connect(&args.db_path).execute().await?;
    tracing::info!("Connected to LanceDB");

    let embedder = LlamaCppInference {
        base_url: url::Url::parse("https://llama-cpp.rgmtrv.my.id")?,
        client: reqwest::Client::new(),
        dim: 2048,
    };
    let dim = embedder.dim;

    // let embedder = OpenClipInference { vis };
    // let dim = embedder.get_dim::<i32>().expect("Failed to get dimension") as usize;

    db.embedding_registry()
        .register("custom", Arc::new(embedder))?;
    tracing::debug!("Registered custom embedding function");

    let table = db::get_or_create_table(&db, "result", "vector", "custom").await?;
    let schema = Arc::new(Schema::new(vec![
        arrow_schema::Field::new("img", DataType::Binary, false),
        arrow_schema::Field::new("filename", DataType::Utf8, false),
    ]));

    tracing::debug!(
        n_images = args.images.len(),
        "Resolving image paths to filename pairs"
    );
    let path_name_pair = args
        .images
        .into_iter()
        .map(|x| eyre::Ok((get_filename(&x)?, x)).map(|(x, y)| (y, x)))
        .collect::<eyre::Result<Vec<_>>>()?;

    let namelist = path_name_pair
        .iter()
        .map(|(_, n)| n.clone())
        .map(|s| format!("'{}'", s.replace("'", "''")))
        .collect::<Vec<_>>()
        .join(", ");

    let existing_query = table.query().only_if(format!("filename IN ({namelist})"));

    let existing_batches: Vec<RecordBatch> = existing_query
        .execute()
        .await?
        .try_collect::<Vec<RecordBatch>>()
        .await?;

    let existing: Vec<String> = existing_batches
        .iter()
        .flat_map(|batch| {
            batch
                .column_by_name("filename")
                .expect("Can't find filename column")
                .as_string::<i32>()
                .iter()
                .flatten()
                .map(|s: &str| s.to_string())
        })
        .collect::<Vec<_>>();

    tracing::info!(n_existing = existing.len(), "Found already-indexed images");

    let not_exist = path_name_pair
        .into_iter()
        .filter(|(_, n)| !existing.contains(&n))
        .collect::<Vec<_>>();

    tracing::info!(n_new = not_exist.len(), "Images to index");

    if !not_exist.is_empty() {
        let pb = MPB.add(ProgressBar::new(not_exist.len() as u64));
        pb.set_style(
                indicatif::ProgressStyle::with_template(
                    "{msg} {spinner:.green} [{elapsed_precise}] [{bar:40.cyan/blue}] {pos}/{len} ({eta})",
                )
                .unwrap()
                .progress_chars("#>-"),
            );
        pb.set_message("Processing images");

        let schema2 = Arc::clone(&schema);
        let stream = futures::stream::iter(not_exist)
            .chunks(16)
            .then(move |chunk| {
                let schema = schema2.clone();
                let pb = pb.clone();
                async move {
                    let mut img_builder = arrow_array::builder::BinaryBuilder::new();
                    let mut name_builder = arrow_array::builder::StringBuilder::new();

                    for (path, name) in &chunk {
                        let bytes =
                            tokio::fs::read(path)
                                .await
                                .map_err(|e| lancedb::Error::Other {
                                    message: e.to_string(),
                                    source: Default::default(),
                                })?;
                        img_builder.append_value(&bytes);
                        name_builder.append_value(name);
                    }

                    pb.inc(chunk.len() as u64);

                    Ok::<_, lancedb::Error>(RecordBatch::try_new(
                        schema.clone(),
                        vec![
                            Arc::new(img_builder.finish()),
                            Arc::new(name_builder.finish()),
                        ],
                    )?)
                }
            });

        let reader: SendableRecordBatchStream =
            Box::pin(SimpleRecordBatchStream::new(stream, schema.clone()));

        db::add_batches(&table, reader).await?;
        tracing::info!("All images indexed successfully");
    } else {
        tracing::warn!("No new images to index, all images already exist in table");
    }

    let load_pb = MPB.add(ProgressBar::new(0));
    load_pb.set_style(
        ProgressStyle::with_template("{msg} [{bar:40.cyan/blue}] {pos}/{len} batches ({elapsed})")
            .unwrap()
            .progress_chars("#>-"),
    );

    let (filenames, data) = clustering::load_vectors(&table, dim, &load_pb).await?;

    let spin_pb = MPB.add(ProgressBar::new_spinner());
    spin_pb.set_style(ProgressStyle::with_template("{msg} {spinner:.green} ({elapsed})").unwrap());
    spin_pb.set_message("Clustering");
    spin_pb.enable_steady_tick(std::time::Duration::from_millis(100));

    let mut hdbscan = HDbscan {
        min_samples: 3,
        min_cluster_size: 3,
        ..HDbscan::default()
    };
    let (clusters, outliers, _scores) = hdbscan.fit(&data, None);

    spin_pb.finish_with_message(format!(
        "Clustering done — {} clusters, {} outliers",
        clusters.len(),
        outliers.len()
    ));

    clustering::print_clusters(&filenames, &data, &clusters, &outliers);

    Ok(())
}
