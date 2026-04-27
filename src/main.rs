use std::{
    borrow::Cow,
    sync::{Arc, LazyLock},
};

use arrow_array::{cast::AsArray, Array, FixedSizeListArray, Float32Array, RecordBatch};
use arrow_data::ArrayData;
use arrow_schema::{DataType, Field, Schema};
use color_eyre::Report;
use eyre::ContextCompat;
use futures::{StreamExt, TryStreamExt};
use indicatif::{MultiProgress, ProgressBar, ProgressStyle};
use lancedb::{
    arrow::{SendableRecordBatchStream, SimpleRecordBatchStream},
    embeddings::{EmbeddingDefinition, EmbeddingFunction},
    query::{ExecutableQuery, QueryBase},
};
use ndarray::{concatenate, Array2, ArrayView2, Axis};
use open_clip_inference::VisionEmbedder;
use petal_clustering::{Fit, HDbscan};
use petal_neighbors::distance::Euclidean;

mod init;

#[cfg(target_env = "musl")]
#[global_allocator]
static GLOBAL: mimalloc::MiMalloc = mimalloc::MiMalloc;

// const MODEL: &str = "ViT-SO400M-16-SigLIP2-384-ONNX";
const MODEL: &str = "RuteNL/MobileCLIP2-S2-OpenCLIP-ONNX";

#[derive(Debug)]
struct OpenClipInference {
    vis: VisionEmbedder,
}

impl OpenClipInference {
    fn get_dim<T>(&self) -> Result<T, T::Error>
    where
        T: TryFrom<usize>,
    {
        T::try_from(self.vis.config.model_cfg.embed_dim)
    }

    pub fn compute_inner(&self, source: Arc<dyn Array>) -> eyre::Result<Float32Array> {
        tracing::trace!(
            len = source.len(),
            nullable = source.is_nullable(),
            "compute_inner called"
        );

        if source.is_nullable() {
            eyre::bail!("Expected non-nullable data type")
        }

        if !matches!(source.data_type(), DataType::Binary) {
            eyre::bail!("Expected Binary data type")
        };

        if source.len() == 0 {
            tracing::debug!(
                "Empty source array, returning empty embeddings (schema inference probe)"
            );
            return Ok(Float32Array::from(Vec::<f32>::new()));
        }

        tracing::debug!(n_images = source.len(), "Decoding images for embedding");

        let inputs = source
            .as_binary::<i32>()
            .into_iter()
            .map(|b| {
                let bytes = b.wrap_err("we already asserted that the array is non-nullable")?;
                image::load_from_memory(bytes).map_err(Report::from)
            })
            .collect::<eyre::Result<Vec<_>>>()?;

        tracing::trace!(
            n_images = inputs.len(),
            "Images decoded, running vision embedder"
        );

        let embeds = self.vis.embed_images(&inputs)?;

        tracing::trace!("Embeddings computed, flattening result");

        let flat = embeds
            .as_slice()
            .wrap_err("Embedded result is not contigous")?;

        tracing::debug!(
            n_embeddings = inputs.len(),
            flat_len = flat.len(),
            "Embeddings ready"
        );

        Ok(Float32Array::from(flat.to_vec()))
    }
}

impl EmbeddingFunction for OpenClipInference {
    fn name(&self) -> &str {
        "custom"
    }

    fn source_type(&self) -> lancedb::Result<std::borrow::Cow<'_, DataType>> {
        Ok(Cow::Owned(DataType::Binary))
    }

    fn dest_type(&self) -> lancedb::Result<std::borrow::Cow<'_, DataType>> {
        Ok(Cow::Owned(DataType::new_fixed_size_list(
            DataType::Float32,
            self.get_dim().expect("Failed to get dimension"),
            false,
        )))
    }

    fn compute_source_embeddings(&self, source: Arc<dyn Array>) -> lancedb::Result<Arc<dyn Array>> {
        tracing::debug!(n = source.len(), "Computing source embeddings");
        let len = source.len();
        let n_dims: i32 = self.get_dim().expect("Failed to get dimensions");
        let inner = self
            .compute_inner(source)
            .map_err(|e| lancedb::Error::Other {
                message: e.to_string(),
                source: Some(e.into()),
            })?;

        let fsl = DataType::new_fixed_size_list(DataType::Float32, n_dims, false);

        let arraydata = ArrayData::builder(fsl)
            .len(len)
            .add_child_data(inner.into_data())
            .build()?;

        tracing::trace!(
            len,
            n_dims,
            "Source embeddings built into FixedSizeListArray"
        );

        Ok(Arc::new(FixedSizeListArray::from(arraydata)))
    }

    fn compute_query_embeddings(&self, input: Arc<dyn Array>) -> lancedb::Result<Arc<dyn Array>> {
        tracing::debug!(n = input.len(), "Computing query embeddings");
        let arr = self
            .compute_inner(input)
            .map_err(|e| lancedb::Error::Other {
                message: e.to_string(),
                source: Some(e.into()),
            })?;

        tracing::trace!("Query embeddings ready");

        Ok(Arc::new(arr))
    }
}

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

    let embedder = OpenClipInference { vis };
    let dim = embedder.get_dim().expect("Failed to get dimension");
    db.embedding_registry()
        .register("custom", Arc::new(embedder))?;
    tracing::debug!("Registered custom embedding function");

    let schema = Arc::new(Schema::new(vec![
        Field::new("img", DataType::Binary, false),
        Field::new("filename", DataType::Utf8, false),
    ]));
    tracing::trace!(?schema, "Schema defined");

    tracing::debug!("Opening or creating 'result' table");
    let table = match db.open_table("result").execute().await {
        Ok(t) => {
            tracing::info!("Opened existing table 'result'");
            t
        }
        Err(e) => {
            tracing::info!(error = %e, "Table 'result' not found, creating it");
            db.create_empty_table("result", schema.clone())
                .add_embedding(EmbeddingDefinition::new("img", "custom", Some("vector")))?
                .execute()
                .await?
        }
    };

    tracing::debug!(
        n_images = args.images.len(),
        "Resolving image paths to filename pairs"
    );
    let path_name_pair = args
        .images
        .into_iter()
        .map(|x| eyre::Ok((get_filename(&x)?, x)).map(|(x, y)| (y, x)))
        .collect::<eyre::Result<Vec<_>>>()?;
    tracing::trace!(n = path_name_pair.len(), "Path/name pairs resolved");

    tracing::debug!(
        n = path_name_pair.len(),
        "Querying existing filenames in table"
    );
    let namelist = path_name_pair
        .iter()
        .map(|(_, n)| n.clone())
        .map(|s| format!("'{}'", s.replace("'", "''")))
        .collect::<Vec<_>>()
        .join(", ");

    let existing_query = table
        .query()
        .only_if(format!("filename IN ({namelist})"))
        .execute()
        .await?;

    let existing_batches = existing_query
        .collect::<Vec<_>>()
        .await
        .into_iter()
        .collect::<Result<Vec<_>, _>>()?;

    let existing = existing_batches
        .iter()
        .flat_map(|batch| {
            batch
                .column_by_name("filename")
                .expect("Can't fine filename column")
                .as_string::<i32>()
                .iter()
                .flatten()
                .map(|s| s.to_string())
        })
        .collect::<Vec<_>>();

    tracing::info!(n_existing = existing.len(), "Found already-indexed images");
    tracing::trace!(existing = ?existing, "Existing filenames");

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
        tracing::debug!("Building record batch stream");
        let stream = futures::stream::iter(not_exist)
                .chunks(16)
                .then(move |chunk| {
                    let schema = schema2.clone();
                    let pb = pb.clone();
                    async move {
                        tracing::trace!(chunk_size = chunk.len(), "Processing batch chunk");
                        let mut img_builder = arrow_array::builder::BinaryBuilder::new();
                        let mut name_builder = arrow_array::builder::StringBuilder::new();

                        for (path, name) in &chunk {
                            tracing::trace!(path = %path.display(), name, "Reading image file");
                            let bytes = tokio::fs::read(path)
                                .await
                                .map_err(|e| {
                                    tracing::error!(path = %path.display(), error = %e, "Failed to read image file");
                                    lancedb::Error::Other {
                                        message: e.to_string(),
                                        source: Default::default(),
                                    }
                                })?;
                            tracing::trace!(name, bytes = bytes.len(), "Image read successfully");
                            img_builder.append_value(&bytes);
                            name_builder.append_value(name);
                        }

                        pb.inc(chunk.len() as u64);

                        tracing::debug!(chunk_size = chunk.len(), "Batch chunk ready, building RecordBatch");
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

        tracing::info!("Streaming record batches into table");
        table.add(reader).execute().await?;
        tracing::info!("All images indexed successfully");
    } else {
        tracing::warn!("No new images to index, all images already exist in table");
    }

    let batches = table
        .query()
        .execute()
        .await?
        .try_collect::<Vec<_>>()
        .await?;

    // we know exactly how many batches, so a real bar works here
    let load_pb = MPB.add(ProgressBar::new(batches.len() as u64));
    load_pb.set_style(
        ProgressStyle::with_template("{msg} [{bar:40.cyan/blue}] {pos}/{len} batches ({elapsed})")
            .unwrap()
            .progress_chars("#>-"),
    );
    load_pb.set_message("Loading vectors");

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
        load_pb.inc(1);
    }
    load_pb.finish_with_message("Vectors loaded");

    let data = Array2::from_shape_vec((n_rows, dim), flat)?;

    // HDBSCAN has no progress callbacks, so a spinner is the best we can do
    let spin_pb = MPB.add(ProgressBar::new_spinner());
    spin_pb.set_style(ProgressStyle::with_template("{msg} {spinner:.green} ({elapsed})").unwrap());
    spin_pb.set_message("Clustering");
    spin_pb.enable_steady_tick(std::time::Duration::from_millis(100));

    // // This one's for Photos
    // let mut hdbscan = HDbscan {
    //     min_cluster_size: 10,   // avoid splitting one "food" category into 5 micro-clusters
    //     min_samples: 5,   // be lenient — natural scenes bleed into each other
    //     ..HDbscan::default()
    // };

    // This one's for screenshots
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

    use colored::Colorize;

    for (cluster_id, indices) in &clusters {
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
    for &i in &outliers {
        println!("  {} {}", "·".dimmed(), filenames[i].dimmed());
    }

    Ok(())
}
