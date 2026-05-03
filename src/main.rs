use std::sync::LazyLock;

use color_eyre::Report;
use eyre::{ContextCompat, WrapErr};
use indicatif::{MultiProgress, ProgressBar, ProgressStyle};
use open_clip_inference::VisionEmbedder;
use petal_clustering::{Fit, HDbscan};
use tokio_util::sync::CancellationToken;

mod clustering;
mod db;
mod embeddings;
mod init;
mod ndarray_compat;

use crate::embeddings::{
    Embedder, EmbedderExt, LlamaCppInference, OllamaInference, OpenClipInference,
};

#[cfg(target_env = "musl")]
#[global_allocator]
static GLOBAL: mimalloc::MiMalloc = mimalloc::MiMalloc;

fn get_table_name(mode: init::EmbedMode, model: &str) -> String {
    let mode_str = format!("{:?}", mode).to_lowercase();
    let model_sanitized = model
        .replace(|c: char| !c.is_alphanumeric(), "_")
        .to_lowercase();
    format!("results_{}_{}", mode_str, model_sanitized)
}

fn get_filename(p: impl AsRef<std::path::Path> + std::fmt::Debug) -> eyre::Result<String> {
    Ok(p.as_ref()
        .file_name()
        .wrap_err_with(|| format!("Invalid path: {p:?}"))?
        .to_string_lossy()
        .to_string())
}

static MPB: LazyLock<MultiProgress> = LazyLock::new(|| MultiProgress::new());

async fn download_hf_repo(
    model: impl ToString,
    cachedir: Option<std::path::PathBuf>,
) -> eyre::Result<std::path::PathBuf> {
    let model = model.to_string();

    let cache = match cachedir {
        Some(p) => hf_hub::Cache::new(p.into()),
        None => hf_hub::Cache::from_env(), // reads HF_HOME, falls back to ~/.cache/huggingface/hub
    };

    let normalized_model_path = std::path::Path::new(&model)
        .components()
        .map(|c| c.as_os_str().to_string_lossy())
        .collect::<Vec<_>>()
        .join("--");

    let target_dir = cache.path().join(normalized_model_path);

    tracing::debug!(?target_dir, "downloading file");

    let cacherepo = cache.model(model.clone());
    let hf_api = hf_hub::api::tokio::ApiBuilder::from_cache(cache)
        .high()
        .build()?;
    let repo = hf_api.model(model);
    let repo_info = repo.info().await?;
    let files = repo_info.siblings.into_iter().map(|x| x.rfilename);

    #[derive(Clone)]
    struct HfProgressBar(indicatif::ProgressBar);
    impl hf_hub::api::tokio::Progress for HfProgressBar {
        async fn init(&mut self, size: usize, filename: &str) {
            self.0 = MPB.add(
                ProgressBar::new(size as u64)
                    .with_style(
                        ProgressStyle::with_template(
                            "{spinner:.cyan} [{elapsed_precise}] [{bar:40.cyan/blue}] {bytes}/{total_bytes} ({bytes_per_sec}, {eta}) {msg}",
                        )
                        .expect("ProgressBar style template error")
                        .progress_chars("█▉▊▋▌▍▎▏ "),
                    )
                    .with_message(filename.to_string()),
            );
        }
        async fn update(&mut self, size: usize) {
            self.0.inc(size as u64);
        }
        async fn finish(&mut self) {
            self.0.finish_and_clear();
        }
    }

    for f in files {
        if cacherepo.get(&f).is_none() {
            repo.download_with_progress(&f, HfProgressBar(ProgressBar::new(0)))
                .await?;
        }
    }

    Ok(cacherepo.pointer_path(&repo_info.sha))
}

async fn get_embedder(args: &init::Args) -> eyre::Result<Box<dyn Embedder + Send + Sync>> {
    match args.embed_mode {
        init::EmbedMode::OpenClip => {
            tracing::info!(model = args.model, "Loading OpenClip vision embedder");

            let model_dir = download_hf_repo(args.model.clone(), None).await?; // TODO: Make this as args

            let vis = VisionEmbedder::from_local_dir(&model_dir)
                .with_execution_providers(&[
                    ort::ep::WebGPU::default().build(),
                    ort::ep::DirectML::default().build(),
                ])
                .build()?;
            Ok(Box::new(OpenClipInference { vis }) as Box<dyn Embedder + Send + Sync>)
        }
        init::EmbedMode::LlamaCpp => {
            let base_url = args
                .base_url
                .as_ref()
                .wrap_err("LlamaCpp requires --base-url")?
                .clone();
            Ok(Box::new(LlamaCppInference {
                base_url,
                client: reqwest::Client::new(),
                dim: args.dims.wrap_err("Dim args not defined")?, // TODO: Make configurable if needed
            }) as Box<dyn Embedder + Send + Sync>)
        }
        init::EmbedMode::Ollama => {
            let base_url = args
                .base_url
                .as_ref()
                .wrap_err("Ollama requires --base-url")?
                .clone();
            Ok(Box::new(OllamaInference {
                base_url,
                client: reqwest::Client::new(),
                model: args.model.clone(),
                dim: args.dims.wrap_err("Dim args not defined")?, // TODO: Make configurable if needed
            }) as Box<dyn Embedder + Send + Sync>)
        }
    }
}

fn is_image(path: &std::path::PathBuf) -> bool {
    match path.extension().and_then(|ext| ext.to_str()) {
        Some(ext) => {
            matches!(
                ext.to_ascii_lowercase().as_str(),
                "jpg" | "jpeg" | "png" | "gif" | "bmp" | "webp" | "tiff"
            )
        }
        None => false,
    }
}

#[tracing::instrument]
#[tokio::main]
async fn main() -> Result<(), Report> {
    let args = init::initialize()?;
    let table_name = get_table_name(args.embed_mode, &args.model);

    let embedder = get_embedder(&args).await?;
    let dim = embedder.dim();

    let cancel_token = CancellationToken::new();
    let cloned_token = cancel_token.clone();
    tokio::spawn(async move {
        #[cfg(unix)]
        {
            use tokio::signal::unix::{signal, SignalKind};
            let mut sigterm = signal(SignalKind::terminate()).unwrap();
            let mut sigint = signal(SignalKind::interrupt()).unwrap();

            tokio::select! {
                _ = sigterm.recv() => {
                    tracing::info!("Received SIGTERM, shutting down gracefully");
                }
                _ = sigint.recv() => {
                    tracing::info!("Received SIGINT, shutting down gracefully");
                }
            }
        }

        #[cfg(windows)]
        {
            tokio::signal::ctrl_c().await.unwrap();
            tracing::info!("Received Ctrl+C, shutting down gracefully");
        }

        cloned_token.cancel();
    });

    tracing::debug!("Connecting to Turso database");
    let (db, conn) = db::init_table(&args.get_db_str(), dim as u32, &table_name).await?;
    tracing::info!(table = table_name, "Connected to Turso");

    let images: Vec<_> = args.images.into_iter().filter(is_image).collect();

    tracing::debug!(
        n_images = images.len(),
        "Resolving image paths to filename pairs"
    );
    let path_name_pair = images
        .into_iter()
        .map(|x| eyre::Ok((get_filename(&x)?, x)).map(|(x, y)| (y, x)))
        .collect::<eyre::Result<Vec<_>>>()?;

    let namelist = path_name_pair
        .iter()
        .map(|(_, n)| n.clone())
        .map(|s| format!("'{}'", s.replace("'", "''")))
        .collect::<Vec<_>>()
        .join(", ");

    let mut existing_rows = conn
        .query(
            &format!("SELECT filename FROM {table_name} WHERE filename IN ({namelist})"),
            turso::params![],
        )
        .await?;
    let mut existing = Vec::new();
    while let Some(row) = existing_rows.next().await? {
        existing.push(row.get::<String>(0)?);
    }

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
        pb.enable_steady_tick(std::time::Duration::from_millis(100));

        let mut indexed_any = false;
        for chunk in not_exist.chunks(args.chunk) {
            if cancel_token.is_cancelled() {
                break;
            }

            let mut img_data = Vec::new();
            for (path, name) in chunk {
                let bytes = tokio::fs::read(path).await?;
                img_data.push((name.clone(), bytes));
            }

            let embeddings = embedder.embed(
                img_data
                    .iter()
                    .map(|(_, b)| b.as_slice())
                    .collect::<Vec<_>>(),
            )?;

            for (i, (_, name)) in chunk.iter().enumerate() {
                let start = i * dim;
                let end = (i + 1) * dim;
                let emb = &embeddings.values()[start..end];
                let emb_bytes: Vec<u8> = emb.iter().flat_map(|f| f.to_ne_bytes()).collect();

                conn.execute(
                    &format!(
                        "INSERT OR IGNORE INTO {table_name} (filename, embedding) VALUES (?, ?)"
                    ),
                    turso::params![name.clone(), emb_bytes],
                )
                .await?;
            }

            pb.inc(chunk.len() as u64);
            indexed_any = true;
        }

        if indexed_any {
            tracing::info!("Pushing changes to Turso...");
            db.push().await?;
        }

        if cancel_token.is_cancelled() {
            tracing::info!("Graceful shutdown: indexing stopped, saved partial progress");
            return Ok(());
        }
        tracing::info!("All images indexed successfully");
    } else {
        if cancel_token.is_cancelled() {
            return Ok(());
        }
        tracing::warn!("No new images to index, all images already exist in database");
    }

    let load_pb = MPB.add(ProgressBar::new(0));
    load_pb.set_style(
        ProgressStyle::with_template("{msg} [{bar:40.cyan/blue}] {pos}/{len} ({elapsed})")
            .unwrap()
            .progress_chars("#>-"),
    );

    let (filenames, data) = match clustering::load_vectors(&conn, &table_name, dim).await {
        Ok(res) => res,
        Err(e) if cancel_token.is_cancelled() => {
            return Ok(());
        }
        Err(e) => return Err(e),
    };

    tracing::info!(
        n_vectors = filenames.len(),
        "Vectors loaded successfully from database"
    );

    let spin_pb = MPB.add(ProgressBar::new_spinner());
    spin_pb.set_style(ProgressStyle::with_template("{msg} {spinner:.green} ({elapsed})").unwrap());
    spin_pb.enable_steady_tick(std::time::Duration::from_millis(100));

    let pca_dim = clustering::optimize_pca_dim(&data);

    todo!();

    let embedding = if args.pca_dim > 0 {
        spin_pb.set_message("Dimensionality reduction (PCA)");
        tracing::debug!(
            target_dim = args.pca_dim,
            "Starting PCA dimensionality reduction"
        );
        let mut pca = petal_decomposition::PcaBuilder::new(args.pca_dim).build();
        let emb = pca.fit_transform(&data)?;
        tracing::debug!("PCA reduction complete");
        emb
    } else {
        tracing::info!("Skipping PCA dimensionality reduction as requested");
        data.clone()
    };

    spin_pb.set_message("Clustering (HDBSCAN)");
    tracing::debug!("Starting HDBSCAN clustering");
    let mut hdbscan = HDbscan {
        min_samples: 3,
        min_cluster_size: 3,
        ..HDbscan::default()
    };
    let (clusters, outliers, outlier_scores) = hdbscan.fit(&embedding, None);
    let noise_ratio = outliers.len() as f32 / embedding.nrows() as f32;

    spin_pb.finish_with_message(format!(
        "Clustering done — {} clusters, {} outliers",
        clusters.len(),
        outliers.len()
    ));

    clustering::print_clusters(&filenames, &data, &clusters, &outliers);

    tracing::info!("Final sync with Turso...");
    db.push().await?;

    Ok(())
}
