use std::{ops::Deref, path::PathBuf, str::FromStr};

use clap::{Parser, ValueEnum};
use color_eyre::Report;

mod progbar_logwriter;

#[derive(Parser)]
#[command(author, version, about, long_about = None)]
pub struct Args {
    /// Verbosity log
    #[arg(short, long, action = clap::ArgAction::Count)]
    pub verbose: u8,

    /// PCA target dimension. Set to 0 to skip PCA.
    #[arg(short = 'p', long, default_value = "256")]
    pub pca_dim: usize,

    /// Number of images to process in each batch.
    #[arg(short = 'c', long, default_value = "8")]
    pub chunk: usize,

    /// Embedding mode
    #[arg(long, value_enum, default_value_t = EmbedMode::OpenClip)]
    pub embed_mode: EmbedMode,

    /// HF model ID (OpenClip) or model name (Ollama)
    #[arg(long, default_value = "RuteNL/MobileCLIP2-S2-OpenCLIP-ONNX")]
    pub model: String,

    /// Base URL for inference server
    #[arg(
        long,
        required_if_eq("embed_mode", "llama-cpp"),
        required_if_eq("embed_mode", "ollama")
    )]
    pub base_url: Option<url::Url>,

    pub images: ImagePaths,
}

#[derive(Copy, Clone, PartialEq, Eq, PartialOrd, Ord, ValueEnum, Debug)]
pub enum EmbedMode {
    OpenClip,
    LlamaCpp,
    Ollama,
}

#[derive(Clone)]
pub struct ImagePaths(Vec<PathBuf>);

impl Deref for ImagePaths {
    type Target = Vec<PathBuf>;
    fn deref(&self) -> &Self::Target {
        &self.0
    }
}

impl IntoIterator for ImagePaths {
    type Item = PathBuf;
    type IntoIter = std::vec::IntoIter<PathBuf>;

    fn into_iter(self) -> Self::IntoIter {
        self.0.into_iter()
    }
}

impl FromStr for ImagePaths {
    type Err = String;

    fn from_str(path: &str) -> Result<Self, Self::Err> {
        let p = PathBuf::from(path);

        if p.is_file() {
            Ok(ImagePaths(vec![p]))
        } else if p.is_dir() {
            let files = std::fs::read_dir(&p)
                .map_err(|e| e.to_string())?
                .filter_map(|entry| {
                    let entry = entry.ok()?;
                    let path = entry.path();
                    path.is_file().then_some(path)
                })
                .collect();
            Ok(ImagePaths(files))
        } else {
            Err(format!("{path:?} is not a valid file or directory"))
        }
    }
}

const VERBOSE_LEVELS: &[&str] = &["info", "debug", "trace"];

macro_rules! pkg_name {
    () => {
        env!("CARGO_PKG_NAME").replace('-', "_")
    };
}
pub fn initialize() -> Result<Args, Report> {
    dotenvy::dotenv()?;

    use tracing_error::ErrorLayer;
    use tracing_subscriber::prelude::*;
    use tracing_subscriber::{fmt, EnvFilter};

    color_eyre::install()?;

    let args = Args::parse();

    let crate_level = args
        .verbose
        .min(VERBOSE_LEVELS.len() as u8)
        .checked_sub(1)
        .map(|i| VERBOSE_LEVELS[i as usize])
        .unwrap_or("warn");

    // Try to build from RUST_LOG, or fall back to a base "warn"
    let env_filter = EnvFilter::try_from_default_env()
        .unwrap_or_else(|_| EnvFilter::new("warn"))
        .add_directive(format!("{}={}", pkg_name!(), crate_level).parse().unwrap());

    let mpb_writer = move || -> Box<dyn std::io::Write> {
        Box::new(progbar_logwriter::ProgressBarLogWriter::new(
            std::io::stderr(),
            crate::MPB.clone(),
        ))
    };

    let fmt_layer = fmt::layer()
        .with_writer(mpb_writer)
        .with_level(true)
        .with_thread_ids(args.verbose > 1)
        .with_thread_names(args.verbose > 2);

    tracing_subscriber::registry()
        .with(fmt_layer)
        .with(env_filter)
        .with(ErrorLayer::default())
        .init();

    Ok(args)
}
