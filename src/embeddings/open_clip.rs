use std::sync::OnceLock;
use std::time::Instant;

use arrow_array::{cast::AsArray, ArrayRef, Float32Array};
use arrow_schema::DataType;
use color_eyre::Report;
use eyre::ContextCompat;
use indicatif::{ProgressBar, ProgressStyle};
use open_clip_inference::VisionEmbedder;
use ort::execution_providers::{
    DirectMLExecutionProvider, ExecutionProvider, WebGPUExecutionProvider,
};

use super::common::Embedder;

#[derive(Debug)]
pub struct OpenClipInference {
    pub vis: VisionEmbedder,
}

fn log_device_info() {
    let directml = DirectMLExecutionProvider::default();
    let webgpu = WebGPUExecutionProvider::default();

    let dml_available = directml.is_available().unwrap_or(false);
    let wgpu_available = webgpu.is_available().unwrap_or(false);

    let device = match (dml_available, wgpu_available) {
        (true, _) => "GPU (DirectML)",
        (_, true) => "GPU (WebGPU)",
        _ => "CPU",
    };

    tracing::info!(
        device,
        directml = dml_available,
        webgpu = wgpu_available,
        "ORT execution provider status"
    );
}

impl OpenClipInference {
    fn log_device_once(&self) {
        static LOGGED: OnceLock<()> = OnceLock::new();
        LOGGED.get_or_init(|| log_device_info());
    }
}

impl Embedder for OpenClipInference {
    fn dim(&self) -> usize {
        self.vis.config.model_cfg.embed_dim as usize
    }

    #[tracing::instrument(skip(self, source))]
    fn embed_array(&self, source: ArrayRef) -> eyre::Result<Float32Array> {
        tracing::trace!(
            len = source.len(),
            nullable = source.is_nullable(),
            "embed_array called"
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

        self.log_device_once();

        // Decode images with progress bar
        tracing::debug!(n_images = source.len(), "Decoding images for embedding");

        let pb = crate::MPB.add(ProgressBar::new(source.len() as u64));
        pb.set_style(
            ProgressStyle::with_template(
                "[{elapsed_precise}] {bar:40.cyan/blue} {pos}/{len} images decoded ({eta} remaining)",
            )
            .unwrap()
            .progress_chars("█▉▊▋▌▍▎▏  "),
        );

        let decode_start = Instant::now();
        let inputs = source
            .as_binary::<i32>()
            .into_iter()
            .map(|b| {
                let bytes = b.wrap_err("we already asserted that the array is non-nullable")?;
                let img = image::load_from_memory(bytes).map_err(Report::from)?;
                pb.inc(1);
                Ok(img)
            })
            .collect::<eyre::Result<Vec<_>>>()?;
        pb.finish_and_clear();

        tracing::debug!(
            n_images = inputs.len(),
            elapsed_ms = decode_start.elapsed().as_millis(),
            "Image decoding complete"
        );

        // Run embedder with timing
        tracing::debug!(n_images = inputs.len(), "Starting vision embedder");
        let embed_start = Instant::now();

        let embeds = self.vis.embed_images(&inputs)?;

        let embed_elapsed = embed_start.elapsed();
        tracing::info!(
            n_images = inputs.len(),
            elapsed_ms = embed_elapsed.as_millis(),
            ms_per_image = embed_elapsed.as_millis() / inputs.len().max(1) as u128,
            "Vision embedder completed"
        );

        tracing::trace!("Flattening embeddings result");
        let flat = embeds
            .as_slice()
            .wrap_err("Embedded result is not contiguous")?;

        tracing::debug!(
            n_embeddings = inputs.len(),
            flat_len = flat.len(),
            "Embeddings ready"
        );

        Ok(Float32Array::from(flat.to_vec()))
    }
}
