use std::sync::Arc;
use arrow_array::{cast::AsArray, Array, Float32Array};
use arrow_schema::DataType;
use color_eyre::Report;
use eyre::ContextCompat;
use open_clip_inference::VisionEmbedder;

#[derive(Debug)]
pub struct OpenClipInference {
    pub vis: VisionEmbedder,
}

impl OpenClipInference {
    pub fn get_dim<T>(&self) -> Result<T, T::Error>
    where
        T: TryFrom<usize>,
    {
        T::try_from(self.vis.config.model_cfg.embed_dim)
    }

    #[tracing::instrument(skip(self, source))]
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
