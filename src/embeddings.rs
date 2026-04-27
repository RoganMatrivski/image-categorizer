use std::{borrow::Cow, sync::Arc};

use arrow_array::{
    builder::Float32Builder, cast::AsArray, Array, ArrayRef, FixedSizeListArray, Float32Array,
};
use arrow_data::ArrayData;
use arrow_schema::DataType;
use color_eyre::Report;
use eyre::{Context, ContextCompat};
use lancedb::embeddings::EmbeddingFunction;
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

#[derive(serde::Deserialize)]
struct EmbedResult {
    pub index: usize,
    pub embedding: Vec<Vec<f32>>,
}

pub struct LlamaCppInference {
    pub base_url: url::Url,
    pub client: reqwest::Client,
    pub dim: usize,
}

impl LlamaCppInference {
    fn compute_inner(&self, source: ArrayRef) -> eyre::Result<Float32Array> {
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
                use base64::{engine::general_purpose, Engine as _};

                let bytes = b.wrap_err("we already asserted that the array is non-nullable")?;
                let b64 = general_purpose::STANDARD.encode(bytes);
                eyre::Ok(b64)
            })
            .collect::<eyre::Result<Vec<_>>>()?;

        tracing::trace!(
            n_images = inputs.len(),
            "Images encoded as base64, running vision embedder"
        );

        let payloads = inputs
            .into_iter()
            .map(|s| {
                serde_json::json!({
                    "content": "Image: [img-1]",
                    "image_data": [
                        {
                            "id": 1,
                            "data": s,
                        }
                    ]
                })
            })
            .collect::<Vec<_>>();

        let client = self.client.clone();
        let url = self
            .base_url
            .join("/embedding")
            .wrap_err("Failed to join base_url with '/embedding'")?;

        tokio::task::block_in_place(move || {
            tokio::runtime::Handle::current().block_on(async move {
                let mut vecres = vec![];
                for p in payloads {
                    let url = url.clone();
                    let res: Vec<EmbedResult> = client
                        .post(url)
                        .json(&p)
                        .send()
                        .await
                        .map_err(|e| eyre::eyre!("llama-server request failed: {e}"))?
                        .error_for_status()
                        .map_err(|e| eyre::eyre!("llama-server returned error: {e}"))?
                        .json()
                        .await
                        .map_err(|e| eyre::eyre!("failed to parse llama-server response: {e}"))?;

                    let embed_res = res.get(0).wrap_err("Failed to get embedding result")?;

                    vecres.push(centroid(embed_res.embedding.clone()));
                }

                let mut builder = Float32Builder::new();
                for res in vecres {
                    builder.append_slice(&res);
                }

                Ok(builder.finish())
            })
        })
    }
}

fn centroid(vectors: Vec<Vec<f32>>) -> Vec<f32> {
    let rows = vectors.len();
    let cols = vectors[0].len();

    let flat: Vec<f32> = vectors.into_iter().flatten().collect();
    let arr = ndarray::Array2::from_shape_vec((rows, cols), flat).unwrap();

    arr.mean_axis(ndarray::Axis(0)).unwrap().to_vec()
}
