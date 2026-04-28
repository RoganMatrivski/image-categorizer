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

    #[tracing::instrument(skip(self, source))]
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

    #[tracing::instrument(skip(self, input))]
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
    pub _index: Option<usize>,
    pub embedding: Vec<Vec<f32>>,
}

#[derive(Debug)]
pub struct LlamaCppInference {
    pub base_url: url::Url,
    pub client: reqwest::Client,
    pub dim: usize,
}

impl LlamaCppInference {
    #[tracing::instrument(skip(self, source))]
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

        tracing::debug!(n_images = source.len(), "Encoding images for embedding");

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
            "Images encoded as base64, preparing payloads"
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
                use futures::StreamExt;

                let mut stream = futures::stream::iter(payloads)
                    .map(|p| {
                        let client = client.clone();
                        let url = url.clone();
                        async move {
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
                                .map_err(|e| {
                                    eyre::eyre!("failed to parse llama-server response: {e}")
                                })?;

                            let embed_res =
                                res.get(0).wrap_err("Failed to get embedding result")?;
                            centroid(embed_res.embedding.clone())
                        }
                    })
                    .buffer_unordered(4);

                let mut vecres = vec![];
                while let Some(res) = stream.next().await {
                    vecres.push(res?);
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

fn centroid(vectors: Vec<Vec<f32>>) -> eyre::Result<Vec<f32>> {
    if vectors.is_empty() {
        eyre::bail!("Cannot compute centroid of empty vector list");
    }
    let rows = vectors.len();
    let cols = vectors[0].len();

    let flat: Vec<f32> = vectors.into_iter().flatten().collect();
    let arr = ndarray::Array2::from_shape_vec((rows, cols), flat)
        .map_err(|e| eyre::eyre!("Failed to create ndarray: {e}"))?;

    Ok(arr.mean_axis(ndarray::Axis(0)).unwrap().to_vec())
}

impl EmbeddingFunction for LlamaCppInference {
    fn name(&self) -> &str {
        "llamacpp"
    }

    fn source_type(&self) -> lancedb::Result<std::borrow::Cow<'_, DataType>> {
        Ok(Cow::Owned(DataType::Binary))
    }

    fn dest_type(&self) -> lancedb::Result<std::borrow::Cow<'_, DataType>> {
        Ok(Cow::Owned(DataType::new_fixed_size_list(
            DataType::Float32,
            self.dim as i32,
            false,
        )))
    }

    #[tracing::instrument(skip(self, source))]
    fn compute_source_embeddings(&self, source: Arc<dyn Array>) -> lancedb::Result<Arc<dyn Array>> {
        tracing::debug!(n = source.len(), "Computing source embeddings");
        let len = source.len();
        let n_dims: i32 = self.dim as i32;
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

    #[tracing::instrument(skip(self, input))]
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

#[derive(Debug)]
pub struct OllamaInference {
    pub base_url: url::Url,
    pub client: reqwest::Client,
    pub model: String,
    pub dim: usize,
}

#[derive(serde::Deserialize)]
struct OllamaEmbedResponse {
    pub embeddings: Vec<Vec<f32>>,
}

impl OllamaInference {
    #[tracing::instrument(skip(self, source))]
    fn compute_inner(&self, source: ArrayRef) -> eyre::Result<Float32Array> {
        if source.is_nullable() {
            eyre::bail!("Expected non-nullable data type")
        }
        if !matches!(source.data_type(), DataType::Binary) {
            eyre::bail!("Expected Binary data type")
        };
        if source.len() == 0 {
            return Ok(Float32Array::from(Vec::<f32>::new()));
        }

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

        let client = self.client.clone();
        let url = self
            .base_url
            .join("/api/embed")
            .wrap_err("Failed to join base_url with '/api/embed'")?;

        tokio::task::block_in_place(move || {
            tokio::runtime::Handle::current().block_on(async move {
                use futures::StreamExt;
                let mut stream = futures::stream::iter(inputs)
                    .map(|img_b64| {
                        let client = client.clone();
                        let url = url.clone();
                        let model = self.model.clone();
                        async move {
                            // Ollama supports multimodal embeddings if the model supports it.
                            // Note: Ollama's /api/embed might not directly support image input in all versions.
                            // Some versions might need /api/generate with a specific prompt.
                            let payload = serde_json::json!({
                                "model": model,
                                "input": "describe this image",
                                "images": [img_b64]
                            });

                            let res: OllamaEmbedResponse = client
                                .post(url)
                                .json(&payload)
                                .send()
                                .await
                                .map_err(|e| eyre::eyre!("ollama request failed: {e}"))?
                                .error_for_status()
                                .map_err(|e| eyre::eyre!("ollama returned error: {e}"))?
                                .json()
                                .await
                                .map_err(|e| eyre::eyre!("failed to parse ollama response: {e}"))?;

                            centroid(res.embeddings)
                        }
                    })
                    .buffer_unordered(4);

                let mut vecres = vec![];
                while let Some(res) = stream.next().await {
                    vecres.push(res?);
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

impl EmbeddingFunction for OllamaInference {
    fn name(&self) -> &str {
        "ollama"
    }

    fn source_type(&self) -> lancedb::Result<std::borrow::Cow<'_, DataType>> {
        Ok(Cow::Owned(DataType::Binary))
    }

    fn dest_type(&self) -> lancedb::Result<std::borrow::Cow<'_, DataType>> {
        Ok(Cow::Owned(DataType::new_fixed_size_list(
            DataType::Float32,
            self.dim as i32,
            false,
        )))
    }

    #[tracing::instrument(skip(self, source))]
    fn compute_source_embeddings(&self, source: Arc<dyn Array>) -> lancedb::Result<Arc<dyn Array>> {
        let len = source.len();
        let n_dims: i32 = self.dim as i32;
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

        Ok(Arc::new(FixedSizeListArray::from(arraydata)))
    }

    #[tracing::instrument(skip(self, input))]
    fn compute_query_embeddings(&self, input: Arc<dyn Array>) -> lancedb::Result<Arc<dyn Array>> {
        let arr = self
            .compute_inner(input)
            .map_err(|e| lancedb::Error::Other {
                message: e.to_string(),
                source: Some(e.into()),
            })?;
        Ok(Arc::new(arr))
    }
}
