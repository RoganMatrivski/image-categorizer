use std::{borrow::Cow, sync::Arc};
use arrow_array::{builder::Float32Builder, cast::AsArray, Array, ArrayRef, Float32Array, FixedSizeListArray};
use arrow_data::ArrayData;
use arrow_schema::DataType;
use eyre::{Context, ContextCompat};
use lancedb::embeddings::EmbeddingFunction;
use super::common::centroid;

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
