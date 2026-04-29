mod common;
pub mod llama_cpp;
pub mod ollama;
pub mod open_clip;

pub use common::{Embedder, EmbedderExt, ToEmbedInput};
pub use llama_cpp::LlamaCppInference;
pub use ollama::OllamaInference;
pub use open_clip::OpenClipInference;
