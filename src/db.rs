use std::sync::Arc;
use arrow_schema::{DataType, Field, Schema};
use lancedb::{
    arrow::SendableRecordBatchStream,
    table::Table,
    Connection,
    embeddings::EmbeddingDefinition,
};

pub async fn get_or_create_table(
    db: &Connection,
    table_name: &str,
    embedding_col: &str,
    function_name: &str,
) -> eyre::Result<Table> {
    match db.open_table(table_name).execute().await {
        Ok(t) => {
            tracing::info!(table = table_name, "Opened existing table");
            Ok(t)
        }
        Err(_) => {
            tracing::info!(table = table_name, "Table not found, creating it");
            let schema = Arc::new(Schema::new(vec![
                Field::new("img", DataType::Binary, false),
                Field::new("filename", DataType::Utf8, false),
            ]));
            
            db.create_empty_table(table_name, schema)
                .add_embedding(EmbeddingDefinition::new("img", function_name, Some(embedding_col)))?
                .execute()
                .await
                .map_err(Into::into)
        }
    }
}

pub async fn add_batches(table: &Table, reader: SendableRecordBatchStream) -> eyre::Result<()> {
    table.add(reader).execute().await?;
    Ok(())
}
