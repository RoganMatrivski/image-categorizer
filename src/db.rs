use turso::{sync::Builder, sync::Database, Connection};

pub async fn init_table(
    db_connstr: &str,
    dims: u32,
    table_name: &str,
) -> eyre::Result<(Database, Connection)> {
    let db = Builder::new_remote(db_connstr)
        .with_remote_url(&std::env::var("TURSO_DATABASE_URL")?)
        .with_auth_token(&std::env::var("TURSO_AUTH_TOKEN")?)
        .build()
        .await?;

    let conn = db.connect().await?;

    db.pull().await?;

    conn.execute(
        r#"
            CREATE TABLE IF NOT EXISTS ?1 (
                id INTEGER PRIMARY KEY,
                filename TEXT UNIQUE,
                embedding F32_BLOB()
            );
        "#,
        turso::params![table_name, dims],
    )
    .await?;

    Ok((db, conn))
}
