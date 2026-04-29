use turso::{sync::Database, Connection, sync::Builder};

pub async fn init_table() -> eyre::Result<(Database, Connection)> {
    let db = Builder::new_remote("app.db")
        .with_remote_url(&std::env::var("TURSO_DATABASE_URL")?)
        .with_auth_token(&std::env::var("TURSO_AUTH_TOKEN")?)
        .build()
        .await?;

    let conn = db.connect().await?;

    db.pull().await?;

    conn.execute(
        r#"
        CREATE TABLE IF NOT EXISTS results (
            id INTEGER PRIMARY KEY,
            filename TEXT UNIQUE,
            embedding BLOB
        );
    "#,
        turso::params![],
    )
    .await?;

    Ok((db, conn))
}
