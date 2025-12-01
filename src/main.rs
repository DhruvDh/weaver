use anyhow::Result;
use weaver::app::{RuntimeOptions, cli, run_app};

#[tokio::main(flavor = "multi_thread")]
async fn main() -> Result<()> {
    let args = cli().run();
    run_app(args, RuntimeOptions::default()).await
}
