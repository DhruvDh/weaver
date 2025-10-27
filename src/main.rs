use anyhow::Result;
use kameo::prelude::*;
use tracing::debug;
use tracing_subscriber::EnvFilter;
use weaver::file_reader::{FileReader, FileReaderQuery};

#[tokio::main(flavor = "multi_thread")]
async fn main() -> Result<()> {
    let filter = EnvFilter::try_from_default_env().unwrap_or_else(|_| EnvFilter::new("info"));

    tracing_subscriber::fmt()
        .with_timer(tracing_subscriber::fmt::time::ChronoLocal::new("%Y-%m-%d %H:%M:%S%.3f".into()))
        .with_env_filter(filter)
        .init();

    debug!("FileReader demo starting");

    let actor = match FileReader::from_env(".") {
        Ok(actor) => actor,
        Err(err) => {
            debug!(
                error = %err,
                "Skipping FileReader demo; set OPENAI_MODEL to enable LLM tools"
            );
            return Ok(());
        }
    };

    let tool_names: Vec<&'static str> = FileReader::tool_names()
        .iter()
        .map(|tool| tool.identifier())
        .collect();

    debug!(
        workspace = %actor.workspace_root().display(),
        tools = ?tool_names,
        "Initialized FileReader with LLM tool bridge for the PreTeXt project"
    );

    let reader = FileReader::spawn(actor);

    let prompt = "Summarize the key goals of the UNCC CS2 PreTeXt project. Highlight any modules \
                  in the `uncc_cs2-pretext-project/source/` tree that look important.";
    debug!(prompt, "Dispatching FileReaderQuery with LLM tool access");

    match reader
        .ask(FileReaderQuery {
            prompt: prompt.to_string(),
        })
        .await
    {
        Ok(content) => {
            println!("{content}");
        }
        Err(err) => {
            eprintln!("FileReader query failed: {err}");
        }
    }

    Ok(())
}
