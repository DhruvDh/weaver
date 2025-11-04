use anyhow::{Result, anyhow};
use kameo::prelude::*;
use tracing::debug;
use tracing_subscriber::EnvFilter;
use weaver::{
    file_reader::{FileReader, FileReaderQuery},
    llm_gateway::LLMGateway,
};

#[tokio::main(flavor = "multi_thread")]
async fn main() -> Result<()> {
    let filter = EnvFilter::try_from_default_env().unwrap_or_else(|_| EnvFilter::new("info"));

    tracing_subscriber::fmt()
        .with_timer(tracing_subscriber::fmt::time::ChronoLocal::new("%Y-%m-%d %H:%M:%S%.3f".into()))
        .with_env_filter(filter)
        .try_init()
        .map_err(|err| anyhow!("failed to initialize tracing subscriber: {err}"))?;

    debug!("FileReader demo starting");

    let gateway = LLMGateway::spawn(LLMGateway::from_env()?);

    let actor = match FileReader::from_env(".", gateway.clone()) {
        Ok(actor) => actor,
        Err(err) => {
            debug!(
                error = %err,
                "Skipping FileReader demo; set OPENAI_MODEL to enable LLM tools"
            );
            return Ok(());
        }
    };

    let tool_names = FileReader::tool_identifiers();

    debug!(
        workspace = %actor.workspace_root().display(),
        tools = ?tool_names,
        "Initialized FileReader with LLM tool bridge for the PreTeXt project"
    );

    let reader = FileReader::spawn(actor);

    let prompt = "Summarize the key goals of the UNCC CS2 PreTeXt project. Highlight any modules \
                  in the `uncc_cs2-pretext-project/source/` tree that look important. Please do \
                  make effective use of the `delegate_tasks` tools for all tasks, in parrallel if \
                  possible.";
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
