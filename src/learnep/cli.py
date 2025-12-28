import typer
import os
import shutil
import json
from .orchestrator import LearnEPOrchestrator

app = typer.Typer(help="LearnEP: NEP Active Learning Framework")


@app.command()
def run(
    config: str = typer.Argument(..., help="Path to configuration YAML"),
    restart_from: int = typer.Option(
        None, help="Restart form a specific iteration (clears subsequent progress)"
    ),
):
    """
    Start the active learning workflow.
    """
    if not os.path.exists(config):
        typer.echo(f"Error: Config file {config} not found.")
        raise typer.Exit(code=1)

    orchestrator = LearnEPOrchestrator(config)
    orchestrator.run(restart_from)


@app.command()
def status(work_dir: str = typer.Option("./work", help="Working directory")):
    """
    Check the status of the active learning process.
    """
    status_file = os.path.join(work_dir, "status.json")
    if os.path.exists(status_file):
        with open(status_file, "r") as f:
            data = json.load(f)
        last = data.get("last_completed", -1)
        print(f"Work Dir: {work_dir}")
        print(f"Status: Iteration {last} Completed. (Next: {last + 1})")
    else:
        print(f"Work Dir: {work_dir} - No status file found (Not started or clean).")


@app.command()
def init(output: str = typer.Option("config.yaml", help="Output filename")):
    """
    Generate a sample configuration file.
    """
    # Assuming config_example.yaml is in package or we write a default one?
    # For now, we can just write a minimal template or copy the one we made.
    # Since we are running from source root, we can check 'learnep/config_example.yaml'

    src = os.path.join(os.path.dirname(__file__), "config_example.yaml")
    if os.path.exists(src):
        shutil.copy2(src, output)
        print(f"Generated {output}")
    else:
        print("Error: Template not found.")


if __name__ == "__main__":
    app()
