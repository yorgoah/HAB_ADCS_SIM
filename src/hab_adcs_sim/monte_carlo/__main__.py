"""Command line for Monte Carlo campaigns.

    python -m hab_adcs_sim.monte_carlo generate config/campaigns/smoke.toml
    python -m hab_adcs_sim.monte_carlo run monte_carlo/smoke --workers 3
    python -m hab_adcs_sim.monte_carlo status monte_carlo/smoke
"""

from pathlib import Path

import click

from .generate import DEFAULT_ROOT, generate_from_file
from .run import STAGES, campaign_status, run_campaign
from .spec import SpecError, load_spec


@click.group()
def cli() -> None:
    """Generate, run and inspect Monte Carlo campaigns."""


@cli.command()
@click.argument("spec_path", type=click.Path(exists=True, dir_okay=False, path_type=Path))
@click.option(
    "--root",
    default=DEFAULT_ROOT,
    type=click.Path(file_okay=False, path_type=Path),
    show_default=True,
    help="Directory campaigns are written under.",
)
@click.option("--force", is_flag=True, help="Overwrite a campaign generated from a different spec.")
def generate(spec_path: Path, root: Path, force: bool) -> None:
    """Write one directory per run from a campaign spec."""
    try:
        spec = load_spec(spec_path)
        campaign_dir, created = generate_from_file(spec_path, root=root, force=force)
    except SpecError as exc:
        raise click.ClickException(str(exc)) from None

    click.echo(
        f"{spec.name}: {len(spec.cases)} case(s) x {spec.runs_per_case} run(s) "
        f"= {spec.total_runs} run(s); {created} newly written."
    )
    click.echo(f"Campaign at {campaign_dir}")


@cli.command()
@click.argument("campaign_dir", type=click.Path(exists=True, file_okay=False, path_type=Path))
@click.option("--stage", type=click.Choice(STAGES), default="all", show_default=True)
@click.option(
    "--workers",
    default=3,
    show_default=True,
    help="Parallel processes, limited by RAM since each run keeps its log in memory. "
    "0 runs serially.",
)
@click.option("--case", "only_case", default=None, help="Restrict to a single case directory.")
@click.option("--rerun", is_flag=True, help="Redo work that is already complete.")
def run(campaign_dir: Path, stage: str, workers: int, only_case: str | None, rerun: bool) -> None:
    """Simulate and analyse the runs of a generated campaign."""
    totals = run_campaign(
        campaign_dir,
        stage=stage,
        workers=workers,
        only_case=only_case,
        rerun=rerun,
        echo=click.echo,
    )
    click.echo(
        f"done: {totals['done']}, failed: {totals['failed']}, skipped: {totals['skipped']}"
    )
    if totals["failed"]:
        click.echo("Failed runs left an error.txt in their run directory.")


@cli.command()
@click.argument("campaign_dir", type=click.Path(exists=True, file_okay=False, path_type=Path))
def status(campaign_dir: Path) -> None:
    """Show how far each case has got."""
    table = campaign_status(campaign_dir)
    if table.empty:
        click.echo("No generated runs found.")
        return
    click.echo(table.to_string(index=False))
    click.echo(
        f"totals: {int(table['generated'].sum())} generated, "
        f"{int(table['simulated'].sum())} simulated, "
        f"{int(table['analysed'].sum())} analysed, "
        f"{int(table['failed'].sum())} failed"
    )


if __name__ == "__main__":
    cli()
