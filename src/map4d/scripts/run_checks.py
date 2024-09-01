"""Run checks in the github actions file locally.

Adapted from https://github.com/nerfstudio-project/nerfstudio/blob/main/nerfstudio/scripts/github/run_actions.py
"""
import subprocess
import sys

import tyro
import yaml
from nerfstudio.utils.rich_utils import CONSOLE
from rich.style import Style

LOCAL_TESTS = ["isort", "black", "lint", "test"]


def run_command(command: str, continue_on_fail: bool = False) -> bool:
    """Run a command kill actions if it fails

    Args:
        command: command to run
        continue_on_fail: whether to continue running commands if the current one fails.
    """
    ret_code = subprocess.call(command, shell=True)
    if ret_code != 0:
        CONSOLE.print(f"[bold red]Error: `{command}` failed.")
        if not continue_on_fail:
            sys.exit(1)
    return ret_code == 0


def run_github_actions_file(filename: str, continue_on_fail: bool = False):
    """Run a github actions file locally.

    Args:
        filename: Which yml github actions file to run.
        continue_on_fail: Whether or not to continue running actions commands if the current one fails
    """
    with open(filename, "rb") as f:
        my_dict = yaml.safe_load(f)

    steps = my_dict["jobs"]["core_checks"]["steps"]
    success = True
    for step in steps:
        if "name" in step and step["name"] in LOCAL_TESTS:
            curr_command = step["run"].replace("\n", ";").replace("\\", "")

            curr_command = curr_command.replace(" --check", "")
            if "ruff check" in curr_command:
                curr_command = curr_command.replace(" --diff", "")
                curr_command = curr_command.replace(" --output-format=github", "")

            CONSOLE.line()
            CONSOLE.rule(f"[bold green]Running: {curr_command}")
            success = success and run_command(curr_command, continue_on_fail=continue_on_fail)
        else:
            skip_name = step["name"] if "name" in step else step["uses"]
            CONSOLE.print(f"Skipping {skip_name}")

    CONSOLE.line()
    if success:
        CONSOLE.line()
        CONSOLE.rule(characters="=")
        CONSOLE.print("[bold green]:TADA: :TADA: :TADA: ALL CHECKS PASSED :TADA: :TADA: :TADA:", justify="center")
        CONSOLE.rule(characters="=")
    else:
        CONSOLE.line()
        CONSOLE.rule(characters="=", style=Style(color="red"))
        CONSOLE.print("[bold red]:skull: :skull: :skull: ERRORS FOUND :skull: :skull: :skull:", justify="center")
        CONSOLE.rule(characters="=", style=Style(color="red"))


def run_code_checks(continue_on_fail: bool = False):
    """Run a github actions file locally."""
    # core code checks
    run_github_actions_file(filename=".github/workflows/ci.yml", continue_on_fail=continue_on_fail)


def entrypoint():
    """Entrypoint for use with pyproject scripts."""
    tyro.extras.set_accent_color("bright_yellow")
    tyro.cli(run_code_checks)


if __name__ == "__main__":
    entrypoint()
