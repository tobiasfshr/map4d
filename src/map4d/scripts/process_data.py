from typing import Union

import tyro
from nerfstudio.utils.rich_utils import CONSOLE
from typing_extensions import Annotated

from map4d.scripts.datasets.argoverse2 import ProcessArgoverse2
from map4d.scripts.datasets.kitti import ProcessKITTI
from map4d.scripts.datasets.vkitti2 import ProcessVKITTI2
from map4d.scripts.datasets.waymo import ProcessWaymo

Commands = Union[
    Annotated[ProcessArgoverse2, tyro.conf.subcommand(name="av2")],
    Annotated[ProcessKITTI, tyro.conf.subcommand(name="kitti")],
    Annotated[ProcessVKITTI2, tyro.conf.subcommand(name="vkitti2")],
    Annotated[ProcessWaymo, tyro.conf.subcommand(name="waymo")],
]


def entrypoint():
    """Entrypoint for use with pyproject scripts."""
    tyro.extras.set_accent_color("bright_yellow")
    try:
        tyro.cli(Commands).main()
    except RuntimeError as e:
        CONSOLE.log("[bold red]" + e.args[0])


if __name__ == "__main__":
    entrypoint()
