"""Automatic conversion of python code to markdown files for the documentation."""

import argparse
import os
import pathlib
import shutil
from subprocess import DEVNULL, STDOUT, CalledProcessError, check_call

from tqdm import tqdm

from baybe.telemetry import VARNAME_TELEMETRY_ENABLED

parser = argparse.ArgumentParser()
parser.add_argument(
    "-t",
    "--target_dir",
    help="Destination directory in which the build will be saved (relative).\
    Note that building the documentation actually happens within the doc folder.\
    After building the documentation, it will be copied to this folder.\
    Default is a subfolder 'docs' placed in `build`.",
    default="./build/docs",
)
parser.add_argument(
    "-p",
    "--include_private",
    help="Include private methods in the documentation. Default is false.",
    action="store_true",
)
parser.add_argument(
    "--debug",
    help="Activate debugging mode by not suppressing the output of conversion.",
    action="store_true",
)
parser.add_argument(
    "-e",
    "--ignore_examples",
    help="Ignore the examples and do not include them into the documentation.",
    action="store_true",
)
parser.add_argument(
    "-w",
    "--include_warnings",
    help="Include warnings when processing the examples. The default is ignoring them.",
    action="store_true",
)


# Parse input arguments
args = parser.parse_args()
DESTINATION_DIR = args.target_dir
DEBUG = args.debug
INCLUDE_PRIVATE = args.include_private
IGNORE_EXAMPLES = args.ignore_examples
INCLUDE_WARNINGS = args.include_warnings

# We adjust the environment variable if we decide to ignore warnings
if not INCLUDE_WARNINGS:
    os.environ["PYTHONWARNINGS"] = "ignore"

# Disable telemtetry
os.environ[VARNAME_TELEMETRY_ENABLED] = "false"
# Directories where Sphinx will always put the build, sdk and autosummary data
build_dir = pathlib.Path("docs/build")
sdk_dir = pathlib.Path("docs/sdk")
autosummary_dir = pathlib.Path("docs/_autosummary")
destination_dir = pathlib.Path(DESTINATION_DIR)


def create_example_documentation(example_dest_dir: str, debug: bool):
    """Create the documentation version of the examples files.

    Note that this deletes the destination directory if it already exists.

    Args:
        example_dest_dir: The destination directory.
        debug: Flag indicating whether conversion should be run in debug mode with
            more information.
    """
    # Folder where the .md files created are stored
    examples_directory = pathlib.Path(example_dest_dir)

    # if the destination directory already exists it is deleted
    if examples_directory.is_dir():
        shutil.rmtree(examples_directory)

    # Copy the examples folder in the destination directory
    shutil.copytree("examples", examples_directory)

    # List all directories in the examples folder
    ex_directories = [d for d in examples_directory.iterdir() if d.is_dir()]

    # For the toctree of the top level example folder, we need to keep track of all
    # folders. We thus write the header here and populate it during the execution of the
    # examples
    ex_file = "# Examples\n\nThese examples show how to use BayBE.\n\n```{toctree}\n"

    # Iterate over the directories. Only print output in debug mode.
    for sub_directory in (pbar := tqdm(ex_directories, disable=not debug)):
        # Get the name of the current folder
        # Format it by replacing underscores and capitalizing the words
        folder_name = sub_directory.stem
        formatted = " ".join(word.capitalize() for word in folder_name.split("_"))

        # Attach the link to the folder to the top level toctree.
        ex_file += formatted + f"<{folder_name}/{folder_name}>\n"

        # We need to create a file for the inclusion of the folder
        subdir_toctree = f"# {folder_name}\n\n" + "```{toctree}\n"

        # Set description of progressbar
        pbar.set_description("Overall progress")

        # list all .py files in the subdirectory that need to be converted
        py_files = list(sub_directory.glob("**/*.py"))

        # Iterate through the individual example files
        for file in (inner_pbar := tqdm(py_files, leave=False, disable=not debug)):
            # Include the name of the file to the toctree
            # Format it by replacing underscores and capitalizing the words
            file_name = file.stem
            formatted = " ".join(word.capitalize() for word in file_name.split("_"))
            # Remove duplicate "constraints" for the files in the constraints folder.
            if "Constraints" in folder_name and "Constraints" in formatted:
                formatted = formatted.replace("Constraints", "")

            # Also format the Prodsum name to Product/Sum
            if "Prodsum" in formatted:
                formatted = formatted.replace("Prodsum", "Product/Sum")
            subdir_toctree += formatted + f"<{file_name}>\n"

            # Set description for progress bar
            inner_pbar.set_description(f"Progressing {folder_name}")

            # Create the Markdown file:

            # 1. Convert the file to jupyter notebook
            check_call(["p2j", file], stdout=DEVNULL, stderr=STDOUT)

            notebook_path = file.with_suffix(".ipynb")

            # 2. Execute the notebook and convert to markdown.
            convert_execute = [
                "jupyter",
                "nbconvert",
                "--execute",
                "--to",
                "notebook",
                "--inplace",
                notebook_path,
            ]
            to_markdown = ["jupyter", "nbconvert", "--to", "markdown", notebook_path]

            # Check whether the debug flag is being used.
            if debug:
                check_call(
                    convert_execute,
                )
                check_call(to_markdown)
            else:
                check_call(convert_execute, stdout=DEVNULL, stderr=STDOUT)
                check_call(
                    to_markdown,
                    stdout=DEVNULL,
                    stderr=STDOUT,
                )

            # CLEANUP
            # Remove all lines that try to include a png file
            markdown_path = file.with_suffix(".md")
            with open(markdown_path, "r", encoding="UTF-8") as markdown_file:
                lines = markdown_file.readlines()

            lines = [line for line in lines if "![png]" not in line]

            # Rewrite the file
            with open(markdown_path, "w", encoding="UTF-8") as markdown_file:
                markdown_file.writelines(lines)

        # Write last line of toctree file for this directory and write the file
        subdir_toctree += "```"
        with open(
            sub_directory / f"{sub_directory.name}.md", "w", encoding="UTF-8"
        ) as f:
            f.write(subdir_toctree)

    # Write last line of top level toctree file and write the file
    ex_file += "```"
    with open(
        examples_directory / f"{examples_directory.name}.md", "w", encoding="UTF-8"
    ) as f:
        f.write(ex_file)

    # Remove remaining files and subdirectories from the destination directory
    # Remove any not markdown files
    for file in examples_directory.glob("**/*"):
        if file.is_file() and file.suffix != ".md":
            file.unlink(file)

    # Remove any remaining empty subdirectories
    for subdirectory in examples_directory.glob("*/*"):
        if subdirectory.is_dir() and not any(subdirectory.iterdir()):
            subdirectory.rmdir()


# Collect all of the directories and delete them if they still exist.
directories = [sdk_dir, autosummary_dir, build_dir, destination_dir]

for directory in directories:
    if directory.is_dir():
        shutil.rmtree(directory)

# The call for checking external links.
link_call = [
    "sphinx-build",
    "-b",
    "linkcheck",
    "docs",
    build_dir,
    "-D",
    f"autodoc_default_options.private_members={INCLUDE_PRIVATE}",
]
# The actual call that will be made to build the documentation
building_call = [
    "sphinx-build",
    "-b",
    "html",
    "docs",
    build_dir,
    "-D",
    f"autodoc_default_options.private_members={INCLUDE_PRIVATE}",
    "-n",  # Being nitpicky
]

# Process examples if required.
if not IGNORE_EXAMPLES:
    create_example_documentation(example_dest_dir="docs/examples", debug=DEBUG)


try:
    check_call(
        link_call if DEBUG else link_call + ["-Q"],
    )
    # We only want to fail if we are not in debug mode
    check_call(
        building_call if DEBUG else building_call + ["-q", "-W"],
    )
except CalledProcessError:
    print(
        """One of the processes raised a critical error. Re-running with more output."""
    )
    check_call(link_call)
    check_call(building_call)

# Clean the other files
for directory in [sdk_dir, autosummary_dir]:
    if directory.is_dir():
        shutil.rmtree(directory)

documentation = pathlib.Path(build_dir)
shutil.move(documentation, destination_dir)

# If we decided to not ignore the examples, we delete the created markdown files
if not IGNORE_EXAMPLES:
    example_directory = pathlib.Path("docs/examples")
    if example_directory.is_dir():
        shutil.rmtree(example_directory)

# If we decided to ignore warnings, we now do no want to ignore them anymore
if not INCLUDE_WARNINGS:
    os.environ["PYTHONWARNING"] = "default"
