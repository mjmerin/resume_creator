# RenderCV Resume Template

A reusable system for maintaining one career profile and generating multiple targeted resumes with [RenderCV](https://github.com/rendercv/rendercv). Profile facts and reusable achievement bullets live in one file; each variant selects and arranges the content needed for a particular role.

Generated RenderCV YAML, Typst, PNG, and PDF files are build artifacts and are not committed.

## Recommended Local AI Models

This project was built using a Macbook Pro M1 Max with 32 GB of unified memory. 

`qwen3.5:27b-q4_K_M` was determined to be the best for this workflow and that particular machine. 

For slower machines I recommend using `qwen3.5:4b-q4_K_M`


## Prerequisites

- [uv 0.12.5 or later](https://docs.astral.sh/uv/getting-started/installation/)
- `make`

## Quick start

```sh
make setup
```

`make setup` installs the pinned dependencies and creates `data/profile.yaml` from the fictional sample. Then:

1. Replace every sample value in `data/profile.yaml` with your own information.
2. Edit `variants/general.yaml` to select your role and bullet IDs.
3. Build the resume:

```sh
make all
```

The PDF is written to `dist/`. Its default filename is derived from the profile name and variant slug, such as `Alex_Rivera_general.pdf`.

## Repository layout

```text
config/design.yaml           Shared RenderCV design
data/profile.example.yaml    Fictional starter profile (tracked)
data/profile.yaml            Your local profile (ignored)
variants/general.yaml        Starter resume variant
scripts/build.py             Generates RenderCV input and renders resumes
pyproject.toml               Direct dependency pins and required uv version
requirements.lock            Fully resolved runtime dependencies
build/                       Generated sources and previews (ignored)
dist/                        Generated PDFs (ignored)
```

## Commands

```sh
make init                       # Create data/profile.yaml if it is missing
make all                        # Build every variant
make sources                    # Generate RenderCV YAML without rendering
make variant VARIANT=general    # Build one variant
make match                      # Paste a job posting; Ctrl-D submits it to local Ollama
make match JOB=job-posting.txt  # Evaluate a job-posting text file
make match VARIANT=general JOB=job-posting.txt  # Evaluate a selected resume variant
make clean                      # Remove build/ and dist/
```

## Local AI Setup

1. Install `ollama`

```sh
brew install ollama
```

2. Start the Ollama Server
```sh
ollama serve
```

3. Download your favorite model. In our case we are using `qwen3.5:27b-q4_K_M`

This model is chosen based on a Macbook Pro M1 Max with 32 GB of unified memory. 

```sh
ollama run qwen3.5:27b-q4_K_M
```

## Local AI job matching

`make match` compares the selected resume variant with a job posting using a local
[Ollama](https://ollama.com/) model. It sends the role-relevant resume content (not
your contact details) and job posting only to the Ollama server on your Mac, then
prints and saves a Markdown report under `build/matches/`. The report includes a
calibrated match score out of 100, evidence-backed matches, unverified gaps, keyword
coverage, and truthful tailoring suggestions.

Job matching uses only Python's standard library, so it does not require `make setup`
or any downloaded Python packages. `make setup` is only needed to render resume PDFs.

The default model is `qwen3.5:27b-q4_K_M`, matching the model in the setup
instructions. Override it when needed:

```sh
OLLAMA_MODEL=qwen3.5:27b-q4_K_M make match JOB=job-posting.txt
OLLAMA_HOST=http://127.0.0.1:11434 make match JOB=job-posting.txt
```

If Ollama is not already running, launch it with `ollama serve`. Use a resume variant
for the version of the resume you intend to submit; the score reflects the selected
bullets and skills rather than every item in your full profile.


## Profile format

`data/profile.yaml` contains:

- `identity`: name, email, and phone number
- `roles`: employment entries with reusable, uniquely named bullets
- `education`: one education entry
- `projects`: zero or more projects with an optional summary and reusable bullets

Use `data/profile.example.yaml` as the schema reference. The real profile is ignored so contact details and career history are not accidentally committed to this template repository.

## Adding a variant

Copy `variants/general.yaml`, choose a unique `slug`, write a targeted summary and skills section, then select role and bullet IDs from `data/profile.yaml`.

The optional `output_name` and `pdf_title` fields override generated values. If omitted, both are derived from the profile identity and variant slug.

## Customizing the design

Edit `config/design.yaml` to change typography, spacing, colors, margins, section titles, and entry templates. The shared design is applied to every variant.
