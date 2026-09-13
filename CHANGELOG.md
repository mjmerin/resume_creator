# Changelog

All notable changes to this template are documented here.

## [0.1.4] - 2026-09-13

### Added

- Selectable Ollama, OpenAI, and Anthropic providers for résumé job matching.
- Secure interactive API-key entry and standard environment-variable support for cloud providers.
- Strict structured-output requests and regression tests for OpenAI Responses and Anthropic Messages APIs.

### Changed

- Include the selected provider and data destination in generated match reports.
- Allow `make match` to accept pasted terminal input as documented.

## [0.1.3] - 2026-09-13

### Changed

- Resolve selected résumé content in shared Python code before job-match requests.
- Calculate weighted match scores deterministically from requirement evidence classifications.
- Harden Ollama requests with a system policy, schema grounding, and host-aware privacy reporting.
- Stream Ollama match responses, reserve context for structured output, and report timeouts or context exhaustion clearly.

### Added

- A 20-posting, human-labeled model benchmark and regression tests for matching behavior.

## [0.1.2] - 2026-09-13

### Added

- Dependency-free local Ollama job matching with a 0-100 score, evidence, gaps, keyword coverage, and tailoring recommendations.
- Ignore rules for private profile data, virtual environments, and generated resume artifacts.

### Changed

- Allow uv 0.12.5 and later instead of requiring one exact patch release.

## [0.1.1] - 2026-09-10

### Changed
- Updated the dependency for `uv` so we don't have to download a specific version.
- Updated `README.md`.

## [0.1.0] - 2026-08-30

### Added

- Reproducible RenderCV build pipeline with pinned dependencies.
- Fictional profile and general resume variant examples.
- Support for shared career facts and reusable achievement bullets.
- Automatic, identity-derived PDF titles and output filenames.
- Documentation for setup, customization, and privacy-safe usage.
