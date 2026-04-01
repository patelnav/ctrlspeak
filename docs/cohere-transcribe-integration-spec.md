# Cohere Transcribe Integration Spec

Last updated: 2026-04-01
Status: Proposed handoff spec

## Goal

Add support for `CohereLabs/cohere-transcribe-03-2026` to `ctrlspeak` in two phases:

1. Phase 1: direct native `transformers` integration, exposed as `cohere`, with CPU-first behavior.
2. Phase 2: Apple Silicon optimized MLX integration, exposed as `cohere-mlx`, based on learnings from `mlx-audio` PR #605 without taking `mlx-audio` as a runtime dependency.

This document is intentionally implementation-oriented so another agent can complete the work without redoing the investigation.

## Research Summary

### Findings that should drive the implementation

- The model can be integrated directly through Hugging Face `transformers`.
- The correct path is the native `transformers` implementation in `transformers>=5.4.0`.
- Do not use `trust_remote_code=True` for the production integration path.
- The older remote-code path loaded but produced unusable output in testing because its processor behavior differed from the native implementation.
- Native `transformers 5.4.0` on CPU produced the expected transcript on Cohere's bundled English demo clip.
- Native `transformers 5.4.0` on MPS returned an empty string in local testing. Treat MPS as unsupported for this model until validated.
- The model checkpoint is large: `model.safetensors` is about 4.13 GB.
- `mlx-audio` PR #605 was merged on March 26, 2026, but the `v0.4.1` tag does not include Cohere support.
- The `mlx-audio` implementation is not a thin wrapper. It depends on `mlx_audio` DSP, loader, tokenizer, and utility code, so it should be treated as a reference port, not as a small dependency to bolt on.

### Local validation notes

Validated on 2026-04-01:

- Native CPU path with `transformers 5.4.0` and `torch 2.11.0`: works.
- Native MPS path with `transformers 5.4.0`: loads but returns empty transcription.
- Remote-code path: loaded but produced garbage output.

Example expected transcript from the Cohere demo clip:

`If not, there will be a big crisis between you and the European Parliament.`

## Shared Foundation

Phase 2 gets much easier if Phase 1 stops relying on string-substring routing alone.

### Recommended model metadata refactor

Introduce a small registry layer, either in `models/factory.py` or a new `models/registry.py`, with one entry per selectable model.

Each entry should contain:

- `alias`
- `repo_id`
- `backend`
- `description`
- `requires`
- `supports_translation`
- `supports_streaming`
- `preferred_device_strategy`

Minimum new backends needed:

- `transformers_whisper`
- `transformers_cohere`
- `mlx_parakeet`
- existing NeMo/Nemotron backends as appropriate
- later `mlx_cohere`

Reason:

- Phase 1 needs `cohere` to map to the real repo id.
- Phase 2 needs `cohere-mlx` to use the same upstream repo but a different backend.
- The current alias-only-to-repo mapping in `models/factory.py` cannot represent that cleanly.

If the registry refactor is judged too large for Phase 1, the fallback is:

- keep the existing alias map
- add a synthetic alias routing rule for `cohere-mlx`
- but do not pass synthetic ids into Hugging Face APIs

The registry approach is preferred.

## Existing Touch Points

These are the main files that Phase 1 or Phase 2 are expected to touch:

- `state.py`
- `models/factory.py`
- `models/base_model.py`
- `model_loader.py`
- `transcription.py`
- `cli.py`
- `ctrlspeak.py`
- `ui/state.py`
- `ui/screens/model_selection.py`
- `ui/widgets/device_info.py`
- `README.md`
- `pyproject.toml`
- `requirements-whisper.txt`

Likely new files:

- `models/cohere.py` or `models/cohere_transcribe.py`
- `requirements-cohere.txt` or a new shared HF requirements file
- `tests/test_cohere_model.py`

Phase 2 likely new files:

- `models/cohere_mlx.py`
- `models/cohere_mlx_audio.py`
- `models/cohere_mlx_config.py`
- `models/cohere_mlx_tokenizer.py`

## Phase 1 Spec: Native Transformers Backend

### Objective

Add a `cohere` model alias that works directly through native `transformers` on CPU and integrates with the existing model loading, CLI, history, and UI flows.

### In Scope

- Cold start load with `--model cohere`
- UI model list support
- UI hot-swap support
- `--list-models` support
- language-aware transcription
- explicit handling for unsupported translation behavior
- dependency/install docs

### Out of Scope

- MPS acceleration
- streaming
- translation
- changing the audio segmentation pipeline
- changing the clipboard/paste flow

### Required behavior

#### Alias and model identity

Add:

- alias `cohere`
- canonical repo id `CohereLabs/cohere-transcribe-03-2026`

If the metadata registry refactor is done, define:

- `backend = transformers_cohere`
- `supports_translation = false`
- `supports_streaming = false`
- `preferred_device_strategy = cpu_only`

If the refactor is not done, still add:

- a canonical constant in `state.py`
- the alias in the factory
- exact branching in the factory before generic fallback logic

#### Device behavior

For this backend, prefer CPU even on Apple Silicon.

Required Phase 1 behavior:

- if MPS is available, log that Cohere is intentionally running on CPU because MPS is not validated
- do not silently run on MPS in the default path

#### Dependency behavior

Do not rely on the current `requirements-whisper.txt` floor of `transformers>=4.43.0`.

Preferred Phase 1 packaging:

- add a dedicated `cohere` optional dependency group
- add a dedicated `requirements-cohere.txt`

Minimum Phase 1 dependencies:

- `transformers>=5.4.0`
- `accelerate>=0.30.0`
- `librosa>=0.11.0`
- `sentencepiece>=0.2.0`
- `protobuf>=6`

If the project wants to share one Hugging Face extra for both Whisper and Cohere, that is acceptable, but the floor must be raised high enough for native Cohere support.

#### Backend implementation

Create a new backend class derived from `BaseSTTModel`.

Required implementation characteristics:

- use native `transformers` classes only
- do not pass `trust_remote_code=True`
- use `AutoProcessor.from_pretrained(repo_id)`
- use `AutoModelForSpeechSeq2Seq.from_pretrained(repo_id, ...)`
- use the official processor path that constructs `decoder_input_ids` from `language`
- do not use the broken remote-code `model.transcribe(...)` helper

Recommended `transcribe_batch()` flow:

1. validate `source_lang`
2. reject or explicitly warn on `target_lang != source_lang`
3. load each audio file to waveform
4. call `processor(audio, sampling_rate=16000, return_tensors="pt", language=source_lang, punctuation=True)`
5. capture `audio_chunk_index` if present
6. move tensors to the chosen device and model dtype
7. call `model.generate(..., max_new_tokens=256)`
8. decode via `processor.decode(..., skip_special_tokens=True, language=source_lang, audio_chunk_index=audio_chunk_index)`
9. normalize result into the existing string list contract

#### Language behavior

Supported languages should match the model card:

- `ar`
- `de`
- `el`
- `en`
- `es`
- `fr`
- `it`
- `ja`
- `ko`
- `nl`
- `pl`
- `pt`
- `vi`
- `zh`

Behavior requirements:

- if `source_lang` is unsupported, raise a user-facing error
- if `target_lang != source_lang`, raise a user-facing error that Cohere Transcribe does not support translation

The current app exposes `--target-lang`; this backend must not pretend to translate.

#### CLI and UI behavior

Update:

- `--list-models`
- model selection UI
- any help text that enumerates models
- README install and usage docs

The UI should label the model honestly, for example:

- `cohere -> CohereLabs/cohere-transcribe-03-2026`

Optional note in docs/UI:

- `CPU-first`
- `~4.1 GB download`
- `multilingual transcription only`

### Acceptance criteria

Phase 1 is done when all of the following are true:

- `ctrlspeak --list-models` shows `cohere`
- the Textual model list shows `cohere`
- `ctrlspeak --model cohere --source-lang en` loads the model on a machine with the required deps
- the backend runs on CPU by default even when MPS is present
- the backend returns non-empty text for a valid English utterance
- the backend errors clearly if `target_lang != source_lang`
- the model can be hot-swapped from the UI
- docs mention the larger dependency/download size and CPU-first behavior

### Test plan

Add tests that do not require downloading the real 4 GB model in CI.

#### Unit tests

- alias/registry entry test for `cohere`
- factory dispatch test for the Cohere backend
- dependency-gating test with missing `transformers` or old version
- `target_lang != source_lang` error behavior
- supported-language validation

#### Mocked backend tests

Mock `AutoProcessor` and `AutoModelForSpeechSeq2Seq` and assert:

- native APIs are used
- `trust_remote_code` is not used
- `language=source_lang` is passed
- `punctuation=True` is passed unless a future UI exposes this option
- decoding uses `audio_chunk_index` when present

#### Optional manual smoke test

Manual-only, not required for CI:

- run with Cohere's demo wav or a short local clip
- verify non-empty output
- verify CPU fallback on Apple Silicon

## Phase 2 Spec: MLX Backend

### Objective

Add an Apple Silicon optimized MLX backend for the same upstream model, exposed as `cohere-mlx`.

### In Scope

- native MLX inference path
- Apple Silicon focused implementation
- tokenizer/audio frontend/chunking port sufficient for `ctrlspeak`
- no `mlx-audio` runtime dependency

### Out of Scope

- shipping `mlx-audio` wholesale
- implementing every `mlx-audio` utility or output type
- changing Phase 1 alias `cohere` to auto-switch to MLX

Keep aliases explicit:

- `cohere` remains the Phase 1 native-transformers backend
- `cohere-mlx` is the new Phase 2 Apple Silicon backend

Do not silently change `cohere` semantics in Phase 2.

### Source material to use

Primary reference:

- `mlx-audio` PR #605

Important follow-up:

- inspect current upstream `main` for any fixes after the PR merge
- specifically review the follow-up PR referenced on GitHub that removed a PyTorch dependency from Cohere ASR buffer loading

### Porting guidance

Treat the PR as a reference implementation, not a package dependency.

The following concepts are useful and should likely be ported in reduced form:

- Conformer + decoder MLX model implementation
- tokenizer prompt construction
- audio frontend feature extraction
- energy-based long-form chunk splitting
- per-language join behavior for `ja` and `zh`

Do not copy unrelated `mlx-audio` framework pieces unless strictly necessary.

### Required behavior

#### Alias and backend

Add:

- alias `cohere-mlx`
- backend `mlx_cohere`

If a model registry is introduced in Phase 1, this becomes straightforward:

- alias `cohere-mlx`
- repo id `CohereLabs/cohere-transcribe-03-2026`
- backend `mlx_cohere`
- `requires = mlx + sentencepiece + any Cohere-specific MLX helpers`

#### Dependencies

Do not add `mlx-audio` itself as a dependency.

Use the project's own dependency set, likely extending the MLX extra with only what is truly needed.

Likely additions for Phase 2:

- `sentencepiece`
- `safetensors`
- possibly `scipy` or `librosa` if resampling is required and no better local option is introduced

Keep the MLX dependency surface smaller than `mlx-audio` if possible.

#### Implementation shape

Recommended file split:

- `models/cohere_mlx.py`: backend adapter implementing `BaseSTTModel`
- `models/cohere_mlx_model.py` or equivalent: main MLX architecture
- `models/cohere_mlx_audio.py`: frontend features
- `models/cohere_mlx_tokenizer.py`: prompt tokens and decode helpers
- `models/cohere_mlx_config.py`: config dataclasses or config parsing

The backend adapter should conform to the existing `BaseSTTModel` contract used by `transcription.py`.

#### Performance and behavior

The MLX backend should be the first Cohere path officially intended for Apple Silicon.

Required behavior:

- runs on Apple Silicon without PyTorch model inference
- produces non-empty transcripts for short English test clips
- supports the same 14 languages as the upstream model
- explicitly rejects translation just like Phase 1
- uses the same segmentation queue contract as existing batch backends

### Acceptance criteria

Phase 2 is done when all of the following are true:

- `ctrlspeak --list-models` shows `cohere-mlx`
- the UI shows `cohere-mlx`
- `ctrlspeak --model cohere-mlx --source-lang en` loads on Apple Silicon with the required deps
- the backend produces non-empty output for a short speech clip
- there is no `mlx-audio` runtime dependency
- the implementation is derived from the PR learnings but is locally owned by `ctrlspeak`

### Test plan

#### Unit tests

- tokenizer prompt token construction
- long-form chunk split helper
- chunk join behavior for no-space languages
- factory dispatch for `cohere-mlx`
- translation rejection behavior

#### Smoke tests

Manual smoke tests are expected here.

Recommended:

- short English clip
- one non-English clip
- alias selection from the UI
- hot-swap from a non-Cohere model to `cohere-mlx`

CI should avoid downloading the full model unless a separate opt-in marker is added.

## Explicit Decisions

These are intentional decisions, not open questions:

- Phase 1 should use native `transformers>=5.4.0`, not remote code.
- Phase 1 should default to CPU for this model.
- Phase 1 should not claim translation support.
- Phase 2 should not depend on `mlx-audio` as a runtime package.
- Phase 2 should add `cohere-mlx` rather than silently rewriting `cohere`.

## Open Questions For The Implementing Agent

These are acceptable implementation choices:

- whether to add a dedicated `cohere` extra or merge Cohere and Whisper under a shared HF extra
- whether to introduce a new `models/registry.py` or keep the metadata in `models/factory.py`
- whether to add a small user-facing note in the model selection UI about large download size
- whether to add a manual smoke-test helper script for Cohere under `tests/` or a separate dev script

## Done Checklist

- Add `cohere` Phase 1 backend
- Add dependency/install path for Phase 1
- Update CLI/UI/docs for `cohere`
- Add tests for Phase 1
- Add `cohere-mlx` Phase 2 backend
- Port only the necessary MLX code
- Update CLI/UI/docs for `cohere-mlx`
- Add tests for Phase 2
