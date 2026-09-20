"""The credential seam, and the provider seam behind it.

Credentials come IN. This module never reads a token store, never imports another tool to
find one, and never assumes an identity -- the CLI resolves defaults from the environment
and passes them down explicitly. That is what stops a second tool repeating the
cross-module auth import that made the old monolith's modules non-independent.

Two providers, chosen the same way the transcription layer chooses between AssemblyAI and
local Whisper: one config object, one factory, and nothing above this file knows which one
ran. Bedrock is what this machine has today; a plain API key is a one-flag switch away.
"""

from __future__ import annotations

import os
from dataclasses import dataclass

BEDROCK = "bedrock"
ANTHROPIC = "anthropic"
PROVIDERS = (BEDROCK, ANTHROPIC)

# Bedrock model ids carry an "anthropic." prefix; the bare id is the first-party form.
# Cross-region INFERENCE PROFILE ids put a region scope in front of that ("us.anthropic...",
# "global.anthropic..."), so the test is "does this id already name anthropic", not "does it
# start with the prefix" -- the latter turns a profile id into "anthropic.us.anthropic...".
BEDROCK_PREFIX = "anthropic."

# Per provider, because the same model is named differently on each -- and because the two
# accounts do not grant the same models.
#
# First-party: Claude Opus 5, the current default and what this should use everywhere.
# Bedrock: probed live against this account (2026-09-20) -- opus-5, opus-4-8, opus-4-7 and
# sonnet-5 all return 403 "not available for this account". 4.6 is the most capable id that
# actually answers. Access is granted per-account in the Bedrock console, not in code, so
# raise this the moment Opus 5 is enabled there.
DEFAULT_MODELS = {
    ANTHROPIC: "claude-opus-5",
    BEDROCK: "us.anthropic.claude-opus-4-6-v1",
}


@dataclass(frozen=True)
class ModelConfig:
    """Everything the notes layer needs to reach a model. Passed in, never discovered.

    `region` and `profile` are Bedrock's; `api_key` is the first-party API's. Each is
    ignored by the other provider rather than being an error, so switching provider is a
    single flag and not a rewrite of the call site.
    """

    provider: str = BEDROCK
    model: str | None = None
    region: str | None = None
    profile: str | None = None
    api_key: str | None = None

    def __post_init__(self) -> None:
        if self.provider not in PROVIDERS:
            raise ValueError(f"unknown provider {self.provider!r}; expected one of {PROVIDERS}")

    @property
    def resolved_model(self) -> str:
        """The id to send, defaulted per provider and adjusted only where Bedrock needs it."""
        model = self.model or DEFAULT_MODELS[self.provider]
        if self.provider != BEDROCK:
            return model
        # A bare first-party id needs the prefix; an inference profile id already names
        # anthropic and must be left exactly alone.
        return model if BEDROCK_PREFIX in model else f"{BEDROCK_PREFIX}{model}"


def resolve_provider(explicit: str | None = None, env: dict | None = None) -> str:
    """Which provider to use when the caller did not say.

    An API key in the environment means the first-party API is configured, and that is the
    simpler path, so it wins. Otherwise Bedrock, which is what an AWS-shaped machine has.
    """
    if explicit:
        return explicit
    environ = os.environ if env is None else env
    return ANTHROPIC if environ.get("ANTHROPIC_API_KEY") else BEDROCK


def build_client(config: ModelConfig):
    """The client for this config. Nothing above this function knows which provider ran.

    Imported lazily so that every pure module -- and every test of them -- keeps working
    with the SDK absent and no credentials configured.

    For Bedrock this is AnthropicBedrock, not AnthropicBedrockMantle, despite Mantle being
    the recommended client for new code: Mantle returns 404 for every cross-region
    inference profile id, and this account reaches Claude only through those (a bare
    on-demand id returns "isn't supported with on-demand throughput"). Probed both ways
    live before choosing. Revisit if Mantle gains profile support.
    """
    if config.provider == ANTHROPIC:
        from anthropic import Anthropic

        # A None api_key is not a failure: the SDK then resolves ANTHROPIC_API_KEY, an
        # ANTHROPIC_AUTH_TOKEN, or a logged-in profile on its own.
        return Anthropic(api_key=config.api_key) if config.api_key else Anthropic()

    from anthropic import AnthropicBedrock

    return AnthropicBedrock(aws_region=config.region, aws_profile=config.profile)
