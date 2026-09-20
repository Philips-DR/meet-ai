"""The credential seam.

Credentials come IN. This module never reads a token store, never imports another tool to
find one, and never assumes an identity -- the CLI resolves defaults from the environment
and passes them down explicitly. That is what stops a second tool repeating the
cross-module auth import that made the old monolith's modules non-independent.
"""

from __future__ import annotations

from dataclasses import dataclass

# Bedrock model ids carry an "anthropic." prefix; the bare id is the first-party form.
BEDROCK_PREFIX = "anthropic."
DEFAULT_MODEL = "claude-opus-5"


@dataclass(frozen=True)
class ModelConfig:
    """Everything the notes layer needs to reach a model. Passed in, never discovered.

    `profile` is here because a machine with several AWS profiles and no [default] is the
    normal case, not an edge one -- leaving the SDK to guess produced a bare "could not
    resolve AWS credentials" with nothing to act on.
    """

    region: str
    model: str = DEFAULT_MODEL
    profile: str | None = None

    @property
    def bedrock_model_id(self) -> str:
        if self.model.startswith(BEDROCK_PREFIX):
            return self.model
        return f"{BEDROCK_PREFIX}{self.model}"


def build_client(config: ModelConfig):
    """A Bedrock client for the given config.

    Imported lazily so that every pure module above -- and every test of them -- keeps
    working with the SDK absent and no credentials configured.
    """
    from anthropic import AnthropicBedrockMantle

    return AnthropicBedrockMantle(aws_region=config.region, aws_profile=config.profile)
