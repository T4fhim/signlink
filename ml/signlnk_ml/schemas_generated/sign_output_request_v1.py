# AUTO-GENERATED from packages/schemas/sign_output_request.v1.json. Do not edit; run `pnpm gen:types`.

from __future__ import annotations

from pydantic import BaseModel, ConfigDict, Field, RootModel


class Gloss(RootModel[str]):
    root: str = Field(..., min_length=1)


class SignOutputRequestV1(BaseModel):
    model_config = ConfigDict(
        extra='allow',
    )
    lang: str = Field(
        ...,
        description='Sign language code (ISO 639-3), read from config; never hard-coded.',
        pattern='^[a-z]{2,3}$',
    )
    glosses: list[Gloss]
    source_text: str
