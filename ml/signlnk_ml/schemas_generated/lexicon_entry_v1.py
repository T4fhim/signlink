# AUTO-GENERATED from packages/schemas/lexicon_entry.v1.json. Do not edit; run `pnpm gen:types`.

from __future__ import annotations

from enum import StrEnum

from pydantic import AnyUrl, BaseModel, ConfigDict, Field


class Status(StrEnum):
    draft = 'draft'
    in_review = 'in_review'
    approved = 'approved'
    released = 'released'
    rejected = 'rejected'


class Translation(BaseModel):
    model_config = ConfigDict(
        extra='allow',
    )
    lang: str = Field(..., pattern='^[a-z]{2,3}$')
    text: str = Field(..., min_length=1)
    sense: str | None = None


class Phonology(BaseModel):
    model_config = ConfigDict(
        extra='allow',
    )
    handshape_dom: str | None = None
    handshape_nondom: str | None = None
    sign_type: str | None = None
    location: str | None = None
    movement: str | None = None
    contact: bool | None = None
    non_manual: list[str] | None = None


class Recognition(BaseModel):
    model_config = ConfigDict(
        extra='allow',
    )
    enabled: bool
    n_examples: int = Field(..., ge=0)
    n_signers: int = Field(..., ge=0)


class Clip(BaseModel):
    model_config = ConfigDict(
        extra='allow',
    )
    clip_id: str = Field(..., min_length=1)
    license: str = Field(..., min_length=1)
    signer_id: str = Field(..., min_length=1)


class ExternalRef(BaseModel):
    model_config = ConfigDict(
        extra='allow',
    )
    source: str = Field(..., min_length=1)
    id: str = Field(..., min_length=1)
    url: AnyUrl


class LexiconEntryV1(BaseModel):
    model_config = ConfigDict(
        extra='allow',
    )
    gloss_id: str = Field(..., pattern='^[a-z]{2,3}:.+$')
    lang: str = Field(..., pattern='^[a-z]{2,3}$')
    id_gloss: str = Field(..., min_length=1)
    variant: int = Field(..., ge=1)
    translations: list[Translation]
    phonology: Phonology | None = None
    lexical_class: str | None = None
    register_tags: list[str] | None = None
    regional_tags: list[str] | None = None
    status: Status
    recognition: Recognition | None = None
    clips: list[Clip] | None = None
    external_refs: list[ExternalRef] | None = None
    license: str = Field(..., min_length=1)
    revision: int = Field(..., ge=1)
