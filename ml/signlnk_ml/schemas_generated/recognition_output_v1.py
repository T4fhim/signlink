# AUTO-GENERATED from packages/schemas/recognition_output.v1.json. Do not edit; run `pnpm gen:types`.

from __future__ import annotations

from pydantic import BaseModel, ConfigDict, Field


class TopKEntry(BaseModel):
    model_config = ConfigDict(
        extra='allow',
    )
    gloss_id: str = Field(..., pattern='^[a-z]{2,3}:.+$')
    p: float = Field(..., ge=0.0, le=1.0)


class RecognitionOutputV1(BaseModel):
    model_config = ConfigDict(
        extra='allow',
    )
    gloss: str = Field(..., description='ID-gloss, e.g. THANK-YOU.', min_length=1)
    gloss_id: str = Field(
        ...,
        description='Language-qualified gloss id: <lang>:<ID-GLOSS>.',
        pattern='^[a-z]{2,3}:.+$',
    )
    confidence: float = Field(..., ge=0.0, le=1.0)
    t_start: int = Field(..., description='Sign start, ms.', ge=0)
    t_end: int = Field(..., description='Sign end, ms.', ge=0)
    top_k: list[TopKEntry] = Field(
        ..., description='Highest-probability candidates, best first.'
    )
    model_version: str = Field(..., description='SemVer, e.g. isr-0.3.1.', min_length=1)
    lexicon_version: str = Field(
        ..., description='CalVer YYYY.MM.N.', pattern='^[0-9]{4}\\.[0-9]{1,2}\\.[0-9]+$'
    )
