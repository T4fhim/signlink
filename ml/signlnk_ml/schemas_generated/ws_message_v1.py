# AUTO-GENERATED from packages/schemas/ws_message.v1.json. Do not edit; run `pnpm gen:types`.

from __future__ import annotations

from enum import StrEnum
from typing import Any
from uuid import UUID

from pydantic import BaseModel, ConfigDict, Field


class Type(StrEnum):
    caption = 'caption'
    gloss = 'gloss'
    sign_seq = 'sign_seq'


class WsMessageV1(BaseModel):
    model_config = ConfigDict(
        extra='allow',
    )
    type: Type
    session: UUID
    payload: dict[str, Any] = Field(..., description='Shape depends on `type`.')
