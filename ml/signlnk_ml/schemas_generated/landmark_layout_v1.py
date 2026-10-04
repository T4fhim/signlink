# AUTO-GENERATED from packages/schemas/landmark_layout.v1.json. Do not edit; run `pnpm gen:types`.

from __future__ import annotations

from enum import StrEnum

from pydantic import BaseModel, ConfigDict, Field, RootModel


class Anchors(RootModel[int]):
    root: int = Field(..., ge=0)


class Source(StrEnum):
    hand_left = 'hand_left'
    hand_right = 'hand_right'
    pose = 'pose'
    face = 'face'


class Indice(RootModel[int]):
    root: int = Field(..., ge=0)


class LegacyHolisticIndice(RootModel[int | None]):
    root: int | None = Field(..., ge=0)


class Group(BaseModel):
    model_config = ConfigDict(
        extra='allow',
    )
    name: str = Field(..., min_length=1)
    source: Source
    indices: list[Indice] = Field(
        ...,
        description='Landmark indices in the MediaPipe Tasks output, in output order.',
    )
    legacy_holistic_indices: list[LegacyHolisticIndice | None] | None = Field(
        None,
        description='Matching indices in the legacy Holistic layout (Kaggle ISLR); null where there is no counterpart. Same length as indices.',
    )


class LandmarkLayoutV1(BaseModel):
    model_config = ConfigDict(
        extra='allow',
    )
    layout_id: str = Field(..., pattern='^slk-landmarks-[a-z0-9.-]+$')
    n_landmarks: int = Field(
        ..., description='N in the float32 [T, N, 3] tensor.', ge=1
    )
    groups: list[Group] = Field(
        ..., description='Concatenated in order to form the N landmarks.', min_length=1
    )
    anchors: dict[str, Anchors] | None = Field(
        None,
        description='Named indices into the output layout, e.g. left_shoulder and right_shoulder for normalization.',
    )
