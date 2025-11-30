from pydantic import BaseModel, Field
from typing import Optional, List
from enum import Enum


class ContentMode(str, Enum):
    KID = "kid"
    ADULT = "adult"


class VideoRequest(BaseModel):
    project_description: str = Field(..., min_length=5, max_length=200)
    content_mode: ContentMode = ContentMode.KID
    target_duration: int = Field(default=180, ge=60, le=600)
    expected_steps: int = Field(default=5, ge=3, le=8)
    background_music_id: Optional[str] = None


class JobStatus(str, Enum):
    QUEUED = "queued"
    PROCESSING = "processing"
    COMPLETED = "completed"
    FAILED = "failed"


class VideoResponse(BaseModel):
    job_id: str
    status: JobStatus
    message: str
    video_url: Optional[str] = None
    thumbnail_url: Optional[str] = None


class ScriptScene(BaseModel):
    dialogue: str
    visual_desc: str
    duration_sec: int


class ScriptOutput(BaseModel):
    title: str
    scenes: List[ScriptScene]
    total_duration_sec: int


class PromptEnhancement(BaseModel):
    enhanced_prompts: List[str]


class QualityEvaluation(BaseModel):
    score: float = Field(ge=0.0, le=10.0)
    feedback: str
    pass_check: bool
