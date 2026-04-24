"""Data loading for student profile, performance history, study materials, and tests."""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path

DATA_DIR = Path(__file__).resolve().parent.parent.parent / "data"


@dataclass(frozen=True)
class Material:
    material_id: str
    topic: str
    title: str
    content_type: str
    difficulty: str
    estimated_minutes: int
    description: str

    def searchable_text(self) -> str:
        return f"{self.topic}. {self.title}. {self.description}"

    def to_dict(self) -> dict:
        return {
            "material_id": self.material_id,
            "topic": self.topic,
            "title": self.title,
            "content_type": self.content_type,
            "difficulty": self.difficulty,
            "estimated_minutes": self.estimated_minutes,
        }


@dataclass
class DataStore:
    profile: dict
    performance: dict
    materials: list[Material]
    tests: list[dict]

    @property
    def student_id(self) -> str:
        return self.profile["student_id"]

    def topic_score(self, topic: str) -> float | None:
        for row in self.performance.get("topic_performance", []):
            if row["topic"] == topic:
                return float(row["score_percentage"])
        return None

    def subject_score(self, subject: str) -> float | None:
        for row in self.performance.get("subject_performance", []):
            if row["subject"] == subject:
                return float(row["overall_score_percentage"])
        return None


def load_data(data_dir: Path = DATA_DIR) -> DataStore:
    profile = json.loads((data_dir / "student_profile.json").read_text())
    performance = json.loads((data_dir / "performance_history.json").read_text())
    materials_raw = json.loads((data_dir / "study_materials.json").read_text())["materials"]
    tests = json.loads((data_dir / "upcoming_tests.json").read_text())["upcoming_tests"]

    materials = [Material(**m) for m in materials_raw]
    return DataStore(profile=profile, performance=performance, materials=materials, tests=tests)
