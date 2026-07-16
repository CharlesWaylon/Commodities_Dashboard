import json
from pathlib import Path


def test_milestones_json_shape():
    data = json.loads(Path("config/roadmap_milestones.json").read_text())
    assert isinstance(data, list) and data
    for m in data:
        assert set(m) >= {"label", "status"}
        assert m["status"] in ("done", "in_progress", "planned")


def test_alpha_feedback_model():
    from database.models import AlphaFeedback
    cols = {c.name for c in AlphaFeedback.__table__.columns}
    assert {"id", "created_at", "page", "message", "contact"} <= cols
