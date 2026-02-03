"""Generate unique student_id for students (STU-{college_id}-{seq})."""
from sqlalchemy.orm import Session

from database.models import User, UserRole


def generate_student_id(db: Session, college_id: int) -> str:
    """Generate unique student_id for a college: STU-{college_id}-{seq}."""
    existing = (
        db.query(User.student_id)
        .filter(
            User.college_id == college_id,
            User.role == UserRole.STUDENT,
            User.student_id.isnot(None),
        )
        .all()
    )
    prefix = f"STU-{college_id}-"
    max_n = 0
    for (sid,) in existing:
        if sid and sid.startswith(prefix):
            try:
                n = int(sid[len(prefix) :])
                max_n = max(max_n, n)
            except ValueError:
                pass
    return f"{prefix}{max_n + 1:04d}"
