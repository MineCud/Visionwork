"""考勤业务逻辑：时段判定、重复签到冷却、事件落盘。

这一层与「识别」解耦：识别只负责回答“这是谁”，考勤只负责回答“该不该记、
记成什么状态”。时间通过参数注入，便于单元测试与离线回放。
"""
from __future__ import annotations

import csv
from dataclasses import asdict, dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Sequence

from . import config


@dataclass
class AttendanceEvent:
    """一条考勤事件。"""

    student_id: str
    name: str = ""
    class_name: str = ""
    status: str = config.STATUS_NORMAL
    period: str = ""
    similarity: float = 0.0
    timestamp: str = ""
    snapshot_path: str = ""

    def as_dict(self) -> Dict[str, str]:
        row = asdict(self)
        row["similarity"] = f"{self.similarity:.4f}"
        return row


class AttendanceTracker:
    """签到状态机：负责「时段 → 状态」与「冷却去重」。"""

    def __init__(
        self,
        cooldown_seconds: float = config.COOLDOWN_SECONDS,
        periods: Sequence[Dict[str, str]] = config.ATTENDANCE_PERIODS,
    ) -> None:
        self.cooldown = float(cooldown_seconds)
        self.periods = list(periods)
        self._last: Dict[str, float] = {}

    # ------------------------------------------------------------ 时段 / 状态
    def period_at(self, now: Optional[datetime] = None) -> Optional[Dict[str, str]]:
        """返回当前时刻所属签到时段，非签到时段返回 None。"""
        current = (now or datetime.now()).strftime("%H:%M")
        for period in self.periods:
            if period["start"] <= current <= period["end"]:
                return dict(period)
        return None

    def status_at(self, now: Optional[datetime] = None) -> str:
        """返回 Normal / Late / Off-hours。"""
        now = now or datetime.now()
        period = self.period_at(now)
        if period is None:
            return config.STATUS_OFF_HOURS
        return config.STATUS_LATE if now.strftime("%H:%M") > period.get("late_after", period["end"]) else config.STATUS_NORMAL

    # ------------------------------------------------------------ 冷却去重
    def can_checkin(self, student_id: str, now: Optional[datetime] = None) -> bool:
        stamp = (now or datetime.now()).timestamp()
        return (stamp - self._last.get(student_id, 0.0)) >= self.cooldown

    def checkin(self, student_id: str, now: Optional[datetime] = None) -> bool:
        """尝试记一次签到；冷却中返回 False。"""
        now = now or datetime.now()
        if not self.can_checkin(student_id, now):
            return False
        self._last[student_id] = now.timestamp()
        return True

    def remaining(self, student_id: str, now: Optional[datetime] = None) -> float:
        """剩余冷却秒数（无冷却返回 0）。"""
        stamp = (now or datetime.now()).timestamp()
        return max(0.0, self.cooldown - (stamp - self._last.get(student_id, 0.0)))

    def reset(self, student_id: Optional[str] = None) -> None:
        if student_id is None:
            self._last.clear()
        else:
            self._last.pop(student_id, None)


class AttendanceLogger:
    """考勤事件 CSV 记录器（替代原系统的数据库写入）。"""

    FIELDS = ("timestamp", "student_id", "name", "class_name", "status",
              "period", "similarity", "snapshot_path")

    def __init__(self, path: Optional[str | Path] = None) -> None:
        self.path = Path(path or config.ATTENDANCE_LOG)
        self.path.parent.mkdir(parents=True, exist_ok=True)
        if not self.path.exists():
            with self.path.open("w", encoding="utf-8-sig", newline="") as fh:
                csv.DictWriter(fh, fieldnames=list(self.FIELDS)).writeheader()

    def append(self, event: AttendanceEvent) -> None:
        row = event.as_dict()
        with self.path.open("a", encoding="utf-8-sig", newline="") as fh:
            csv.DictWriter(fh, fieldnames=list(self.FIELDS)).writerow(
                {k: row.get(k, "") for k in self.FIELDS}
            )

    def read(self) -> List[Dict[str, str]]:
        if not self.path.exists():
            return []
        with self.path.open("r", encoding="utf-8-sig", newline="") as fh:
            return list(csv.DictReader(fh))

    def clear(self) -> None:
        with self.path.open("w", encoding="utf-8-sig", newline="") as fh:
            csv.DictWriter(fh, fieldnames=list(self.FIELDS)).writeheader()
