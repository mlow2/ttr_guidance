"""Optional logger for CBF-QP solve outcomes, exported to CSV."""

import csv
import os
import time as _time


class CBFLogger:
    """Accumulates per-QP-solve records in memory and writes them to CSV."""

    FIELDNAMES = ['sim_time', 'solver_type', 'group_id', 'vehicles', 'status',
                  'stage_index', 'stage_name']

    def __init__(self):
        self.records = []
        self._current_time = 0.0
        self._step_start_time = _time.perf_counter()
        self._priority_rank_map: dict[int, int] | None = None

    def set_priority_map(self, priority_list: list[int] | None):
        """Map vehicle ids to priority ranks (index 0 = highest); None clears the map."""
        if priority_list is None:
            self._priority_rank_map = None
        else:
            self._priority_rank_map = {vid: rank for rank, vid in enumerate(priority_list)}

    def vid_label(self, vid: int) -> int:
        """Return the priority rank for *vid*, or *vid* itself if no map is set."""
        if self._priority_rank_map is not None:
            return self._priority_rank_map.get(vid, vid)
        return vid

    def set_time(self, t):
        """Set the simulation time for subsequent log() calls."""
        self._current_time = float(t)
        self._step_start_time = _time.perf_counter()

    @staticmethod
    def _format_value(value):
        if value is None:
            return ''
        if isinstance(value, float):
            return f"{value:.6f}"
        return str(value)

    @classmethod
    def _field_sort_key(cls, fieldname):
        if fieldname in cls.FIELDNAMES:
            return (0, cls.FIELDNAMES.index(fieldname), 0)
        if fieldname.startswith('num_'):
            return (1, fieldname, 0)
        if fieldname.startswith('slack_p'):
            parts = fieldname[len('slack_p'):].split('_')
            if len(parts) == 2 and parts[0].isdigit() and parts[1].isdigit():
                return (3, int(parts[0]), int(parts[1]))
        return (4, fieldname, 0)

    def log(self, solver_type, group_id, vehicles, status, **extra_fields):
        """Record one QP solve outcome; extra fields become additional CSV columns."""
        record = {
            'sim_time': f"{self._current_time:.6f}",
            'solver_type': solver_type,
            'group_id': group_id,
            'vehicles': ','.join(str(self.vid_label(v)) for v in vehicles),
            'status': status or 'unknown',
        }
        elapsed_ms = (_time.perf_counter() - self._step_start_time) * 1000.0
        record['wall_clock_ms'] = f"{elapsed_ms:.3f}"
        for key, value in extra_fields.items():
            record[key] = self._format_value(value)
        self.records.append(record)

    def _get_fieldnames_for_export(self):
        extra_fieldnames = set()
        for record in self.records:
            extra_fieldnames.update(record.keys())
        ordered_fieldnames = sorted(extra_fieldnames, key=self._field_sort_key)
        return ordered_fieldnames

    def to_csv(self, path):
        """Write all accumulated records to a CSV file."""
        os.makedirs(os.path.dirname(path) or '.', exist_ok=True)
        fieldnames = self._get_fieldnames_for_export()
        with open(path, 'w', newline='') as f:
            writer = csv.DictWriter(f, fieldnames=fieldnames)
            writer.writeheader()
            writer.writerows(self.records)

    @staticmethod
    def load_csv(path):
        """Load records from a CSV file and return a list of dicts."""
        with open(path, 'r', newline='') as f:
            reader = csv.DictReader(f)
            return list(reader)
