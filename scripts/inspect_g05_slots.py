"""#RSNA #Kaggle #Pesquisa — read-only own-session audit; no T4 reservation."""
import argparse
from datetime import datetime, timedelta, timezone
import json
from pathlib import Path

from kaggle.api.kaggle_api_extended import KaggleApi
from scripts.launch_h46_exact_source import quota_record


def main(output):
    if output.exists():
        raise FileExistsError('Preserve resource observation receipt')
    api = KaggleApi(); api.authenticate()
    now = datetime.now(timezone.utc)
    recent = api.kernels_list(mine=True, sort_by='dateRun', page_size=100)
    candidates = []
    for kernel in recent:
        date = kernel.last_run_time
        if date.tzinfo is None: date = date.replace(tzinfo=timezone.utc)
        if kernel.enable_gpu and date >= now-timedelta(hours=24):
            state = api.kernels_status(kernel.ref)
            candidates.append({'slug': kernel.ref, 'last_run_utc': date.isoformat(),
                               'status': getattr(state.status, 'name', str(state.status))})
    oldest = min((k.last_run_time for k in recent), default=now)
    if oldest.tzinfo is None: oldest = oldest.replace(tzinfo=timezone.utc)
    coverage = len(recent) < 100 or oldest < now-timedelta(hours=24)
    active = [r for r in candidates if r['status'] not in ('COMPLETE', 'ERROR', 'CANCELLED')]
    value = {'utc': now.isoformat(), 'quota': quota_record(api), 'owned_recent_scanned': len(recent),
             'recent_24h_list_covered': coverage, 'recent_gpu_sessions': candidates,
             'active_or_queued_recent_gpu_sessions': active,
             'own_recent_gpu_queue_clear': coverage and not active,
             'gpu_hardware_slot_guaranteed': False,
             'limitation': 'Read-only API exposes own latest commit runs/quota, not guaranteed T4 capacity or unsaved interactive sessions. Recheck at dispatch; do not launch a probe merely to reserve a slot.',
             'gpu_dispatches': 0, 'paid_services_started': 0, 'submissions_created': 0}
    output.parent.mkdir(parents=True, exist_ok=True)
    with output.open('x') as handle: json.dump(value, handle, indent=2)
    print(json.dumps(value, indent=2))


if __name__ == '__main__':
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--output', type=Path, required=True)
    main(p.parse_args().output)
