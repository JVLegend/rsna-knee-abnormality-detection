"""#RSNA #Kaggle #Pesquisa — read-only resource/job/deadline receipt, no dispatch.

Use the existing cached Kaggle 2.2 runtime, not the system's older 2.1 package.
"""
import argparse
from datetime import datetime, timezone
import json
from pathlib import Path
import shutil

from kaggle.api.kaggle_api_extended import KaggleApi
from scripts.launch_h46_exact_source import quota_record


def main(output):
    if output.exists():
        raise FileExistsError(output)
    api = KaggleApi()
    api.authenticate()
    competition = api.competitions_list(search='rsna-knee-abnormality-detection').competitions
    if len(competition) != 1 or competition[0].title != 'RSNA Knee Abnormality Detection':
        raise ValueError('Competition identity ambiguous')
    c = competition[0]
    submissions = api.competition_submissions('rsna-knee-abnormality-detection', page_size=20)
    best = next(s for s in submissions if s.ref == 56696639)
    if str(best.public_score) != '0.943' or str(best.status).rsplit('.', 1)[-1] != 'COMPLETE':
        raise ValueError('H46 not reconciled')
    jobs = {}
    for slug in ['jvlegend/rsna-knee-h46-exact-public-source',
                 'jvlegend/rsna-knee-g04-resolution-preflight',
                 'jvlegend/rsna-knee-v03-scale1000']:
        state = api.kernels_status(slug)
        jobs[slug] = {'status': getattr(state.status, 'name', str(state.status)),
                      'failure_message': state.failure_message}
    quota = quota_record(api)  # timedelta.total_seconds includes days, unlike buggy JSON.
    disk = shutil.disk_usage(Path.cwd())
    now = datetime.now(timezone.utc)
    value = {'utc': now.isoformat(), 'deadline_utc': str(c.deadline),
             'entry_deadline_utc': str(c.new_entrant_deadline),
             'daily_submission_limit': c.max_daily_submissions,
             'utc_today_submissions_observed': sum(s.date.date() == now.date() for s in submissions),
             'quota': quota, 'jobs': jobs,
             'gpu_slot_availability': 'not_proven_by_quota; recheck_server_at_dispatch',
             'h46': {'ref': best.ref, 'status': str(best.status), 'public_score': str(best.public_score)},
             'external_disk_free_bytes': disk.free,
             'dispatches': 0, 'submissions_created': 0, 'paid_services_started': 0}
    output.parent.mkdir(parents=True, exist_ok=True)
    with output.open('x') as handle:
        json.dump(value, handle, indent=2)
    print(json.dumps(value, indent=2))


if __name__ == '__main__':
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--output', type=Path, required=True)
    main(p.parse_args().output)
