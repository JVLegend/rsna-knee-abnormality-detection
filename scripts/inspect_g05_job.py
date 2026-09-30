"""#RSNA #Kaggle #Dados — bounded read-only reconciliation of the single CPU audit."""
import argparse
import json
from pathlib import Path
import signal
import time
import requests

from kaggle.api.kaggle_api_extended import KaggleApi

SLUG = 'jvlegend/rsna-knee-g05-full-headers-v1'
PIXELS_SLUG = 'jvlegend/rsna-knee-g05-paired-pixels-v1'


def main(args):
    api = KaggleApi(); api.authenticate()
    slug = args.slug
    state = api.kernels_status(slug)
    status = getattr(state.status, 'name', str(state.status))
    print(json.dumps({'slug': slug, 'status': status, 'failure_message': state.failure_message}), flush=True)
    if status in ('COMPLETE', 'ERROR'):
        pattern = (r'^g05_headers(\.json|_receipt\.json|_failure\.json|_progress\.json)$' if slug == SLUG else
                   r'^g05_(pixels_\d{4}\.npz|pixels_receipt\.json|geometry\.json|duplicates\.json|extraction_failure\.json)$')
        print(api.kernels_output(slug, str(args.output), file_pattern=pattern, force=False))
    elif args.logs:
        yielded = False
        def alarm(*_):
            nonlocal yielded
            yielded = True
            raise TimeoutError('Bounded log observation complete')
        old = signal.signal(signal.SIGALRM, alarm); signal.alarm(20)
        recent = []
        try:
            for i, event in enumerate(api.kernels_logs_stream(slug)):
                recent.append(str(event)[:700])
                recent = recent[-5:]
                if i >= 120:
                    break
        except TimeoutError:
            print('LOG_OBSERVATION_YIELDED', flush=True)
        except requests.exceptions.ConnectionError:
            # urllib3 wraps a timed-out SSE read. This is a local observation
            # deadline, not evidence the remote audit failed.
            if not yielded:
                raise
            print('LOG_OBSERVATION_YIELDED', flush=True)
        finally:
            signal.alarm(0); signal.signal(signal.SIGALRM, old)
            for event in recent:
                print(event, flush=True)


if __name__ == '__main__':
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--output', type=Path, required=True)
    p.add_argument('--logs', action='store_true')
    p.add_argument('--slug', default=SLUG, choices=[SLUG, PIXELS_SLUG])
    main(p.parse_args())
