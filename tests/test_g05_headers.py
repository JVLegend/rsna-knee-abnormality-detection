"""#RSNA #Kaggle #Pesquisa — synthetic headers; no actual confirmation access."""
import ast
from pathlib import Path

import pydicom
from pydicom.dataset import FileDataset, FileMetaDataset
from pydicom.uid import ExplicitVRLittleEndian, MRImageStorage
import pytest

from scripts.g05_header_runtime import HEADER_TAGS, hashed, scan_study
from scripts.prepare_g05_headers import assemble


def write_header(path, patient='patient-a', study='1.2.3', series='1.2.3.4', sop='1.2.3.4.5'):
    path.parent.mkdir(parents=True, exist_ok=True)
    meta = FileMetaDataset(); meta.TransferSyntaxUID = ExplicitVRLittleEndian
    ds = FileDataset(str(path), {}, file_meta=meta, preamble=b'\0'*128)
    ds.SOPClassUID = MRImageStorage
    ds.PatientID = patient; ds.StudyInstanceUID = study; ds.SeriesInstanceUID = series
    ds.SOPInstanceUID = sop; ds.StudyDate = '20200101'; ds.PatientIdentityRemoved = 'YES'
    ds.DeidentificationMethod = 'synthetic test protocol not dataset evidence'
    ds.PatientName = 'NEVER_READ'; ds.Rows = 1; ds.Columns = 1; ds.BitsAllocated = 8
    ds.PixelData = b'\xff\0'
    ds.save_as(path, enforce_file_format=True)


def test_reads_every_header_and_unselected_series_without_pixels(tmp_path, monkeypatch):
    uid = '1.2.3'; metadata = []
    for j in (4, 6):
        series = uid+'.'+str(j)
        metadata.append({'StudyInstanceUID': uid, 'SeriesInstanceUID': series})
        for k in (1, 2):
            write_header(tmp_path/uid/series/(str(k)+'.dcm'), series=series, sop=series+'.'+str(k))
    original = pydicom.dcmread
    calls = []
    def checked(*a, **kw):
        assert kw['stop_before_pixels'] and kw['specific_tags'] == HEADER_TAGS
        ds = original(*a, **kw); assert 'PixelData' not in ds and 'PatientName' not in ds
        calls.append(a[0]); return ds
    monkeypatch.setattr(pydicom, 'dcmread', checked)
    row = {'StudyInstanceUID': uid, 'split': 'confirmation', 'series': [{'series_uid': uid+'.4'}]}
    record, sops, _, _, _ = scan_study(tmp_path, row, metadata, lambda: None)
    assert len(calls) == record['headers_verified'] == 4
    assert record['all_headers_consistent'] and len(record['all_series_uids']) == 2
    assert record['patient_hash'] == hashed('patient-a', 'RSNA-G05-patient')
    assert len(sops) == 4 and 'patient-a' not in str(record)


@pytest.mark.parametrize('fault', ['patient', 'series', 'missing', 'sop', 'placeholder'])
def test_inconsistent_or_incomplete_headers_fail_closed(tmp_path, fault):
    uid = '1.2.3'; series = uid+'.4'; folder = tmp_path/uid/series
    write_header(folder/'1.dcm', series=series, sop=series+'.1')
    write_header(folder/'2.dcm', patient='UNKNOWN' if fault == 'placeholder' else 'different' if fault == 'patient' else 'patient-a',
                 series=series+'.99' if fault == 'series' else series,
                 sop=series+'.1' if fault == 'sop' else series+'.2')
    metadata = [{'StudyInstanceUID': uid, 'SeriesInstanceUID': series}]
    if fault == 'missing': metadata.append({'StudyInstanceUID': uid, 'SeriesInstanceUID': uid+'.6'})
    record, *_ = scan_study(tmp_path, {'StudyInstanceUID': uid, 'split': 'train', 'series': [{'series_uid': series}]}, metadata, lambda: None)
    assert not record['all_headers_consistent'] and record['issues']


def test_builder_contains_1600_ids_but_no_labels_or_training():
    source = assemble(); spec = ast.literal_eval(ast.parse(source).body[0].value)
    assert len(spec['rows']) == 1600 and not spec['gpu'] and not spec['confirmation_evaluated']
    assert all('labels' not in r and 'report_hash' not in r for r in spec['rows'])
    assert {s: sum(r['split'] == s for r in spec['rows']) for s in ['train', 'development', 'confirmation']} == {'train': 1000, 'development': 300, 'confirmation': 300}
    assert 'AdamW' not in source and 'torch' not in source and 'PixelData' not in HEADER_TAGS
    assert 'stable_patient_key_verified\': False' in source


def test_audit_budget_interrupts_instead_of_certifying(tmp_path):
    uid = '1.2.3'; series = uid+'.4'
    write_header(tmp_path/uid/series/'1.dcm')
    def guard(): raise TimeoutError('synthetic budget')
    with pytest.raises(TimeoutError):
        scan_study(tmp_path, {'StudyInstanceUID': uid, 'series': []}, [{'StudyInstanceUID': uid, 'SeriesInstanceUID': series}], guard)
