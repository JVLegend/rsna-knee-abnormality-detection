"""Constantes oficiais da competição."""

KEY_COLUMN = "StudyInstanceUID"

TARGET_COLUMNS = [
    "ACL",
    "MCL",
    "Medial Meniscus",
    "Lateral Meniscus",
    "Medial OA",
    "Lateral OA",
    "PF OA",
    "Effusion",
    "Synovitis",
    "Baker's",
    "Contusion",
    "Fracture",
]

EXPECTED_TRAIN_COLUMNS = ["StudyInstanceUID", "PatientSex", "Report", *TARGET_COLUMNS]
EXPECTED_SERIES_COLUMNS = [
    "StudyInstanceUID",
    "SeriesInstanceUID",
    "Fluid_Sensitive",
    "Fat_Suppression",
    "Anatomical_Plane",
]
