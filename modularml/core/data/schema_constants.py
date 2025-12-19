from typing import Final, Literal

# ======================================================
# Domain vocabulary
# ======================================================
DOMAIN_FEATURES: Final[str] = "features"
DOMAIN_TARGETS: Final[str] = "targets"
DOMAIN_TAGS: Final[str] = "tags"
DOMAIN_SAMPLE_ID: Final[str] = "sample_id"
DOMAIN_OUTPUTS: Final[str] = "outputs"

ALL_DOMAINS: Final[tuple[str, ...]] = (
    DOMAIN_FEATURES,
    DOMAIN_TARGETS,
    DOMAIN_TAGS,
    DOMAIN_SAMPLE_ID,
    DOMAIN_OUTPUTS,
)
T_ALL_DOMAINS = Literal[
    "features",
    "targets",
    "tags",
    "sample_id",
    "outputs",
]

# ======================================================
# Representation vocabulary
# ======================================================
REP_RAW: Final[str] = "raw"
REP_TRANSFORMED: Final[str] = "transformed"

ALL_REPS: Final[tuple[str, ...]] = (
    REP_RAW,
    REP_TRANSFORMED,
)
T_ALL_REPS = Literal[
    "raw",
    "transformed",
]


# ======================================================
# Label / naming rules
# ======================================================
INVALID_LABEL_CHARACTERS: Final[set[str]] = {".", " ", "/", "\\", ":"}

MML_FILE_VERSION = "1.0"
MML_EXTENSION = ".mml"
