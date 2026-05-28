import os

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

ARCFACE_CKPT = os.path.join(
    PROJECT_ROOT, "deepfake_generators", "face_idloss", "model_ir_se50.pth")

_DATASETS_DIR = os.path.join(PROJECT_ROOT, "datasets")

CELEBA_HQ_DIR = os.path.join(_DATASETS_DIR, "CelebAHQ")
CELEBA_HQ_TEST_DIR = os.path.join(CELEBA_HQ_DIR, "source")
CELEBA_HQ_TRAIN_DIR = os.path.join(CELEBA_HQ_DIR, "source")

CELEBAHQ_DIR = {
    'base': CELEBA_HQ_DIR,
    'train': CELEBA_HQ_TRAIN_DIR,
    'val': CELEBA_HQ_TEST_DIR,
    'test': CELEBA_HQ_TEST_DIR,
    'attrs': os.path.join(CELEBA_HQ_DIR, 'CelebAMask-HQ-attribute-anno.txt'),
    'identity': os.path.join(CELEBA_HQ_DIR, 'CelebA-HQ-to-CelebA-mapping.txt'),
}

DATASETS = {
    "ffhq": {
        "image_dir": os.path.join(_DATASETS_DIR, "FFHQ", "source"),
        "target": os.path.join(_DATASETS_DIR, "FFHQ", "target", "69999.png"),
    },
    "celeba": {
        "image_dir": os.path.join(_DATASETS_DIR, "CelebAHQ", "source"),
        "target": os.path.join(_DATASETS_DIR, "CelebAHQ", "target", "29550.jpg"),
    },
    "vggface2hq": {
        "image_dir": os.path.join(_DATASETS_DIR, "VGGFace2-HQ", "source"),
        "target": os.path.join(_DATASETS_DIR, "VGGface2", "target", "0066_01.jpg"),
    },
}

FFHQ_TARGET_69999 = DATASETS["ffhq"]["target"]
SIMSWAP_TARGET = DATASETS["celeba"]["target"]
