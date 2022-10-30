import argparse
from pathlib import Path
import warnings
from io import BytesIO

import numpy as np
from torch.utils.data import DataLoader
import uuid

from dataset import get_dataset
from model import load_model_from_path
from model.renderer import save_mesh_as_gif
from utils import path_mkdir
from utils.path import MODELS_PATH
from utils.logger import print_log
from utils.mesh import save_mesh_as_obj, normalize
from utils.pytorch import get_torch_device


BATCH_SIZE = 32
N_WORKERS = 4
PRINT_ITER = 1
SAVE_GIF = False
warnings.filterwarnings("ignore")


def reconstruct(model, input):
    assert model is not None and input is not None

    device = get_torch_device()
    m = load_model_from_path(MODELS_PATH / model).to(device)
    m.eval()
    print_log(f"Model {model} loaded: input img_size is set to {m.init_kwargs['img_size']}")


    data = get_dataset(input)(split='test', img_size=m.init_kwargs['img_size'])
    loader = DataLoader(data, batch_size=BATCH_SIZE, num_workers=N_WORKERS, shuffle=False)


    print_log(f"Found {len(data)} images in the folder")
    print_log("Starting reconstruction...")

    out = path_mkdir('demo_rec')
    reconstruction_count = 0
    for j, (inp, _) in enumerate(loader):
        imgs = inp['imgs'].to(device)
        meshes = m.predict_mesh_pose_bkg(imgs)[0]

        B, d, e = len(imgs), m.T_init[-1], np.mean(m.elev_range)


        print('B is ', B)


        for k in range(B):
            reconstruction_count += 1
            mcenter = normalize(meshes[k])
            filename = str(uuid.uuid4())
            save_mesh_as_obj(mcenter, out / f'{filename}_mesh.obj')

    print_log("Done!")

    if reconstruction_count > 0:
        path = Path(out / f'{filename}_mesh.obj')
        print('Reconstructed is file =', path.is_file())
        buffer = BytesIO(path.read_bytes())
        path.unlink()
        Path(out / f'{filename}_mesh.mtl').unlink()
        Path(out / f'{filename}_mesh.png').unlink()
        return buffer

