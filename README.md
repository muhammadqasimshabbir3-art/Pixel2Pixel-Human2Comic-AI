# Pixel2Pixel — Human → Comic (pix2pix conditional GAN)

This repository contains code, pretrained weights and visualizations for a progressive pix2pix-style conditional GAN that converts human face/portrait images into a comic / stylized illustration. The implementation (in the included Jupyter notebook) uses a U-Net generator with attention / CBAM skip gates, a Patch Discriminator, and several perceptual and frequency-domain losses. Progressive training is used to go from low resolution (64px) up to 256px.

Contents
- `pix2pix-conditional-gan-in-human2comic.ipynb` — the main notebook containing model definitions, training loop, evaluation and visualization.
- Pretrained weights (trained with the notebook):
  - `generator_best.pth` — best generator (by validation) saved during training
  - `generator_final.pth` — final generator
  - `generator_ema_best.pth` — best EMA (exponential moving average) generator
  - `generator_ema_final.pth` — final EMA generator
  - `discriminator_final.pth` — final discriminator
- Visualizations and logs:
  - `vis_stage1_64px.png`, `vis_stage2_128px.png`, `vis_stage3_256px.png` — sample results saved after each progressive stage
  - `loss_curves.png`, `loss_curves_v6.png` — training loss curves
  - `__results___files/` — example input / output image pairs
- `app.py` — (placeholder) small app entry (empty in this repo)

Quick checklist
- [ ] Create a Python environment (recommended: venv or conda)
- [ ] Install required packages (see Requirements)
- [ ] Launch the notebook `pix2pix-conditional-gan-in-human2comic.ipynb` and run cells to reproduce training/visualizations, or use the pretrained weights for inference

Requirements
The notebook and model use PyTorch. A minimal set of packages to run inference / the notebook:

 - Python 3.8+
 - torch (CPU or CUDA build compatible with your machine) — tested with torch 1.12+ (install the appropriate CUDA build if you have an NVIDIA GPU)
 - torchvision
 - pillow
 - numpy
 - matplotlib
 - pandas
 - tqdm
 - scikit-image

You can create and install into a virtual environment as follows (Linux / macOS):

```bash
python -m venv .venv
source .venv/bin/activate
pip install --upgrade pip
# Install common dependencies; pick the correct torch+cuda wheel from https://pytorch.org
pip install torchvision pillow numpy matplotlib pandas tqdm scikit-image jupyterlab
pip install torch       # or follow instructions at pytorch.org for the right CUDA build
```

Usage — run the notebook
1. Open `pix2pix-conditional-gan-in-human2comic.ipynb` (e.g. with JupyterLab):

```bash
jupyter lab pix2pix-conditional-gan-in-human2comic.ipynb
```

2. Run cells in order. The notebook contains full model definitions (generator, discriminator), training loop, progressive training schedule and visualization helpers. Training can be reproduced entirely from the notebook; it will save checkpoints under the configured `save_dir`.

Quick inference example (use pretrained EMA generator)
If you only want to run inference with the provided weights, copy the `HumanToComicGenerator` class from the notebook into a Python file (for example `model.py`) or run the notebook cells that define it. Then use the snippet below to load the EMA generator and convert an image.

```python
import torch
from PIL import Image
import torchvision.transforms as T

# Import your model class (copy from notebook) e.g.:
# from model import HumanToComicGenerator

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# Create generator and set depth to target resolution (example: 256px)
gen = HumanToComicGenerator(ngf=64, dropout=0.3).to(device)
gen.set_depth(256, n_res_blocks=6)  # 64->2, 128->4, 256->6 in the notebook's config

# Load EMA weights (recommended)
state = torch.load('generator_ema_best.pth', map_location=device)
gen.load_state_dict(state)
gen.eval()

# Prepare an input image (resized to 256×256 for this example)
img = Image.open('input.jpg').convert('RGB').resize((256, 256))
tf = T.Compose([T.ToTensor(), T.Normalize([0.5,0.5,0.5], [0.5,0.5,0.5])])
inp = tf(img).unsqueeze(0).to(device)

with torch.no_grad():
	out = gen(inp)

# Denormalize and save
out = out.clamp(-1, 1).cpu()[0]
out = (out * 0.5 + 0.5).permute(1, 2, 0).numpy()
out_img = Image.fromarray((out * 255).astype('uint8'))
out_img.save('output_comic.png')
```

Notes & tips
- If you use a GPU, install the CUDA-enabled PyTorch build matching your driver. The notebook uses mixed precision (AMP) in training; for inference you can run in full precision or with torch.cuda.amp.autocast for speed.
- The model is progressive: call `set_depth(resolution, n_res_blocks=...)` before inference to match the resolution the weights expect. The notebook uses the following res -> n_res_blocks mapping: 64→2, 128→4, 256→6.
- Use the EMA weights (`generator_ema_best.pth` or `generator_ema_final.pth`) for slightly more stable outputs.
- The input images are normalized to [-1, 1] in the notebook (`Normalize([0.5]*3, [0.5]*3)` corresponds to x -> x*2-1 when using ToTensor + Normalize as above).

Pretrained results
- See `vis_stage1_64px.png`, `vis_stage2_128px.png`, `vis_stage3_256px.png` for sample qualitative results saved during training.

Reproducing training
- The notebook implements the full training pipeline including: progressive training stages, TTUR optimizer settings, EMA shadow model, VGG perceptual loss and a Frequency loss to encourage comic-style edges. To reproduce training, run the notebook end-to-end with a dataset arranged as the notebook expects (see the notebook cells for the dataset loading and transforms).

Acknowledgements
- This project follows a pix2pix / U-Net + PatchGAN design and includes several modern improvements (self-attention, CBAM skip gates, frequency loss and EMA). See the notebook for references embedded in the markdown cells.

License & contact
- No license file is included in this repository. If you plan to reuse the code or weights in a public project, please check usage constraints and add an appropriate LICENSE.
- For questions or help reproducing results, open an issue or contact the repository maintainer.


