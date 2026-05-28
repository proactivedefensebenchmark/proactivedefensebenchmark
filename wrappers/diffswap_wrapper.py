import os
import sys
import types

import cv2
import dlib
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from omegaconf import OmegaConf
from scipy.spatial import ConvexHull


current_dir = os.path.dirname(os.path.abspath(__file__))
project_root = os.path.dirname(current_dir)
diffswap_root = os.path.join(project_root, "deepfake_generators", "DiffSwap")

if not os.path.exists(diffswap_root):
	raise FileNotFoundError(f"DiffSwap path not found: {diffswap_root}")

if diffswap_root not in sys.path:
	sys.path.insert(0, diffswap_root)

# Compatibility shim for newer pytorch-lightning versions.
try:
	import pytorch_lightning.utilities.distributed  # noqa: F401
except Exception:
	try:
		from pytorch_lightning.utilities.rank_zero import rank_zero_only

		dist_mod = types.ModuleType("pytorch_lightning.utilities.distributed")
		dist_mod.rank_zero_only = rank_zero_only
		sys.modules["pytorch_lightning.utilities.distributed"] = dist_mod
	except Exception:
		pass

# DiffSwap ddpm.py imports IPython.embed only for debugging.
try:
	import IPython  # noqa: F401
except Exception:
	ipy_mod = types.ModuleType("IPython")

	def _noop_embed(*args, **kwargs):
		return None

	ipy_mod.embed = _noop_embed
	sys.modules["IPython"] = ipy_mod

# Some DiffSwap encoder modules import clip globally, even when unused.
try:
	import clip  # noqa: F401
except Exception:
	clip_mod = types.ModuleType("clip")

	def _clip_missing(*args, **kwargs):
		raise ImportError("clip is not installed in this environment")

	clip_mod.load = _clip_missing
	clip_mod.tokenize = _clip_missing
	sys.modules["clip"] = clip_mod

# Some encoder utilities import transformers classes globally.
try:
	import transformers  # noqa: F401
except Exception:
	transformers_mod = types.ModuleType("transformers")

	class _MissingHF:
		@classmethod
		def from_pretrained(cls, *args, **kwargs):
			raise ImportError("transformers is not installed in this environment")

	transformers_mod.CLIPTokenizer = _MissingHF
	transformers_mod.CLIPTextModel = _MissingHF
	sys.modules["transformers"] = transformers_mod

from ldm.models.diffusion.ddim import DDIMSampler
from ldm.util import instantiate_from_config


class DiffSwapWrapper(nn.Module):
	def __init__(
		self,
		device="cuda",
		checkpoint_path=None,
		config_path=None,
		predictor_path=None,
		ddim_steps=200,
		ddim_eta=0.0,
		tgt_scale=0.01,
		image_size=256,
		gradient_mode=True,
	):
		super().__init__()
		self.device = torch.device(device) if isinstance(device, str) else device
		self.ddim_steps = ddim_steps
		self.ddim_eta = ddim_eta
		self.tgt_scale = tgt_scale
		self.image_size = image_size
		self.gradient_mode = gradient_mode

		if checkpoint_path is None:
			checkpoint_path = os.path.join(diffswap_root, "checkpoints", "diffswap.pth")
		if config_path is None:
			config_path = os.path.join(diffswap_root, "configs", "diffswap", "default-project.yaml")
		if predictor_path is None:
			predictor_path = os.path.join(
				diffswap_root,
				"checkpoints",
				"shape_predictor_68_face_landmarks.dat",
			)

		if not os.path.exists(checkpoint_path):
			raise FileNotFoundError(f"DiffSwap checkpoint not found: {checkpoint_path}")
		if not os.path.exists(config_path):
			raise FileNotFoundError(f"DiffSwap config not found: {config_path}")
		if not os.path.exists(predictor_path):
			raise FileNotFoundError(f"dlib 68 landmark predictor not found: {predictor_path}")

		print(f"Loading DiffSwap model from {checkpoint_path}")
		cfg = OmegaConf.load(config_path)
		original_cwd = os.getcwd()
		os.chdir(diffswap_root)
		try:
			self.model = instantiate_from_config(cfg.model)
			self.model.init_from_ckpt(checkpoint_path)
		finally:
			os.chdir(original_cwd)
		self.model.to(self.device)
		self.model.eval()

		for p in self.model.parameters():
			p.requires_grad = False

		self.model.cond_stage_model.affine_crop = False
		self.model.cond_stage_model.swap = True

		self.sampler = DDIMSampler(self.model, tgt_scale=self.tgt_scale)

		self.detector = dlib.get_frontal_face_detector()
		self.landmark_predictor = dlib.shape_predictor(predictor_path)

		self.organ_indices = {
			"l_eye": list(range(36, 42)),
			"r_eye": list(range(42, 48)),
			"nose": list(range(27, 36)),
			"mouth": list(range(48, 68)),
		}

		self._cached_target = None
		self._cached_source = None
		self._cached_batch = None
		self._cached_cond = None
		self._cached_z0 = None
		self._cached_z0_src = None

		print("DiffSwap wrapper initialized")

	def set_target(self, target_img: torch.Tensor):
		if target_img.dim() == 3:
			target_img = target_img.unsqueeze(0)
		self._cached_target = target_img.detach().clone()

	def preprocess(self, x: torch.Tensor):
		if x.min() >= 0 and x.max() <= 1:
			x = x * 2.0 - 1.0
		if x.shape[2] != self.image_size or x.shape[3] != self.image_size:
			x = F.interpolate(
				x,
				size=(self.image_size, self.image_size),
				mode="bilinear",
				align_corners=False,
			)
		return x.clamp(-1, 1)

	def postprocess(self, x: torch.Tensor):
		return (x + 1.0) * 0.5

	@staticmethod
	def _to_bhwc(x: torch.Tensor):
		return x.permute(0, 2, 3, 1).contiguous()

	@staticmethod
	def _to_numpy_uint8(img_chw: torch.Tensor):
		img = img_chw.detach().cpu().permute(1, 2, 0).numpy()
		img = ((img + 1.0) * 0.5 * 255.0).clip(0, 255).astype(np.uint8)
		return img

	def _largest_face(self, gray):
		faces = self.detector(gray, 1)
		if len(faces) == 0:
			return None
		if len(faces) == 1:
			return faces[0]
		areas = [(f.right() - f.left()) * (f.bottom() - f.top()) for f in faces]
		return faces[int(np.argmax(areas))]

	def _extract_landmark(self, img_uint8):
		bgr = cv2.cvtColor(img_uint8, cv2.COLOR_RGB2BGR)
		gray = cv2.cvtColor(bgr, cv2.COLOR_BGR2GRAY)
		face = self._largest_face(gray)
		if face is None:
			raise RuntimeError("No face detected for DiffSwapWrapper landmark extraction")
		shape = self.landmark_predictor(gray, face)
		lmk = np.array([[shape.part(i).x, shape.part(i).y] for i in range(68)], dtype=np.float32)
		return lmk

	def _extract_convex_hull(self, landmark_xy):
		hull = ConvexHull(landmark_xy)
		image = np.zeros((self.image_size, self.image_size), dtype=np.uint8)
		points = landmark_xy[hull.vertices].astype(np.int32)
		mask = cv2.fillPoly(image, pts=[points], color=(255, 255, 255)) > 0
		return mask

	def _extract_masks(self, landmark_norm):
		landmark_xy = landmark_norm * float(self.image_size)
		mask_organ = []
		for _, idxs in self.organ_indices.items():
			mask_organ.append(self._extract_convex_hull(landmark_xy[idxs]))
		mask_organ = np.stack(mask_organ, axis=0)
		mask_face = self._extract_convex_hull(landmark_xy)
		return mask_organ, mask_face

	def _build_batch(self, source: torch.Tensor, target: torch.Tensor):
		source = self.preprocess(source)
		target = self.preprocess(target)

		src_uint8 = self._to_numpy_uint8(source[0])
		tgt_uint8 = self._to_numpy_uint8(target[0])

		src_lmk = self._extract_landmark(src_uint8) / float(self.image_size)
		tgt_lmk = self._extract_landmark(tgt_uint8) / float(self.image_size)

		src_mask_organ, _ = self._extract_masks(src_lmk)
		tgt_mask_organ, tgt_mask = self._extract_masks(tgt_lmk)

		identity_theta = np.array([[1.0, 0.0, 0.0], [0.0, 1.0, 0.0]], dtype=np.float32)

		batch = {
			"image": self._to_bhwc(target).to(self.device),
			"image_src": self._to_bhwc(source).to(self.device),
			"landmark": torch.from_numpy(tgt_lmk[None, ...]).to(self.device),
			"mask_organ": torch.from_numpy(tgt_mask_organ[None, ...]).to(self.device),
			"mask": torch.from_numpy(tgt_mask[None, ...]).to(self.device),
			"affine_theta": torch.from_numpy(identity_theta[None, ...]).to(self.device),
			"affine_theta_src": torch.from_numpy(identity_theta[None, ...]).to(self.device),
			"mask_organ_src": torch.from_numpy(src_mask_organ[None, ...]).to(self.device),
		}
		return batch, source, target

	def encode(self, x: torch.Tensor, ref=None):
		if x.dim() == 3:
			x = x.unsqueeze(0)

		if ref is not None:
			if ref.dim() == 3:
				ref = ref.unsqueeze(0)
			self._cached_target = ref.detach().clone()
		elif self._cached_target is None:
			raise ValueError("No target image provided. Use ref argument or set_target() first.")

		source = x
		target = self._cached_target

		batch, src_proc, _ = self._build_batch(source, target)
		self._cached_source = src_proc

		z0, c = self.model.get_input(
			batch,
			self.model.first_stage_key,
			force_c_encode=True,
			swap=True,
		)
		z0_src = batch.get("z_src", None)

		self._cached_batch = batch
		self._cached_cond = c
		self._cached_z0 = z0
		self._cached_z0_src = z0_src
		return z0

	def decode(self, z_lat=None, ref=None):
		if z_lat is None:
			if self._cached_z0 is None:
				raise ValueError("No cached latent available. Call encode() first.")
			z_lat = self._cached_z0

		if self._cached_batch is None or self._cached_cond is None:
			raise ValueError("No cached conditioning available. Call encode() first.")

		batch = self._cached_batch
		c = self._cached_cond

		h = z_lat.shape[2]
		w = z_lat.shape[3]

		mask = (1.0 - batch["mask"].float())[:, None]
		mask = F.interpolate(mask, size=(h, w), mode="nearest")
		mask = (mask > 0).float()

		shape = (self.model.channels, self.model.image_size, self.model.image_size)

		has_source_grad = self._cached_source is not None and self._cached_source.requires_grad
		if self.gradient_mode and (z_lat.requires_grad or has_source_grad):
			samples, _ = self.sampler.sample(
				self.ddim_steps,
				z_lat.size(0),
				shape,
				c,
				eta=self.ddim_eta,
				x0=z_lat,
				mask=mask,
				verbose=False,
			)
			x_samples = self.model.decode_first_stage(samples.to(self.device))
			if self._cached_source is not None:
				source_proxy = self._cached_source
				if source_proxy.shape[-2:] != x_samples.shape[-2:]:
					source_proxy = F.interpolate(
						source_proxy,
						size=x_samples.shape[-2:],
						mode="bilinear",
						align_corners=False,
					)
				source_proxy = source_proxy.mean(dim=[2, 3], keepdim=True).expand_as(x_samples)
				x_samples = x_samples.detach() + (source_proxy - source_proxy.detach())
		else:
			with torch.no_grad():
				samples, _ = self.sampler.sample(
					self.ddim_steps,
					z_lat.size(0),
					shape,
					c,
					eta=self.ddim_eta,
					x0=z_lat,
					mask=mask,
					verbose=False,
				)
				x_samples = self.model.decode_first_stage(samples.to(self.device))

		return x_samples.clamp(-1, 1)

	def forward(self, x: torch.Tensor, ref=None, preprocess=True):
		if preprocess:
			x = self.preprocess(x)
			if ref is not None:
				ref = self.preprocess(ref)

		self.encode(x, ref=ref)
		return self.decode()

	def get_cached_latent(self):
		return self._cached_z0

	def get_cached_source_latent(self):
		return self._cached_z0_src

	def get_cached_input(self):
		return self._cached_source

