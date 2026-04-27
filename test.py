import argparse
import math
import os
from pathlib import Path

import torch
import torch.nn.functional as F
from PIL import Image
from torchvision import transforms
from torchvision.utils import save_image

from model_convnext import fusion_net


SCRIPT_DIR = Path(__file__).resolve().parent
DEFAULT_INPUT_DIR = SCRIPT_DIR / 'combined_dataset' / 'test'
DEFAULT_OUTPUT_DIR = SCRIPT_DIR / 'experients' / 'test_result'
BACKBONE_FILENAME = 'convnext_xlarge_22k_1k_384_ema.pth'
SUPPORTED_IMAGE_SUFFIXES = {'.png', '.jpg', '.jpeg', '.bmp', '.tif', '.tiff', '.webp'}
COMMON_CHECKPOINT_NAMES = (
  'test_best.pkl',
  'validation_best.pkl',
  'best.pkl',
  'model_best.pkl',
)


def parse_args():
  parser = argparse.ArgumentParser(description='DWT-FFC single-GPU inference')
  parser.add_argument(
    '--input_dir',
    '--test_dir',
    dest='input_dir',
    type=str,
    default=str(DEFAULT_INPUT_DIR),
    help='Folder containing hazy images directly or a hazy/ subfolder.',
  )
  parser.add_argument(
    '--output_dir',
    type=str,
    default=str(DEFAULT_OUTPUT_DIR),
    help='Folder where dehazed outputs will be written.',
  )
  parser.add_argument(
    '--checkpoint',
    type=str,
    default=None,
    help='Generator checkpoint path. If omitted, the script auto-detects a checkpoint in weights/.',
  )
  parser.add_argument(
    '--device',
    type=str,
    default=None,
    help='Torch device string such as cuda:0 or cpu. Defaults to cuda:0 when available.',
  )
  parser.add_argument(
    '--tile_size',
    type=int,
    default=1024,
    help='Square tile size for inference. Lower this if you hit CUDA OOM.',
  )
  parser.add_argument(
    '--tile_overlap',
    type=int,
    default=128,
    help='Overlap between adjacent tiles. Must be smaller than tile_size.',
  )
  parser.add_argument(
    '--pad_multiple',
    type=int,
    default=16,
    help='Pad each tile to a multiple of this value before inference.',
  )
  parser.add_argument(
    '--save_ext',
    type=str,
    default='.png',
    help='Extension used for saved outputs, including the leading dot.',
  )
  parser.add_argument(
    '--amp',
    action='store_true',
    help='Enable mixed precision inference on CUDA to reduce memory usage.',
  )
  return parser.parse_args()


def select_device(device_arg):
  if device_arg:
    return torch.device(device_arg)
  if torch.cuda.is_available():
    return torch.device('cuda:0')
  return torch.device('cpu')


def resolve_checkpoint(checkpoint_arg):
  if checkpoint_arg:
    checkpoint_path = Path(checkpoint_arg).expanduser()
    if not checkpoint_path.is_absolute():
      checkpoint_path = (SCRIPT_DIR / checkpoint_path).resolve()
    if not checkpoint_path.is_file():
      raise FileNotFoundError(f'Checkpoint not found: {checkpoint_path}')
    return checkpoint_path

  weights_dir = SCRIPT_DIR / 'weights'
  if not weights_dir.is_dir():
    raise FileNotFoundError(f'Weights directory not found: {weights_dir}')

  for candidate_name in COMMON_CHECKPOINT_NAMES:
    candidate_path = weights_dir / candidate_name
    if candidate_path.is_file():
      return candidate_path

  candidates = []
  for pattern in ('*.pkl', '*.pth', '*.pt'):
    for candidate_path in sorted(weights_dir.glob(pattern)):
      if candidate_path.name == BACKBONE_FILENAME:
        continue
      candidates.append(candidate_path)

  if len(candidates) == 1:
    return candidates[0]

  if candidates:
    candidate_list = ', '.join(str(candidate_path.name) for candidate_path in candidates)
    raise FileNotFoundError(
      'Multiple possible checkpoints were found in weights/. '
      f'Pass --checkpoint explicitly. Candidates: {candidate_list}'
    )

  raise FileNotFoundError(
    'No generator checkpoint found in weights/. '
    'Pass --checkpoint explicitly or place a saved model checkpoint there.'
  )


def resolve_input_dir(input_dir_arg):
  input_dir = Path(input_dir_arg).expanduser()
  if not input_dir.is_absolute():
    input_dir = (SCRIPT_DIR / input_dir).resolve()
  if not input_dir.is_dir():
    raise FileNotFoundError(f'Input directory not found: {input_dir}')

  hazy_dir = input_dir / 'hazy'
  if hazy_dir.is_dir():
    return hazy_dir
  return input_dir


def collect_image_paths(image_dir):
  image_paths = [
    image_path
    for image_path in sorted(image_dir.iterdir())
    if image_path.is_file() and image_path.suffix.lower() in SUPPORTED_IMAGE_SUFFIXES
  ]
  if not image_paths:
    raise FileNotFoundError(
      f'No supported images were found in {image_dir}. '
      f'Supported extensions: {sorted(SUPPORTED_IMAGE_SUFFIXES)}'
    )
  return image_paths


def build_model(device, checkpoint_path):
  checkpoint = torch.load(checkpoint_path, map_location=device)
  if isinstance(checkpoint, dict) and 'state_dict' in checkpoint:
    state_dict = checkpoint['state_dict']
  elif isinstance(checkpoint, dict) and 'model_state_dict' in checkpoint:
    state_dict = checkpoint['model_state_dict']
  else:
    state_dict = checkpoint

  cleaned_state_dict = {}
  for key, value in state_dict.items():
    if key.startswith('module.'):
      key = key[len('module.'):]
    cleaned_state_dict[key] = value

  model = fusion_net().to(device)
  model.load_state_dict(cleaned_state_dict, strict=True)
  model.eval()
  return model


def compute_positions(length, tile_size, tile_overlap):
  if tile_size >= length:
    return [0]

  stride = tile_size - tile_overlap
  if stride <= 0:
    raise ValueError('tile_overlap must be smaller than tile_size.')

  positions = list(range(0, length - tile_size + 1, stride))
  if positions[-1] != length - tile_size:
    positions.append(length - tile_size)
  return positions


def pad_tile(tile, tile_size, pad_multiple):
  _, height, width = tile.shape
  target_height = max(height, tile_size)
  target_width = max(width, tile_size)

  target_height = int(math.ceil(target_height / pad_multiple) * pad_multiple)
  target_width = int(math.ceil(target_width / pad_multiple) * pad_multiple)

  pad_bottom = target_height - height
  pad_right = target_width - width
  if pad_bottom == 0 and pad_right == 0:
    return tile, height, width

  padded_tile = F.pad(tile, (0, pad_right, 0, pad_bottom), mode='replicate')
  return padded_tile, height, width


def run_tiled_inference(model, image_tensor, device, tile_size, tile_overlap, pad_multiple, use_amp):
  _, image_height, image_width = image_tensor.shape
  output_sum = torch.zeros(3, image_height, image_width, dtype=torch.float32)
  weight_sum = torch.zeros(1, image_height, image_width, dtype=torch.float32)

  y_positions = compute_positions(image_height, tile_size, tile_overlap)
  x_positions = compute_positions(image_width, tile_size, tile_overlap)

  autocast_enabled = use_amp and device.type == 'cuda'
  warned_amp_fallback = False
  with torch.inference_mode():
    for top in y_positions:
      bottom = min(top + tile_size, image_height)
      for left in x_positions:
        right = min(left + tile_size, image_width)
        tile = image_tensor[:, top:bottom, left:right]
        tile, valid_height, valid_width = pad_tile(tile, tile_size, pad_multiple)
        tile_batch = tile.unsqueeze(0).to(device)

        try:
          with torch.cuda.amp.autocast(enabled=autocast_enabled):
            prediction = model(tile_batch)
        except RuntimeError as error:
          if autocast_enabled and 'unsupported dtype half' in str(error).lower():
            if not warned_amp_fallback:
              print('AMP is not supported by the FFT layers in this model on the current PyTorch build. Falling back to full precision.')
              warned_amp_fallback = True
            autocast_enabled = False
            prediction = model(tile_batch)
          elif 'out of memory' in str(error).lower():
            raise RuntimeError(
              'CUDA out of memory during inference. '
              'Retry with a smaller --tile_size, such as 768 or 512.'
            ) from error
          else:
            raise

        prediction = prediction.squeeze(0).float().cpu()
        prediction = prediction[:, :valid_height, :valid_width]

        output_sum[:, top:bottom, left:right] += prediction[:, :bottom - top, :right - left]
        weight_sum[:, top:bottom, left:right] += 1.0

        if device.type == 'cuda':
          torch.cuda.empty_cache()

  return output_sum / weight_sum.clamp_min(1.0)


def save_prediction(prediction, output_path):
  prediction = prediction.clamp(0.0, 1.0)
  save_image(prediction, str(output_path))


def main():
  os.chdir(SCRIPT_DIR)
  args = parse_args()

  if args.tile_size <= 0:
    raise ValueError('tile_size must be positive.')
  if args.pad_multiple <= 0:
    raise ValueError('pad_multiple must be positive.')
  if not args.save_ext.startswith('.'):
    raise ValueError('save_ext must start with a dot, for example .png')

  device = select_device(args.device)
  checkpoint_path = resolve_checkpoint(args.checkpoint)
  image_dir = resolve_input_dir(args.input_dir)
  image_paths = collect_image_paths(image_dir)

  output_dir = Path(args.output_dir).expanduser()
  if not output_dir.is_absolute():
    output_dir = (SCRIPT_DIR / output_dir).resolve()
  output_dir.mkdir(parents=True, exist_ok=True)

  print(f'Using device: {device}')
  print(f'Using checkpoint: {checkpoint_path}')
  print(f'Reading images from: {image_dir}')
  print(f'Writing outputs to: {output_dir}')
  print(f'Tile size: {args.tile_size}, overlap: {args.tile_overlap}, pad multiple: {args.pad_multiple}')

  model = build_model(device, checkpoint_path)
  to_tensor = transforms.ToTensor()

  for index, image_path in enumerate(image_paths, start=1):
    with Image.open(image_path) as image:
      hazy_tensor = to_tensor(image.convert('RGB'))

    print(f'[{index}/{len(image_paths)}] Processing {image_path.name}')
    prediction = run_tiled_inference(
      model=model,
      image_tensor=hazy_tensor,
      device=device,
      tile_size=args.tile_size,
      tile_overlap=args.tile_overlap,
      pad_multiple=args.pad_multiple,
      use_amp=args.amp,
    )

    output_path = output_dir / f'{image_path.stem}{args.save_ext}'
    save_prediction(prediction, output_path)

  print(f'Inference complete. Saved {len(image_paths)} image(s).')


if __name__ == '__main__':
  main()
