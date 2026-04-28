import cv2
import kornia
import torch
from kornia.geometry.transform import get_rotation_matrix2d, warp_affine


def resize_frame(frame, target_height: int, target_width: int):
    if frame.shape[0] != target_height or frame.shape[1] != target_width:
        frame = cv2.resize(frame, (target_width, target_height))
    return frame


def augment_with_camera_movement(
    frames_tensor: torch.Tensor,
    max_rotation: float = 1.2,
    max_translation: float = 15.0,
) -> torch.Tensor:
    n_frames, _, height, width = frames_tensor.shape
    orig_dtype = frames_tensor.dtype
    if frames_tensor.max() > 1.0:
        frames_tensor = frames_tensor.float() / 255.0

    frame_numbers = torch.arange(
        n_frames, dtype=torch.float32, device=frames_tensor.device
    )
    rotations = max_rotation * torch.sin(2 * torch.pi * frame_numbers / n_frames)
    translations_x = max_translation * torch.sin(
        2 * torch.pi * frame_numbers / n_frames
    )
    translations_y = max_translation * torch.cos(
        2 * torch.pi * frame_numbers / n_frames
    )

    center = torch.tensor([width / 2, height / 2], device=frames_tensor.device)
    center = center.unsqueeze(0).repeat(n_frames, 1)
    scales = torch.ones(n_frames, 2, device=frames_tensor.device)
    matrices = get_rotation_matrix2d(center=center, angle=rotations, scale=scales)
    matrices[:, :, 2] += torch.stack((translations_x, translations_y), dim=1)

    augmented = warp_affine(
        frames_tensor,
        matrices,
        dsize=(height, width),
        mode="bilinear",
        padding_mode="zeros",
        align_corners=True,
    ).clamp(0.0, 1.0)
    if orig_dtype == torch.uint8:
        return (augmented * 255).round().to(torch.uint8)
    return augmented


def crop_video(
    frames_tensor: torch.Tensor,
    crop_size_h: int,
    crop_size_w: int,
) -> torch.Tensor:
    n_frames, _, height, width = frames_tensor.shape
    if crop_size_h >= height or crop_size_w >= width:
        raise ValueError("Crop size must be smaller than the original size.")

    device = frames_tensor.device
    orig_dtype = frames_tensor.dtype
    if frames_tensor.dtype == torch.uint8:
        frames_tensor = frames_tensor.float() / 255.0

    x_start = torch.randint(0, width - crop_size_w + 1, (1,), device=device).item()
    y_start = torch.randint(0, height - crop_size_h + 1, (1,), device=device).item()
    src_box = torch.tensor(
        [
            [
                [x_start, y_start],
                [x_start + crop_size_w - 1, y_start],
                [x_start + crop_size_w - 1, y_start + crop_size_h - 1],
                [x_start, y_start + crop_size_h - 1],
            ]
        ],
        dtype=torch.float32,
        device=device,
    ).expand(n_frames, -1, -1)
    dst_box = torch.tensor(
        [[[0, 0], [width - 1, 0], [width - 1, height - 1], [0, height - 1]]],
        dtype=torch.float32,
        device=device,
    ).expand(n_frames, -1, -1)

    cropped = kornia.geometry.transform.crop_by_boxes(
        frames_tensor,
        src_box=src_box,
        dst_box=dst_box,
        mode="bilinear",
        align_corners=False,
    ).clamp(0.0, 1.0)
    if orig_dtype == torch.uint8:
        return (cropped * 255.0).round().to(torch.uint8)
    return cropped
