import os
from tqdm import tqdm
import torch
from torch.utils.data import DataLoader
from frames_dataset import PairedDataset
from logger import Logger, Visualizer
import imageio
from scipy.spatial import ConvexHull
import numpy as np
from sync_batchnorm import DataParallelWithCallback
from modules.keypoint_detector import draw_kp, norm_mask
import torch.nn.functional as F


def normalize_kp(kp_source, kp_driving, kp_driving_initial, adapt_movement_scale=False,
                 use_relative_movement=False, use_relative_jacobian=False):
    if adapt_movement_scale:
        source_area = ConvexHull(kp_source['value'][0].data.cpu().numpy()).volume
        driving_area = ConvexHull(kp_driving_initial['value'][0].data.cpu().numpy()).volume
        adapt_movement_scale = np.sqrt(source_area) / np.sqrt(driving_area)
    else:
        adapt_movement_scale = 1

    kp_new = {k: v for k, v in kp_driving.items()}

    if use_relative_movement:
        kp_value_diff = (kp_driving['value'] - kp_driving_initial['value'])
        kp_value_diff *= adapt_movement_scale
        kp_new['value'] = kp_value_diff + kp_source['value']

        if use_relative_jacobian:
            jacobian_diff = torch.matmul(kp_driving['jacobian'], torch.inverse(kp_driving_initial['jacobian']))
            kp_new['jacobian'] = torch.matmul(jacobian_diff, kp_source['jacobian'])

    return kp_new


def animate(config, generator, kp_detector, checkpoint, log_dir, dataset, kp_after_softmax):
    log_dir = os.path.join(log_dir, 'animation')
    png_dir = os.path.join(log_dir, 'png')
    animate_params = config['animate_params']
    frame_size = config['dataset_params']['frame_shape'][0]
    latent_size = int(config['model_params']['common_params']['scale_factor'] * frame_size)
    dataset = PairedDataset(initial_dataset=dataset, number_of_pairs=animate_params['num_pairs'])
    dataloader = DataLoader(dataset, batch_size=1, shuffle=False, num_workers=1)

    if checkpoint is not None:
        Logger.load_cpk(checkpoint, generator=generator, kp_detector=kp_detector)
    else:
        raise AttributeError("Checkpoint should be specified for mode='animate'.")

    if not os.path.exists(log_dir):
        os.makedirs(log_dir)

    if not os.path.exists(png_dir):
        os.makedirs(png_dir)

    if torch.cuda.is_available():
        generator = DataParallelWithCallback(generator)
        kp_detector = DataParallelWithCallback(kp_detector)

    generator.eval()
    kp_detector.eval()

    for it, x in tqdm(enumerate(dataloader)):
        with torch.no_grad():
            predictions = []
            visualizations = []

            driving_video = x['driving_video']
            source_frame = x['source_video'][:, :, 0, :, :]

            kp_source = kp_detector(source_frame)
            kp_driving_initial = kp_detector(driving_video[:, :, 0])

            for frame_idx in range(driving_video.shape[2]):
                driving_frame = driving_video[:, :, frame_idx]
                kp_driving = kp_detector(driving_frame)

                if kp_after_softmax:
                    kp_norm = normalize_kp(kp_source=kp_source, kp_driving=kp_driving,
                                           kp_driving_initial=kp_driving_initial,
                                           **animate_params['normalization_params'])
                    kp_source = draw_kp([latent_size, latent_size], kp_source)
                    kp_norm = draw_kp([latent_size, latent_size], kp_norm)
                    kp_source = norm_mask(latent_size, kp_source)
                    kp_norm = norm_mask(latent_size, kp_norm)
                    out = generator(source_frame, kp_source=kp_source, kp_driving=kp_norm)
                    kp_norm_int = F.interpolate(kp_norm, size=source_frame.shape[2:], mode='bilinear',
                                                align_corners=False)
                    out['kp_norm_int'] = kp_norm_int.repeat(1, 3, 1, 1)
                else:
                    out = generator(source_frame, kp_source=kp_source, kp_driving=kp_driving)

                out['kp_driving'] = kp_driving
                out['kp_source'] = kp_source

                predictions.append(np.transpose(out['low_res_prediction'].data.cpu().numpy(), [0, 2, 3, 1])[0])
                predictions.append(np.transpose(out['upscaled_prediction'].data.cpu().numpy(), [0, 2, 3, 1])[0])

                visualization = Visualizer(**config['visualizer_params']).visualize(source=source_frame,
                                                                                    driving=driving_frame, out=out)
                visualization = visualization
                visualizations.append(visualization)

            predictions = np.concatenate(predictions, axis=1)
            result_name = "-".join([x['driving_name'][0], x['source_name'][0]])
            imageio.imsave(os.path.join(png_dir, result_name + '.png'), (255 * predictions).astype(np.uint8))

            image_name = result_name + animate_params['format']
            imageio.mimsave(os.path.join(log_dir, image_name), visualizations)
