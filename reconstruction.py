import os
from tqdm import tqdm
import torch
from torch.utils.data import DataLoader
from logger import Logger, Visualizer
import numpy as np
import imageio
from sync_batchnorm import DataParallelWithCallback


def reconstruction(config, generator, kp_detector, checkpoint, log_dir, dataset):
    png_dir = os.path.join(log_dir, 'reconstruction/png')
    log_dir = os.path.join(log_dir, 'reconstruction')

    if checkpoint is not None:
        Logger.load_cpk(checkpoint, generator=generator, kp_detector=kp_detector)
    else:
        raise AttributeError("Checkpoint should be specified for mode='reconstruction'.")
    dataloader = DataLoader(dataset, batch_size=1, shuffle=False, num_workers=1)

    if not os.path.exists(log_dir):
        os.makedirs(log_dir)

    if not os.path.exists(png_dir):
        os.makedirs(png_dir)

    loss_list = []
    if torch.cuda.is_available():
        generator = DataParallelWithCallback(generator)
        kp_detector = DataParallelWithCallback(kp_detector)

    generator.eval()
    kp_detector.eval()
    recon_gen_dir = os.path.join(log_dir, 'recon_gen')
    os.makedirs(recon_gen_dir, exist_ok=False)  #

    for it, x in tqdm(enumerate(dataloader)):
        if config['reconstruction_params']['num_videos'] is not None:
            if it > config['reconstruction_params']['num_videos']:
                break
        with torch.no_grad():
            predictions = []
            visualizations = []
            if torch.cuda.is_available():
                x['video'] = x['video'].cuda()
            kp_source = kp_detector(x['video'][:, :, 0])

            video_gen_dir = os.path.join(recon_gen_dir, x['name'][0])
            os.makedirs(video_gen_dir, exist_ok=False)  #

            for frame_idx in range(x['video'].shape[2]):
                source = x['video'][:, :, 0]
                driving = x['video'][:, :, frame_idx]
                kp_driving = kp_detector(driving)
                out = generator(source, kp_source=kp_source, kp_driving=kp_driving)
                out['kp_source'] = kp_source
                out['kp_driving'] = kp_driving
                predictions.append(np.transpose(out['upscaled_prediction'].data.cpu().numpy(), [0, 2, 3, 1])[0])

                visualization = Visualizer(**config['visualizer_params']).visualize(source=source,
                                                                                    driving=driving, out=out)
                visualizations.append(visualization)

                loss_list.append(torch.abs(out['upscaled_prediction'] - driving).mean().cpu().numpy())

                frame_name = str(frame_idx).zfill(7) + '.png'
                upscaled_prediction = out['upscaled_prediction'].data.cpu().numpy()
                upscaled_prediction = np.transpose(upscaled_prediction, [0, 2, 3, 1])
                upscaled_prediction = (255 * upscaled_prediction).astype(np.uint8)
                imageio.imsave(os.path.join(video_gen_dir, frame_name), upscaled_prediction[0])

            predictions = np.concatenate(predictions, axis=1)
            imageio.imsave(os.path.join(png_dir, x['name'][0] + '.png'), (255 * predictions).astype(np.uint8))

            image_name = x['name'][0] + config['reconstruction_params']['format']
            imageio.mimsave(os.path.join(log_dir, image_name), visualizations)

    print(f'Reconstruction loss: {np.mean(loss_list)}')
