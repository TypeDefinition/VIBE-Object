from lib.dataset.inference import Inference
from torch.utils.data import DataLoader
import torch
import numpy as np
import os
import joblib

from lib.models.vibe import VIBE_Demo
from lib.data_utils.kp_utils import convert_kps
from lib.utils.smooth_pose import smooth_pose

from lib.utils.demo_utils import (
    smplify_runner,
    convert_crop_coords_to_orig_img,
    convert_crop_cam_to_orig_img,
    video_to_images,
    download_ckpt,
)

class PreProcessPersonData:
    def __init__(self, _bboxes, _joints2D, _frames, _id):
        self.bboxes = _bboxes
        self.joints2D = _joints2D
        self.frames = _frames
        self.id = _id


class VidSMPLParamCreator:
    def __init__(self, _vidFilePath, _vibeConfigs):
        self.vidFilePath = _vidFilePath
        self.vibeConfigs = _vibeConfigs

        self.device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')
        self.image_folder, num_frames, img_shape = video_to_images(self.vidFilePath, return_info=True)
        self.orig_height, self.orig_width = img_shape[:2]
        self.bbox_scale = 1.1

        self.model = VIBE_Demo(
            seqlen=16,
            n_layers=2,
            hidden_size=1024,
            add_linear=True,
            use_residual=True,
        ).to(self.device)

        # ========= Load pretrained weights ========= #
        pretrained_file = download_ckpt(use_3dpw=False)
        ckpt = torch.load(pretrained_file, map_location='cpu')
        print(f'Performance of pretrained model on 3DPW: {ckpt["performance"]}')
        ckpt = ckpt['gen_state_dict']
        self.model.load_state_dict(ckpt, strict=False)
        self.model.eval()
        print(f'Loaded pretrained weights from \"{pretrained_file}\"')

    '''
        Arguments:
            _people: [{
                bboxes = [[cx, cy, h, w] ...]
                joints2D = []
                frames = [frame number the person appears in]
                id = person identification number
            }...] Array of people objects
            _outputPath: file path to dump the pkl file

        return:
            vibe_results (results of pkl file): dictionary 
                {person id: {
                    'pred_cam': pred_cam,
                    'orig_cam': orig_cam,
                    'verts': pred_verts,
                    'pose': pred_pose,
                    'betas': pred_betas,
                    'joints3d': pred_joints3d,
                    'joints2d': _joints2D,
                    'joints2d_img_coord': joints2d_img_coord,
                    'bboxes': bboxes,
                    'frame_ids': frames,
                }...}
    '''
    def processPeopleInVid(self, _people, _outputPath):
        vibe_results = {}
        for person in _people:
            vibe_results[person.id] = self.createVidPersonParams(person.bboxes, person.joints2D, person.frames)
    
        # print(f'VIBE FPS: {fps:.2f}')
        # total_time = time.time() - total_time
        # print(f'Total time spent: {total_time:.2f} seconds (including model loading time).')
        # print(f'Total FPS (including model loading time): {num_frames / total_time:.2f}.')

        # print(f'Saving output results to \"{os.path.join(output_path, "vibe_output.pkl")}\".')

        joblib.dump(vibe_results, os.path.join(_outputPath, "vibe_output.pkl"))

        return vibe_results
    

    '''
        This is PER person
        params:
            _bboxes: all bounding boxes of the human throughout video
            _joints2D: all joints2D detected of the human throughout the video
            _frames: all detected frame number
    '''
    def createVidPersonParams(self, _bboxes = None, _joints2D = None, _frames = []):
        dataset = Inference(
            image_folder= self.image_folder,
            frames= _frames,
            bboxes= _bboxes,
            joints2d= _joints2D,
            scale= self.bbox_scale,
        )

        bboxes = dataset.bboxes
        frames = dataset.frames
        has_keypoints = True if _joints2D is not None else False

        dataloader = DataLoader(dataset, batch_size=self.vibeConfigs["batch_size"], num_workers=16)

        with torch.no_grad():
            pred_cam, pred_verts, pred_pose, pred_betas, pred_joints3d, smpl_joints2d, norm_joints2d = [], [], [], [], [], [], []

            for batch in dataloader:
                if has_keypoints:
                    batch, nj2d = batch
                    norm_joints2d.append(nj2d.numpy().reshape(-1, 21, 3))

                batch = batch.unsqueeze(0)
                batch = batch.to(self.device)

                batch_size, seqlen = batch.shape[:2]
                output = self.model(batch)[-1]

                pred_cam.append(output['theta'][:, :, :3].reshape(batch_size * seqlen, -1))
                pred_verts.append(output['verts'].reshape(batch_size * seqlen, -1, 3))
                pred_pose.append(output['theta'][:,:,3:75].reshape(batch_size * seqlen, -1))
                pred_betas.append(output['theta'][:, :,75:].reshape(batch_size * seqlen, -1))
                pred_joints3d.append(output['kp_3d'].reshape(batch_size * seqlen, -1, 3))
                smpl_joints2d.append(output['kp_2d'].reshape(batch_size * seqlen, -1, 2))

            pred_cam = torch.cat(pred_cam, dim=0)
            pred_verts = torch.cat(pred_verts, dim=0)
            pred_pose = torch.cat(pred_pose, dim=0)
            pred_betas = torch.cat(pred_betas, dim=0)
            pred_joints3d = torch.cat(pred_joints3d, dim=0)
            smpl_joints2d = torch.cat(smpl_joints2d, dim=0)
            del batch

        # ========= [Optional] run Temporal SMPLify to refine the results ========= #
        if self.vibeConfigs["runSimplify"] and has_keypoints:
            norm_joints2d = np.concatenate(norm_joints2d, axis=0)
            norm_joints2d = convert_kps(norm_joints2d, src='staf', dst='spin')
            norm_joints2d = torch.from_numpy(norm_joints2d).float().to(self.device)

            # Run Temporal SMPLify
            update, new_opt_vertices, new_opt_cam, new_opt_pose, new_opt_betas, \
            new_opt_joints3d, new_opt_joint_loss, opt_joint_loss = smplify_runner(
                pred_rotmat=pred_pose,
                pred_betas=pred_betas,
                pred_cam=pred_cam,
                j2d=norm_joints2d,
                device= self.device,
                batch_size=norm_joints2d.shape[0],
                pose2aa=False,
            )

            # update the parameters after refinement
            print(f'Update ratio after Temporal SMPLify: {update.sum()} / {norm_joints2d.shape[0]}')
            pred_verts = pred_verts.cpu()
            pred_cam = pred_cam.cpu()
            pred_pose = pred_pose.cpu()
            pred_betas = pred_betas.cpu()
            pred_joints3d = pred_joints3d.cpu()
            pred_verts[update] = new_opt_vertices[update]
            pred_cam[update] = new_opt_cam[update]
            pred_pose[update] = new_opt_pose[update]
            pred_betas[update] = new_opt_betas[update]
            pred_joints3d[update] = new_opt_joints3d[update]

        elif self.vibeConfigs["runSimplify"] and not has_keypoints:
            print('[WARNING] You need to enable pose tracking to run Temporal SMPLify algorithm!')
            print('[WARNING] Continuing without running Temporal SMPLify!..')

        # ========= Save results to a pickle file ========= #
        pred_cam = pred_cam.cpu().numpy()
        pred_verts = pred_verts.cpu().numpy()
        pred_pose = pred_pose.cpu().numpy()
        pred_betas = pred_betas.cpu().numpy()
        pred_joints3d = pred_joints3d.cpu().numpy() # 300 frames, 49 bones, 3 floats each. Uses spin_joint_names.
        smpl_joints2d = smpl_joints2d.cpu().numpy()

        # Runs 1 Euro Filter to smooth out the results
        if self.vibeConfigs["smooth_results"]:
            min_cutoff = self.vibeConfigs["smooth_min_cutoff"] # 0.004
            beta = self.vibeConfigs["smooth_beta"] # 1.5
            print(f'Running smoothing on person min_cutoff: {min_cutoff}, beta: {beta}')

        # [VIBE-Object]
        # Over here, the joints are smoothed out.
        pred_verts, pred_pose, pred_joints3d = smooth_pose(pred_pose, pred_betas,
                                                               min_cutoff=min_cutoff, beta=beta)

        orig_cam = convert_crop_cam_to_orig_img(
            cam=pred_cam,
            bbox=bboxes,
            img_width= self.orig_width,
            img_height= self.orig_height
        )

        joints2d_img_coord = convert_crop_coords_to_orig_img(
            bbox=bboxes,
            keypoints=smpl_joints2d,
            crop_size=224,
        )

        output_dict = {
            'pred_cam': pred_cam,
            'orig_cam': orig_cam,
            'verts': pred_verts,
            'pose': pred_pose,
            'betas': pred_betas,
            'joints3d': pred_joints3d,
            'joints2d': _joints2D,
            'joints2d_img_coord': joints2d_img_coord,
            'bboxes': bboxes,
            'frame_ids': frames,
        }

        return output_dict