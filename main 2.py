import sys
import argparse
import cv2
from lib.preprocess import h36m_coco_format, revise_kpts
from lib.hrnet.helper import hrnet_pose   # gen_video_kpts
import os
import numpy as np
import torch
import torch.nn as nn
import glob
from tqdm import tqdm
import copy
import time
import json

sys.path.append(os.getcwd())
from demo.lib.utils import normalize_screen_coordinates, camera_to_world
from model.MotionAGFormer import MotionAGFormer

import matplotlib
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import matplotlib.gridspec as gridspec

plt.switch_backend('agg')
matplotlib.rcParams['pdf.fonttype'] = 42
matplotlib.rcParams['ps.fonttype'] = 42

def show2Dpose(kps, img):
    """
    Draws 2D pose connections on an image.

    Parameters:
    kps: ndarray of shape (1, 1, 17, 3)
        Keypoints with batch, person, joint, and (x, y, confidence).
    img: ndarray of shape (H, W, 3)
        The image to draw on.

    Returns:
    img: ndarray
        The image with the pose overlay.
    """
    # Connections between keypoints
    connections = [[0, 1], [1, 2], [2, 3], [0, 4], [4, 5],
                   [5, 6], [0, 7], [7, 8], [8, 9], [9, 10],
                   [8, 11], [11, 12], [12, 13], [8, 14], [14, 15], [15, 16]]
    
    # Left/right marker for colors
    LR = np.array([0, 0, 0, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 0, 0, 0], dtype=bool)
    lcolor = (255, 0, 0)  # Blue for left
    rcolor = (0, 0, 255)  # Red for right
    thickness = 3

    # Extract the first person from the batch (assuming only one person)
    keypoints = kps[0, 0, :, :2]  # Shape: (17, 2), ignoring confidence

    for j, c in enumerate(connections):
        # Start and end points of the connection
        start = tuple(map(int, keypoints[c[0]]))
        end = tuple(map(int, keypoints[c[1]]))
        
        # Draw connection line
        cv2.line(img, start, end, lcolor if LR[j] else rcolor, thickness)
        
        # Draw keypoints as circles
        cv2.circle(img, start, radius=3, color=(0, 255, 0), thickness=-1)
        cv2.circle(img, end, radius=3, color=(0, 255, 0), thickness=-1)
    
    return img


def show3Dpose(vals, ax):
    ax.view_init(elev=15., azim=70)

    lcolor = (0, 0, 1)
    rcolor = (1, 0, 0)

    I = np.array([0, 0, 1, 4, 2, 5, 0, 7, 8, 8, 14, 15, 11, 12, 8, 9])
    J = np.array([1, 4, 2, 5, 3, 6, 7, 8, 14, 11, 15, 16, 12, 13, 9, 10])

    LR = np.array([0, 1, 0, 1, 0, 1, 0, 0, 0, 1, 0, 0, 1, 1, 0, 0], dtype=bool)

    for i in np.arange(len(I)):
        x, y, z = [np.array([vals[I[i], j], vals[J[i], j]]) for j in range(3)]
        ax.plot(x, y, z, lw=2, color=lcolor if LR[i] else rcolor)

    RADIUS = 0.72
    RADIUS_Z = 0.7

    xroot, yroot, zroot = vals[0, 0], vals[0, 1], vals[0, 2]
    ax.set_xlim3d([-RADIUS + xroot, RADIUS + xroot])
    ax.set_ylim3d([-RADIUS + yroot, RADIUS + yroot])
    ax.set_zlim3d([-RADIUS_Z + zroot, RADIUS_Z + zroot])
    ax.set_aspect('auto')

    white = (1.0, 1.0, 1.0, 0.0)
    ax.xaxis.set_pane_color(white)
    ax.yaxis.set_pane_color(white)
    ax.zaxis.set_pane_color(white)

    ax.tick_params('x', labelbottom=False)
    ax.tick_params('y', labelleft=False)
    ax.tick_params('z', labelleft=False)

def get_pose2D(frame, output_dir):
    print("Generating 2D pose...")
    current_time = time.time()
    keypoints, scores = hrnet_pose(frame, det_dim=416, num_peroson=1, gen_output=True)
    # Check if keypoints or scores are empty
    if keypoints.size == 0 or scores.size == 0:
        print("No keypoints detected. Saving placeholder 2D keypoints.")
        keypoints = np.zeros((1, 1, 17, 3))  # Placeholder with shape (1, T, 17, 3) for compatibility
        valid_frames = [False]  # Mark the frame as invalid
        current_time = 0
    else:
        # Convert keypoints and scores to desired format
        keypoints, scores, valid_frames = h36m_coco_format(keypoints, scores)
        # single_img = keypoints.copy().squeeze()
        # single_img_path = output_dir + 'single_img.npy'
        # np.save(single_img_path, single_img)
        keypoints = np.concatenate((keypoints, scores[..., None]), axis=-1)

    # Ensure output directory exists
    output_dir += 'input_2D/'
    os.makedirs(output_dir, exist_ok=True)

    # Save the keypoints as a compressed file
    output_npz = output_dir + 'keypoints.npz'
    np.savez_compressed(output_npz, reconstruction=keypoints)
    print("--------------2D Pose Generation Complete--------------")
    
    return current_time


def merge_img(frame, output_dir):
    pose_dir = os.path.join(output_dir, 'mypose2D/')
    img_2d = os.path.join(pose_dir, 'img_2D.png')

    if not img_2d:
        print(f"Error: No images found in {pose_dir}.")
        return

    # Get the first image from the pose directory to determine size
    img = cv2.imread(img_2d)
    if img is None:
        print(f"Error: Unable to read the first image {img_2d}.")
        return

    # Ensure the pose images match the frame's size
    frame_height, frame_width = 240, 320
    
    img = cv2.resize(img, (frame_width, frame_height))
    frame = cv2.resize(frame, (frame_width, frame_height))

    size = (frame_width * 2, frame_height)  # Combined image size (side-by-side)

    # Concatenate the frame and pose image horizontally
    combined_img = cv2.hconcat([frame, img])

    return combined_img


def resample(n_frames):
    even = np.linspace(0, n_frames, num=243, endpoint=False)
    result = np.floor(even)
    result = np.clip(result, a_min=0, a_max=n_frames - 1).astype(np.uint32)
    return result

def turn_into_clips(keypoints):
    clips = []
    n_frames = keypoints.shape[1]
    if n_frames <= 243:
        new_indices = resample(n_frames)
        clips.append(keypoints[:, new_indices, ...])
        downsample = np.unique(new_indices, return_index=True)[1]
    else:
        for start_idx in range(0, n_frames, 243):
            keypoints_clip = keypoints[:, start_idx:start_idx + 243, ...]
            clip_length = keypoints_clip.shape[1]
            if clip_length != 243:
                new_indices = resample(clip_length)
                clips.append(keypoints_clip[:, new_indices, ...])
                downsample = np.unique(new_indices, return_index=True)[1]
            else:
                clips.append(keypoints_clip)
    return clips, downsample

@torch.no_grad()
def get_pose3D(img, output_dir):
    args, _ = argparse.ArgumentParser().parse_known_args()
    args.n_layers, args.dim_in, args.dim_feat, args.dim_rep, args.dim_out = 16, 3, 128, 512, 3
    args.mlp_ratio, args.act_layer = 4, nn.GELU
    args.attn_drop, args.drop, args.drop_path = 0.0, 0.0, 0.0
    args.use_layer_scale, args.layer_scale_init_value, args.use_adaptive_fusion = True, 0.00001, True
    args.num_heads, args.qkv_bias, args.qkv_scale = 8, False, None
    args.hierarchical = False
    args.use_temporal_similarity, args.neighbour_num, args.temporal_connection_len = True, 2, 1
    args.use_tcn, args.graph_only = False, False
    args.n_frames = 243
    args = vars(args)

    device = torch.device("cpu")

    ## Reload 
    model = nn.DataParallel(MotionAGFormer(**args)).to(device)

    # Put the pretrained model of MotionAGFormer in 'checkpoint/'
    model_path = sorted(glob.glob(os.path.join('checkpoint', 'motionagformer-b-h36m.pth.tr')))[0]

    pre_dict = torch.load(model_path, map_location=device)
    model.load_state_dict(pre_dict['model'], strict=True)

    model.eval()

    ## input
    keypoints = np.load(output_dir + 'input_2D/keypoints.npz', allow_pickle=True)['reconstruction']
    clips, downsample = turn_into_clips(keypoints)

    img_size = img.shape

    input_2D = keypoints

    image = show2Dpose(input_2D, copy.deepcopy(img))

    output_dir_2D = output_dir +'mypose2D/'
    os.makedirs(output_dir_2D, exist_ok=True)
    cv2.imwrite(output_dir_2D + 'img_2D.png', image)

    
    print('\nGenerating 3D pose...')
        
    if np.all(keypoints == 0):
        print("No valid 3D keypoints found. Showing 3d blank image.")
        placeholder = np.zeros((240, 320, 3), dtype=np.uint8)
        return placeholder
    
    for idx, clip in enumerate(clips):
        input_2D = normalize_screen_coordinates(clip, w=img_size[1], h=img_size[0]) 
        input_2D_aug = flip_data(input_2D)
        
        input_2D = torch.from_numpy(input_2D.astype('float32')).to(device)
        input_2D_aug = torch.from_numpy(input_2D_aug.astype('float32')).to(device)

        output_3D_non_flip = model(input_2D) 
        output_3D_flip = flip_data(model(input_2D_aug))
        output_3D = (output_3D_non_flip + output_3D_flip) / 2

        if idx == len(clips) - 1:
            output_3D = output_3D[:, downsample]

        output_3D[:, :, 0, :] = 0
        post_out_all = output_3D[0].cpu().detach().numpy()
        
        for j, post_out in enumerate(post_out_all):
            rot =  [0.1407056450843811, -0.1500701755285263, -0.755240797996521, 0.6223280429840088]
            rot = np.array(rot, dtype='float32')
            post_out = camera_to_world(post_out, R=rot, t=0)
            post_out[:, 2] -= np.min(post_out[:, 2])
            max_value = np.max(post_out)
            post_out /= max_value

            fig = plt.figure(figsize=(9.6, 5.4))
            # print(fig)
            gs = gridspec.GridSpec(1, 1)
            gs.update(wspace=-0.00, hspace=0.05) 
            ax = plt.subplot(gs[0], projection='3d')
            
            
            # Prepare the JSON structure
            json_structure = {
                "Human 3d pose data": {
                    "Human pose data": {},
                    "3d pose axis data": {
                        "X-axis coordinate": post_out[:, 0].tolist(),
                        "Y-axis coordinate": post_out[:, 1].tolist(),
                        "Z-axis coordinate": post_out[:, 2].tolist()
                    }
                }
            }

            # Convert to JSON string
            json_string = json.dumps(json_structure, indent=4)

            # Print the JSON string
            print(json_string)
            
            
            show3Dpose(post_out, ax)

            output_dir_3D = output_dir +'mypose3D/'
            os.makedirs(output_dir_3D, exist_ok=True)
            str(('%04d'% (idx * 243 + j)))
            plt.savefig(output_dir_3D + 'img_3D.png', dpi=200, format='png', bbox_inches='tight')
            plt.close(fig)

    path_3d = os.path.join(output_dir_3D,'img_3D.png')
    pose_3d = cv2.imread(path_3d)
    
    frame_height, frame_width = 240, 320
    pose_3d = cv2.resize(pose_3d, (frame_width, frame_height))
    print('----------------Generating 3D pose successful!----------------')
    
    return pose_3d

    
def flip_data(data, left_joints=[1, 2, 3, 14, 15, 16], right_joints=[4, 5, 6, 11, 12, 13]):
    """
    data: [N, F, 17, D] or [F, 17, D]
    """
    flipped_data = copy.deepcopy(data)
    flipped_data[..., 0] *= -1  # flip x of all joints
    flipped_data[..., left_joints + right_joints, :] = flipped_data[..., right_joints + left_joints, :]  # Change orders
    return flipped_data

def showimage(ax, img):
    ax.set_xticks([])
    ax.set_yticks([]) 
    plt.axis('off')
    ax.imshow(img)



