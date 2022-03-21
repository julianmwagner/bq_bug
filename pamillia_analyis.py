import numpy as np
import pandas as pd

import re
import os.path
import os
from pathlib import Path
import glob
import tqdm
import tqdm.notebook

import cv2
import skimage
import skimage.io
import skimage.util
import skimage.morphology


def img_to_rgb(img):
    img = skimage.img_as_float32(img)*254
    return(skimage.color.gray2rgb(img).astype(np.uint8))

def ant_blobs_dists(metadata, bg_ims, bg_im_dict, thresh = 0.15, ind=15, ine=(15000,), min_size=1000, x1_crop=120, x2_crop=1225, ball_rads=(20, 30, 30, 30, 20),
                    buffer_spme=20, frame_rate=25, parts = ['bh', 'bt1', 'bt2', 'ba1', 'ba2'], likelihood_thresh = 0.9, give_ims=False, save_vid=False,
                    vid_name="~/Downloads/close_vid.mp4", network='DLC_resnet50_pamillia_onlyAug5shuffle1_250000', erode_amount=10, dilate_amount=10, return_ims=True, part_shift_thresh=100, vid_col='video_file_local'):
    
    balls=[skimage.morphology.disk(b_rad) for b_rad in ball_rads]
    
    file = metadata.iloc[ind][vid_col]
    file_an_out = re.sub('.mp4', '_dist_im_an.csv', file)
    df_dlc = pd.read_hdf(metadata.iloc[ind]['DLC_file'])
    
    for i in range(len(parts)):
        df_dlc[(network, parts[i], 'part_dist')] = 0
        all_dis = []
        for j in range(len(parts)):
            if i == j:
                continue
            p_dis = np.sqrt(np.square(df_dlc[(network, parts[i], 'x')] -
                                 df_dlc[(network, parts[j], 'x')]) + 
                       np.square(df_dlc[(network, parts[i], 'y')] -
                                 df_dlc[(network, parts[j], 'y')]) )
            df_dlc[(network, parts[i], 'part_dist' + parts[i])] = p_dis
            all_dis.append(p_dis)
        df_dlc[(network, parts[i], 'part_dist')] = np.percentile(np.vstack(tup=tuple(all_dis)).T, axis=1, q=100/len(parts)-1, interpolation='nearest')    
    
    start_time = metadata.iloc[ind]['SPME_start_time'] + buffer_spme
    stop_time = metadata.iloc[ind]['SPME_stop_time'] - buffer_spme

    cap = cv2.VideoCapture(file)
    
    _, f1 = cap.read()
    
    if save_vid:
        x1, x2, y1, y2 = (0, x2_crop-x1_crop, 0, f1.shape[0])
        file_out = vid_name
        fourcc = cv2.VideoWriter_fourcc(*'mp4v')
        vid_out = cv2.VideoWriter(file_out, fourcc, 25, (x2-x1,y2-y1), True)
    
    all_min_dists = []
    all_min_dists_points = []
    all_ims = []
    for frame_ind in tqdm.tqdm(ine):
        x_points = []
        y_points = []
        for part in parts:
            if ((df_dlc.iloc[frame_ind][(network, part, 'likelihood')] > likelihood_thresh) and
                (df_dlc.iloc[frame_ind][(network, part, 'x')]>x1_crop ) and
                (df_dlc.iloc[frame_ind][(network, part, 'x')]< x2_crop) and
                (df_dlc.iloc[frame_ind][(network, part, 'part_dist')]<part_shift_thresh)):
                
                x_points.append(df_dlc.iloc[frame_ind][(network, part, 'x')])
                y_points.append(df_dlc.iloc[frame_ind][(network, part, 'y')])
        x_points = np.array(x_points)
        y_points = np.array(y_points)
        
        cap.set(1, frame_ind)
        ret, frame = cap.read()
        frame = skimage.img_as_float32(frame[:, :, 0])
        im = cv2.subtract(frame, skimage.img_as_float32(bg_ims[bg_im_dict[metadata.iloc[ind]['video_file']]][0]))[:, x1_crop:x2_crop]
        
        ball_pos = np.zeros(im.shape)
        
        for xp, yp, b in zip(x_points-x1_crop, y_points, balls):
            c1, c2 = (int(yp)-int(np.floor(b.shape[0]/2)), int(yp)+int(np.ceil(b.shape[0]/2)))
            c3, c4 = (int(xp)-int(np.floor(b.shape[1]/2)), int(xp)+int(np.ceil(b.shape[1]/2)))
            ball_c1 = np.abs(min(c1, 0))
            ball_c2 = b.shape[0]+min(im.shape[0]-c2, 0)
            ball_c3 = np.abs(min(c3, 0))
            ball_c4 = b.shape[0]+min(im.shape[1]-c4, 0)
            c1 = max(c1, 0)
            c3 = max(c3, 0)
            
            im[c1:c2, c3:c4] = np.multiply(im[c1:c2, c3:c4], 1-b[ball_c1:ball_c2, ball_c3:ball_c4])
            ball_pos[c1:c2, c3:c4] += b[ball_c1:ball_c2, ball_c3:ball_c4]

        im_bw = im > thresh
        im_bw = skimage.morphology.remove_small_objects(im_bw, min_size=min_size)
        im_bw_small = skimage.morphology.erosion(im_bw, skimage.morphology.disk(erode_amount))
        im_bw_small = skimage.morphology.dilation(im_bw_small, skimage.morphology.disk(dilate_amount))
        points = np.array(np.where(im_bw_small)).T
        dist = np.inf
        m_ind = 0
        b_ind = 0
        for xb, yb, bi in zip((x_points-x1_crop), y_points, range(len(y_points))):
            dists = np.sqrt(np.square(points[:, 1] - xb) + np.square(points[:, 0] - yb))
            if dists.min() < dist:
                b_ind = bi
                dist = dists.min()
                m_ind = dists.argmin()
        all_min_dists.append(dist)
        all_min_dists_points.append(points[m_ind])
        if give_ims:
            c_im = frame[:, x1_crop:x2_crop].copy()
            if len(x_points) > 0:
                for line_pos in (0, -1, -2, 1, 2, -3, 3, 4, -4):
                    rr, cc = skimage.draw.line(int(y_points[b_ind]) + line_pos,
                                               int((x_points-x1_crop)[b_ind]) + line_pos,
                                               int(points[:, 0][m_ind]) + line_pos,
                                               int(points[:, 1][m_ind]) + line_pos)
                    try:
                        c_im[rr, cc] = 1
                    except:
                        pass

                line_pos=0
                rr, cc = skimage.draw.line(int(y_points[b_ind]) + line_pos,
                                           int((x_points-x1_crop)[b_ind]) + line_pos,
                                           int(points[:, 0][m_ind]) + line_pos,
                                           int(points[:, 1][m_ind]) + line_pos)
                c_im[rr, cc] = 0
        if return_ims:
            all_ims.append(c_im)
        if save_vid:
            color_im = img_to_rgb(c_im)
            color_im[im_bw_small, 2] = 1
            color_im[ball_pos>0, 1] = 1
            vid_out.write(color_im)
    
    print("Finished analysis for "+file)
    df_labs = pd.DataFrame({'frame_inds':ine, 'all_dists':all_min_dists, 'blob_points_x': np.array(all_min_dists_points)[:,0], 'blob_points_y': np.array(all_min_dists_points)[:,1]}).to_csv(file_an_out)
    if save_vid:
        vid_out.release()
    if return_ims:
        return(all_min_dists, all_min_dists_points, all_ims)
    else: 
        return(all_min_dists, all_min_dists_points)

