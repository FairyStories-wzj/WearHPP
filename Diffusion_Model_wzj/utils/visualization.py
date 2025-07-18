"""
This code is adopted from:
https://github.com/wei-mao-2019/gsps/blob/main/motion_pred/utils/visualization.py
"""

"""
把这玩意改成2D的话，就会脱胎换骨，容易改崩了，所以我的选择是在画图的时候给z轴都设置成0
"""
import os
import matplotlib
# matplotlib.use('TkAgg')
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation, writers
from mpl_toolkits.mplot3d import Axes3D
import numpy as np
import pickle
from vpython import *
import time

from datetime import datetime


def render_animation(skeleton, poses_generator, algos, t_hist, fix_0=True, azim=0.0, output=None, mode='pred', size=2, ncol=5,
                     bitrate=3000, fix_index=None):
    """
    Render an animation. The supported output modes are:
     -- 'interactive': display an interactive figure
                       (also works on notebooks if associated with %matplotlib inline)
     -- 'html': render the animation as HTML5 video. Can be displayed in a notebook using HTML(...).
     -- 'filename.mp4': render and export the animation as an h264 video (requires ffmpeg).
     -- 'filename.gif': render and export the animation a gif file (requires imagemagick).
    """
    if mode == 'switch':
        fix_0 = False
    if fix_index is not None:
        fix_list = [
            [1, 2, 3],  #
            [4, 5, 6],
            [7, 8, 9, 10],
            [11, 12, 13],
            [14, 15, 16],
            [1, 2, 3, 4, 5, 6],
            [7, 8, 9, 10, 11, 12, 13, 14, 15, 16]
        ]
        fix_i = fix_list[fix_index]
        fix_col = 'darkblue'
    else:
        fix_i = None
    all_poses, joint_deleted, mean, std = next(poses_generator)  # 修改：有些关节可以删除
    algo = algos[0] if len(algos) > 0 else next(iter(all_poses.keys()))
    t_total = next(iter(all_poses.values())).shape[0]
    poses = dict(filter(lambda x: x[0] in {'gt', 'context'} or algo == x[0].split('_')[0] or x[0].startswith('gt'),
                        all_poses.items()))

    plt.ioff()
    nrow = int(np.ceil(len(poses) / ncol))
    fig = plt.figure(figsize=(size * ncol, size * nrow))
    ax_2d = []
    lines_2d = []
    trajectories = []
    radius = 8
    for index, (title, data) in enumerate(poses.items()):
        # if index >= 2:
        #     ax = fig.add_subplot(nrow, ncol, 3, projection='3d')
        # else:
        #     ax = fig.add_subplot(nrow, ncol, index+1, projection='3d')
        # ax = fig.add_subplot(nrow, ncol, index+1, projection='3d')
        # ax.view_init(elev=15., azim=azim)  修改：三维改二维，改变视角
        ax = fig.add_subplot(nrow, ncol, index + 1)
        # ax.view_init(elev=90, azim=-90)
        # ax.set_xlim3d([-radius / 2, radius / 2])
        # ax.set_zlim3d([0, radius])
        # ax.set_ylim3d([-radius / 2, radius / 2])
        # ax.set_xlim([-radius / 2, radius / 2])
        # ax.set_ylim([-radius / 2, radius / 2])
        ax.set_xlim([0, 1500])
        ax.set_ylim([0, 1500])
        ax.set_xticks([])
        ax.set_yticks([])
        # ax.set_xticklabels([])
        # ax.set_yticklabels([])
        # ax.set_zticklabels([])
        # ax.dist = 5.0
        if mode == 'switch':
            if index == 0:
                ax.set_title('target', y=1.0, fontsize=12)
        if mode == 'pred' or 'fix' in mode or mode == 'control' or mode == 'zero_shot':
            if index == 0 or index == 1:
                ax.set_title(title, y=1.0, fontsize=12)
        ax.set_aspect('equal')  # 保持比例一致
        ax.set_axis_off()
        ax.patch.set_alpha(0.0)
        ax_2d.append(ax)
        lines_2d.append([])
        trajectories.append(data[:, 0, [0, 1]])
    fig.tight_layout(h_pad=15,w_pad=15)
    fig.subplots_adjust(wspace=-0.4, hspace=0.4)
    poses = list(poses.values())

    anim = None
    initialized = False
    animating = True
    find = 0
    hist_lcol, hist_mcol, hist_rcol = 'gray', 'black', 'red'
    pred_lcol, pred_mcol, pred_rcol = 'purple', 'black', 'green'
    tran_lcol, tran_mcol, tran_rcol = 'orange', 'black', 'blue'

    parents = skeleton.parents()

    # scatter_plots = [[] for _ in range(len(ax_2d))]  # 每个子图一个列表

    def update_video(i):
        print(f"正在生成动画的第{i}帧")
        nonlocal initialized
        # nonlocal scatter_plots
        if mode == 'switch':
            if i < t_hist:
                lcol, mcol, rcol = hist_lcol, hist_mcol, hist_rcol
            elif i > 75:
                lcol, mcol, rcol = tran_lcol, pred_mcol, tran_rcol
            else:
                lcol, mcol, rcol = pred_lcol, tran_mcol, pred_rcol
        else:
            if i < t_hist:
                lcol, mcol, rcol = hist_lcol, hist_mcol, hist_rcol
            else:
                lcol, mcol, rcol = pred_lcol, pred_mcol, pred_rcol

        for n, ax in enumerate(ax_2d):
            if fix_0 and n == 0 and i >= t_hist:
                continue
            if fix_0 and n % ncol == 0 and i >= t_hist:
                continue

            # 修改：三维改二维
            # trajectories[n] = poses[n][:, 0, [0, 1, 2]]
            # ax.set_xlim3d([-radius / 2 + trajectories[n][i, 0], radius / 2 + trajectories[n][i, 0]])
            # ax.set_ylim3d([-radius / 2 + trajectories[n][i, 1], radius / 2 + trajectories[n][i, 1]])
            # ax.set_zlim3d([-radius / 2 + trajectories[n][i, 2], radius / 2 + trajectories[n][i, 2]])

            trajectories[n] = poses[n][:, 0, [0, 1]]  # shape: [Frame, 2]
            # x_center, y_center = trajectories[n][i, 0] * global_max, trajectories[n][i, 1] * global_max
            # x_center, y_center = 0, 0

            # ax.set_xlim([-radius / 2 + x_center, radius / 2 + x_center])
            # ax.set_ylim([-radius / 2 + y_center, radius / 2 + y_center])
            ax.set_xlim([0, 1500])
            ax.set_ylim([0, 1500])

            # ax.plot([0, 0.1],
            #         [0, 0],
            #         [0, 0], c='r')
            # ax.plot([0, 0],
            #         [0, 0.1],
            #         [0, 0], c='g')
            # ax.plot([0, 0],
            #         [0, 0],
            #         [0, 0.1], c='b')
        if not initialized:
            for j, j_parent in enumerate(parents):
                if j_parent == -1:
                    continue

                if j in skeleton.joints_right():
                    col = rcol
                elif j in skeleton.joints_left():
                    col = lcol
                else:
                    col = mcol

                if fix_i is not None and j in fix_i:
                    col = fix_col

                for n, ax in enumerate(ax_2d):
                    pos = poses[n][i]
                    # if j in fix_list[n // ncol] and ((n // ncol) * ncol < n < (n // ncol) * ncol + ncol):
                    #     col = fix_col

                    # if j in fix_list[n // ncol] and ((n // ncol) * ncol < n < ((n // ncol) + 1) * ncol):
                    #     lines_2d[n].append(ax.plot([pos[j, 0], pos[j_parent, 0]],
                    #                                [pos[j, 1], pos[j_parent, 1]],
                    #                                [pos[j, 2], pos[j_parent, 2]], zdir='z', c=fix_col, linewidth=3.0))
                    # else:
                    lines_2d[n].append(ax.plot([pos[j, 0] * std[j][0] + mean[j][0], pos[j_parent, 0] * std[j_parent][0] + mean[j_parent][0]],
                                               [pos[j, 1] * std[j][1] + mean[j][1], pos[j_parent, 1] * std[j_parent][1] + mean[j_parent][1]],
                                               c=col, linewidth=1.0))
                    # if n == 0:
                    # if j == 1:
                    #     for tx in ax.texts:
                    #         tx.remove()
                    #     for tx in ax.texts:
                    #         tx.remove()
                    # ax.text(pos[j, 0], pos[j, 1], pos[j, 2], f'{j}', None)
            # 替代整个 "for j, j_parent in enumerate(parents): ..." 结构
            # for n, ax in enumerate(ax_2d):
            #     # pos = poses[n][i]  # shape: [Joint, 2]
            #     # 修改：删掉不要的关节
            #     pos = np.array([poses[n][i][joint] for joint in range(len(poses[n][i])) if joint not in joint_deleted])
            #
            #     for j in range(pos.shape[0]):
            #         # 设定颜色逻辑（没有 skeleton 时你可以直接用索引划分，或者全用一种颜色）
            #         if fix_i is not None and j in fix_i:
            #             col = fix_col
            #         elif j % 3 == 0:
            #             col = rcol
            #         elif j % 3 == 1:
            #             col = lcol
            #         else:
            #             col = mcol
            #
            #         scat = ax.scatter(pos[j, 0], pos[j, 1], c=col, s=2)  # s 控制点大小
            #         scatter_plots[n].append(scat)

            initialized = True
        else:
            for j, j_parent in enumerate(parents):
                if j_parent == -1:
                    continue

                if j in skeleton.joints_right():
                    col = rcol
                elif j in skeleton.joints_left():
                    col = lcol
                else:
                    col = mcol

                if fix_i is not None and j in fix_i:
                    col = fix_col

                for n, ax in enumerate(ax_2d):
                    if fix_0 and n == 0 and i >= t_hist:
                        continue
                    if fix_0 and n % ncol == 0 and i >= t_hist:
                        continue

                    # if j in fix_list[n // ncol] and ((n // ncol) * ncol < n < (n // ncol) * ncol + ncol):
                    #     col = fix_col

                    pos = poses[n][i]
                    x_array = np.array([pos[j, 0] * std[j][0] + mean[j][0], pos[j_parent, 0] * std[j_parent][0] + mean[j_parent][0]])
                    y_array = np.array([pos[j, 1] * std[j][1] + mean[j][1], pos[j_parent, 1] * std[j_parent][1] + mean[j_parent][1]])
                    # print(x_array, y_array)
                    # z_array = np.array([pos[j, 2], pos[j_parent, 2]])
                    # lines_2d[n][j - 1][0].set_data_3d(x_array, y_array, z_array)
                    lines_2d[n][j - 1][0].set_data(x_array, y_array)
                    # if j in fix_list[n // ncol] and ((n // ncol) * ncol < n < ((n // ncol) + 1) * ncol):
                    #     lines_2d[n][j - 1][0].set_color(fix_col)
                    # else:
                    lines_2d[n][j - 1][0].set_color(col)

                    # if j == 1:
                    #     for tx in ax.texts:
                    #         tx.remove()
                    #     for tx in ax.texts:
                    #         tx.remove()
                    # ax.text(pos[j, 0], pos[j, 1], pos[j,https://zjwsite.github.io/ 2], f'{j}', None)
            # for n, ax in enumerate(ax_2d):
            #     if fix_0 and n == 0 and i >= t_hist:
            #         continue
            #     if fix_0 and n % ncol == 0 and i >= t_hist:
            #         continue
            #
            #     # pos = poses[n][i]  # shape: [Joint, 3]
            #     # 修改：删掉不要的关节
            #     pos = np.array([poses[n][i][joint] for joint in range(len(poses[n][i])) if joint not in joint_deleted])
            #
            #     # 清除上一个时间步的散点
            #     for scat in scatter_plots[n]:
            #         scat.remove()
            #     scatter_plots[n] = []
            #
            #     for j in range(pos.shape[0]):
            #         # 设置颜色逻辑（根据编号简单分组）
            #         if fix_i is not None and j in fix_i:
            #             col = fix_col
            #         elif j % 3 == 0:
            #             col = rcol
            #         elif j % 3 == 1:
            #             col = lcol
            #         else:
            #             col = mcol
            #
            #         scat = ax.scatter(pos[j, 0], pos[j, 1], c=col, s=2)  # s 控制点大小
            #         scatter_plots[n].append(scat)

    def show_animation():
        nonlocal anim
        if anim is not None:
            anim.event_source.stop()
        anim = FuncAnimation(fig, update_video, frames=np.arange(0, poses[0].shape[0]), interval=0, repeat=True)
        plt.draw()

    def reload_poses():
        nonlocal poses
        poses = dict(filter(lambda x: x[0] in {'gt', 'context'} or algo == x[0].split('_')[0] or x[0].startswith('gt'),
                            all_poses.items()))
        if x[0] in {'gt', 'context'}:
            for ax, title in zip(ax_2d, poses.keys()):
                ax.set_title(title, y=1.0, fontsize=12)
        if mode == 'switch':
            if x[0] in {algo + '_0'}:
                for ax, title in zip(ax_2d, poses.keys()):
                    ax.set_title('target', y=1.0, fontsize=12)
        
        poses = list(poses.values())

    # def save_figs():
    #     nonlocal algo, find
    #     old_algo = algo
    #     for algo in algos:
    #         reload_poses()
    #         update_video(t_total - 1)
    #         fig.savefig('out/%d_%s.png' % (find, algo), dpi=400, transparent=True)
    #     algo = old_algo
    #     find += 1
    
    def save_figs():
        nonlocal algo, find
        old_algo = algo
        os.makedirs('out_svg', exist_ok=True)
        suffix = datetime.now().strftime('%Y-%m-%d_%H:%M:%S.%f')[:-3]
        os.makedirs('out_svg_' + suffix, exist_ok=True)
        for algo in algos:
            reload_poses()
            for i in range(0, t_total + 1, 10):
                if i == 0:
                    update_video(0)
                else:
                    update_video(i - 1)
                fig.savefig('out_svg_' + suffix + '/%d_%s_%d.svg' % (find, algo, i), transparent=True)
        algo = old_algo
        find += 1

    def on_key(event):
        nonlocal algo, all_poses, animating, anim

        if event.key == 'd':
            all_poses = next(poses_generator)
            reload_poses()
            show_animation()
        elif event.key == 'c':
            save()
        elif event.key == ' ':
            if animating:
                anim.event_source.stop()
            else:
                anim.event_source.start()
            animating = not animating
        elif event.key == 'v':  # save images
            if anim is not None:
                anim.event_source.stop()
                anim = None
            save_figs()
        elif event.key.isdigit():
            algo = algos[int(event.key) - 1]
            reload_poses()
            show_animation()

    def save():
        nonlocal anim

        # fps = 50  # 修改：fps改成15
        fps = 15
        anim = FuncAnimation(fig, update_video, frames=np.arange(0, poses[0].shape[0]), interval=1000 / fps,
                             repeat=False)
        os.makedirs(os.path.dirname(output), exist_ok=True)
        if output.endswith('.mp4'):
            Writer = writers['ffmpeg']
            writer = Writer(fps=fps, metadata={}, bitrate=bitrate)
            anim.save(output, writer=writer)
        elif output.endswith('.gif'):
            anim.save(output, dpi=80, writer='pillow')
        else:
            raise ValueError('Unsupported output format (only .mp4 and .gif are supported)')
        print(f'video saved to {output}!')

    fig.canvas.mpl_connect('key_press_event', on_key)
    
    save()
    show_animation()
    plt.show()
    plt.close()
    
    # save_figs()