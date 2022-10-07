import os
import glob


if __name__ == '__main__':
    result_path = 'exps/srncar_08182022/test_nuscenes_use_instance_opt_pose_model_w_rgb_bgloss/*'
    tgt_paths = glob.glob(result_path)

    for tgt_path in tgt_paths:
        if os.path.isfile(tgt_path):
            continue

        cmd = f'ffmpeg -r 20 -f image2 -i {tgt_path}/opt%03d.png -pix_fmt yuv420p {tgt_path}.mp4'
        print(cmd)
        os.system(cmd)