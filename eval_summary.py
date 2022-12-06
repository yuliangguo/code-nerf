import torch
import numpy as np

if __name__ == '__main__':
    result_file = 'exps_nuscenes_autorf/vehicle.car.v1.0-trainval.use_instance.bsize6_2022_12_02/test_nuscenes_use_instance_opt_pose_full/codes+poses.pth'

    saved_result = torch.load(result_file, map_location=torch.device('cpu'))

    psnr_all = []
    for psnr in saved_result['psnr_eval'].values():
        psnr_all.append(psnr[0])
    print(f'Avg psnr error: {np.mean(np.array(psnr_all))}')

    R_err_all = []
    T_err_all = []
    for R_err in saved_result['R_eval'].values():
        R_err_all.append(R_err.numpy())
    for T_err in saved_result['T_eval'].values():
        T_err_all.append(T_err.numpy())

    print(f'Avg R error: {np.mean(np.array(R_err_all))}, Avg T error: {np.mean(np.array(T_err_all))}')
