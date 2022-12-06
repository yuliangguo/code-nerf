import torch
import numpy as np

if __name__ == '__main__':
    result_file = './exps/srncar/test_rpert0.1_tpert0.1_nops200_nview2/codes+poses.pth'

    saved_result = torch.load(result_file, map_location=torch.device('cpu'))

    psnr_all = []
    for psnr in saved_result['psnr_eval'].values():
        psnr_all.append(psnr[0].squeeze().numpy())

    R_err_all = []
    T_err_all = []
    for R_err in saved_result['R_eval'].values():
        R_err_all.append(R_err[0].squeeze().numpy())
    for T_err in saved_result['T_eval'].values():
        T_err_all.append(T_err[0].squeeze().numpy())

    print(f'Avg R error: {np.mean(np.array(R_err_all))}, Avg T error: {np.mean(np.array(T_err_all))}')
