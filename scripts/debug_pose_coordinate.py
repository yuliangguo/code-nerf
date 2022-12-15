import torch
import numpy as np
import pytorch3d.transforms.rotation_conversions as rot_trans

if __name__ == '__main__':

    # initial error could from o2c or c2o
    init_error_from_c2o = True

    # points in camera coordinates
    X_c = np.asarray([[1, 5, 10],
                      [2, -4, 8],
                      [30, 9, 7],
                      [-4, -10, -3],
                      [5, 30, -8]], dtype=np.float32).transpose()
    X_c = torch.from_numpy(X_c)

    # gt pose, rotation is based on axis-angle representation
    rot_vec_o2c_gt = torch.tensor([0, 0, 1], dtype=torch.float32)
    trans_vec_o2c_gt = torch.tensor([20, 5, 1], dtype=torch.float32)
    R_o2c_gt = rot_trans.axis_angle_to_matrix(rot_vec_o2c_gt)
    T_o2c_gt = trans_vec_o2c_gt.unsqueeze(-1)
    R_c2o_gt = torch.transpose(R_o2c_gt, dim0=-2, dim1=-1)
    T_c2o_gt = - R_c2o_gt @ T_o2c_gt
    rot_vec_c2o_gt = rot_trans.matrix_to_axis_angle(R_c2o_gt)
    trans_vec_c2o_gt = T_c2o_gt.squeeze()
    X_o_gt = R_c2o_gt @ X_c + T_c2o_gt

    # pred pose
    ratio = 1.2
    rot_vec_o2c_pred = torch.tensor([0.2, 0.5, 0.8], dtype=torch.float32).detach().requires_grad_()
    trans_vec_o2c_pred = torch.tensor([20*ratio, 5*ratio, 1*ratio], dtype=torch.float32).detach().requires_grad_()
    # trans_vec_o2c_pred = ratio * trans_vec_o2c_gt.clone().detach().requires_grad_()
    R_o2c_pred = rot_trans.axis_angle_to_matrix(rot_vec_o2c_pred)
    T_o2c_pred = trans_vec_o2c_pred.unsqueeze(-1)
    R_c2o_pred = torch.transpose(R_o2c_pred, dim0=-2, dim1=-1)
    T_c2o_pred = - R_c2o_pred @ T_o2c_pred
    rot_vec_c2o_pred = rot_trans.matrix_to_axis_angle(R_c2o_pred).detach().requires_grad_()
    trans_vec_c2o_pred = T_c2o_pred.squeeze().detach().requires_grad_()
    if init_error_from_c2o:
        trans_vec_c2o_pred = torch.tensor([-15.0134*ratio,  14.1279*ratio,  -1.0000*ratio], dtype=torch.float32).detach().requires_grad_()
        trans_vec_o2c_pred = -R_o2c_pred @ trans_vec_c2o_pred.unsqueeze(-1)
        trans_vec_o2c_pred = trans_vec_o2c_pred.squeeze().detach().requires_grad_()

    lr_rot = 0.001
    lr_trans = 0.01

    # optimizer1 = torch.optim.AdamW([
    #     {'params': rot_vec_o2c_pred, 'lr': 0.1},
    #     {'params': trans_vec_o2c_pred, 'lr': 0.1}
    # ])
    # optimizer1 = torch.optim.Adagrad([
    #     {'params': rot_vec_o2c_pred},
    #     {'params': trans_vec_o2c_pred}
    # ], lr=0.01)

    # optimizer2 = torch.optim.AdamW([
    #     {'params': rot_vec_c2o_pred, 'lr': 0.1},
    #     {'params': trans_vec_c2o_pred, 'lr': 0.1}
    # ])
    # optimizer2 = torch.optim.Adagrad([
    #     {'params': rot_vec_c2o_pred},
    #     {'params': trans_vec_c2o_pred}
    # ], lr=0.01)

    # optimization
    print(f'iter: 0, loss1: Inf, rot_vec_o2c_gt: {rot_vec_o2c_gt}, rot_vec_o2c_pred: {rot_vec_o2c_pred}, trans_vec_o2c_gt: {trans_vec_o2c_gt}, trans_vec_o2c_pred: {trans_vec_o2c_pred}')
    print(f'iter: 0, loss2: inf, rot_vec_c2o_gt: {rot_vec_c2o_gt}, rot_vec_c2o_pred: {rot_vec_c2o_pred}, trans_vec_c2o_gt: {trans_vec_c2o_gt}, trans_vec_c2o_pred: {trans_vec_c2o_pred}')

    for iter in range(0, 100):
        rot_vec_o2c_pred.retain_grad()
        trans_vec_o2c_pred.retain_grad()
        rot_vec_c2o_pred.retain_grad()
        trans_vec_c2o_pred.retain_grad()
        #######################################    optimize in o2c space  ###########################################
        # optimizer1.zero_grad()
        R_o2c1 = rot_trans.axis_angle_to_matrix(rot_vec_o2c_pred)
        R_o2c1.retain_grad()
        T_o2c1 = trans_vec_o2c_pred.unsqueeze(-1)
        T_o2c1_copy = torch.clone(T_o2c1).detach()
        T_o2c1.retain_grad()
        R_c2o1 = torch.transpose(R_o2c1, dim0=-2, dim1=-1)
        R_c2o1.retain_grad()
        T_c2o1 = - R_c2o1 @ T_o2c1
        T_c2o1.retain_grad()
        X_o1 = R_c2o1 @ X_c + T_c2o1

        loss1 = torch.mean(torch.sum((X_o1 - X_o_gt) ** 2, dim=0))
        loss1.backward()  # backward just compute all the gradients, but no change to the variables
        # optimizer1.step()  # step will collect the gradients and apply lr to update declared parameters to optimize
        print('rot_vec_o2c_pred.grad')
        print(rot_vec_o2c_pred.grad)
        print('R_o2c1.grad')
        print(R_o2c1.grad)
        print('R_c2o1.grad')
        print(R_c2o1.grad)
        # Naive gradient decent update
        rot_vec_o2c_pred = rot_vec_o2c_pred - rot_vec_o2c_pred.grad * lr_rot
        trans_vec_o2c_pred = trans_vec_o2c_pred - trans_vec_o2c_pred.grad * lr_trans

        print('T_o2c1.grad')
        print(T_o2c1.grad)

        # print('T_o2c actual change')
        # print(T_o2c1 - T_o2c1_copy)
        #
        # print('T o2c actual change divided by grad')
        # print((T_o2c1 - T_o2c1_copy) / T_o2c1.grad)

        #######################################    optimize in c2o space  ###########################################
        # optimizer2.zero_grad()
        R_c2o2 = rot_trans.axis_angle_to_matrix(rot_vec_c2o_pred)
        R_c2o2.retain_grad()
        T_c2o2 = trans_vec_c2o_pred.unsqueeze(-1)
        T_c2o2_copy = torch.clone(T_c2o2).detach()
        T_c2o2.retain_grad()

        print('trans diff before optimization')
        print(-torch.transpose(R_c2o2, dim0=-2, dim1=-1) @ T_c2o2 - T_o2c1_copy)

        X_o2 = R_c2o2 @ X_c + T_c2o2
        loss2 = torch.mean(torch.sum((X_o2 - X_o_gt) ** 2, dim=0))
        loss2.backward()
        # backward just compute all the gradients, but no change to the variables
        # optimizer2.step()  # step will collect the gradients and apply lr to update declared parameters to optimize

        print('rot_vec_c2o_pred.grad')
        print(rot_vec_c2o_pred.grad)
        print('R_c2o2.grad')
        print(R_c2o2.grad)
        # Naive gradient decent update
        rot_vec_c2o_pred = rot_vec_c2o_pred - rot_vec_c2o_pred.grad * lr_rot
        trans_vec_c2o_pred = trans_vec_c2o_pred - trans_vec_c2o_pred.grad * lr_trans

        print('trans gradiant diff')
        print(-torch.transpose(R_c2o2, dim0=-2, dim1=-1) @ T_c2o2.grad - T_o2c1.grad)

        print('trans diff after optimization v1')
        print(-torch.transpose(R_c2o2, dim0=-2, dim1=-1) @ trans_vec_c2o_pred.unsqueeze(-1) - trans_vec_o2c_pred.unsqueeze(-1))

        print('T_c2o2.grad')
        print(T_c2o2.grad)

        # print('T_c2o actual change')
        # print(T_c2o2 - T_c2o2_copy)
        #
        # print('T_c2o actual change divided by grad')
        # print((T_c2o2 - T_c2o2_copy) / T_c2o2.grad)

        print(f'iter: {iter+1}, loss1: {loss1}, rot_vec_o2c_gt: {rot_vec_o2c_gt}, rot_vec_o2c_pred: {rot_vec_o2c_pred}, trans_vec_o2c_gt: {trans_vec_o2c_gt}, trans_vec_o2c_pred: {trans_vec_o2c_pred}')
        print(f'iter: {iter+1}, loss2: {loss2}, rot_vec_c2o_gt: {rot_vec_c2o_gt}, rot_vec_c2o_pred: {rot_vec_c2o_pred}, trans_vec_c2o_gt: {trans_vec_c2o_gt}, trans_vec_c2o_pred: {trans_vec_c2o_pred}')

