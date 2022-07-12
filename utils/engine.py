import torch
import numpy as np
import matplotlib.pyplot as plt


def tensor_2_img(target, output):
    b, c, h, w = target.shape
    b_index = np.random.randint(low=0, high=b)
    cmap = plt.get_cmap('jet')
    target_data = target.cpu()[b_index].squeeze().numpy()
    output_data = output.data.cpu()[b_index].squeeze().numpy()
    denominator = max(target_data.max(), output_data.max())
    target_rgba = cmap(target_data / denominator)
    target_rgb = torch.from_numpy(np.delete(target_rgba, 3, 2)).permute(2, 0, 1)
    output_rgba = cmap(output_data / denominator)
    output_rgb = torch.from_numpy(np.delete(output_rgba, 3, 2)).permute(2, 0, 1)

    return target_rgb, output_rgb


def arr_2_heat_map(hist_arr):
    hist_arr = np.vstack(hist_arr)
    heat_map = []
    for hist_map in hist_arr:
        cmap = plt.get_cmap('jet')
        denominator = hist_arr.max()
        rgba = cmap(hist_map / denominator)
        rgb = torch.from_numpy(np.delete(rgba, 3, 2)).permute(2, 0, 1)
        heat_map.append(rgb)

    return torch.cat(heat_map, dim=2)


def euclidean_dist(x, y):
    dist = torch.cdist(x, y, p=2)
    return dist


def generate_hist_2d(x, y):
    hist_arr = []
    for xi, yi in zip(x, y):
        xx = euclidean_dist(xi, xi)
        yy = euclidean_dist(yi, yi)
        # hist_2d = np.histogram2d(xx.cpu().detach().numpy().flatten(), yy.cpu().detach().numpy().flatten(), bins=128,range=[[0, 1.2], [0, 1.2]], normed=True)
        hist_2d = np.histogram2d(xx.cpu().detach().numpy().flatten(), yy.cpu().detach().numpy().flatten(), bins=192, normed=True)
        hist_arr.append(hist_2d[0][np.newaxis,:])
    return hist_arr


def trainer(model, optimizer, train_loader, criterion, epoch, visual=True):
    loss_w_avg = 0
    loss_avg = 0
    iter_num = 0
    x = []
    y = []
    for feats in train_loader:
        feats = feats.cuda()
        c_feats = model(feats)
        loss_w, loss = criterion(x=feats, y=c_feats)

        loss_w_avg += float(loss_w)
        loss_avg += float(loss)
        iter_num += 1

        optimizer.zero_grad()
        # torch.nn.utils.clip_grad_value_(model.parameters(), 5.0)
        loss_w.backward()
        optimizer.step()

        if iter_num == 1:
            x.append(feats)
            y.append(c_feats)

        if iter_num % (len(train_loader) // 2) == 0:
            print('epoch:{} || iter:{} || loss_w:{} || loss:{}'.format(epoch, iter_num, loss_w_avg / iter_num,
                                                                       loss_avg / iter_num))
            x.append(feats)
            y.append(c_feats)

    if visual:
        hist_arr = generate_hist_2d(x, y)
        heat_map = arr_2_heat_map(hist_arr)
        hist_arr_x = generate_hist_2d(x, x)
        heat_map_x = arr_2_heat_map(hist_arr_x)
        hist_arr_y = generate_hist_2d(y, y)
        heat_map_y = arr_2_heat_map(hist_arr_y)
        return loss_w_avg / iter_num, loss_avg / iter_num, heat_map, heat_map_x, heat_map_y
    else:
        return loss_w_avg / iter_num, loss_avg / iter_num


def val(model, val_loader, criterion):
    loss_w_avg = 0
    loss_avg = 0
    iter_num = 0
    with torch.no_grad():
        for feats in val_loader:
            feats = feats.cuda()
            c_feats = model(feats)
            loss_w, loss = criterion(x=feats, y=c_feats)

            loss_w_avg += float(loss_w)
            loss_avg += float(loss)
            iter_num += 1

    return loss_w_avg / iter_num, loss_avg / iter_num