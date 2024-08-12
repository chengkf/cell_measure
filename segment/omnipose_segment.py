import math
import pandas as pd
from cellpose_omni import models, io, plot
from scipy.ndimage import find_objects
from plotly.subplots import make_subplots
import plotly.graph_objs as go
import matplotlib.pyplot as plt
import numpy as np
import skimage.io
import omnipose
import cv2
import os

class omnipose_segment():
    def __init__(self):
        pass

    def point_connect(self, pots):
        """
        完成所有点的连接，返回所有线段的list(线段中包含两个点的索引)和最短路径值，要求所有线路之和最短。
        例如输入 pots = [[0, 0], [2, 0], [3, 2], [3, -2]]
        返回 paths = [[0, 1], [1, 2], [1, 3]], length =
        """
        l = len(pots)
        if l <= 1:
            return [], 0

        con = [pots[0]]  # 已经连线的点集，先随便放一个点进去
        not_con = pots[1:]  # 还没连线的点集
        paths = []  # 所有连线
        length_total = 0  # 总连线长度
        for _ in range(l - 1):  # 共 l-1 条连线
            # 得到下一条连线的两点a、b 及其距离length_ab
            a, b = con[0], not_con[0]  # 先任意选两个点
            length_ab = math.sqrt((a[0] - b[0]) ** 2 + (a[1] - b[1]) ** 2)
            for m in con:
                for n in not_con:
                    lg = math.sqrt((m[0] - n[0]) ** 2 + (m[1] - n[1]) ** 2)
                    if lg < length_ab:  # 如果有更短的
                        length_ab = lg
                        a, b = m, n

            # 记录
            paths.append([pots.index(a), pots.index(b)])  # 记录连线ab
            con.append(b)  # 已连接点集中记录点b
            not_con.remove(b)  # 未连接点集中删除点b
            length_total += length_ab  # 记录总长度
        #length_total += 30
        return paths, length_total

    def save_all(self, data, xlsx_path):
        t = 1
        for value in data:
            if isinstance(value, float):
                data[t] = "{:.12f}".format(value)
                t = t + 1
        df = pd.read_excel(xlsx_path)
        df = pd.concat([df,data], ignore_index=True)
        # 将工作簿保存为xlsx文件
        df.to_excel(xlsx_path, index=False)

    def draw_scatterchart(self,df):

        x = df['编号']
        y1 = df['细胞面积']
        y2 = df['细胞周长']
        y3 = df['细胞长度']
        y4 = df['细胞宽度']

        # 创建四个轨迹
        trace1 = go.Scatter(x=x, y=y1, mode='markers', name='细胞面积')
        trace2 = go.Scatter(x=x, y=y2, mode='markers', name='细胞周长')
        trace3 = go.Scatter(x=x, y=y3, mode='markers', name='细胞长度')
        trace4 = go.Scatter(x=x, y=y4, mode='markers', name='细胞宽度')

        # 创建子图对象
        fig = make_subplots(rows=1, cols=2)

        # 将轨迹添加到子图中
        fig.add_trace(trace1, row=1, col=1)
        fig.add_trace(trace2, row=1, col=2)
        fig.add_trace(trace3, row=1, col=2)
        fig.add_trace(trace4, row=1, col=2)
        #fig.update_layout(height=800, width=1000, title_text="折线图")

        return fig

    def draw_boxchart(self, filenames, dfs, str):
        fig = go.Figure()
        for df,filename in zip(dfs, filenames):
            y1 = df[str]
            fig.add_trace(go.Box(y=y1, name=filename))
        return fig


    def base_segment(self, data):
        area, perimeter, length, dimas = [], [], [], []
        imgs = [data]  # 读取文件夹中所有图片
        model = models.CellposeModel(gpu=True, pretrained_model='segment/model0511', nclasses=4)
        masks, flows, _ = model.eval(imgs, channels=[1, 0], mask_threshold=0.0, flow_threshold=0.0, diameter=30.0,
                                         invert=False, cluster=True, net_avg=False, do_3D=False, omni=True)
        maski = masks[0]
        flowi = flows[0][0]
        fig = plt.figure(figsize=(12, 5))
        img, outlines, overlay = plot.show_segmentation(fig, imgs[0], maski, flowi,channels=[1, 0], display=False)
        outX, outY = np.nonzero(outlines)
        imgout = img.copy()
        imgout[outX, outY] = np.array([0, 255, 0])
        slices = find_objects(maski.astype(int))

        for i, si in enumerate(slices):
            if si is not None:

                sr, sc = si
                cv2.putText(imgout, str(i), ((sc.start+sc.stop)//2, (sr.start+sr.stop)//2), cv2.FONT_HERSHEY_COMPLEX, 1, (100, 200, 200), 2)
                mask = (maski[sr, sc] == (i + 1)).astype(np.uint8)
                dima = omnipose.core.diameters(mask)
                mask = np.pad(mask, (2, 2), 'constant')
                contours, hierarchy = cv2.findContours(mask, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)

                kernel = np.ones((3, 3), dtype=np.uint8)
                mask = cv2.erode(mask, kernel, iterations=2)
                flow = flowi[sr, sc]
                flow[flow < 150] = 0
                for x in range(flow.shape[0]):
                    for y in range(flow.shape[1]):
                        if flow[x][y].any() != 0:  # or mask[x][y]*255 == 0:
                            flow[x][y] = [255, 255, 255]
                flow = cv2.cvtColor(flow, cv2.COLOR_BGR2GRAY)
                flow = np.pad(flow, (2, 2), 'constant')
                flow = 255 - flow
                for x in range(flow.shape[0]):
                    for y in range(flow.shape[1]):
                        if mask[x][y] == 0:
                            flow[x][y] = 0
                point = []
                index_y, index_x = np.where(flow == 255)
                for idx in range(len(index_x)):  # 像素点的坐标
                    point.append([index_x[idx], index_y[idx]])
                paths, length_total = self.point_connect(point)
                area.append(cv2.contourArea(contours[0]))
                perimeter.append(cv2.arcLength(contours[0], True))
                length.append(length_total + dima)
                dimas.append(dima)
                # length.append(length_total if length_total != 0 else 30)

        mean_area = sum(area) / len(area)
        min_area = min(area)
        max_area = max(area)

        mean_perimeter = sum(perimeter) / len(perimeter)
        min_perimeter = min(perimeter)
        max_perimeter = max(perimeter)

        mean_length = sum(length) / len(length)
        min_length = min(length)
        max_length = max(length)

        mean_dimas = sum(dimas) / len(dimas)
        min_dimas = min(dimas)
        max_dimas = max(dimas)

        df = pd.DataFrame(range(1, len(area) + 1), columns=['编号'])
        df = pd.concat([df, pd.DataFrame(area, columns=['细胞面积'])], axis=1)
        df = pd.concat([df, pd.DataFrame(perimeter, columns=['细胞周长'])], axis=1)
        df = pd.concat([df, pd.DataFrame(length, columns=['细胞长度'])], axis=1)
        df = pd.concat([df, pd.DataFrame(dimas, columns=['细胞宽度'])], axis=1)
        df_mean = pd.DataFrame({'编号': '平均值', '细胞面积': mean_area, '细胞周长': mean_perimeter, '细胞长度': mean_length, '细胞宽度': mean_dimas}, index=[0])
        df_max = pd.DataFrame({'编号': '最大值', '细胞面积': max_area, '细胞周长': max_perimeter, '细胞长度': max_length, '细胞宽度': max_dimas}, index=[0])
        df_min = pd.DataFrame({'编号': '最小值', '细胞面积': min_area, '细胞周长': min_perimeter, '细胞长度': min_length, '细胞宽度': min_dimas}, index=[0])
        df = pd.concat([df, df_mean], ignore_index=True)
        df = pd.concat([df, df_max], ignore_index=True)
        df = pd.concat([df, df_min], ignore_index=True)
        return df, imgout

    def batch_segment(self, images, filenames):
        imgouts, dfs, mean_dirs, min_dirs, max_dirs = [], [], [], [], []
        area, perimeter, length, diams = [], [], [], []
        mean_dir, min_dir, max_dir = pd.DataFrame(), pd.DataFrame(), pd.DataFrame()
        model = models.CellposeModel(gpu=True, pretrained_model='segment/model0511', nclasses=4)
        mean_sum, min_sum, max_sum = pd.DataFrame(), pd.DataFrame(), pd.DataFrame()

        masks, flows, _ = model.eval(images, channels=[1, 0], mask_threshold=0.0, flow_threshold=0.0, diameter=30.0,
                                     invert=False, cluster=True, net_avg=False, do_3D=False, omni=True)
        nimg = len(images)
        for idx, file in zip(range(nimg), filenames):

            area.append([])
            perimeter.append([])
            length.append([])
            diams.append([])

            maski = masks[idx]
            flowi = flows[idx][0]
            fig = plt.figure(figsize=(12, 5))
            img, outlines, overlay = plot.show_segmentation(fig, images[idx], maski, flowi, channels=[1, 0],
                                                            display=False)
            outX, outY = np.nonzero(outlines)
            imgout = img.copy()
            imgout[outX, outY] = np.array([0, 255, 0])
            imgouts.append(imgout)
            slices = find_objects(maski.astype(int))
            for i, si in enumerate(slices):
                if si is not None:
                    sr, sc = si
                    cv2.putText(imgout, str(i), ((sc.start + sc.stop) // 2, (sr.start + sr.stop) // 2),
                                cv2.FONT_HERSHEY_COMPLEX, 1, (100, 200, 200), 2)
                    mask = (maski[sr, sc] == (i + 1)).astype(np.uint8)
                    diam = omnipose.core.diameters(mask)
                    mask = np.pad(mask, (2, 2), 'constant')
                    contours, hierarchy = cv2.findContours(mask, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)

                    kernel = np.ones((3, 3), dtype=np.uint8)
                    mask = cv2.erode(mask, kernel, iterations=2)
                    flow = flowi[sr, sc]
                    flow[flow < 150] = 0
                    for x in range(flow.shape[0]):
                        for y in range(flow.shape[1]):
                            if flow[x][y].any() != 0:  # or mask[x][y]*255 == 0:
                                flow[x][y] = [255, 255, 255]
                    flow = cv2.cvtColor(flow, cv2.COLOR_BGR2GRAY)
                    flow = np.pad(flow, (2, 2), 'constant')
                    flow = 255 - flow
                    for x in range(flow.shape[0]):
                        for y in range(flow.shape[1]):
                            if mask[x][y] == 0:
                                flow[x][y] = 0
                    point = []
                    index_y, index_x = np.where(flow == 255)
                    for x in range(len(index_x)):  # 像素点的坐标
                        point.append([index_x[x], index_y[x]])
                    paths, length_total = self.point_connect(point)

                    area[idx].append(cv2.contourArea(contours[0]))
                    perimeter[idx].append(cv2.arcLength(contours[0], True))
                    length[idx].append(length_total + diam)
                    diams[idx].append(diam)
            mean_area = sum(area[idx]) / len(area[idx])
            min_area = min(area[idx])
            max_area = max(area[idx])

            mean_perimeter = sum(perimeter[idx]) / len(perimeter[idx])
            min_perimeter = min(perimeter[idx])
            max_perimeter = max(perimeter[idx])

            mean_length = sum(length[idx]) / len(length[idx])
            min_length = min(length[idx])
            max_length = max(length[idx])

            mean_diams = sum(diams[idx]) / len(diams[idx])
            min_diams = min(diams[idx])
            max_diams = max(diams[idx])

            df = pd.DataFrame(range(1, len(area[idx]) + 1), columns=['编号'])
            df = pd.concat([df, pd.DataFrame(area[idx], columns=['细胞面积'])], axis=1)
            df = pd.concat([df, pd.DataFrame(perimeter[idx], columns=['细胞周长'])], axis=1)
            df = pd.concat([df, pd.DataFrame(length[idx], columns=['细胞长度'])], axis=1)
            df = pd.concat([df, pd.DataFrame(diams[idx], columns=['细胞宽度'])], axis=1)
            df_mean = pd.DataFrame(
                {'编号': '平均值', '细胞面积': mean_area, '细胞周长': mean_perimeter, '细胞长度': mean_length,
                 '细胞宽度': mean_diams}, index=[0])
            df_max = pd.DataFrame(
                {'编号': '最大值', '细胞面积': max_area, '细胞周长': max_perimeter, '细胞长度': max_length,
                 '细胞宽度': max_diams}, index=[0])
            df_min = pd.DataFrame(
                {'编号': '最小值', '细胞面积': min_area, '细胞周长': min_perimeter, '细胞长度': min_length,
                 '细胞宽度': min_diams}, index=[0])
            df = pd.concat([df, df_mean], ignore_index=True)
            df = pd.concat([df, df_max], ignore_index=True)
            df = pd.concat([df, df_min], ignore_index=True)
            df_mean = pd.DataFrame(
                {'编号': file, '细胞面积': mean_area, '细胞周长': mean_perimeter, '细胞长度': mean_length,
                 '细胞宽度': mean_diams}, index=[0])
            df_max = pd.DataFrame(
                {'编号': file, '细胞面积': max_area, '细胞周长': max_perimeter, '细胞长度': max_length,
                 '细胞宽度': max_diams}, index=[0])
            df_min = pd.DataFrame(
                {'编号': file, '细胞面积': min_area, '细胞周长': min_perimeter, '细胞长度': min_length,
                 '细胞宽度': min_diams}, index=[0])
            mean_dir = pd.concat([mean_dir, df_mean], ignore_index=True)
            min_dir = pd.concat([min_dir, df_min], ignore_index=True)
            max_dir = pd.concat([max_dir, df_max], ignore_index=True)
            dfs.append(df)

        return mean_dir, min_dir, max_dir, [filenames, images, imgouts, dfs, mean_dir, min_dir, max_dir]

