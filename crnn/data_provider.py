import os
import cv2
import math
import random
import shutil
import numpy as np
from tqdm import trange
from collections import Counter
from crnn import charset_generate
from multiprocessing import Process
from crnn import config as crnn_config
from PIL import Image, ImageDraw, ImageFont


class TextCut(object):
    def __init__(self,
                 org_images_path,
                 org_labels_path,
                 cut_train_images_path,
                 cut_train_labels_path,
                 cut_test_images_path,
                 cut_test_labels_path,
                 train_test_ratio=0.8,
                 filter_ratio=1.5,
                 filter_height=25,
                 is_transform=True,
                 angle_range=[-15.0, 15.0],
                 write_mode='w',
                 use_blank=False,
                 num_process=1):
        """
            对ICPR原始图像进行切图
            :param org_images_path: ICPR数据集原始图像路径，[str]
            :param org_labels_path: ICPR数据集原始label路径，[str]
            :param cut_train_images_path: 训练集切图的保存路径，[str]
            :param cut_train_labels_path: 训练集切图对应label的保存路径，[str]
            :param cut_test_images_path: 测试集切图的保存路径，[str]
            :param cut_test_labels_path: 测试集切图对应label的保存路径，[str]
            :param train_test_ratio: 训练测试数据集比例，[float]
            :param filter_ratio: 图片过滤的高宽比例，高于该比例的图片将被过滤，default:1.5 ，[float]
            :param filter_height:高度过滤，切图后的图像高度低于该值的将被过滤掉，[int]
            :param is_transform: 是否进行仿射变换，default:True [boolean]
            :param angle_range: 不进行仿射变换的角度范围default:[-15.0, 15.0]，[list]
            :param write_mode: 数据写入模式，'w':write,'a':add，[str]
            :param use_blank: 是否使用空格,[boolean]
            :param num_process: 并行处理的进程数
            :return:
        """
        self.org_images_path = org_images_path
        self.org_labels_path = org_labels_path
        self.cut_train_images_path = cut_train_images_path
        self.cut_train_labels_path = cut_train_labels_path
        self.cut_test_images_path = cut_test_images_path
        self.cut_test_labels_path = cut_test_labels_path
        self.train_test_ratio = train_test_ratio
        self.filter_ratio = filter_ratio
        self.filter_height = filter_height
        self.is_transform = is_transform
        self.angle_range = angle_range
        assert write_mode in ['w', 'a'], "write mode should be 'w'(write) or 'a'(add)"
        self.write_mode = write_mode
        self.use_blank = use_blank
        self.num_process = num_process
        self.org_labels_list = None
        super().__init__()

    def data_load(self, org_images_list):
        """
        对ICPR图像做文本切割处理
        :param org_images_list: 原始图片文件名
        :return:
        """
        data_len = len(org_images_list)
        train_test_offset = data_len * self.train_test_ratio
        for data_i in range(len(org_images_list)):
            org_image_path = org_images_list[data_i]
            org_image_name = os.path.basename(org_image_path)[:-4]
            org_label_path = org_image_name + ".txt"
            if org_label_path not in self.org_labels_list:
                continue
            org_image = Image.open(os.path.join(self.org_images_path, org_image_path))
            with open(os.path.join(self.org_labels_path, org_label_path), 'r', encoding='utf-8') as fr:
                org_label = fr.read().split('\n')
            cut_images_list, cut_labels_list = self.cut_text(org_image, org_label,
                                                             self.filter_ratio,
                                                             self.is_transform,
                                                             self.angle_range)
            if data_i < train_test_offset:
                img_save_path = self.cut_train_images_path
                label_save_path = self.cut_train_labels_path
            else:
                img_save_path = self.cut_test_images_path
                label_save_path = self.cut_test_labels_path
            for i in range(len(cut_images_list)):
                cut_img = cut_images_list[i]
                if cut_img.shape[0] >= self.filter_height:
                    cut_img = Image.fromarray(cut_img)
                    cut_img = cut_img.convert('RGB')
                    cut_label = cut_labels_list[i]
                    cut_img_name = org_image_name + '_' + str(i) + '.jpg'
                    cut_img.save(os.path.join(img_save_path, cut_img_name))
                    with open(label_save_path, 'a', encoding='utf-8') as fa:
                        fa.write(cut_img_name + '\t' + cut_label + '\n')

    def data_load_multi_process(self, num_process=None):
        """
        多进程对ICPR图像做文本切割处理
        :param num_process:进程数，默认16,[int]
        :return:
        """
        if num_process is None:
            num_process = self.num_process
        org_images_list = os.listdir(self.org_images_path)
        self.org_labels_list = os.listdir(self.org_labels_path)
        # clear label.txt at first step
        check_path([self.cut_train_images_path,
                    self.cut_train_labels_path,
                    self.cut_test_images_path,
                    self.cut_test_labels_path])
        if self.write_mode == 'w':
            clear_content([self.cut_train_images_path,
                           self.cut_train_labels_path,
                           self.cut_test_images_path,
                           self.cut_test_labels_path])
        all_data_len = len(org_images_list)
        data_offset = all_data_len // num_process
        processes = list()
        for data_i in trange(0, all_data_len, data_offset):
            if data_i + data_offset >= all_data_len:
                processes.append(Process(target=self.data_load, args=(org_images_list[data_i:],)))
            else:
                processes.append(Process(target=self.data_load, args=(org_images_list[data_i:data_i + data_offset],)))
        for process in processes:
            process.start()
        for process in processes:
            process.join()

    def cut_text(self, image, labels, filter_ratio, is_transform, angle_range):
        """
        文本切图
        :param image: 原始图像，[array]
        :param labels: 文本的label，[str]
        :param filter_ratio: 图片过滤的高宽比例，高于该比例的图片将被过滤，e.g. 1.5 ，[float]
        :param is_transform: 是否进行仿射变换，[boolean]
        :param angle_range: 不进行仿射变换的角度范围e.g.[-15.0, 15.0]，[list]
        :return:
        """
        cut_images = list()
        cut_labels = list()
        w, h = image.size
        for label in labels:
            if label == '':
                continue
            label_text = label.split(',')
            text = label_text[-1]
            if not self.use_blank:
                text = text.replace(' ', '')
            if text == '###' or text == '★' or text == '':
                continue
            position = self.reorder_vertexes(
                np.array([[round(float(label_text[i])), round(float(label_text[i + 1]))] for i in range(0, 8, 2)]))
            position = np.reshape(position, 8).tolist()
            left = max(min([position[i] for i in range(0, 8, 2)]), 0)
            right = min(max([position[i] for i in range(0, 8, 2)]), w)
            top = max(min([position[i] for i in range(1, 8, 2)]), 0)
            bottom = min(max([position[i] for i in range(1, 8, 2)]), h)
            if (bottom - top) / (right - left + 1e-3) > filter_ratio:
                continue
            image = np.asarray(image)
            cut_image = image[top:bottom, left:right]
            if is_transform:
                trans_img = self.transform(image, position, angle_range)
                if trans_img is not None:
                    cut_image = trans_img
            cut_images.append(cut_image)
            cut_labels.append(text)
        return cut_images, cut_labels

    def transform(self, image, position, angle_range):
        """
        仿射变换
        :param image: 原始图像，[array]
        :param position: 文本所在的位置e.g.[x0,y0,x1,y1,x2,y2]，[list]
        :param angle_range: 不进行仿射变换的角度范围e.g.[-15.0, 15.0]，[list]
        :return: 变换后的图像
        """
        from_points = [position[2:4], position[4:6]]
        width = round(float(self.calc_dis(position[2:4], position[4:6])))
        height = round(float(self.calc_dis(position[2:4], position[0:2])))
        to_points = [[0, 0], [width, 0]]
        from_mat = self.list2col_matrix(from_points)
        to_mat = self.list2col_matrix(to_points)
        tran_m, tran_b = self.get_transform(from_mat, to_mat)
        probe_vec = np.matrix([1.0, 0.0]).transpose()
        probe_vec = tran_m * probe_vec
        scale = np.linalg.norm(probe_vec)
        angle = 180.0 / np.pi * math.atan2(probe_vec[1, 0], probe_vec[0, 0])
        if (angle > angle_range[0]) and (angle < angle_range[1]):
            return None
        else:
            from_center = position[2:4]
            to_center = [0, 0]
            dx = to_center[0] - from_center[0]
            dy = to_center[1] - from_center[1]
            trans_m = cv2.getRotationMatrix2D((from_center[0], from_center[1]), -1 * angle, scale)
            trans_m[0][2] += dx
            trans_m[1][2] += dy
            dst = cv2.warpAffine(image, trans_m, (int(width), int(height)))
            return dst

    def get_transform(self, from_shape, to_shape):
        """
        计算变换矩阵A,使得y=A*x
        :param from_shape: 变换之前的形状x，形式为矩阵，[list]
        :param to_shape: 变换之后的形状y，形式为矩阵，[list]
        :return: A
        """
        assert from_shape.shape[0] == to_shape.shape[0] and from_shape.shape[0] % 2 == 0
        sigma_from = 0.0
        sigma_to = 0.0
        cov = np.matrix([[0.0, 0.0], [0.0, 0.0]])
        # compute the mean and cov
        from_shape_points = from_shape.reshape(from_shape.shape[0] // 2, 2)
        to_shape_points = to_shape.reshape(to_shape.shape[0] // 2, 2)
        mean_from = from_shape_points.mean(axis=0)
        mean_to = to_shape_points.mean(axis=0)
        for i in range(from_shape_points.shape[0]):
            temp_dis = np.linalg.norm(from_shape_points[i] - mean_from)
            sigma_from += temp_dis * temp_dis
            temp_dis = np.linalg.norm(to_shape_points[i] - mean_to)
            sigma_to += temp_dis * temp_dis
            cov += (to_shape_points[i].transpose() - mean_to.transpose()) * (from_shape_points[i] - mean_from)
        sigma_from = sigma_from / to_shape_points.shape[0]
        sigma_to = sigma_to / to_shape_points.shape[0]
        cov = cov / to_shape_points.shape[0]
        # compute the affine matrix
        s = np.matrix([[1.0, 0.0], [0.0, 1.0]])
        u, d, vt = np.linalg.svd(cov)
        if np.linalg.det(cov) < 0:
            if d[1] < d[0]:
                s[1, 1] = -1
            else:
                s[0, 0] = -1
        r = u * s * vt
        c = 1.0
        if sigma_from != 0:
            c = 1.0 / sigma_from * np.trace(np.diag(d) * s)
        tran_b = mean_to.transpose() - c * r * mean_from.transpose()
        tran_m = c * r
        return tran_m, tran_b

    def list2col_matrix(self, pts_list):
        """
        列表转为列矩阵
        :param pts_list:点列表e.g[x0,y0,x1,y1,x2,y1],[list]
        :return:
        """
        assert len(pts_list) > 0
        col_mat = []
        for i in range(len(pts_list)):
            col_mat.append(pts_list[i][0])
            col_mat.append(pts_list[i][1])
        col_mat = np.matrix(col_mat).transpose()
        return col_mat

    def calc_dis(self, point1, point2):
        """
        计算两个点的欧式距离
        :param point1:二维坐标e.g.[12.3, 34.1],list
        :param point2:二维坐标e.g.[12.3, 34.1],list
        :return:两个点的欧式距离
        """
        return np.sqrt((point2[1] - point1[1]) ** 2 + (point2[0] - point1[0]) ** 2)

    def reorder_vertexes(self, xy_list):
        """
        对文本线的四个顶点坐标进行重新排序，按照逆时针排序
        :param xy_list: 文本线的四个顶点坐标, [array]
        :return:
        """
        reorder_xy_list = np.zeros_like(xy_list)

        # 确定第一个顶点的坐标，选择横坐标最小的作为第一个顶点
        ordered = np.argsort(xy_list, axis=0)
        xmin1_index = ordered[0, 0]
        xmin2_index = ordered[1, 0]
        if xy_list[xmin1_index, 0] == xy_list[xmin2_index, 0]:
            if xy_list[xmin1_index, 1] <= xy_list[xmin2_index, 1]:
                reorder_xy_list[0] = xy_list[xmin1_index]
                first_v = xmin1_index
            else:
                reorder_xy_list[0] = xy_list[xmin2_index]
                first_v = xmin2_index
        else:
            reorder_xy_list[0] = xy_list[xmin1_index]
            first_v = xmin1_index

        # 计算另外三个顶点与第一个顶点的正切，将值处于中间的顶点作为第三个顶点
        others = list(range(4))
        others.remove(first_v)
        k = np.zeros((len(others),))
        for index, i in zip(others, range(len(others))):
            k[i] = (xy_list[index, 1] - xy_list[first_v, 1]) \
                   / (xy_list[index, 0] - xy_list[first_v, 0] + crnn_config.epsilon)
        k_mid = np.argsort(k)[1]
        third_v = others[k_mid]
        reorder_xy_list[2] = xy_list[third_v]

        # 比较第二个顶点与第四个顶点与第一个顶点的正切与第三个顶点与第一个顶点的正切的大小，
        # 将大于中间值的顶点作为第二个顶点，另一个作为第四个顶点
        others.remove(third_v)
        b_mid = xy_list[first_v, 1] - k[k_mid] * xy_list[first_v, 0]
        second_v, fourth_v = 0, 0
        for index, i in zip(others, range(len(others))):
            # delta = y - (k * x + b)
            delta_y = xy_list[index, 1] - (k[k_mid] * xy_list[index, 0] + b_mid)
            if delta_y > 0:
                second_v = index
            else:
                fourth_v = index
        reorder_xy_list[1] = xy_list[second_v]
        reorder_xy_list[3] = xy_list[fourth_v]

        # 判断是否需要对顶点进行旋转，当第一个顶点是四边形的左下顶点时，则按照逆时针旋转一个单位
        k13 = k[k_mid]
        k24 = (xy_list[second_v, 1] - xy_list[fourth_v, 1]) / (
                xy_list[second_v, 0] - xy_list[fourth_v, 0] + crnn_config.epsilon)
        if k13 < k24:
            tmp_x, tmp_y = reorder_xy_list[3, 0], reorder_xy_list[3, 1]
            for i in range(2, -1, -1):
                reorder_xy_list[i + 1] = reorder_xy_list[i]
            reorder_xy_list[0, 0], reorder_xy_list[0, 1] = tmp_x, tmp_y
        return [reorder_xy_list[1], reorder_xy_list[0], reorder_xy_list[3], reorder_xy_list[2]]


class ImageGenerate(object):
    def __init__(self,
                 img_base_path,
                 font_style_path,
                 text_size_limit,
                 font_size,
                 font_color,
                 train_images_path,
                 train_labels_path,
                 test_images_path,
                 test_labels_path,
                 train_test_ratio,
                 num_samples,
                 dictionary_file,
                 margin=20,
                 write_mode='w',
                 use_blank=False,
                 num_process=1):
        """
        生成类代码图像
        :param img_base_path: 背景文件夹路径，[str]
        :param font_style_path: 字体文件夹路径，包括中英文字体文件夹，[dict]
        :param text_size_limit: 文本字符个数范围列表e.g.[1,8]，[list]
        :param font_size: 文本字体大小列表e.g.[24,32,36]，[list]
        :param font_color: 文本字体颜色列表e.g.[[0, 0, 0], [255, 36, 36]]，[list]
        :param train_images_path: 训练集图片保存路径，[str]
        :param train_labels_path: 训练集标签保存路径，[str]
        :param test_images_path:测试集图片保存路径，[str]
        :param test_labels_path:测试集标签保存路径，[str]
        :param train_test_ratio: 训练集测试集比例，[float]
        :param num_samples: 生成样本总数，[int]
        :param dictionary_file: 字典文件路径,[str]
        :param margin: 文本离背景图的边距
        :param write_mode: 数据写入模式，'w':write,'a':add，[str]
        :param use_blank: 是否使用空格,[boolean]
        :param num_process: 并行生成样本的进程数
        """
        self.img_base_path = img_base_path
        self.font_style_path = font_style_path
        self.text_size_limit = text_size_limit
        self.font_size = font_size
        self.font_color = font_color
        self.train_images_path = train_images_path
        self.train_labels_path = train_labels_path
        self.test_images_path = test_images_path
        self.test_labels_path = test_labels_path
        self.train_test_ratio = train_test_ratio
        self.num_samples = num_samples
        self.dictionary_file = dictionary_file
        assert write_mode in ['w', 'a'], "write mode should be 'w'(write) or 'a'(add)"
        self.write_mode = write_mode
        self.use_blank = use_blank
        self.num_process = num_process
        self.margin = margin
        self.base_image_paths = None
        self.list_words = None
        self.used_ch_word = list()
        self.ch_fonts_list = os.listdir(self.font_style_path['ch'])
        self.en_fonts_list = os.listdir(self.font_style_path['en'])
        super().__init__()

    def generate_image(self, start_end):
        """
        生成样本图片并保存
        :param start_end: 开始ID和结尾ID的list,[list]
        :return:
        """
        # check dir and files
        train_test_offset = start_end[0] + (start_end[1] - start_end[0]) * self.train_test_ratio
        for i in range(start_end[0], start_end[1]):
            # get base image by order
            base_img_path = self.base_image_paths[
                (i - start_end[0]) * len(self.base_image_paths) // (start_end[1] - start_end[0])]

            # choice font_color depend on base image
            if os.path.basename(base_img_path).split('_')[1] == '0':
                font_color = random.choice(self.font_color[3:])
            elif os.path.basename(base_img_path).split('_')[1] == '1':
                font_color = random.choice(self.font_color[0:6] + self.font_color[12:])
            elif os.path.basename(base_img_path).split('_')[1] == '2':
                font_color = random.choice(self.font_color[0:12] + self.font_color[15:])
            elif os.path.basename(base_img_path).split('_')[1] == '3':
                font_color = random.choice(self.font_color[0:16])

            # create image draw
            base_img = Image.open(base_img_path)
            base_img_width, base_img_height = base_img.size
            draw = ImageDraw.Draw(base_img)
            while 1:
                try:
                    # randomly choice font size
                    font_size = random.choice(self.font_size)
                    # randomly choice words str
                    words_str_len = random.randint(self.text_size_limit[0], self.text_size_limit[1])
                    only_latin, words_str = self.get_word_str(words_str_len)
                    # randomly choice font style
                    if only_latin:
                        font_style_path = random.choice(self.en_fonts_list)
                        font_style_path = os.path.join(self.font_style_path['en'], font_style_path)
                    else:
                        font_style_path = random.choice(self.ch_fonts_list)
                        font_style_path = os.path.join(self.font_style_path['ch'], font_style_path)

                    font = ImageFont.truetype(font_style_path, font_size)
                    words_str_width, words_str_height = draw.textsize(words_str, font)
                    x0 = random.randint(self.margin, base_img_width - self.margin - words_str_width)
                    y0 = random.randint(self.margin, base_img_height - self.margin - words_str_height)
                    draw.text((x0, y0), words_str, tuple(font_color), font=font)
                    # save Image
                    x_left = x0 - random.randint(0, self.margin)
                    y_top = y0 - random.randint(0, self.margin)
                    x_right = x0 + words_str_width + random.randint(0, self.margin)
                    y_bottom = y0 + words_str_height + random.randint(0, self.margin)
                    base_img = np.asarray(base_img)[:, :, 0:3]
                    image = base_img[y_top:y_bottom, x_left:x_right]
                    image = Image.fromarray(image)
                    if i < train_test_offset:
                        image_dir = self.train_images_path
                        labels_path = self.train_labels_path
                    else:
                        image_dir = self.test_images_path
                        labels_path = self.test_labels_path
                    image_name = 'img_' + str(i).zfill(len(str(self.num_samples))) + '.jpg'
                    image_save_path = os.path.join(image_dir, image_name)
                    image.save(image_save_path)
                    # save labels
                    with open(labels_path, 'a', encoding='utf-8')as fa:
                        fa.write(image_name + '\t' + words_str + '\n')
                    break
                except Exception as e:
                    continue

    def generate_image_multi_process(self, num_process=None):
        """
        多进程生成样本图片并保存
        :return:
        """
        if num_process is None:
            num_process = self.num_process
        self.base_image_paths = [os.path.join(self.img_base_path, img) for img in
                                 os.listdir(self.img_base_path)]
        words = [Counter(extract_words_i) for extract_words_i in
                 self.extract_words(open(self.dictionary_file, encoding="utf-8").read())]
        self.list_words = [list(words_i.keys()) for words_i in words]
        # check dir and files
        check_path([self.train_images_path,
                    self.train_labels_path,
                    self.test_images_path,
                    self.test_labels_path])
        if self.write_mode == 'w':
            clear_content([self.train_images_path,
                           self.train_labels_path,
                           self.test_images_path,
                           self.test_labels_path])
        data_offset = self.num_samples // num_process
        processes = list()
        for i in trange(0, self.num_samples, data_offset):
            if i + data_offset >= self.num_samples:
                processes.append(Process(target=self.generate_image, args=([i, self.num_samples],)))
            else:
                processes.append(Process(target=self.generate_image, args=([i, i + data_offset],)))
        for process in processes:
            process.start()
        for process in processes:
            process.join()

    def extract_words(self, text):
        """
        提取文字
        :param text:all char about en and ch divided by \n
        :return:word_list,e.g[['1','2',..],['a','b',...,'A','B',...],[',','!',...],['甲','风',...]]
        """
        words_list = text.split('\n')
        words_list = [i.replace(' ', '') for i in words_list]
        words_list = [[j for j in i] for i in words_list]
        if self.use_blank:
            words_list.append([' '])
        return words_list

    def get_word_str(self, length):
        """
        generate word str randomly
        :param length: length of word str
        :return:
        """
        word_str = ''
        self.used_ch_word = list()
        only_latin = False
        # only latin char
        if random.random() < 0.2:
            for i in range(length):
                if self.use_blank and (i == 0 or i == length - 1):
                    words_list_i = random.choice(self.list_words[:3])
                else:
                    if self.use_blank and random.random() < 0.2:
                        words_list_i = random.choice(self.list_words[:3] + self.list_words[-1])
                    else:
                        words_list_i = random.choice(self.list_words[:3])
                word_str += random.choice(words_list_i)
            only_latin = True
        else:
            for i in range(length):
                if self.use_blank and (i == 0 or i == length - 1):
                    words_list_i = random.choice(self.list_words[:-1])
                else:
                    if self.use_blank and random.random() < 0.2:
                        words_list_i = random.choice(self.list_words)
                    else:
                        words_list_i = random.choice(self.list_words[:-1])
                word_str += random.choice(words_list_i)
        return only_latin, word_str


def check_path(path_list):
    """
    检查路径列表中的路径是否存在，如不存在就生存文件夹或者文件
    :param path_list: path list,[list]
    :return:
    """
    for path in path_list:
        if not os.path.exists(path) and '.' not in path[2:]:
            os.mkdir(path)
        elif not os.path.exists(path) and '.' in path[2:]:
            with open(path, 'w', encoding='utf-8') as fw:
                fw.write('')


def clear_content(path_list):
    """
    清空文件夹和文件内容
    :param path_list: path list,[list]
    :return:
    """
    for path in path_list:
        if os.path.isdir(path):
            shutil.rmtree(path)
            os.mkdir(path)
        elif os.path.isfile(path):
            os.remove(path)
            with open(path, 'w', encoding='utf-8') as fw:
                fw.write('')


def do_text_cut(write_mode):
    print("{0}".format('text cutting...').center(100, '='))
    print('train_test_ratio={0}\nfilter_ratio={1}\nfilter_height={2}'
          '\nis_transform={3}\nangle_range={4}\nwrite_mode={5}\nuse_blank={6}\nnum_process={7}'.format(
        crnn_config.train_test_ratio,
        crnn_config.filter_ratio,
        crnn_config.filter_height,
        crnn_config.is_transform,
        crnn_config.angle_range,
        write_mode,
        crnn_config.use_blank,
        crnn_config.num_process))
    print('=' * 100)
    text_cut = TextCut(org_images_path=crnn_config.org_images_path,
                       org_labels_path=crnn_config.org_labels_path,
                       cut_train_images_path=crnn_config.cut_train_images_path,
                       cut_train_labels_path=crnn_config.cut_train_labels_path,
                       cut_test_images_path=crnn_config.cut_test_images_path,
                       cut_test_labels_path=crnn_config.cut_test_labels_path,
                       train_test_ratio=crnn_config.train_test_ratio,
                       filter_ratio=crnn_config.filter_ratio,
                       filter_height=crnn_config.filter_height,
                       is_transform=crnn_config.is_transform,
                       angle_range=crnn_config.angle_range,
                       write_mode=write_mode,
                       use_blank=crnn_config.use_blank,
                       num_process=crnn_config.num_process
                       )
    text_cut.data_load_multi_process()


def do_image_generate(write_mode):
    print("{0}".format('image generating...').center(100, '='))
    print('train_test_ratio={0}\nnum_samples={1}\nmargin={2}\nwrite_mode={3}\nuse_blank={4}\nnum_process={5}'
          .format(crnn_config.train_test_ratio, crnn_config.num_samples, crnn_config.margin, write_mode,
                  crnn_config.use_blank,
                  crnn_config.num_process))
    image_generate = ImageGenerate(img_base_path=crnn_config.base_img_dir,
                                   font_style_path=crnn_config.font_style_path,
                                   text_size_limit=crnn_config.text_size_limit,
                                   font_size=crnn_config.font_size,
                                   font_color=crnn_config.font_color,
                                   train_images_path=crnn_config.train_images_path,
                                   train_labels_path=crnn_config.train_label_path,
                                   test_images_path=crnn_config.test_images_path,
                                   test_labels_path=crnn_config.test_label_path,
                                   train_test_ratio=crnn_config.train_test_ratio,
                                   num_samples=crnn_config.num_samples,
                                   dictionary_file=crnn_config.dictionary_file,
                                   margin=crnn_config.margin,
                                   write_mode=write_mode,
                                   use_blank=crnn_config.use_blank,
                                   num_process=crnn_config.num_process)
    image_generate.generate_image_multi_process()


def do_generate_charset(label_path, charset_path):
    """
    生成字符集文件
    :param label_path: 训练的label地址
    :param charset_path: 字符集文件地址
    :return:
    """
    print("{0}".format('charset generating...').center(100, '='))
    print('label_path={0}\ncharset_path={1}'.format(label_path, charset_path))
    print('=' * 100)
    charset_generate.generate_charset(label_path, charset_path)


if __name__ == '__main__':
    do_text_cut(write_mode='w')
    do_image_generate(write_mode='a')
    # do_generate_charset(crnn_config.train_label_path, crnn_config.charset_path)