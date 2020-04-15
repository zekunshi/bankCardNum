import tqdm
from crnn import config as crnn_config


def generate_charset(labels_path, charset_path):
    """
    generate char dictionary with text label
    :param labels_path:label_path: path of your text label
    :param charset_path: path for restore char dict
    :return:
    """
    with open(labels_path, 'r', encoding='utf-8') as fr:
        lines = fr.read().split('\n')
    dic = str()
    for label in tqdm.tqdm(lines[:-1]):
        for char in label:
            if char in dic:
                continue
            else:
                dic += char
    with open(charset_path, 'w', encoding='utf-8')as fw:
        fw.write(dic)


if __name__ == '__main__':
    label_path = crnn_config.train_label_path
    char_dict_path = crnn_config.charset_path
    generate_charset(label_path, char_dict_path)