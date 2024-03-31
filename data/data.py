import cv2
import logging
from pathlib import Path
import os


class Data:
    def __init__(self, logger: logging.Logger, base_path:  Path ):
        self.logger = logger
        self.classes = self.get_classes(base_path)
        self.links = self.get_files_links(base_path)

    def load_img(self, path: Path):
        self.logger.debug('load img: %s', path)
        return cv2.imread(str(path))

    def get_files_links(self,  base_path) -> list[(Path, int)]:
        return [(base_path/folder/item,  i) for i, folder in  enumerate(os.listdir(base_path)) for  item in os.listdir(base_path/folder)]

    def load_files (self, index:  int, batch_size:  int):
        return  [(cv2.imread(str(link[0])),  link[1])for link in self.links[index*batch_size: ] ]

    def get_classes(self,  base_path: Path):
        return [name for name in os.listdir(base_path)if os.path.isdir(base_path / name)]





logger = logging.getLogger(__name__)
data = Data(logger, Path('./dataset/train'))
data.get_files_links(Path('./dataset/train'))
print(data.get_classes(Path('./dataset/train')))
print(data.load_files(  1, 32 ))