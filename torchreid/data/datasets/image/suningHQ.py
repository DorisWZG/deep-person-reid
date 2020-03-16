from __future__ import division, print_function, absolute_import
import re
import glob
import os.path as osp
import warnings

from ..dataset import ImageDataset


class SuningHQ(ImageDataset):
    """Market1501.

    Reference:
        Zheng et al. Scalable Person Re-identification: A Benchmark. ICCV 2015.

    URL: `<http://www.liangzheng.org/Project/project_reid.html>`_
    
    Dataset statistics:
        - identities: 1501 (+1 for background).
        - images: 12936 (train) + 3368 (query) + 15913 (gallery).
    """
    _junk_pids = [0, -1]
    dataset_dir = 'autostore'

    def __init__(self, root='', **kwargs):
        self.root = osp.abspath(osp.expanduser(root))
        self.dataset_dir = osp.join(self.root, self.dataset_dir)

        # allow alternative directory structure
        self.data_dir = self.dataset_dir

        self.train_dir = osp.join(self.data_dir, 'bounding_box_train')
        self.query_dir = osp.join(self.data_dir, 'query')
        self.gallery_dir = osp.join(self.data_dir, 'bounding_box_test')

        required_files = [
            self.data_dir, self.train_dir, self.query_dir, self.gallery_dir
        ]
        self.check_before_run(required_files)

        train = self.process_dir(self.train_dir,relabel=True)
        query = self.process_dir(self.query_dir,relabel=False)
        gallery = self.process_dir(self.gallery_dir, relabel=False)

        super(SuningHQ, self).__init__(train, query, gallery, **kwargs)

    def process_dir(self, dir_path,relabel=False):
        img_paths = glob.glob(osp.join(dir_path, '*.jpg'))

        pid_container = set()
        for img_path in img_paths:
            image_name = img_path.split("/")[-1]
            split_image_name = image_name.split("_")
            pid,camid = int(split_image_name[0]), int(split_image_name[1])
            print(img_path,pid,camid)
            if pid == -1:
                continue # junk images are just ignored
            pid_container.add(pid)

        pid2label = {pid: label for label, pid in enumerate(pid_container)}

        data = []
        for img_path in img_paths:
            image_name = img_path.split("/")[-1]
            split_image_name = image_name.split("_")
            pid,camid = int(split_image_name[0]), int(split_image_name[1])
            if pid == -1:
                continue # junk images are just ignored
            # assert 0 <= pid <= 1501 # pid == 0 means background
            # assert 1 <= camid <= 6
            camid -= 1 # index starts from 0
            if relabel:
                pid = pid2label[pid]
            data.append((img_path, pid, camid))

        return data
