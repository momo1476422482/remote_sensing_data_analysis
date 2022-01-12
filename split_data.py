import codecs
import shapely.geometry as shgeo
from pathlib import Path
import numpy as np
from typing import List, Dict, Tuple
import cv2
import pandas as pd


class split_data:
    def __init__(self,
                 basepath: Path,
                 outpath: Path,
                 code: str = "utf-8",
                 gap: int = 56,
                 subsize: int = 256,
                 thresh: float = 0.7,
                 choosebestpoint: bool = True,
                 ext: str = ".png",
                 save_mode: str = 'txt'
                 ):
        """
        :param basepath: base path for dota data
        :param outpath: output base path for dota data,
        the basepath and outputpath have the similar subdirectory, 'images' and 'labelTxt'
        :param code: encodeing format of txt file
        :param gap: overlap between two patches
        :param subsize: subsize of patch
        :param thresh: the thresh determine whether to keep the instance if the instance is cut down in the process of split
        :param choosebestpoint: used to choose the first point for the
        :param ext: ext for the image format
        :param save_mode: the format of the saved label files

        """
        self.basepath = basepath
        self.outpath = outpath
        self.code = code
        self.gap = gap
        self.subsize = subsize
        self.slide = self.subsize - self.gap
        self.thresh = thresh
        self.ext = ext

        self.imagepath = self.basepath / "images"
        self.labelpath = self.basepath / "labelTxt"
        self.outimagepath = self.outpath / "images"
        self.outlabelpath = self.outpath / "labelTxt"
        self.choosebestpoint = choosebestpoint
        self.save_mode = save_mode

        if not self.outpath.is_dir():
            self.outpath.mkdir(parents=True, exist_ok=False)
        self.outimagepath.mkdir(parents=False, exist_ok=False)
        self.outlabelpath.mkdir(parents=False, exist_ok=False)

    # ================================================================
    @staticmethod
    def data2poly(path_label: Path) -> List[Dict]:
        pass

    # =====================================================================================
    @staticmethod
    def convert_poly_coordinate_orig2sub(left: float, up: float, coord_poly_orig: List[Tuple]) -> np.ndarray:

        coord_poly_sub = coord_poly_orig
        for i in range(int(len(coord_poly_orig) / 2)):
            coord_poly_sub[i * 2] = int(coord_poly_orig[i * 2][0] - left), int(coord_poly_orig[i * 2][1] - left)
            coord_poly_sub[i * 2 + 1] = int(coord_poly_orig[i * 2 + 1][0] - up), int(coord_poly_orig[i * 2 + 1][1] - up)
        return coord_poly_sub

    # =====================================================================================
    @staticmethod
    def choose_best_pointorder_fit_another(poly1: np.ndarray, poly2: np.ndarray) -> np.ndarray:
        """
        To make the two polygons best fit with each point
        """
        x1 = poly1[0]
        y1 = poly1[1]
        x2 = poly1[2]
        y2 = poly1[3]
        x3 = poly1[4]
        y3 = poly1[5]
        x4 = poly1[6]
        y4 = poly1[7]
        combinate = [
            np.array([x1, y1, x2, y2, x3, y3, x4, y4]),
            np.array([x2, y2, x3, y3, x4, y4, x1, y1]),
            np.array([x3, y3, x4, y4, x1, y1, x2, y2]),
            np.array([x4, y4, x1, y1, x2, y2, x3, y3]),
        ]
        dst_coordinate = np.array(poly2)
        distances = np.array([np.sum((coord - dst_coordinate) ** 2) for coord in combinate])
        sorted = distances.argsort()
        return combinate[sorted[0]]

    # =====================================================================================
    @staticmethod
    def calculate_inter_iou(poly1, poly2) -> Tuple[shgeo.Polygon, float]:
        inter_poly = poly1.intersection(poly2)
        inter_area = inter_poly.area
        poly1_area = poly1.area
        inter_iou = inter_area / poly1_area
        return inter_poly, inter_iou

    # ====================================================================================
    def get_poly_sub_img(self, subimgname: str, objects: List[Dict], left: float, up: float, right: float,
                         down: float) -> List[Dict]:
        """
        Get the list of label of objects existing on the subimage
        :param subimgname:
        :param objects:
        :param left:
        :param up:
        :param right:
        :param down:
        :return:
        """
        object_struct = {}
        objects_sub_img = []
        imgpoly = shgeo.Polygon([(left, up), (right, up), (right, down), (left, down)])

        for obj in objects:

            gtpoly = shgeo.Polygon([(obj["poly"][0][0], obj["poly"][0][1]), (obj["poly"][1][0], obj["poly"][1][1]),
                                    (obj["poly"][2][0], obj["poly"][2][1]), (obj["poly"][3][0], obj["poly"][3][1]),
                                    ])

            if gtpoly.area <= 0:
                continue
            inter_poly, inter_iou = self.calculate_inter_iou(gtpoly, imgpoly)

            if inter_iou >= self.thresh:
                coord_poly_orig = list(shgeo.polygon.orient(inter_poly, sign=1).exterior.coords)[0:-1]
                if len(coord_poly_orig) == 4:
                    if self.choosebestpoint:
                        coord_poly_orig = self.choose_best_pointorder_fit_another(
                            coord_poly_orig, obj["poly"]
                        )
                    coord_poly_sub = self.convert_poly_coordinate_orig2sub(left, up, coord_poly_orig)
                    coord_poly_sub = self.filter_coordinate(coord_poly_sub)
                    object_struct['imagename'] = subimgname
                    object_struct['poly'] = coord_poly_sub
                    gtpoly = shgeo.Polygon(object_struct['poly'])
                    object_struct['area'] = gtpoly.area
                    object_struct['name'] = obj['name']
                    objects_sub_img.append(object_struct)
        return objects_sub_img

    # =====================================================================================
    def crop_patch_label(self, subimgname: str, objects: List[Dict], left: float, up: float, right: float,
                         down: float) -> None:
        """
        get the corresponding label text file of the subimage (patch) and save it
        :param objects:
        :param subimgname: name of sub image
        :param left:
        :param up:
        :param right:
        :param down:
        :return:
        """
        objects_patch = self.get_poly_sub_img(subimgname, objects, left, up, right, down)
        if self.save_mode == 'txt':
            path_output = self.outlabelpath / str(subimgname + ".txt")
            with open(path_output, 'w') as f_out:
                for obj in objects_patch:
                    f_out.write(" ".join(list(map(str, obj['poly']))) + " " + obj["name"] + " " + str(
                        obj["difficult"]) + "\n")
        if self.save_mode == 'csv':
            dataframe = pd.DataFrame(objects_patch)
            dataframe.to_csv(self.outlabelpath / str(subimgname + ".csv"))

    # =====================================================================================
    def crop_patch_img(self, img: np.ndarray, subimgname: str, left: float, up: float, right: float,
                       down: float) -> None:
        """
        get the sub_image and save it
        :param img:
        :param subimgname:
        :param left:
        :param up:
        :param right:
        :param down:
        :return:
        """
        path_output = self.outimagepath / str(subimgname + self.ext)
        subimg = img[up: down, left: right]
        cv2.imwrite(str(path_output), subimg)

    # =====================================================================================
    def crop_one_patch(self, resizeimg: np.ndarray, objects: List[Dict], subimgname: str, left: float, up: float,
                       right: float,
                       down: float) -> None:
        self.crop_patch_img(resizeimg, subimgname, left, up, right, down)
        self.crop_patch_label(subimgname, objects, left, up, right, down)

    # =====================================================================================
    def filter_coordinate(self, coor: List[Tuple]) -> np.ndarray:
        coor = np.array(coor)
        for index, item in enumerate(coor):
            for i in range(2):
                if item[i] < 1:

                    coor[index][i] = 1
                elif item[i] >= self.subsize:
                    coor[index][i] = self.subsize
        return list(map(tuple, coor))

    # =====================================================================================
    def split_one_image(self, name: str, extent: str, rate: float) -> None:
        """
            split a single image and ground truth
            :param name: image name
            :param rate: the resize scale for the image
            :param extent: the image format

            """

        img = cv2.imread(str(self.imagepath / str(name + extent)))

        if np.shape(img) == ():
            return
        label_name = self.labelpath / str(name + ".txt")
        objects = self.data2poly(label_name)
        for obj in objects:
            obj["poly"] = list(map(lambda x: rate * x, obj["poly"]))

        if rate != 1:
            resizeimg = cv2.resize(
                img, None, fx=rate, fy=rate, interpolation=cv2.INTER_CUBIC
            )
        else:
            resizeimg = img
        outbasename = name + "__" + str(rate) + "__"
        weight = np.shape(resizeimg)[1]
        height = np.shape(resizeimg)[0]

        left, up = 0, 0
        while left < weight:
            if left + self.subsize >= weight:
                left = max(weight - self.subsize, 0)
            up = 0
            while up < height:
                if up + self.subsize >= height:
                    up = max(height - self.subsize, 0)
                right = min(left + self.subsize, weight - 1)
                down = min(up + self.subsize, height - 1)
                subimgname = outbasename + str(left) + "___" + str(up)
                # self.f_sub.write(name + ' ' + subimgname + ' ' + str(left) + ' ' + str(up) + '\n')

                self.crop_one_patch(resizeimg, objects, subimgname, left, up, right, down)
                if up + self.subsize >= height:
                    break
                else:
                    up = up + self.slide
            if left + self.subsize >= weight:
                break
            else:
                left = left + self.slide

    # =====================================================================================
    def split_images(self, rate: float) -> None:
        pathlist = Path(self.imagepath).glob('**/*' + self.ext)
        for path_data in pathlist:
            self.split_one_image(path_data.stem, self.ext, rate)


class split_dota(split_data):
    def __init__(self):
        super.__init__()
# =====================================================================================
    @staticmethod
    def data2poly(path_label: Path) -> List[Dict]:
        """
             parse the dota ground truth in the format:
             [(x1, y1), (x2, y2), (x3, y3), (x4, y4)]
         """

        objects = []
        f = open(path_label, 'r')
        while True:
            line = f.readline()
            if line:
                splitlines = line.strip().split(' ')
                object_struct = {}
                if len(splitlines) < 9:
                    continue
                if len(splitlines) >= 9:
                    object_struct['name'] = splitlines[8]
                if len(splitlines) == 9:
                    object_struct['difficult'] = '0'
                elif len(splitlines) >= 10:

                    object_struct['difficult'] = splitlines[9]
                object_struct['poly'] = [(float(splitlines[0]), float(splitlines[1])),
                                         (float(splitlines[2]), float(splitlines[3])),
                                         (float(splitlines[4]), float(splitlines[5])),
                                         (float(splitlines[6]), float(splitlines[7]))
                                         ]
                gtpoly = shgeo.Polygon(object_struct['poly'])
                object_struct['area'] = gtpoly.area

                objects.append(object_struct)
            else:
                break
        return objects

class split_nwpu(split_data):
    def __init__(self):
        super.__init__()
# =====================================================================================
    @staticmethod
    def data2poly(path_label: Path) -> List[Dict]:
        """
             parse the dota ground truth in the format:
             [(x1, y1), (x2, y2), (x3, y3), (x4, y4)]
         """

# ===============================================================
if __name__ == '__main__':
    filedir = Path(__file__).parent

    split = split_data(
        (filedir / "dataset" / "train"),
        (filedir / "dataset" / "train_split_test"),
        gap=200,
        subsize=855,
        choosebestpoint=False,
        save_mode='csv'

    )
    split.split_images(1)
