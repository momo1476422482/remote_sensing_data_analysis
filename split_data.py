import codecs
import shapely.geometry as shgeo
from pathlib import Path
import numpy as np
from typing import List, Dict,Tuple
import cv2


class split_data:
    def __init__(self,
                 basepath: Path,
                 outpath: Path,
                 code: str = "utf-8",
                 gap: int = 512,
                 subsize: int = 1024,
                 thresh: float = 0.7,
                 choosebestpoint: bool = True,
                 ext=".png",

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
        :param padding: if to padding the images so that all the images have the same size
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

        if not self.outpath.is_dir():
            self.outpath.mkdir(parents=True, exist_ok=True)
        if not self.outimagepath.is_dir():
            self.outimagepath.mkdir(parents=True, exist_ok=True)
        if not self.outlabelpath.is_dir():
            self.outlabelpath.mkdir(parents=True, exist_ok=True)

    # ================================================================
    @staticmethod
    def dota2poly(path_label: Path) -> List[Dict]:
        """
             parse the dota ground truth in the format:
             [(x1, y1), (x2, y2), (x3, y3), (x4, y4)]
         """
        filename = path_label.name
        objects = []
        f = open(filename, 'r')
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

    # =====================================================================================
    @staticmethod
    def convert_poly_coordinate_orig2sub(left: float, up: float, coord_poly_orig: np.ndarray) -> np.ndarray:
        coord_poly_sub = np.zeros(len(coord_poly_orig))
        for i in range(int(len(coord_poly_orig) / 2)):
            coord_poly_sub[i * 2] = int(coord_poly_orig[i * 2] - left)
            coord_poly_sub[i * 2 + 1] = int(coord_poly_orig[i * 2 + 1] - up)
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

    # =====================================================================================
    def crop_patch_label(self, objects: List[Dict], subimgname: str, left: float, up: float, right: float,
                         down: float) -> None:
        path_output = self.outlabelpath / subimgname + ".txt"
        path_output.mkdir(parents=True, exist_ok=True)
        imgpoly = shgeo.Polygon([(left, up), (right, up), (right, down), (left, down)])

        with path_output.open("w", encoding="utf-8") as f_out:
            for obj in objects:
                gtpoly = shgeo.Polygon([(obj["poly"][0], obj["poly"][1]), (obj["poly"][2], obj["poly"][3]),
                                        (obj["poly"][4], obj["poly"][5]), (obj["poly"][6], obj["poly"][7]),
                                        ])
                if gtpoly.area <= 0:
                    continue
                inter_poly, inter_iou = self.calculate_inter_iou(gtpoly, imgpoly)

                if inter_iou >= self.thresh and len(coord_poly_orig) == 4:
                    coord_poly_orig = list(shgeo.polygon.orient(inter_poly, sign=1).exterior.coords)[0:-1]
                    if self.choosebestpoint:
                        coord_poly_orig = self.choose_best_pointorder_fit_another(
                            coord_poly_orig, obj["poly"]
                        )
                    coord_poly_sub = self.convert_poly_coordinate_orig2sub(left, up, coord_poly_orig)
                    f_out.write(" ".join(list(map(str, coord_poly_sub))) + " " + obj["name"] + " " + str(
                        obj["difficult"]) + "\n")

    # =====================================================================================
    def crop_patch_img(self, img:np.ndarray,subimgname: str, left: float, up: float, right: float,
                         down: float) -> None:
        path_output = self.outimagepath / subimgname + self.ext
        path_output.mkdir(parents=True, exist_ok=True)
        subimg = img[up: down, left: right]
        cv2.imwrite(path_output, subimg)

    # =====================================================================================
    def crop_one_patch(self, resizeimg:np.ndarray,objects: List[Dict], subimgname: str, left: float, up: float, right: float,
                         down: float) -> None:
        self.crop_patch_img(resizeimg, subimgname, left, up, right, down)
        self.crop_patch_label(self, objects, subimgname, left, up, right, down)

    # =====================================================================================
    def filter_coordinate(self, coor: np.ndarray) -> np.ndarray:
        for index, item in enumerate(coor):
            if item <= 1:
                coor[index] = 1
            elif item >= self.subsize:
                coor[index] = self.subsize
        return coor

    def split_one_image(self):
        pass

    def split_images(self):
        pass


def SplitSingle(self, name, rate, extent):
    """
        split a single image and ground truth
    :param name: image name
    :param rate: the resize scale for the image
    :param extent: the image format
    :return:
    """
    img = cv2.imread(os.path.join(self.imagepath, name + extent))
    if np.shape(img) == ():
        return
    fullname = os.path.join(self.labelpath, name + ".txt")
    objects = util.parse_dota_poly2(fullname)
    for obj in objects:
        obj["poly"] = list(map(lambda x: rate * x, obj["poly"]))
        # obj['poly'] = list(map(lambda x: ([2 * y for y in x]), obj['poly']))

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
            self.savepatches(resizeimg, objects, subimgname, left, up, right, down)
            if up + self.subsize >= height:
                break
            else:
                up = up + self.slide
        if left + self.subsize >= weight:
            break
        else:
            left = left + self.slide


def splitdata(self, rate):
    """
    :param rate: resize rate before cut
    """
    imagelist = GetFileFromThisRootDir(self.imagepath)
    imagenames = [
        util.custombasename(x)
        for x in imagelist
        if (util.custombasename(x) != "Thumbs")
    ]
    if self.num_process == 1:
        for name in imagenames:
            self.SplitSingle(name, rate, self.ext)
    else:

        # worker = partial(self.SplitSingle, rate=rate, extent=self.ext)
        worker = partial(
            split_single_warp, split_base=self, rate=rate, extent=self.ext
        )
        self.pool.map(worker, imagenames)
