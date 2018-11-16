import cv2
import numpy as np
from matplotlib import pyplot as plt


class FeatureMatching:
    # 官方教程的目标图片是query image
    def __init__(self, query_image='001.jpg'):
        # 创建SURF探测器，并设置Hessian阈值，由于效果不好，我改成了SIFT方法
        # self.min_hessian = 400（surf方法使用）
        # self.surf = cv2.xfeatures2d.SURF_create(min_hessian)
        self.sift = cv2.xfeatures2d.SIFT_create()
        self.img_query = cv2.imread(query_image, 0)
        # 读取一个目标模板
        if self.img_query is None:
            print("Could not find train image " + query_image)
            raise SystemExit
        self.shape_query = self.img_query.shape[:2]  # 注意，rows，cols，对应的是y和x，后面的角点坐标的x,y要搞清楚
        #  detectAndCompute函数返回关键点和描述符
        self.key_query, self.desc_query = self.sift.detectAndCompute(self.img_query, None)
        # 设置FLANN对象
        FLANN_INDEX_KDTREE = 0
        index_params = dict(algorithm=FLANN_INDEX_KDTREE, trees=5)
        search_params = dict(checks=50)
        self.flann = cv2.FlannBasedMatcher(index_params, search_params)
        # 保存最后一次计算的单应矩阵
        self.last_hinv = np.zeros((3, 3))
        # 保存没有找到目标的帧的数量
        self.num_frames_no_success = 0
        # 最大连续没有找到目标的帧的次数
        self.max_frames_no_success = 5
        self.max_error_hinv = 50.
        # 防止第一次检测到时由于单应矩阵变化过大而退出
        self.first_frame = True

    def _extract_features(self, frame):
        # self.min_hessian = 400
        # sift = cv2.xfeatures2d.SURF_create(self.min_hessian)
        sift = cv2.xfeatures2d.SIFT_create()
        #  detectAndCompute函数返回关键点和描述符，mask为None
        key_train, desc_train = sift.detectAndCompute(frame, None)
        return key_train, desc_train

    def _match_features(self, desc_frame):
        # 函数返回一个训练集和询问集的一致性列表
        matches = self.flann.knnMatch(self.desc_query, desc_frame, k=2)
        # 丢弃坏的匹配
        good_matches = []
        # matches中每个元素是两个对象，分别是与测试的点距离最近的两个点的信息
        # 留下距离更近的那个匹配点
        for m, n in matches:
            if m.distance < 0.7 * n.distance:
                good_matches.append(m)
        return good_matches

    def _detect_corner_points(self, key_frame, good_matches):
        # 将所有好的匹配的对应点的坐标存储下来
        src_points = np.float32([self.key_query[m.queryIdx].pt for m in good_matches]).reshape(-1, 1, 2)
        dst_points = np.float32([key_frame[m.trainIdx].pt for m in good_matches]).reshape(-1, 1, 2)
        H, mask = cv2.findHomography(src_points, dst_points, cv2.RANSAC, 5.0)
        matchesMask = mask.ravel().tolist()
        # 有了H单应性矩阵，我们可以查看源点被映射到img_query中的位置
        # src_corners = np.float32([(0, 0), (self.shape_train[1], 0), (self.shape_train[1], self.shape_train[0]),
        #                           (0, self.shape_train[0])]).reshape(-1, 1, 2)
        h, w = self.img_query.shape[:2]
        src_corners = np.float32([[0, 0], [0, h - 1], [w - 1, h - 1], [w - 1, 0]]).reshape(-1, 1, 2)
        # perspectiveTransform返回点的列表
        dst_corners = cv2.perspectiveTransform(src_corners, H)
        return dst_corners, H, matchesMask

    def _center_keypoints(self, frame, key_frame, good_matches):
        dst_size = frame.shape[:2]
        # 将图片的对象大小缩小到query image的1/2（书里是train image，和官方命名相反而已）
        scale_row = 1. / self.shape_query[0] * dst_size[0] / 2.
        bias_row = dst_size[0] / 4.
        scale_col = 1. / self.shape_query[1] * dst_size[1] / 2.
        bias_col = dst_size[1] / 4.
        # 将每个点应用这样的变换
        src_points = [self.key_query[m.queryIdx].pt for m in good_matches]
        dst_points = [key_frame[m.trainIdx].pt for m in good_matches]
        dst_points = [[x * scale_row + bias_row, y * scale_col + bias_col] for x, y in dst_points]
        Hinv, _ = cv2.findHomography(np.array(src_points), np.array(dst_points), cv2.RANSAC, 5.0)
        img_center = cv2.warpPerspective(frame, Hinv, dst_size, flags=2)
        return img_center

    def _frontal_keypoints(self, frame, H):
        Hinv = np.linalg.inv(H)
        dst_size = frame.shape[:2]
        img_front = cv2.warpPerspective(frame, Hinv, dst_size, flags=2)
        return img_front

    def match(self, frame):
        img_train = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        cv2.waitKey(0)
        shape_train = img_train.shape[:2]  # rows,cols

        # 获得好的matches
        key_train, desc_train = self._extract_features(img_train)
        good_matches = self._match_features(desc_train)

        # 为了让RANSAC方法可以尽快工作，至少需要4个好的匹配，否则视为匹配失败
        if len(good_matches) < 4:
            self.num_frames_no_success += 1
            return False, frame
        # 画出匹配的点
        img_match = cv2.drawMatchesKnn(self.img_query, self.key_query, img_train, key_train, [good_matches], None,
                                       flags=2)
        plt.imshow(img_match), plt.show()

        # 在query_image中找到对应的角点
        dst_corners, Hinv, matchesMask = self._detect_corner_points(key_train, good_matches)
        # 如果这些点位置距离图片内太远（至少20像素），那么意味着我们没有找到我们感兴趣
        # 的目标或者说是目标没有完整的出现在图片内，对于这两种情况，我们都视为False
        dst_ravel = dst_corners.ravel()
        if (dst_ravel > shape_train[0] + 20).any() and (dst_ravel > -20).any() \
                and (dst_ravel > shape_train[1] + 20).any():
            self.num_frames_no_success += 1
            return False, frame

        # 如果4个角点没有围出一个合理的四边形，意味着我们可能没有找到我们的目标。
        # 通过行列式计算四边形面积
        area = 0.
        for i in range(0, 4):
            D = np.array([[1., 1., 1.],
                          [dst_corners[i][0][0], dst_corners[(i + 1) % 4][0][0], dst_corners[(i + 2) % 4][0][0]],
                          [dst_corners[i][0][1], dst_corners[(i + 1) % 4][0][1], dst_corners[(i + 2) % 4][0][1]]])
            area += abs(np.linalg.det(D)) / 2.
        area /= 2.
        # 以下注释部分是书中的计算方式，我使用时是错误的
        # for i in range(0, 4):
        #     next_i = (i + 1) % 4
        #     print(dst_corners[i][0][0])
        #     print(dst_corners[i][0][1])
        #     area += (dst_corners[i][0][0] * dst_corners[next_i][0][1] - dst_corners[i][0][1] * dst_corners[next_i][0][
        #         0]) / 2.
        # 如果面积太大或太小，将它排除
        if area < np.prod(shape_train) / 16. or area > np.prod(shape_train) / 2.:
            self.num_frames_no_success += 1
            return False, frame

        # 如果我们此时发现的单应性矩阵和上一次发现的单应性矩阵变化太大，意味着我们可能找到了
        # 另一个对象，这种情况我们丢弃这个帧并返回False
        # 这里要用到self.max_frames_no_success的，作用就是距离上一次发现的单应性矩阵
        # 不能太久时间，如果时间过长的话，完全可以将上一次的hinv抛弃，使用当前计算得到
        # 的Hinv
        recent = self.num_frames_no_success < self.max_frames_no_success
        similar = np.linalg.norm(Hinv - self.last_hinv) < self.max_error_hinv
        if recent and not similar and not self.first_frame:
            self.num_frames_no_success += self.num_frames_no_success
            return False, frame
        # 第一次检测标志置否
        self.first_frame = False
        self.num_frames_no_success = 0
        self.last_hinv = Hinv

        draw_params = dict(matchColor=(0, 255, 0),  # draw matches in green color
                           singlePointColor=None,
                           matchesMask=matchesMask,  # draw only inliers
                           flags=2)
        img_dst = cv2.polylines(img_train, [np.int32(dst_corners)], True, (0, 255, 255), 5, cv2.LINE_AA)
        img_dst = cv2.drawMatches(self.img_query, self.key_query, img_dst, key_train, good_matches, None,
                                  **draw_params)
        plt.imshow(img_dst)
        plt.show()

        img_center = self._center_keypoints(frame, key_train, good_matches)
        plt.imshow(img_center)
        plt.show()

        # 转换成正面视角
        img_front = self._frontal_keypoints(frame, Hinv)
        plt.imshow(img_front)
        plt.show()
        return True, img_dst





