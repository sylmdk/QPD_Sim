import numpy as np
import cv2
import math
try:
    from skimage.measure import block_reduce
except ImportError:
    def block_reduce(src, block_size, func=np.sum):
        bh, bw = block_size
        h = src.shape[0] // bh * bh
        w = src.shape[1] // bw * bw
        view = src[:h, :w].reshape(h // bh, bh, w // bw, bw)
        if func is np.sum:
            return view.sum(axis=(1, 3))
        return func(func(view, axis=3), axis=1)
try:
    from skimage.transform import rotate as sk_rotate
except ImportError:
    def sk_rotate(src, angle):
        h, w = src.shape[:2]
        center = ((w - 1) / 2.0, (h - 1) / 2.0)
        mat = cv2.getRotationMatrix2D(center, angle, 1.0)
        return cv2.warpAffine(src, mat, (w, h), flags=cv2.INTER_LINEAR, borderMode=cv2.BORDER_CONSTANT)


class QpdSimulate(object):
    def __init__(self, qsc_configs):
        self.ch_rgb = 3
        self.qpd_cell = 2

        self.qsc_configs = qsc_configs

        self.disable = False
        self.blur_sigma = 0.0

        self.params_local_shading = None
        if np.random.rand() < qsc_configs['prob_shading']:
            self.params_local_shading = self.gen_params_shading()

        self.params_local_blur = None
        if np.random.rand() < qsc_configs['prob_blur']:
            self.params_local_blur, blur_sigma = self.gen_params_blur()
            self.blur_sigma += blur_sigma

        self.params_local_reorder = None
        if np.random.rand() < qsc_configs['prob_reorder']:
            self.params_local_reorder, blur_sigma = self.gen_params_reorder()
            self.blur_sigma += blur_sigma

        self.params_local_rdm_mix = None
        if np.random.rand() < qsc_configs['prob_rdm_mix']:
            self.params_local_rdm_mix, blur_sigma = self.gen_params_rdm_mix()
            self.params_fore_local_rdm_mix, _ = self.gen_params_rdm_mix()
            self.blur_sigma += blur_sigma

    def gen_params_shading(self):
        range_h, range_w = self.qsc_configs['reorder_range'], self.qsc_configs['reorder_range']
        color_h, color_w = self.qsc_configs['reorder_color'], self.qsc_configs['reorder_color']
        if np.random.rand() < 0.3:
            if np.random.rand() < 0.5:
                range_h = 0.01
            else:
                range_w = 0.01
        reorder_range_h = np.random.uniform(1.0 - range_h, 1.0 + range_h) + \
                          np.random.uniform(-color_h, color_h, size=(1, 1, self.ch_rgb))

        reorder_range_w = np.random.uniform(1.0 - range_w, 1.0 + range_w) + \
                          np.random.uniform(-color_w, color_w, size=(1, 1, self.ch_rgb))
        return reorder_range_h, reorder_range_w

    def apply_local_shading(self, src_img, params_local_shading):
        size_h, size_w = src_img.shape[:2]
        ch_rgb = self.ch_rgb
        qpd_cell = self.qpd_cell
        reorder_range_h, reorder_range_w = params_local_shading

        # [1, 1, qpd_cell * qpd_cell, ch_rgb]
        shading_gain = np.stack([reorder_range_h * reorder_range_w,
                                 reorder_range_h * (2 - reorder_range_w),
                                 (2 - reorder_range_h) * reorder_range_w,
                                 (2 - reorder_range_h) * (2 - reorder_range_w)], axis=2)

        err_h = np.random.uniform(-0.03, 0.03, size=(2,))
        err_h = np.interp(np.arange(size_h // qpd_cell) / (size_h // qpd_cell - 1), (0, 1), (err_h[0], err_h[1]))
        err_h = err_h.reshape((size_h // qpd_cell, 1, 1, 1))

        err_w = np.random.uniform(-0.03, 0.03, size=(2,))
        err_w = np.interp(np.arange(size_w // qpd_cell) / (size_w // qpd_cell - 1), (0, 1), (err_w[0], err_w[1]))
        err_w = err_w.reshape((1, size_w // qpd_cell, 1, 1))

        shading_map = err_h * err_w * shading_gain
        shading_map /= np.mean(shading_map, axis=2, keepdims=True)

        shading_map = np.reshape(shading_map, (size_h // qpd_cell, size_w // qpd_cell, qpd_cell, qpd_cell, ch_rgb))
        shading_map = np.transpose(shading_map, (0, 2, 1, 3, 4)).reshape((size_h, size_w, ch_rgb))

        dst_img = src_img * shading_map
        return dst_img

    def gen_params_rdm_mix(self):
        size_mat = self.qpd_cell * self.qpd_cell
        if np.random.rand() < self.qsc_configs['prob_flatten']:
            mix_matrix = np.zeros((size_mat, size_mat, self.ch_rgb), dtype=np.float32)
            mat_random = np.random.uniform(0.0, self.qsc_configs['mix_energey'], size=(size_mat, self.ch_rgb))
            mat_random[0, :] = 1.0 - np.sum(mat_random, axis=0) + mat_random[0, :]
            mix_matrix[0, :, :] = mat_random                                # [0.25~? , 0~0.25, 0~0.25, 0~0.25]
            mix_matrix[1, :, :] = np.roll(mat_random, (1, 0), axis=(0, 1))  # [0~0.25, 0.25~? , 0~0.25, 0~0.25]
            mix_matrix[2, :, :] = np.roll(mat_random, (2, 0), axis=(0, 1))  # [0~0.25 , 0~0.25, 0.25~?, 0~0.25]
            mix_matrix[3, :, :] = np.roll(mat_random, (3, 0), axis=(0, 1))  # [0~0.25 , 0~0.25, 0~0.25, 0.25~?]
        else:
            mix_matrix = np.random.uniform(0.0, self.qsc_configs['mix_energey'], size=(size_mat, size_mat, self.ch_rgb))
            for i in range(size_mat):
                mix_matrix[i, i, :] = 1.0 - np.sum(mix_matrix[i, :, :], axis=0) + mix_matrix[i, i, :]
        diag = mix_matrix[np.arange(size_mat), np.arange(size_mat), :]
        return mix_matrix, 1.0 - np.mean(diag)

    def apply_local_rdm_mix(self, src_img, params_local_rdm_mix):
        size_h, size_w = src_img.shape[:2]
        qpd_cell = self.qpd_cell
        mix_matrix = params_local_rdm_mix

        src_img = np.reshape(src_img, (size_h // qpd_cell, qpd_cell, size_w // qpd_cell, qpd_cell, 3))
        src_img = src_img.transpose((0, 2, 4, 1, 3)).reshape((size_h // qpd_cell, size_w // qpd_cell, 3,
                                                              qpd_cell * qpd_cell))
        dst_img = np.zeros_like(src_img)
        for c in range(3):
            dst_img[..., c, :] = src_img[..., c, :] @ mix_matrix[:, :, c]
        dst_img = np.reshape(dst_img, (size_h // qpd_cell, size_w // qpd_cell, 3, qpd_cell, qpd_cell))
        dst_img = dst_img.transpose((0, 3, 1, 4, 2)).reshape((size_h, size_w, 3))
        return dst_img

    def gen_params_reorder(self):
        range_h, range_w = self.qsc_configs['reorder_range'], self.qsc_configs['reorder_range']
        color_h, color_w = self.qsc_configs['reorder_color'], self.qsc_configs['reorder_color']
        if np.random.rand() < 0.3:
            if np.random.rand() < 0.5:
                range_h = 0.01
            else:
                range_w = 0.01
        reorder_range_h = np.random.uniform(1.0 - range_h, 1.0 + range_h) + \
                          np.random.uniform(-color_h, color_h, size=(1, 1, self.ch_rgb))

        reorder_range_w = np.random.uniform(1.0 - range_w, 1.0 + range_w) + \
                          np.random.uniform(-color_w, color_w, size=(1, 1, self.ch_rgb))
        return (reorder_range_h, reorder_range_w), np.mean(reorder_range_h + reorder_range_w) / 2.0

    def apply_local_reorder(self, src_img, params_local_reorder):
        size_h, size_w = src_img.shape[:2]
        ch_rgb = self.ch_rgb
        qpd_cell = self.qpd_cell
        reorder_range_h, reorder_range_w = params_local_reorder

        split_h = np.random.uniform(-0.05, 0.05, size=(2,))
        split_h = np.interp(np.arange(size_w // qpd_cell) / (size_w // qpd_cell - 1),
                            (0, 1), (split_h[0], split_h[1]))
        split_h = reorder_range_h + split_h.reshape((1, size_w // qpd_cell, 1))

        split_w = np.random.uniform(-0.05, 0.05, size=(2,))
        split_w = np.interp(np.arange(size_h // qpd_cell) / (size_h // qpd_cell - 1),
                            (0, 1), (split_w[0], split_w[1]))
        split_w = reorder_range_w + split_w.reshape((size_h // qpd_cell, 1, 1))

        dst_img = np.zeros_like(src_img)
        img_stack = [
            src_img[0::qpd_cell, 0::qpd_cell, :],
            src_img[0::qpd_cell, 1::qpd_cell, :],
            src_img[1::qpd_cell, 0::qpd_cell, :],
            src_img[1::qpd_cell, 1::qpd_cell, :],
        ]

        dst_img[0::qpd_cell, 0::qpd_cell, :] = \
            img_stack[0] * (np.minimum(split_h + 0.0, 1.0) * np.minimum(split_w + 0.0, 1.0)) + \
            img_stack[1] * (np.maximum(split_h - 1.0, 0.0) * np.minimum(split_w + 0.0, 1.0)) + \
            img_stack[2] * (np.minimum(split_h + 0.0, 1.0) * np.maximum(split_w - 1.0, 0.0)) + \
            img_stack[3] * (np.maximum(split_h - 1.0, 0.0) * np.maximum(split_w - 1.0, 0.0))

        dst_img[0::qpd_cell, 1::qpd_cell, :] = \
            img_stack[0] * (np.maximum(1.0 - split_h, 0.0) * np.minimum(split_w + 0.0, 1.0)) + \
            img_stack[1] * (np.minimum(2.0 - split_h, 1.0) * np.minimum(split_w + 0.0, 1.0)) + \
            img_stack[2] * (np.maximum(1.0 - split_h, 0.0) * np.maximum(split_w - 1.0, 0.0)) + \
            img_stack[3] * (np.minimum(2.0 - split_h, 1.0) * np.maximum(split_w - 1.0, 0.0))

        dst_img[1::qpd_cell, 0::qpd_cell, :] = \
            img_stack[0] * (np.minimum(split_h + 0.0, 1.0) * np.maximum(1.0 - split_w, 0.0)) + \
            img_stack[1] * (np.maximum(split_h - 1.0, 0.0) * np.maximum(1.0 - split_w, 0.0)) + \
            img_stack[2] * (np.minimum(split_h + 0.0, 1.0) * np.minimum(2.0 - split_w, 1.0)) + \
            img_stack[3] * (np.maximum(split_h - 1.0, 0.0) * np.minimum(2.0 - split_w, 1.0))

        dst_img[1::qpd_cell, 1::qpd_cell, :] = \
            img_stack[0] * (np.maximum(1.0 - split_h, 0.0) * np.maximum(1.0 - split_w, 0.0)) + \
            img_stack[1] * (np.minimum(2.0 - split_h, 1.0) * np.maximum(1.0 - split_w, 0.0)) + \
            img_stack[2] * (np.maximum(1.0 - split_h, 0.0) * np.minimum(2.0 - split_w, 1.0)) + \
            img_stack[3] * (np.minimum(2.0 - split_h, 1.0) * np.minimum(2.0 - split_w, 1.0))
        return dst_img

    def gen_params_blur_bak(self, sigma, max_shift):
        qpd_cell = self.qpd_cell

        def kernel_bin(kernel, size):
            mask = np.zeros((size, size), dtype=np.float32)
            mask[:size // 2 + 1, :size // 2 + 1] = 1.0
            mask[size // 2, :] *= 0.5
            mask[:, size // 2] *= 0.5

            pos00 = np.sum(kernel * mask)
            pos01 = np.sum(kernel * mask[::-1, :])
            pos10 = np.sum(kernel * mask[:, ::-1])
            pos11 = np.sum(kernel * mask[::-1, ::-1])
            return np.asarray([[pos00, pos01],
                               [pos10, pos11]])

        def gauss_kernel(size=5, sigmaX=1.0, sigmaY=1.0):
            k_X = cv2.getGaussianKernel(ksize=size, sigma=sigmaX)
            k_Y = cv2.getGaussianKernel(ksize=size, sigma=sigmaY)
            kernel = np.outer(k_X, k_Y).astype(np.float32)

            kernel = sk_rotate(kernel, np.random.uniform(0, 180))
            kernel = kernel ** np.random.uniform(0.7, 1.3)
            kernel = kernel / np.sum(kernel)
            return kernel

        mix_matrix = np.zeros((stride, stride, 4))  # 4: R, Gr, Gb, B
        sign = np.asarray([[-1, -1], [1, -1], [-1, 1], [1, 1]])
        direct = sign[np.random.randint(len(sign))]

        focus_shift = np.random.randint(1, ksize - 1 - self.qsc_configs['max_shift'], size=(2,))
        if np.random.rand() < 0.3:
            if np.random.rand() < 0.5:
                focus_shift = (ksize, focus_shift[1])
            else:
                focus_shift = (focus_shift[0], ksize)

        local_shift = np.random.randint(1, max_shift + 1, size=(2,))
        if check_data:
            print('qsc params:', sigma, max_shift)
            print('qsc params:', focus_shift, local_shift)

        for c in range(4):
            sigma_color = sigma + np.random.uniform(0.0, 1.0)
            for i in range(stride):
                kernel = gauss_kernel(ksize, sigma_color + np.random.uniform(0.0, 0.5),
                                      sigma_color + np.random.uniform(0.0, 0.5))
                kernel = np.pad(kernel, ((ksize, ksize), (ksize, ksize)))
                # print(np.round(kernel, 3))

                shift = focus_shift + (local_shift + np.random.randint(0, 2, size=(2,))) * direct * sign[i]
                shift = np.clip(shift, self.qsc_configs['min_shift'], ksize) * sign[i]
                # print(sign[i], 'shift', shift)
                # assert np.sign(shift[0]) == sign[i][0], np.sign(shift[1]) == sign[i][1]

                kernel = np.roll(kernel, shift, axis=(0, 1))
                kernel = kernel_bin(kernel, kernel.shape[0]).reshape(-1)
                kernel = kernel / np.sum(kernel)
                mix_matrix[i, :, c] = kernel
        return mix_matrix

    def gen_params_blur(self):
        qpd_cell = self.qpd_cell

        pad = self.qsc_configs['pad'] // 2 * 2 + 1
        ksize = self.qsc_configs['ksize'] // 2 * 2 + 1
        radius = ksize // 2
        sigma = np.random.uniform(*self.qsc_configs['sigma'])
        ray_center = np.random.randint(-radius // 2, radius // 2 + 1, size=(2, ))

        def gauss_kernel(size=5, sigmaX=1.0, sigmaY=1.0):
            k_X = cv2.getGaussianKernel(ksize=size, sigma=sigmaX)
            k_Y = cv2.getGaussianKernel(ksize=size, sigma=sigmaY)
            kernel = np.outer(k_X, k_Y).astype(np.float32)

            kernel = sk_rotate(kernel, np.random.uniform(0, 180))
            kernel = kernel ** np.random.uniform(0.7, 1.3)
            kernel = kernel / np.sum(kernel)
            return kernel

        sign = np.asarray([[-1, -1], [-1, 1], [1, -1], [1, 1]])
        mix_matrix = np.zeros((qpd_cell * qpd_cell, qpd_cell * qpd_cell, self.ch_rgb))
        for c in range(self.ch_rgb):
            for i in range(qpd_cell * qpd_cell):
                ray_shift = np.random.randint(radius // 2, radius * 2 // 3 + 1, size=(2, ))
                ray_pos = ray_center * np.asarray(sign[i]) + ray_shift
                ray_pos = np.clip(ray_pos, 1, radius)
                rot = round(math.atan2(ray_pos[1], ray_pos[0]) / np.pi * 180)
                end = int(np.round(np.sqrt(np.sum(ray_pos ** 2))))
                start = np.random.randint(1, end + 1)
                length = end - start
                # print(i, sigma, ray_center, ray_shift, ray_pos, rot, start, end, length)
                # print(end, length)

                motion = np.zeros((ksize * 2 + 1, ksize * 2 + 1), dtype=np.float32)
                motion[ksize + start: ksize + end + 1, ksize] = np.random.uniform(1.0 / (length + 1), 1.0, size=(length + 1,))
                motion = sk_rotate(motion, rot)[::-1, ::-1]
                # cv2.imwrite(dst_path + '%d_%d_0_motion.png' % (c, i), motion / np.max(motion) * 255)

                kernel = gauss_kernel(
                    ksize * 2 + 1,
                    sigma + np.random.uniform(0.0, 0.5),
                    sigma + np.random.uniform(0.0, 0.5),
                )
                kernel = cv2.filter2D(kernel.astype(np.float32), -1, motion.astype(np.float32),
                                      borderType=cv2.BORDER_CONSTANT)
                kernel = np.pad(kernel, ((pad, 0), (pad, 0)))
                # cv2.imwrite(dst_path + '%d_%d_1_kernel.png' % (c, i), kernel / np.max(kernel) * 255)

                kernel /= np.sum(kernel)
                if sign[i][0] == -1:
                    kernel = kernel[::-1, :]
                if sign[i][1] == -1:
                    kernel = kernel[:, ::-1]
                # cv2.imwrite(dst_path + '%d_%d_2_local_blur.png' % (c, i), kernel / np.max(kernel) * 255)
                shape = kernel.shape[0]
                mat = block_reduce(kernel, (shape//2, shape//2), np.sum).reshape(-1)
                assert np.argmax(mat) == i
                mix_matrix[i, :, c] = mat
                # print(np.round(mat, 3))
        return {'sigma': sigma, 'mat': mix_matrix}, sigma / 2.0

    def apply_local_blur(self, src_img, params_local_blur):
        size_h, size_w = src_img.shape[:2]
        qpd_cell = self.qpd_cell
        mix_matrix = params_local_blur['mat']

        src_img = np.reshape(src_img, (size_h // qpd_cell, qpd_cell, size_w // qpd_cell, qpd_cell, 3))
        src_img = src_img.transpose((0, 2, 4, 1, 3)).reshape((size_h // qpd_cell, size_w // qpd_cell, 3,
                                                              qpd_cell * qpd_cell))
        dst_img = np.zeros_like(src_img)
        for c in range(3):
            dst_img[..., c, :] = src_img[..., c, :] @ mix_matrix[:, :, c]
        dst_img = np.reshape(dst_img, (size_h // qpd_cell, size_w // qpd_cell, 3, qpd_cell, qpd_cell))
        dst_img = dst_img.transpose((0, 3, 1, 4, 2)).reshape((size_h, size_w, 3))
        return dst_img

    def __call__(self, src_img, mask_fore=None):
        qsc_configs = self.qsc_configs
        dst_img = np.copy(src_img)
        if self.disable:
            return dst_img

        if mask_fore is None:
            mask_fore = np.zeros(src_img.shape[:2], dtype=np.float32)

        # if self.params_local_shading is not None:
        #     dst_img = self.apply_local_shading(dst_img)

        if self.params_local_blur is not None:
            dst_img = self.apply_local_blur(dst_img, self.params_local_blur)

        elif self.params_local_reorder is not None:
            dst_img = self.apply_local_reorder(dst_img, self.params_local_reorder)

        if self.params_local_rdm_mix is not None:
            if np.sum(mask_fore) > 0:
                dst_img_1 = self.apply_local_rdm_mix(dst_img.copy(), self.params_local_rdm_mix)
                dst_img_2 = self.apply_local_rdm_mix(dst_img.copy(), self.params_fore_local_rdm_mix)
                dst_img = dst_img_1 * (1 - mask_fore[..., None]) + dst_img_2 * mask_fore[..., None]
            else:
                dst_img = self.apply_local_rdm_mix(dst_img, self.params_local_rdm_mix)
        return dst_img


if __name__ == '__main__':
    dst_path = './qsc_sim/'

    qsc_configs = {
        'prob_shading': 0.0,

        'prob_blur': 0.6,
        'pad': 1,
        'ksize': 21,
        'sigma': (0.2, 1.8),

        'prob_reorder': 0.9,
        'reorder_range': 0.35,  # (1+0.35) / (1-0.35) ~= 2.0
        'reorder_color': 0.10,  # (1+0.35+0.15) / (1-0.35-0.15) = 3.0

        'prob_rdm_mix': 0.5,
        'mix_energey': 0.25,  ####  [0.25~? , 0~0.25, 0~0.25, 0~0.25]
        'prob_flatten': 0.8,  ####  []
    }
    for k in range(10):
        qpd_sim = QpdSimulate(qsc_configs)

        img = np.zeros((128, 128, 3), dtype=np.float32) + 0.5

        # qpd_sim.gen_params_blur()
        img_qsc = qpd_sim(img)

        print(k, np.mean(img_qsc[:2, :2, :], axis=(0, 1)))
        print(np.max(img_qsc, axis=(0, 1)) / np.min(img_qsc, axis=(0, 1)))
        cv2.imwrite(dst_path + '%d_img_qsc.png' % k,
                    img_qsc[..., ::-1].astype(np.float32) * 255)
        cv2.imwrite(dst_path + '%d_img_qsc_1.png' % k,
                    img_qsc[..., 0].astype(np.float32) * 255)
        cv2.imwrite(dst_path + '%d_img_qsc_2.png' % k,
                    img_qsc[..., 1].astype(np.float32) * 255)
        cv2.imwrite(dst_path + '%d_img_qsc_3.png' % k,
                    img_qsc[..., 2].astype(np.float32) * 255)
