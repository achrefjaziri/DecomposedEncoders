from scipy import signal, ndimage, stats
import numpy as np
import random
import math
import sys
import torch
class GaussianNoiseEstimator():
    def __init__(self, p=0.1):
        self.p = p
        self.sobel1 = [[-1, -2, -1],
                       [0, 0, 0],
                       [1, 2, 1]]
        self.sobel2 = [[-1, 0, 1],
                       [-2, 0, 2],
                       [-1, 0, 1]]

        self.laplacian = [[1, -2, 1],  # LaPlacioan Operator
                          [-2, 4, -2],
                          [1, -2, 1]]

    # returns a mask, for all pixels,
    # if it belongs to p % of the positions with least of the edges
    def homogeneous_regions(self, img, intensity=None, saturation=None):
        # gradient calculation with sobel filters

        g1 = signal.convolve2d(img, self.sobel1, mode='same')
        g2 = signal.convolve2d(img, self.sobel2, mode='same')
        g = np.absolute(g1) + np.absolute(g2)
        # cutoff (don't use them for calculation) extreme areas
        if ((intensity is not None)
                and (np.count_nonzero(intensity > 0.95 * 255) < 0.5 * img.shape[0] * img.shape[1])
                and (np.count_nonzero(intensity < 0.05 * 255) < 0.5 * img.shape[0] * img.shape[1])):
            # dont use positions with high and low intensity I>0.05*255 and I< 0.95*255*3
            g[intensity > 0.95 * 255] = g.max()
            g[intensity < 0.05 * 255] = g.max()
        if saturation is not None:
            # dont use positions with high and low sturation S>0.05 and S< 0.95
            g[saturation > 0.95] = g.max()
            g[saturation < 0.05] = g.max()
        threshhold_g = g.min()  # threshhold = the G value when the accumulated histogram reaches p% of the whole image
        if (not (g.max() == g.min())):
            # compute the histogram of G # TODO check if could be replaced by sort
            hist_g = np.histogram(g, bins=int(g.max() - g.min()), range=(g.min(), g.max()))
            # calculate threshhold
            p_pixels = self.p * img.shape[0] * img.shape[1]
            sum_pixels = 0
            for gi, g_value in enumerate(hist_g[0]):
                sum_pixels += g_value
                if sum_pixels > p_pixels:
                    threshhold_g = g.min() + gi
                    break
        masked_img = np.ma.masked_where(g > threshhold_g, img)
        return masked_img

    def estimateNoise(self, img):
        # see Immerkaer Fast Noise Estimation Algorithm
        H, W = img.shape
        conv = signal.convolve2d(img, self.laplacian, mode='valid')
        abs_values = np.absolute(conv)
        sigma = np.sum(np.sum(abs_values))
        return sigma * np.sqrt(0.5 * np.pi) / (6 * (W - 2) * (H - 2))

    def estimateNoiseExtend(self, img, intensity=None, saturation=None):
        img_suppressed_structures = signal.convolve2d(img, self.laplacian, mode='valid')

        intensity = None if intensity is None else intensity[1:-1, 1:-1]  # pixels lost by convolution
        saturation = None if saturation is None else saturation[1:-1, 1:-1]  # pixels lost by convolution
        edge_mask = self.homogeneous_regions(img[1:-1, 1:-1], intensity, saturation).mask

        abs_residuals = np.ma.array(np.absolute(img_suppressed_structures), mask=edge_mask)
        summed_abs_residuals = np.sum(np.ma.sum(abs_residuals))
        N = (abs_residuals.shape[0] * abs_residuals.shape[1]) - np.count_nonzero(edge_mask)
        # print(summed_abs_residuals, np.count_nonzero(edge_mask), N)
        # print(summed_abs_residuals * np.sqrt(0.5 * np.pi) / (6 * N))
        return summed_abs_residuals * np.sqrt(0.5 * np.pi) / (6 * N)

    def __call__(self, x, intensity=None, saturation=None):
        return self.estimateNoiseExtend(x, intensity, saturation)
class Intensity():
    def __call__(self, x):
        return x.sum(axis=2) // 3


class Saturation():
    def __call__(self, x):
        i = x.sum(axis=2)
        min_c = (np.minimum(np.minimum(x[:, :, 0], x[:, :, 1]), x[:, :, 2]))
        return 1 - np.divide(min_c, i, out=np.zeros(min_c.shape), where=i != 0)


class NormalizedRG():
    def __init__(self, conf=False, p_H0=0.5, p_H1=0.5, randomConf=False):
        self.conf = conf
        self.channels = 2
        self.randomconf = randomConf  # only for testing
        if self.conf:
            self.p_H0 = p_H0
            self.p_H1 = p_H1
            self.noiseEstimator = GaussianNoiseEstimator()
            self.intensity = Intensity()
            self.saturation = Saturation()

    def __call__(self, x):
        x = self.color_correct_rg(x)
        out = rg = self.normalizedRG(x)
        if self.conf:
            if self.randomconf:
                c = np.random.rand(x.shape[0], x.shape[1])
            else:
                c = self.color_confidence_estimation(x)
                if np.isnan(np.concatenate([rg, c], axis=2)).any():
                    print("rg:", np.isnan(rg).any())
                    print("c:", np.isnan(c).any())

                    print("rg:", rg)
                    print("c:", c)

                    sys.exit()
            out = np.concatenate([rg, c], axis=2)
        return out

    def normalizedRG(self, x):
        intensity = x.sum(axis=2, keepdims=True)
        rgb = np.divide(x, intensity, out=np.full(x.shape, 1 / 3), where=intensity != 0)
        return rgb[:, :, :2]

    def find_max_in_epsilon_area(self, values, exp_mode, epsilon, prec=0.01):
        r_hist = np.histogram(values.reshape(-1, 1), int(1 / prec))
        # smooth
        smoothed_r_hist = ndimage.gaussian_filter1d(r_hist[0], 0.1)
        # area
        min, max = 0, len(r_hist[0])
        if np.any(r_hist[1] > 1 / 3 - epsilon):
            min = np.where(r_hist[1] > 1 / 3 - epsilon)[0][0]
        if np.any(r_hist[1] > 1 / 3 + epsilon):
            max = np.where(r_hist[1] > 1 / 3 + epsilon)[0][0]
        if min == max:
            return r_hist[1][min]
        smoothed_r_hist = smoothed_r_hist[min:max]
        # get max
        mode = r_hist[1][min + np.argmax(smoothed_r_hist)]
        return mode

    def rgToRGB(self, rgimg, intensity):
        img = np.zeros((rgimg.shape[0], rgimg.shape[1], 3))
        img[:, :, :2] = rgimg
        img[:, :, 2] = 1 - (rgimg[:, :, 0] + rgimg[:, :, 1])
        img = (img * intensity)
        img[img > 255] = 255  # color correction "error" handling
        return img.astype(np.uint8)

    def color_correct_rg(self, img):
        img_rg = self.normalizedRG(img)
        r_mode = self.find_max_in_epsilon_area(img_rg[:, :, 0], 1 / 3, 0.05, prec=0.01)
        g_mode = self.find_max_in_epsilon_area(img_rg[:, :, 1], 1 / 3, 0.05, prec=0.01)
        rg_c = np.zeros(img_rg.shape)
        rg_c[:, :, 0] = img_rg[:, :, 0] * 1 / 3 * 1 / (r_mode+0.00000001)
        rg_c[:, :, 1] = img_rg[:, :, 1] * 1 / 3 * 1 / (g_mode+0.00000001)
        img = self.rgToRGB(rg_c, img.sum(axis=2, keepdims=True))
        return img

    def color_confidence_estimation(self, img):
        rg = self.normalizedRG(img)
        intensity, saturation = self.intensity(img), self.saturation(img)
        noise = [self.noiseEstimator(img[:, :, i], intensity, saturation) for i in range(img.shape[2])]
        # print(noise)
        d = self.mahalanobi_dist(rg, (1 / 3, 1 / 3), img, noise)

        lambdas = [0, 10, 100, np.median(d) * 0.5]
        weights = [0.25, 0.25, 0.25, 0.25]
        conf = np.expand_dims(
            self.posterior_H1_given_d(d, lambdas[0], lambdas[0], np.median(d) * 1.75, self.p_H0, self.p_H1), axis=2) * \
               weights[0]
        for l, w in zip(lambdas[1:], weights[1:]):
            tmp_conf = self.posterior_H1_given_d(d, l, l, np.median(d) * 1.75, self.p_H0, self.p_H1) * w
            conf = np.concatenate([conf, np.expand_dims(tmp_conf, axis=2)], axis=2)
        p = np.sum(conf, axis=2)
        return np.repeat(p[:, :, np.newaxis], 2, axis=2)  # same confidence for r and g

    def cov(self, img, sigma, r, g):
        var_R = np.square(sigma[0])
        var_G = np.square(sigma[1])
        var_B = np.square(sigma[2])
        R = img[:, :, 0].astype(np.float64)
        G = img[:, :, 1].astype(np.float64)
        B = img[:, :, 2].astype(np.float64)
        eps = 1e-7
        S = R + G + B + eps
        var_I = ((var_R + var_G + var_B) / 3) + eps

        var_r1 = (var_R / var_I) * (1 - 2 * r) + 3 * r * r
        var_r = var_I / np.square(S) * var_r1

        var_g1 = (var_G / var_I) * (1 - 2 * g) + 3 * g * g
        var_g = var_I / np.square(S) * var_g1

        cov_rg1 = -((var_G / var_I) * r) - ((var_R / var_I) * g) + 3 * r * g
        cov_rg = var_I / np.square(S) * cov_rg1
        return np.array([[var_r, cov_rg], [cov_rg, var_g]])

    def mahalanobi_dist(self, rg_measurement, mean, img, sigma):
        eps = 1e-20
        r_mean, g_mean = mean
        cov_ma = self.cov(img, sigma, r_mean, g_mean)
        s_factor = 1 / ((cov_ma[0][0] * cov_ma[1][1]) - np.square(cov_ma[0][1]) + eps)
        s11 = s_factor * cov_ma[1][1]
        s12 = s_factor * - cov_ma[0][1]
        s22 = s_factor * cov_ma[0][0]
        r_diff = rg_measurement[:, :, 0] - r_mean
        g_diff = rg_measurement[:, :, 1] - g_mean
        mah_dist = r_diff * r_diff * s11 + 2 * r_diff * g_diff * s12 + g_diff * g_diff * s22
        return mah_dist

    def moments_noncentral_chi(self, min_lambda, max_lambda):
        a = min_lambda
        b = max_lambda
        if b == 0:
            mu = 0
        else:
            mu = 1 / (b - a) * ((2 * b + (b * b) / 2) - (2 * a + (a * a) / 2))
        var = 4 + 2 * (a + b) - 1 / 6 * a * b + 1 / 12 * (a * a + b * b)
        return mu, var

    def posterior_H1_given_d(self, mah_dist, max_lambdaH0, min_lambda_H1, max_lambdaH1, p_H0, p_H1):
        min_lambdaH0 = 0

        mu_d_H0, var_d_H0 = self.moments_noncentral_chi(min_lambdaH0, max_lambdaH0)
        mu_d_H1, var_d_H1 = self.moments_noncentral_chi(min_lambda_H1, max_lambdaH1)

        p_d_H0 = stats.norm.pdf(mah_dist, mu_d_H0, np.sqrt(var_d_H0))
        p_d_H1 = stats.norm.pdf(mah_dist, mu_d_H1, np.sqrt(var_d_H1))
        out = np.full(p_d_H1.shape, 0.5)
        out[mah_dist > mu_d_H0] = 1
        p = np.divide((p_d_H1 * p_H1), (p_d_H1 * p_H1 + p_d_H0 * p_H0), out=out,
                      where=(p_d_H1 * p_H1 + p_d_H0 * p_H0) > 1e-9)
        return p

    def __str__(self):
        return "rg" + ("(conf)" if self.conf else "")


class LBP(object):
    '''create p neighbouring positions in r distance to center pixel
       threshholding neighboring pixel with center pixel
       if neighbour >= center pixel => 1 else 0
       create decimal number from binary result'''

    def __init__(self, radius=1, points=8, conf=False, no_decimal=False):
        self.radius = radius
        self.points = points
        self.withConf = conf
        if self.withConf:
            self.noiseEstimator = GaussianNoiseEstimator()
        self.no_decimal = no_decimal

    def positions(self):
        a = 2 * math.pi / self.points
        return [(int(round(self.radius * math.cos(a * i))),
                 int(round(self.radius * math.sin(a * i))))
                for i in range(self.points)]

    def noise_prop(self, sigma):
        # Calculate covariance matrix S
        S = np.full((self.points, self.points), sigma ** 2)  # self.points gives dimenion of binary number
        np.fill_diagonal(S, 2 * sigma ** 2)
        return S

    def mahalanobis_distance(self, point, mean, inv_S):
        # print (inv_S)
        return np.matmul(np.matmul(np.transpose(point - mean), inv_S), (point - mean))

    def moments_noncentral_chi(self, min_lambda, max_lambda, debug=False):
        a = min_lambda
        b = max_lambda
        if b == 0:
            mu = 0
        else:
            mu = 1 / (b - a) * ((2 * b + (b * b) / 2) - (2 * a + (a * a) / 2))
        var = 4 + 2 * (a + b) - 1 / 6 * a * b + 1 / 12 * (a * a + b * b)
        return mu, var

    def posterior_H1_given_d(self, mah_dist, max_lambdaH0, min_lambda_H1, max_lambdaH1, p_H0=0.5, p_H1=0.5):
        min_lambdaH0 = 0

        mu_d_H0, var_d_H0 = self.moments_noncentral_chi(min_lambdaH0, max_lambdaH0)
        mu_d_H1, var_d_H1 = self.moments_noncentral_chi(min_lambda_H1, max_lambdaH1)

        p_d_H0 = stats.norm.pdf(mah_dist, mu_d_H0, np.sqrt(var_d_H0))
        p_d_H1 = stats.norm.pdf(mah_dist, mu_d_H1, np.sqrt(var_d_H1))
        out = np.full(p_d_H1.shape, 0.5)
        out[mah_dist > mu_d_H0] = 1
        p = np.divide((p_d_H1 * p_H1), (p_d_H1 * p_H1 + p_d_H0 * p_H0), out=out,
                      where=(p_d_H1 * p_H1 + p_d_H0 * p_H0) > 1e-9)
        return p

    def get_inv_S(self, d, sigma):
        inv_S = np.full((d, d), -1 / ((d + 1) * sigma ** 2))  # self.points gives dimenion of binary number
        np.fill_diagonal(inv_S, d / ((d + 1) * sigma ** 2))
        return inv_S

    def conf(self, intensity, diffs):
        if False:
            return np.ones((diffs.shape[0], diffs.shape[1], 1))
        mah = np.zeros((diffs.shape[0], diffs.shape[1]))
        sigma = self.noiseEstimator(intensity, intensity) + 1e-7
        inv_S = self.get_inv_S(self.points, sigma)
        threshhold = self.mahalanobis_distance(np.full((self.points, 1), 2 * sigma ** 2), np.zeros((self.points, 1)),
                                               inv_S)
        for i in range(diffs.shape[0]):
            for j in range(diffs.shape[1]):
                mah[i, j] = self.mahalanobis_distance(diffs[i, j].reshape(self.points, 1), np.zeros((self.points, 1)),
                                                      inv_S)
        lambdas = [0, 1, 10, np.median(mah) * 0.5]
        weights = [0.25, 0.25, 0.25, 0.25]
        conf = np.expand_dims(self.posterior_H1_given_d(mah, lambdas[0], lambdas[0], np.median(mah) * 1.75), axis=2) * \
               weights[0]
        for l, w in zip(lambdas[1:], weights[1:]):
            tmp_conf = self.posterior_H1_given_d(mah, l, l, np.median(mah) * 1.75) * w
            conf = np.concatenate([conf, np.expand_dims(tmp_conf, axis=2)], axis=2)
        conf = np.sum(conf, axis=2, keepdims=True)
        return conf

    def __call__(self, data):
        # compare the neighboring pixels to that of the central pixel
        neigbour = np.zeros((data.shape[0], data.shape[1], self.points))
        max_x, max_y = data.shape[1], data.shape[0]
        for i, (y, x) in enumerate(self.positions()):
            neigbour[max(0, -y):min(max_y, max_y - y),
            max(0, -x):min(max_x, max_x - x),
            i] = data[max(0, y):min(max_y, max_y + y),
                 max(0, x):min(max_x, max_x + x)]
        diff = neigbour - np.repeat(data[:, :, np.newaxis], self.points, axis=2)  # height,width, num_points

        if self.withConf:
            c = self.conf(data, diff)

        # convert to binary: 0 if lessequall
        diff[diff >= 0] = 1
        diff[diff < 0] = 0
        if self.no_decimal:
            return diff

        #Here do the rotation

        # convert to decimal
        for i in range(self.points):
            diff[:, :, i] = diff[:, :, i] * 2 ** i
        d = np.sum(diff, axis=2)
        out = d / float(2 ** (self.points + 1) - 1)  # max decimal number, defined by number of points
        if self.withConf:
            out = np.concatenate([np.expand_dims(out, axis=2), c], axis=2)
        return out

    def __str__(self):
        return "LBP(radius_%i_points_%i)" % (self.radius, self.points)


class MultipleLBP():
    def __init__(self, parameters, conf=False, no_decimal=False):
        self.lbps = []
        self.conf = conf
        self.parameters = parameters
        for r, p in parameters:
            self.lbps.append(LBP(r, p, conf, no_decimal))
        self.channels = len(self.lbps) if not no_decimal else sum([l.points for l in self.lbps])

        self.intensity = Intensity()

        if conf:
            self.forward = self.forward_with_conf
        else:
            self.forward = self.forward_without_conf
        self.no_decimal = no_decimal

    def forward_without_conf(self, x):

        # convert to grayscale
        x = self.intensity(x)
        print('after intensity', x.shape)
        out = np.zeros((x.shape[0], x.shape[1], self.channels))
        c_it = 0
        for t in self.lbps:
            out_tmp = t(x)
            c_tmp = c_it + out_tmp.shape[2]
            out[:, :, c_it:c_tmp] = out_tmp
            c_it = c_tmp
        return out

    def forward_with_conf(self, x):
        # print("start lbp")
        # convert to grayscale
        # x = self.intensity(x)
        out = np.zeros((x.shape[0], x.shape[1], 2 * len(self.lbps)))
        # print(out.shape)
        for i, t in enumerate(self.lbps):
            li = t(x)
            out[:, :, i] = li[:, :, 0]
            out[:, :, len(self.lbps) + i] = li[:, :, 1]
        # print("done")
        return out

    def __call__(self, x):
        return self.forward(x)

    def __str__(self):
        return "LBP(%s)" % (str(self.parameters))



#These two lambda functions can be passed to transforms.Compose as follows: transforms.Compose([
            #transforms.Resize([input_size, input_size]),
            #transforms.ToTensor(),
            #transforms.Lambda(lbp_lambda)]


def lbp_lambda(x):
  lbp_transform = LBP(radius=3, points=24)
  #print('shape in lbp_lambda',x.shape)
  img_out = torch.Tensor(lbp_transform(x[0].detach().numpy()))
  img_out=torch.unsqueeze(img_out, 0)
  return img_out

def rg_lambda(x):
  rg_norm = NormalizedRG(conf=False)
  #print('shape i lbp_lambda',x.shape)
  img_out = torch.Tensor(rg_norm(x.permute(1,2,0).detach().numpy())).permute(2,0,1)
  #img_out=torch.unsqueeze(img_out, 0)
  return img_out



#for the DTCWT you can use the package provided by https://github.com/fbcotter/pytorch_wavelets