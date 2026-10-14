"""Stereo rectification and matcher adapters used by disparity_gui.py."""
from pathlib import Path
import re
import time
import cv2
import numpy as np
import yaml


class CalibrationLoader(yaml.SafeLoader):
    pass


def _opencv_matrix(loader, node):
    value = loader.construct_mapping(node, deep=True)
    return np.asarray(value['data'], dtype=np.float64).reshape(value['rows'], value['cols'])


CalibrationLoader.add_constructor('tag:yaml.org,2002:opencv-matrix', _opencv_matrix)


def load_calibration(path):
    text = Path(path).read_text()
    text = re.sub(r'^%YAML:1\.0\s*', '', text)
    data = yaml.load(text, Loader=CalibrationLoader)
    if not isinstance(data, dict):
        raise ValueError('Calibration YAML must contain a mapping.')
    aliases = {
        'K1': ('K1', 'camera_matrix_left', 'M1'), 'K2': ('K2', 'camera_matrix_right', 'M2'),
        'D1': ('D1', 'dist_coeffs_left'), 'D2': ('D2', 'dist_coeffs_right'),
        'R': ('R',), 'T': ('T',),
        'R1': ('R1', 'rect_matrix_left', 'rot_matrix_left'),
        'R2': ('R2', 'rect_matrix_right', 'rot_matrix_right'),
        'P1': ('P1', 'proj_matrix_left'), 'P2': ('P2', 'proj_matrix_right'),
    }
    result = {}
    for key, names in aliases.items():
        for name in names:
            if name in data:
                value = data[name]
                if isinstance(value, dict):
                    value = np.asarray(value['data']).reshape(value['rows'], value['cols'])
                result[key] = np.asarray(value, dtype=np.float64)
                if not np.isfinite(result[key]).all():
                    raise ValueError(f'{name} contains nonfinite values.')
                break
    for key in ('K1', 'K2', 'D1', 'D2'):
        if key not in result:
            raise ValueError(f'Missing {key}; supported keys include {aliases[key]}.')
    for key in ('K1', 'K2', 'R', 'R1', 'R2'):
        if key in result and result[key].shape != (3, 3):
            raise ValueError(f'{key} must be 3 × 3.')
    for key in ('P1', 'P2'):
        if key in result and result[key].shape != (3, 4):
            raise ValueError(f'{key} must be 3 × 4.')
    if 'T' in result and (result['T'].size != 3 or np.linalg.norm(result['T']) == 0):
        raise ValueError('T must contain three values and a nonzero stereo baseline.')
    for key in ('K1', 'K2'):
        if result[key][0, 0] <= 0 or result[key][1, 1] <= 0:
            raise ValueError(f'{key} must have positive focal lengths.')
    for key in ('D1', 'D2'):
        if result[key].size not in (4, 5, 8, 12, 14):
            raise ValueError(f'{key} has an unsupported distortion coefficient count.')
    for key in ('image_width', 'image_height', 'distortion_model'):
        if key in data:
            result[key] = data[key]
    return result


def read_pair(left_path, right_path='', layout='Separate images'):
    left = cv2.imread(str(left_path), cv2.IMREAD_COLOR)
    if left is None:
        raise ValueError('Cannot decode the left/combined image. Use PNG, JPEG, TIFF or BMP.')
    if layout == 'Separate images':
        right = cv2.imread(str(right_path), cv2.IMREAD_COLOR)
        if right is None:
            raise ValueError('Cannot decode the right image.')
    elif layout == 'Side by side':
        if left.shape[1] % 2:
            raise ValueError('Side-by-side image width must be even.')
        left, right = np.split(left, 2, axis=1)
    else:
        if left.shape[0] % 2:
            raise ValueError('Top/bottom image height must be even.')
        left, right = np.split(left, 2, axis=0)
    if left.shape != right.shape:
        raise ValueError('Left and right images must have the same dimensions.')
    return left, right


def rectify_pair(left, right, calibration, alpha=0., model='Pinhole', saved=False):
    c = calibration
    size = (left.shape[1], left.shape[0])
    if 'image_width' in c and 'image_height' in c:
        if size != (int(c['image_width']), int(c['image_height'])):
            raise ValueError('Images differ from the calibration resolution. Use native calibration-size images.')
    k1, k2 = c['K1'], c['K2']
    d1, d2 = c['D1'].ravel(), c['D2'].ravel()
    q = None
    if saved:
        if not all(k in c for k in ('R1', 'R2', 'P1', 'P2')):
            raise ValueError('Saved rectification requires R1/R2/P1/P2 or rect_matrix_*/proj_matrix_* keys.')
        r1, r2, p1, p2 = (c[k] for k in ('R1', 'R2', 'P1', 'P2'))
    else:
        if not all(k in c for k in ('R', 'T')):
            raise ValueError('Computing rectification requires stereo R and T. Alternatively use saved rectification.')
        if model == 'Fisheye':
            if d1.size != 4 or d2.size != 4:
                raise ValueError('Fisheye requires four distortion coefficients per camera.')
            r1, r2, p1, p2, q = cv2.fisheye.stereoRectify(
                k1, d1, k2, d2, size, c['R'], c['T'].reshape(3, 1),
                flags=cv2.CALIB_ZERO_DISPARITY, balance=alpha, newImageSize=size)
        else:
            r1, r2, p1, p2, q, _, _ = cv2.stereoRectify(
                k1, d1, k2, d2, size, c['R'], c['T'].reshape(3, 1), alpha=alpha)
    if abs(p2[1, 3]) > abs(p2[0, 3]):
        raise ValueError('Vertical stereo baseline detected. These matchers require horizontal rectification.')
    mapper = cv2.fisheye.initUndistortRectifyMap if model == 'Fisheye' else cv2.initUndistortRectifyMap
    maps = [mapper(k, d, r, p, size, cv2.CV_32FC1)
            for k, d, r, p in ((k1, d1, r1, p1), (k2, d2, r2, p2))]
    images, masks = [], []
    for image, (mx, my) in zip((left, right), maps):
        images.append(cv2.remap(image, mx, my, cv2.INTER_LINEAR))
        masks.append((mx >= 0) & (mx < size[0]-1) & (my >= 0) & (my < size[1]-1))
    return (*images, masks, q)


def binary_factory(name):
    module = getattr(cv2, 'stereo', None)
    for owner in (module, cv2):
        if owner is None:
            continue
        factory = getattr(owner, name + '_create', None)
        if callable(factory):
            return factory
        cls = getattr(owner, name, None)
        if cls is not None and callable(getattr(cls, 'create', None)):
            return cls.create
    return None


def availability():
    result = {'StereoBM': '', 'StereoSGBM': ''}
    for name in ('StereoBinaryBM', 'StereoBinarySGBM'):
        result[name] = '' if binary_factory(name) else 'OpenCV stereo Python bindings unavailable'
    try:
        cuda = hasattr(cv2.cuda, 'createStereoSGM') and cv2.cuda.getCudaEnabledDeviceCount() > 0
    except cv2.error:
        cuda = False
    result['CUDA SGM'] = '' if cuda else 'Requires CUDA-enabled OpenCV and a compatible GPU'
    return result


DEFAULTS = dict(minDisparity=0, numDisparities=128, blockSize=7, P1=392, P2=1568,
                uniquenessRatio=10, speckleWindowSize=100, speckleRange=2,
                disp12MaxDiff=1, preFilterCap=31, textureThreshold=10, mode=0)


def wls_unavailable_reason(algorithm):
    if algorithm not in ('StereoBM', 'StereoSGBM'):
        return 'Confidence-based WLS is supported for StereoBM and StereoSGBM.'
    module = getattr(cv2, 'ximgproc', None)
    if module is None or not all(hasattr(module, name) for name in
                                ('createRightMatcher', 'createDisparityWLSFilter')):
        return 'WLS requires OpenCV contrib with ximgproc Python bindings.'
    return ''


def compute_disparity(left, right, algorithm, params, wls=None):
    p = dict(params)
    use_wls = wls is not None and wls.get('enabled', False)
    if use_wls:
        reason = wls_unavailable_reason(algorithm)
        if reason:
            raise ValueError(reason)
        if wls['lambda'] <= 0 or wls['sigmaColor'] <= 0:
            raise ValueError('WLS lambda and sigmaColor must be positive.')
    n, b = p['numDisparities'], p['blockSize']
    if n <= 0 or n % 16 or n + max(p['minDisparity'], 0) >= left.shape[1]:
        raise ValueError('Disparity count must be a positive multiple of 16 smaller than the image search width.')
    if b % 2 != 1 or b < 1 or b > min(left.shape[:2]):
        raise ValueError('Block size must be odd and fit inside the image.')
    if algorithm.endswith('BM') and 'SGBM' not in algorithm and not 5 <= b <= 255:
        raise ValueError('BM block size must be odd, between 5 and 255.')
    if ('SGBM' in algorithm or algorithm == 'CUDA SGM') and p['P2'] <= p['P1']:
        raise ValueError('P2 must be greater than P1.')
    reason = availability()[algorithm]
    if reason:
        raise ValueError(reason)
    l = cv2.cvtColor(left, cv2.COLOR_BGR2GRAY) if left.ndim == 3 else left
    r = cv2.cvtColor(right, cv2.COLOR_BGR2GRAY) if right.ndim == 3 else right
    started = time.perf_counter()
    if algorithm == 'CUDA SGM':
        if n not in (64, 128, 256):
            raise ValueError('CUDA SGM supports 64, 128 or 256 disparities.')
        matcher = cv2.cuda.createStereoSGM(p['minDisparity'], n, p['P1'], p['P2'], p['uniquenessRatio'], p['mode'])
        gl, gr = cv2.cuda_GpuMat(), cv2.cuda_GpuMat()
        gl.upload(l); gr.upload(r)
        raw = matcher.compute(gl, gr).download()
    else:
        if algorithm == 'StereoBM':
            matcher = cv2.StereoBM_create(n, b)
        elif algorithm == 'StereoSGBM':
            matcher = cv2.StereoSGBM_create()
        elif algorithm == 'StereoBinaryBM':
            matcher = binary_factory(algorithm)(n, b)
        else:
            matcher = binary_factory(algorithm)(p['minDisparity'], n, b)
        keys = ['minDisparity', 'numDisparities', 'blockSize', 'uniquenessRatio',
                'speckleWindowSize', 'speckleRange', 'disp12MaxDiff', 'preFilterCap']
        keys += ['P1', 'P2', 'mode'] if 'SGBM' in algorithm else ['textureThreshold']
        for key in keys:
            getattr(matcher, 'set' + key[0].upper() + key[1:])(p[key])
        if use_wls:
            # Construct before compute: OpenCV adjusts the left matcher for WLS.
            right_matcher = cv2.ximgproc.createRightMatcher(matcher)
            wls_filter = cv2.ximgproc.createDisparityWLSFilter(matcher)
            wls_filter.setLambda(float(wls['lambda']))
            wls_filter.setSigmaColor(float(wls['sigmaColor']))
        raw = matcher.compute(l, r)
        if use_wls:
            right_raw = right_matcher.compute(r, l)
            raw = wls_filter.filter(raw, left, disparity_map_right=right_raw)
    disparity = raw.astype(np.float32)
    if np.issubdtype(raw.dtype, np.integer):
        disparity /= 16.
    valid = np.isfinite(disparity) & (disparity > p['minDisparity'] - 1)
    if use_wls:
        roi_mask = np.zeros(valid.shape, dtype=bool)
        x, y, width, height = wls_filter.getROI()
        roi_mask[y:y+height, x:x+width] = True
        valid &= roi_mask
    return disparity, valid, (time.perf_counter() - started) * 1000


def evaluate(left, right, disparity, valid, masks):
    """Photometric warp error plus coverage; heuristic, not ground-truth accuracy."""
    h, w = disparity.shape
    x, y = np.meshgrid(np.arange(w, dtype=np.float32), np.arange(h, dtype=np.float32))
    xr = x - disparity
    matched_mask = cv2.remap(masks[1].astype(np.uint8), xr, y, cv2.INTER_NEAREST).astype(bool)
    valid = valid & masks[0] & matched_mask & (xr >= 0) & (xr < w-1)
    gray = [cv2.cvtColor(im, cv2.COLOR_BGR2GRAY).astype(np.float32) for im in (left, right)]
    warped = cv2.remap(gray[1], xr, y, cv2.INTER_LINEAR)
    error = np.abs(gray[0] - warped)
    coverage = valid.sum() / max(1, masks[0].sum())
    mae = float(np.mean(error[valid])) if valid.any() else 255.
    score = coverage * (1 - mae / 255.)
    return valid, {'coverage': float(coverage), 'photometric_mae': mae, 'score': float(score)}, error


def colorize(disparity, valid, minimum, count):
    scaled = np.clip((disparity - minimum) * 255 / max(1, count), 0, 255).astype(np.uint8)
    image = cv2.applyColorMap(scaled, cv2.COLORMAP_TURBO)
    image[~valid] = 0
    return image
