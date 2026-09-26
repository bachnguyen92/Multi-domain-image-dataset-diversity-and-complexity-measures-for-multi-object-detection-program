
import numpy as np
import cv2
import os
from scipy.stats import skew, kurtosis
from scipy.stats import norm, entropy
import matplotlib.pyplot as plt
import torch
from PIL import Image
from torchvision import transforms
import torch.nn.functional as F
import random
from skimage.feature import graycomatrix, graycoprops
from itertools import combinations
import pandas as pd
import torch_dct as dct
import torch.fft
import glob

def kl_image_vs_normal(image_path):
    # Đọc ảnh và chuyển sang grayscale
    img = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
    pixels = img.flatten() # type: ignore

    # 1. Phân phối thực nghiệm của ảnh (P)
    hist_p, bins = np.histogram(pixels, bins=256, range=(0, 256), density=True)

    # 2. Phân phối chuẩn lý thuyết (Q)
    mean_img = np.mean(pixels)
    std_img = np.std(pixels)

    # Tạo phân phối chuẩn với cùng mean, std
    x = np.linspace(0, 255, 256)
    q = norm.pdf(x, loc=mean_img, scale=std_img)
    q = q / q.sum()  # Chuẩn hóa thành phân phối xác suất

    # 3. Tính KL divergence
    eps = 1e-10  # Tránh log(0)
    hist_p = np.clip(hist_p, eps, 1)
    q = np.clip(q, eps, 1)

    kl_div = entropy(hist_p, q)

    return kl_div

def compute_sii_exact(image_path):
    """
    Tính Self-Information Index (SII) đúng theo công thức
    """
    num_gray_levels=256
    img = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
    # Histogram
    hist = cv2.calcHist([img], [0], None,
                        [num_gray_levels], [0, num_gray_levels]).flatten()

    # Xác suất
    prob = hist / hist.sum()

    # Chỉ giữ mức xám xuất hiện
    prob = prob[prob > 0]
    G_I = len(prob)

    # Self-information
    SI = -np.log2(prob)

    # H(SI(g)) = entropy
    H_SI = np.sum(prob * SI)

    # Độ lệch chuẩn của SI
    sigma_SI = np.sqrt(
        np.sum((SI - H_SI) ** 2) / G_I
    )

    # Self-Information Index
    SII = sigma_SI / H_SI if H_SI != 0 else 0

    Ksi1 = (np.max(SI)-np.min(SI)) / (np.max(SI)+np.min(SI))
    Ksi2 = (np.max(SI)-np.mean(SI)) / (np.max(SI)+np.mean(SI))
    Ksi3 = (np.max(SI)-np.median(SI)) / (np.max(SI)+np.median(SI))
    US_SI = 1 - np.max(SI)
    SI_R = np.max(SI)/np.min(SI)

    return sigma_SI, SII, Ksi1, Ksi2, Ksi3, SI_R,US_SI

def compute_contrast_diversity_brightness_metrics(image_path):
    # Đọc ảnh grayscale

    img = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE).astype(np.float32) # type: ignore

    if img is None:
        raise ValueError("Không đọc được ảnh từ path")

    hist = cv2.calcHist([img], [0], None, [256], [0, 256]).flatten()
    prob = hist / np.sum(hist)
    levels = np.arange(256)

    img_ori = img.astype(np.float32)

    Kp1 = (np.max(prob)-np.min(prob)) / (np.max(prob)+np.min(prob))
    Kp2 = (np.max(prob)-np.mean(prob)) / (np.max(prob)+np.mean(prob))
    Kp3 = (np.max(prob)-np.median(prob)) / (np.max(prob)+np.median(prob))
    US_p = 1 - np.max(prob)

    img = img.astype(np.float32) / 255.0  # normalize pixel [0,1]

    kurt = kurtosis(img.flatten())
    # ===== Basic stats =====
    I_min = np.min(img)
    I_max = np.max(img)
    I_mean = np.mean(img)

    # ===== 1. Global Contrast =====
    if (I_max + I_min) != 0:
        global_contrast = (I_max - I_min) / (I_max + I_min)
    else:
        global_contrast = 0.0

    # ===== 2. RMS Contrast =====
    rms_contrast = np.sqrt(np.mean((img - I_mean) ** 2))

    # ===== 3. Michelson Contrast =====
    if (I_max + I_min) != 0:
        michelson_contrast = (I_max - I_min) / (I_max + I_min)
    else:
        michelson_contrast = 0.0

    # ===== 4. Coefficient of Variation Contrast =====
    mean = np.mean(img)
    std_img = np.std(img)
    cv = std_img / mean

    # 5 Percentile spread (robust)
    p5 = np.percentile(img, 5)
    p95 = np.percentile(img, 95)
    percentile_spread = p95 - p5

    # 6 Entropy H(I)
    eps = 1e-10
    entropy = -np.sum(prob * np.log2(prob + eps))
    L = 256  # số mức xám
    entropy_max = np.log2(L)  # = 8
    entropy_norm = entropy / entropy_max

    # ===== 6. Negentropy =====
    mu = np.mean(img_ori)
    sigma = np.std(img_ori)
    H_gauss = 0.5 * np.log2(2 * np.pi * np.e * sigma**2)
    J = (H_gauss - entropy) / H_gauss

    # 7 contrast_function
    mean_p = np.sum(levels * prob)
    variance = np.sum(((levels - mean_p) ** 2) * prob)
    std = np.sqrt(variance)
    std_norm = std / 255

    contrast_function = entropy_norm * std_norm

    # 8. Histogram Flatness Measure (HFM)
    hist_nonzero = hist[hist > 0]

    if len(hist_nonzero) > 0:
        geo_mean = np.exp(np.mean(np.log(hist_nonzero)))
        arith_mean = np.mean(hist_nonzero)
        HFM = geo_mean / arith_mean
    else:
        HFM = 0

    # =========================
    # 9. Histogram Spread (HS)
    # =========================
    cdf = np.cumsum(hist)
    total = cdf[-1]

    # tìm Q1 (25%) và Q3 (75%)
    Q1 = np.searchsorted(cdf, 0.25 * total)
    Q3 = np.searchsorted(cdf, 0.75 * total)

    quartile_distance = Q3 - Q1
    intensity_range = 255  # với ảnh 8-bit

    IRQ_HS = quartile_distance / intensity_range

    # 10 Mean_pro
    mean_pro = np.sum(levels * prob)
    # 11 Brightness Difference (BD)
    BD = abs(mean_pro - 128)

    kl_div = kl_image_vs_normal(image_path)

    return {
        "global_contrast": global_contrast,
        "rms_contrast": rms_contrast,
        "michelson_contrast": michelson_contrast,
        "coefficient_variation": cv,
        "percentile_spread": percentile_spread,
        "entropy": entropy_norm,
        "contrast_function": contrast_function,
        "HFM": HFM,
        "IRQ_HS": IRQ_HS, "mean_pro": mean_pro, "BD": BD, "mean": I_mean, "negentropy": J, "kurtosis":kurt, "kl_div": kl_div,
        "Kp1":Kp1, "Kp2":Kp2, "Kp3":Kp3, "US_p":US_p
    }

def compute_glcm_features(image_path):
    img = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
    img = cv2.normalize(img, None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8)

    # giảm levels để ổn định (khuyến nghị)
    levels = 64
    img_q = (img / (256 // levels)).astype(np.uint8)

    glcm = graycomatrix(img_q,
                        distances=[1],
                        angles=[0, np.pi/4, np.pi/2, 3*np.pi/4],
                        levels=levels,
                        symmetric=True,
                        normed=True)

    features = {
        "texture_contrast": graycoprops(glcm, 'contrast')[0, 0],
        "homogeneity": graycoprops(glcm, 'homogeneity')[0, 0],
        "dissimilarity": graycoprops(glcm, 'dissimilarity')[0, 0],
        "ASM": graycoprops(glcm, 'ASM')[0, 0],
        "energy": graycoprops(glcm, 'energy')[0, 0],
        "correlation": graycoprops(glcm, 'correlation')[0, 0]
    }

    return features


def image_to_patch_embeddings(image_path, image_size=640, patch_size=64):
    """
    Input: image_pat (PIL.Image hoặc path tới ảnh)
    Output: tensor shape (num_patches, embedding_dim)
    """
    # nếu là path, convert thành PIL.Image
    if isinstance(image_path, str):
        image_path = Image.open(image_path).convert("RGB")
    elif not isinstance(image_path, Image.Image):
        raise ValueError("image_pat phải là PIL.Image hoặc path tới ảnh")

    transform = transforms.Compose([
        transforms.Resize((image_size, image_size)),
        transforms.ToTensor(),  # (C, H, W)
    ])

    img = transform(image_path)  # (C, H, W)
    C, H, W = img.shape

    if H % patch_size != 0 or W % patch_size != 0:
        raise ValueError("H và W phải chia hết cho patch_size")

    # Chia thành patch
    patches = img.unfold(1, patch_size, patch_size).unfold(2, patch_size, patch_size)
    # (C, num_patch_h, num_patch_w, patch_size, patch_size)
    patches = patches.permute(1, 2, 0, 3, 4).contiguous()  # (num_patch_h, num_patch_w, C, patch_size, patch_size)
    patches = patches.view(-1, C * patch_size * patch_size)  # flatten mỗi patch

    return patches  # (num_patches, embedding_dim)

def vendi_score_full_from_image(image_path, chunk_size=50):

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    # 1. Load embeddings
    embeddings = image_to_patch_embeddings(image_path)
    embeddings = embeddings.to(device).float()

    N = embeddings.shape[0]

    # 2. Normalize (cosine similarity)
    embeddings = torch.nn.functional.normalize(embeddings, dim=1)

    # 3. Allocate K
    K = torch.empty((N, N), dtype=torch.float32, device=device)

    # 4. Compute similarity in chunks
    for i in range(0, N, chunk_size):
        end_i = min(i + chunk_size, N)

        sim = embeddings[i:end_i] @ embeddings.T
        K[i:end_i] = sim

    # 5. Move về CPU để eig ổn định
    K = K.cpu()

    # 6. Eigenvalues
    eigvals = torch.linalg.eigvalsh(K)
    eigvals = torch.clamp(eigvals, min=1e-12)

    # 7. Normalize
    eigvals = eigvals / eigvals.sum()

    # 8. Entropy → VS
    entropy = -(eigvals * torch.log(eigvals)).sum()

    # 9. Effective Degree of Freedom
    DF = 1.0 / (N * torch.sum(eigvals**2))

    return torch.exp(entropy), DF


def image_tensor_to_patches(image_tensor, patch_size=16):

    # Dùng unfold để trích xuất patches hiệu quả
    patches = F.unfold(
        image_tensor.unsqueeze(0),  # Thêm batch dim: (1, C, H, W)
        kernel_size=patch_size,
        stride=patch_size
    )  # shape: (1, C*patch_size*patch_size, num_patches)

    patches = patches.squeeze(0).t()  # shape: (num_patches, C*patch_size*patch_size)

    return patches

def compute_correlation_dimension_fast(embeddings, max_samples=150):
    """Optimized correlation dimension computation"""

    torch.manual_seed(0)
    N = embeddings.shape[0]
    if N > max_samples:
        indices = torch.randperm(N, device=embeddings.device)[:max_samples]
        embeddings = embeddings[indices]
        N = max_samples

    # Normalize embeddings for stable distance computation
    embeddings = F.normalize(embeddings, dim=1)

    # Compute distances in chunks to save memory
    chunk_size = 50
    all_dists = []
    for i in range(0, N, chunk_size):
        end_i = min(i + chunk_size, N)
        chunk_dists = torch.cdist(embeddings[i:end_i], embeddings, p=2)
        all_dists.append(chunk_dists)
    dists = torch.cat(all_dists, dim=0)

    # Use percentile-based epsilon values
    dists_flat = dists[torch.triu(torch.ones_like(dists), diagonal=1).bool()]
    epsilons = torch.quantile(dists_flat, torch.tensor([0.1, 0.3, 0.5, 0.7, 0.9], device=embeddings.device))

    C_values = []
    for eps in epsilons:
        mask = dists < eps
        mask.fill_diagonal_(False)
        C = mask.float().sum() / (N * (N - 1) + 1e-10)
        C_values.append(float(C.item()) + 1e-10)

    log_eps = torch.log(epsilons + 1e-10).cpu().numpy()
    log_C = np.log(C_values)

    if np.std(log_eps) > 1e-6:
        D_corr = float(np.polyfit(log_eps, log_C, 1)[0])
    else:
        D_corr = 1.0

    return float(max(0.1, min(D_corr, 10.0)))

def compute_capacity_dimension_box(embeddings,max_samples=500, num_bins_list=[2,4,8,16,32,64]):
    """
    embeddings: tensor (N, D)
    """

    torch.manual_seed(0)

    N = embeddings.shape[0]
    device = embeddings.device

    # Subsample nếu quá lớn
    if N > max_samples:
        indices = torch.randperm(N, device=device)[:max_samples]
        embeddings = embeddings[indices]
        N = max_samples

    # Normalize (optional nhưng thường nên giữ)
    embeddings = F.normalize(embeddings, dim=1)
    # print( embeddings )
    # min_vals = embeddings.min(dim=0, keepdim=True).values
    # max_vals = embeddings.max(dim=0, keepdim=True).values
    # embeddings = (embeddings - min_vals) / (max_vals - min_vals )  # +eps tránh chia 0

    N_values = []
    epsilons = []

    for bins in num_bins_list:
        # Map vào grid
        idx = torch.floor(embeddings * bins).long()
        idx = torch.clamp(idx, min=0, max=bins - 1)

        # Đếm số box khác nhau
        # unique theo hàng (dim=0)
        unique_boxes = torch.unique(idx, dim=0)
        N_box = unique_boxes.shape[0]

        N_values.append(N_box)
        epsilons.append(1.0 / bins)

    # Convert sang tensor
    N_values = torch.tensor(N_values, dtype=torch.float32, device=device)
    epsilons = torch.tensor(epsilons, dtype=torch.float32, device=device)

    log_N = torch.log(N_values)
    log_inv_eps = torch.log(1.0 / epsilons)

    # Linear regression (polyfit degree 1)
    # slope = cov(x,y) / var(x)
    x = log_inv_eps
    y = log_N

    x_mean = x.mean()
    y_mean = y.mean()

    slope = ((x - x_mean) * (y - y_mean)).sum() / ((x - x_mean) ** 2).sum()

    return slope.item()

def compute_information_dimension_box(embeddings,max_samples=500, num_bins_list=[2,4,8,16,32,64]):
    torch.manual_seed(0)

    N, D = embeddings.shape
    device = embeddings.device

    if N > max_samples:
        indices = torch.randperm(N, device=device)[:max_samples]
        embeddings = embeddings[indices]
        N = max_samples

    # Normalize
    embeddings = F.normalize(embeddings, dim=1)
    # min_vals = embeddings.min(dim=0, keepdim=True).values
    # max_vals = embeddings.max(dim=0, keepdim=True).values
    # embeddings = (embeddings - min_vals) / (max_vals - min_vals)  # +eps tránh chia 0

    H_values = []
    epsilons = []

    for bins in num_bins_list:
        # Map vào grid
        idx = torch.floor(embeddings * bins).long()
        idx = torch.clamp(idx, 0, bins - 1)

        # Đếm số điểm mỗi box
        _, counts = torch.unique(idx, dim=0, return_counts=True)

        # Xác suất
        p = counts.float() / counts.sum()

        # Entropy (CHÚ Ý dấu âm)
        H = (p * torch.log(p + 1e-10)).sum()

        H_values.append(H)
        epsilons.append(1.0 / bins)

    H_values = torch.stack(H_values)
    epsilons = torch.tensor(epsilons, device=device)

    log_eps = torch.log(epsilons)
    log_H = H_values

    # Linear regression (GPU)
    x = log_eps
    y = log_H

    slope = ((x - x.mean()) * (y - y.mean())).sum() / ((x - x.mean())**2).sum()

    return slope.item()

def compute_capacity_dimension_box_for_image(image_path, patch_size=16, max_samples=500):
    """
    Tính correlation dimension cho 1 ảnh bằng cách chia thành patches

    Args:
        image_path: Đường dẫn đến ảnh
        patch_size: Kích thước patch
        stride: Bước nhảy
        max_samples: Số patches tối đa để tính

    Returns:
        float: Correlation dimension của ảnh
    """
    # Load và tiền xử lý ảnh
    img = Image.open(image_path).convert('RGB')

    transform = transforms.Compose([
        transforms.Resize((640, 640)),  # Resize về kích thước chuẩn
        transforms.ToTensor(),           # Chuyển sang tensor [0, 1]
    ])

    img_tensor = transform(img)  # shape: (3, 224, 224)

    # Chuyển ảnh thành các patches
    patches = image_tensor_to_patches(img_tensor, patch_size)
    # print(f"Số patches: {patches.shape[0]}, Mỗi patch có dimension: {patches.shape[1]}")

    # Tính correlation dimension sử dụng hàm có sẵn
    # patches chính là "embeddings" - mỗi patch là 1 điểm trong không gian đặc trưng
    correlation_dim = compute_capacity_dimension_box(patches, max_samples=max_samples)

    return correlation_dim

def compute_information_dimension_box_for_image(image_path, patch_size=16, max_samples=500):
    """
    Tính correlation dimension cho 1 ảnh bằng cách chia thành patches

    Args:
        image_path: Đường dẫn đến ảnh
        patch_size: Kích thước patch
        stride: Bước nhảy
        max_samples: Số patches tối đa để tính

    Returns:
        float: Correlation dimension của ảnh
    """
    # Load và tiền xử lý ảnh
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    img = Image.open(image_path).convert('RGB')

    transform = transforms.Compose([
        transforms.Resize((640, 640)),  # Resize về kích thước chuẩn
        transforms.ToTensor(),           # Chuyển sang tensor [0, 1]
    ])

    img_tensor = transform(img).to(device)  # shape: (3, 224, 224)

    # Chuyển ảnh thành các patches
    patches = image_tensor_to_patches(img_tensor, patch_size)
    # print(f"Số patches: {patches.shape[0]}, Mỗi patch có dimension: {patches.shape[1]}")

    # Tính correlation dimension sử dụng hàm có sẵn
    # patches chính là "embeddings" - mỗi patch là 1 điểm trong không gian đặc trưng
    correlation_dim = compute_information_dimension_box(patches, max_samples=max_samples)

    return correlation_dim

def compute_correlation_dimension_for_image(image_path, patch_size=16, max_samples=500):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    # Load và tiền xử lý ảnh
    img = Image.open(image_path).convert('RGB')

    transform = transforms.Compose([
        transforms.Resize((640, 640)),  # Resize về kích thước chuẩn
        transforms.ToTensor()           # Chuyển sang tensor [0, 1]
    ])

    img_tensor = transform(img).to(device)  # shape: (3, 224, 224)
    # print(img_tensor.min().item(), img_tensor.max().item())

    # Chuyển ảnh thành các patches
    patches = image_tensor_to_patches(img_tensor, patch_size)

    # Tính correlation dimension sử dụng hàm có sẵn
    # patches chính là "embeddings" - mỗi patch là 1 điểm trong không gian đặc trưng
    correlation_dim = compute_correlation_dimension_fast(patches, max_samples=max_samples)

    return correlation_dim

def compute_features(images_dir):

    feature_names = [
    "global_contrast", "rms_contrast", "michelson_contrast", "coefficient_variation","percentile_spread", "contrast_function", "HFM", "IRQ_HS",
    "entropy", "texture_contrast", "homogeneity", "dissimilarity", "ASM", "energy",
    "correlation",
    "mean_pro", "BD", "mean",
    "sigma_SI", "SII", "negentropy", "kurtosis", "kl_div", # "cd" , "vs", "dof", "ci", "ca_d"
    "Kp1", "Kp2", "Kp3", "Ksi1", "Ksi2", "Ksi3", "SI_R", "US_SI", "US_p"

    ]

    img_paths = sorted(glob.glob(os.path.join(images_dir, "*.jpg")))

    all_features = {name: [] for name in feature_names}
    for img_path in img_paths:
        features = compute_contrast_diversity_brightness_metrics(img_path)

        vs = vendi_score_full_from_image(img_path)
        cd = compute_correlation_dimension_for_image(img_path)
        ci = compute_information_dimension_box_for_image(img_path)
        ca_d = compute_capacity_dimension_box_for_image(img_path)

        glcm_features = compute_glcm_features(img_path)
        sigma_SI, SII, Ksi1, Ksi2, Ksi3, SI_R, US_SI = compute_sii_exact(img_path)

        # features["ci"] = ci
        # features["ca_d"] = ca_d
        # features["vs"] = vs[0]
        # features["dof"] = vs[1]
        # features["cd"] = cd
        features["sigma_SI"] = sigma_SI
        features["SII"] = SII
        features["Ksi1"] = Ksi1
        features["Ksi2"] = Ksi2
        features["Ksi3"] = Ksi3
        features["SI_R"] = SI_R
        features["US_SI"] = US_SI

        features.update(glcm_features)

        for name in feature_names:
            all_features[name].append(features[name])

    return all_features

# ================================================ VS and CD
def load_images(folder, image_size=640, max_samples=500):
    random.seed(0)
    
    transform = transforms.Compose([
        transforms.Resize((image_size, image_size)),
        transforms.ToTensor(),
    ])

    all_files = [f for f in os.listdir(folder) if os.path.isfile(os.path.join(folder, f))]
    if not all_files:
        return torch.empty(0, 3 * image_size * image_size)

    random.shuffle(all_files)
    selected_files = all_files[:max_samples]  # Lấy tối đa max_samples

    embeddings = []

    for fname in selected_files:
        path = os.path.join(folder, fname)
        try:
            img = Image.open(path).convert("RGB")
            img = transform(img)  # (3, H, W)
            emb = img.view(-1)    # flatten
            embeddings.append(emb)
        except:
            continue

    if not embeddings:
        return torch.empty(0, 3 * image_size * image_size)

    return torch.stack(embeddings)     # (N, d)

def vendi_score_full_from_folder(folder_patch, chunk_size=50, device="cpu"):
    # 1. Load embeddings
    embeddings = load_images(folder_patch)
    embeddings = embeddings.to(device).float()

    N = embeddings.shape[0]

    if not N:
        return 0

    # 2. Normalize (cosine similarity)
    embeddings = torch.nn.functional.normalize(embeddings, dim=1)

    # 3. Allocate K
    K = torch.empty((N, N), dtype=torch.float32, device=device)

    # 4. Compute similarity in chunks
    for i in range(0, N, chunk_size):
        end_i = min(i + chunk_size, N)

        sim = embeddings[i:end_i] @ embeddings.T
        K[i:end_i] = sim

    # 5. Move về CPU để eig ổn định
    K = K.cpu()

    # 6. Eigenvalues
    eigvals = torch.linalg.eigvalsh(K)
    eigvals = torch.clamp(eigvals, min=1e-12)

    # 7. Normalize
    eigvals = eigvals / eigvals.sum()

    # 8. Entropy → VS
    entropy = -(eigvals * torch.log(eigvals)).sum()


    vs = torch.exp(entropy).cpu().numpy()

    return vs


def image_to_embedding(image_path):
    img = Image.open(image_path).convert('RGB')

    transform = transforms.Compose([
        transforms.Resize((256, 256)),  # nhỏ lại để tránh vector quá lớn
        transforms.ToTensor(),
    ])

    img_tensor = transform(img)  # (3, H, W)
    embedding = img_tensor.view(-1)  # flatten → (C*H*W)

    return embedding

def compute_correlation_dimension_for_folder(image_folder, patch_size=16, max_samples=500):
    embeddings = []

    image_files = [f for f in os.listdir(image_folder) if f.lower().endswith(('.jpg'))]

    for i, fname in enumerate(image_files):
        path = os.path.join(image_folder, fname)

        emb = image_to_embedding(path)
        embeddings.append(emb)

    if not embeddings:
        return 0
    
    embeddings = torch.stack(embeddings)  # shape: (N, D)

    correlation_dim = compute_correlation_dimension_fast(embeddings, max_samples=max_samples)

    return correlation_dim

# ==================================================  CVBB
def load_yolo_bboxes(label_path, img_w, img_h):
    """
    YOLO format: class x_center y_center w h (normalized)
    return: [x1, y1, x2, y2]
    """
    bboxes = []

    if not os.path.exists(label_path):
        return np.array(bboxes)

    with open(label_path, "r") as f:
        for line in f.readlines():
            parts = list(map(float, line.strip().split()))
            if len(parts) != 5:
                continue

            _, xc, yc, w, h = parts

            xc *= img_w
            yc *= img_h
            w *= img_w
            h *= img_h

            x1 = xc - w / 2
            y1 = yc - h / 2
            x2 = xc + w / 2
            y2 = yc + h / 2

            bboxes.append([x1, y1, x2, y2])

    return np.array(bboxes)

def compute_union_area_by_subtraction(bboxes):
    if len(bboxes) == 0:
        return 0.0

    bboxes = np.array(bboxes)

    # Tổng diện tích từng box
    total_area = np.sum((bboxes[:,2] - bboxes[:,0]) * (bboxes[:,3] - bboxes[:,1]))

    # Tính diện tích giao nhau giữa từng cặp box
    overlap_area = 0
    for box1, box2 in combinations(bboxes, 2):
        x_left   = max(box1[0], box2[0])
        y_top    = max(box1[1], box2[1])
        x_right  = min(box1[2], box2[2])
        y_bottom = min(box1[3], box2[3])

        if x_right > x_left and y_bottom > y_top:
            overlap_area += (x_right - x_left) * (y_bottom - y_top)

    # Diện tích hợp
    union_area = total_area - overlap_area
    return union_area

def compute_total_area(bboxes):
    """
    bboxes: numpy array (N,4) [x1, y1, x2, y2]
    return: tổng diện tích tất cả bounding box (không trừ overlap)
    """
    if len(bboxes) == 0:
        return 0.0

    widths = bboxes[:, 2] - bboxes[:, 0]
    heights = bboxes[:, 3] - bboxes[:, 1]

    areas = widths * heights
    return np.sum(areas)

def compute_cvbb_image(bboxes, img_w, img_h):
    if len(bboxes) == 0:
        return 0.0

    # --- centroid ---
    cx = (bboxes[:, 0] + bboxes[:, 2]) / 2
    cy = (bboxes[:, 1] + bboxes[:, 3]) / 2
    centroids = np.stack([cx, cy], axis=1)

    x_min = min(b[0] for b in bboxes)
    y_min = min(b[1] for b in bboxes)
    x_max = max(b[2] for b in bboxes)
    y_max = max(b[3] for b in bboxes)

    dx, dy = x_max - x_min, y_max - y_min

    # --- median centroid ---
    median_centroid = np.median(centroids, axis=0)

    # --- top-left vectors ---
    top_left = bboxes[:, :2]

    # --- distance (top-left - median centroid) ---
    distances = np.linalg.norm(top_left - median_centroid, axis=1)

    # --- spatial dispersion ---
    # spatial_term = np.sqrt(np.mean(distances ** 2))
    spatial_term = np.median(distances)

    # --- normalize by diagonal ---
    diag = np.sqrt(dx**2 + dy**2)
    spatial_norm = spatial_term / (diag / 2)

    # --- scale coefficient (union) ---
    union_area = compute_union_area_by_subtraction(bboxes)
    # union_area = compute_total_area(bboxes)

    enclosing_area = dx*dy
    # img_area = img_w * img_h
    scale_coeff = union_area / enclosing_area

    return spatial_norm * scale_coeff
    # return spatial_norm + scale_coeff

def compute_cvbb_dataset(image_dir, label_dir):
    cvbb_values = []

    for img_name in os.listdir(image_dir):
        if not img_name.lower().endswith((".jpg")):
            continue

        img_path = os.path.join(image_dir, img_name)
        label_path = os.path.join(label_dir, img_name.rsplit(".", 1)[0] + ".txt")

        img = cv2.imread(img_path)
        if img is None:
            continue

        h, w = img.shape[:2]

        bboxes = load_yolo_bboxes(label_path, w, h)

        cvbb = compute_cvbb_image(bboxes, w, h)
        cvbb_values.append(cvbb)
    
    if not cvbb_values:
        return 0, 0

    return np.median(cvbb_values), cvbb_values

# ================================================== Imbalanced metrics

def compute_imbalanced_metrics(labels_dir):

    all_label_files = [f for f in os.listdir(labels_dir) if f.endswith('.txt')]

    class_counts = {}

    for label_file in all_label_files:
        with open(os.path.join(labels_dir, label_file), 'r') as f:
            lines = f.readlines()
            for line in lines:
                cls = int(line.split()[0])  # lớp là số đầu tiên trong dòng
                if cls not in class_counts:
                    class_counts[cls] = 0
                class_counts[cls] += 1

    if not class_counts:  # Trường hợp không có file nào
        return 0.0, 0.0
    
    max_class = max(class_counts.keys()) + 1
    class_counts_array = np.zeros(max_class, dtype=int)
    for cls, count in class_counts.items():
        class_counts_array[cls] = count
    
    nonzero_counts = class_counts_array[class_counts_array > 0]

    if len(nonzero_counts) <= 1:  # Chỉ có 1 lớp
        IR = 1.0 if len(nonzero_counts) == 1 else 0.0
        LRID = 0.0
        return IR, LRID

    # Tính IR (Imbalance Ratio)
    N_max = nonzero_counts.max()
    N_min = nonzero_counts.min()
    IR = N_max / N_min

    # Tính LRID theo công : LRID = -2 * sum(N_i * ln(N / (C * N_i)))  Imbalance Degree
    N = class_counts_array.sum()
    C = len(class_counts_array)    # Tổng số lớp (kể cả lớp trống)
    mask = class_counts_array > 0
    LRID = -2 * np.sum(class_counts_array[mask] * np.log(N / (C * class_counts_array[mask])))

    return IR, LRID

# ================================================ VS in multi-domain
# spatial domain based on ZNSSD (Zero-mean Normalized Sum of Squared Differences)
def load_images_as_embeddings(folder, image_size=640, max_samples=500):
    transform = transforms.Compose([
        transforms.Resize((image_size, image_size)),
        transforms.ToTensor(),
    ])


    all_files = [f for f in os.listdir(folder) if os.path.isfile(os.path.join(folder, f))]
    if len(all_files) > max_samples:
        random.seed(0)  # reproducible
        all_files = random.sample(all_files, max_samples)

    embeddings = []

    for fname in all_files:
        path = os.path.join(folder, fname)

        try:
            img = Image.open(path).convert("RGB")
            img = transform(img)  # (3, H, W)
            emb = img.view(-1)    # flatten
            embeddings.append(emb)
        except:
            continue

    embeddings = torch.stack(embeddings)

    return embeddings    # (N, d)

def vendi_score_znssd(folder_patch, chunk_size=50, device="cpu"):
    embeddings = load_images_as_embeddings(folder_patch)
    embeddings = embeddings.to(device).float()

    N = embeddings.shape[0]

    # --- ZNCC normalization ---
    mean = embeddings.mean(dim=1, keepdim=True)
    centered = embeddings - mean
    nor = torch.norm(centered, dim=1, keepdim=True) + 1e-8
    embeddings = centered / nor

    # # --- Precompute norm^2 ---
    norm2 = torch.sum(embeddings**2, dim=1, keepdim=True)

    K = torch.empty((N, N), dtype=torch.float32, device=device)

    for i in range(0, N, chunk_size):
        end_i = min(i + chunk_size, N)

        dot = embeddings[i:end_i] @ embeddings.T

        znsdd = norm2[i:end_i] + norm2.T - 2 * dot

        K[i:end_i] = znsdd

    # --- convert distance → similarity ---
    K = torch.exp(-K)

    # --- symmetry ---
    # K = (K + K.T) / 2

    K = K.cpu()

    eigvals = torch.linalg.eigvalsh(K)
    eigvals = torch.clamp(eigvals, min=1e-12)
    eigvals = eigvals / eigvals.sum()

    entropy = -(eigvals * torch.log(eigvals)).sum()

    vs = torch.exp(entropy).cpu().numpy()
    EDoF = (1.0 / torch.sum(eigvals**2)).cpu().numpy()

    return vs, EDoF, K

# frequency domain
def split_frequency_bands(dct_img, low_ratio=0.1, mid_ratio=0.3):
    _, H, W = dct_img.shape

    low_h = int(H * low_ratio)
    low_w = int(W * low_ratio)

    mid_h = int(H * mid_ratio)
    mid_w = int(W * mid_ratio)

    # Low frequency (top-left nhỏ)
    low = dct_img[:, :low_h, :low_w]

    # Mid frequency (vùng giữa, trừ low)
    mid = dct_img[:, :mid_h, :mid_w].clone()
    mid[:, :low_h, :low_w] = 0  # bỏ phần low

    # High frequency (phần còn lại)
    high = dct_img.clone()
    high[:, :mid_h, :mid_w] = 0  # bỏ low + mid

    return low, mid, high

def load_images_as_embeddings_fre(folder, image_size=640, max_samples=500):
    # crop_size=320

    transform = transforms.Compose([
        transforms.Resize((image_size, image_size)),
        transforms.Grayscale(),   # chuyển về 1 channel
        transforms.ToTensor(),
    ])

    all_files = [f for f in os.listdir(folder) if os.path.isfile(os.path.join(folder, f))]
    if len(all_files) > max_samples:
        random.seed(0)
        all_files = random.sample(all_files, max_samples)

    embeddings = []
    dc_values = []

    for fname in all_files:
        path = os.path.join(folder, fname)

        try:
            img = Image.open(path).convert("RGB")
            img = transform(img)  # (1, H, W)

            # --- DCT ---
            dct_img = dct.dct_2d(img)

            dc_value = dct_img[..., 0, 0].clone()
            dc_values.append(dc_value)

            # --- remove DC ---
            dct_img[..., 0, 0] = 0

            low, mid, high = split_frequency_bands(dct_img)

            # --- lấy low-frequency (top-left) ---
            # crop = dct_img[:, :crop_size, :crop_size]

            # emb = torch.abs(dct_img.reshape(-1))
            emb = (dct_img.reshape(-1))
            embeddings.append(emb)

        except:
            continue

    DC_max = torch.max(torch.stack(dc_values))
    DC_min = torch.min(torch.stack(dc_values))
    # print(f"dc_values range: {((DC_max-DC_min)/(DC_max+DC_min)):.2f}")

    embeddings = torch.stack(embeddings)

    return embeddings

def vendi_score_znssd_fre(folder_patch, chunk_size=50, device="cpu"):
    embeddings = load_images_as_embeddings_fre(folder_patch)
    embeddings = embeddings.to(device).float()

    N = embeddings.shape[0]

    # --- ZNCC normalization ---
    mean = embeddings.mean(dim=1, keepdim=True)
    centered = embeddings - mean
    nor = torch.norm(centered, dim=1, keepdim=True) + 1e-8
    embeddings = centered / nor

    # # --- Precompute norm^2 ---
    norm2 = torch.sum(embeddings**2, dim=1, keepdim=True)

    K = torch.empty((N, N), dtype=torch.float32, device=device)

    for i in range(0, N, chunk_size):
        end_i = min(i + chunk_size, N)

        dot = embeddings[i:end_i] @ embeddings.T

        znsdd = norm2[i:end_i] + norm2.T - 2 * dot

        K[i:end_i] = znsdd

    # --- convert distance → similarity ---
    K = torch.exp(-K)

    K = K.cpu()

    eigvals = torch.linalg.eigvalsh(K)
    eigvals = torch.clamp(eigvals, min=1e-12)
    eigvals = eigvals / eigvals.sum()

    entropy = -(eigvals * torch.log(eigvals)).sum()

    vs = torch.exp(entropy).cpu().numpy()
    EDoF = (1.0 / torch.sum(eigvals**2)).cpu().numpy()

    return vs, EDoF, K

# phase domain based on GPOC

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

def load_images_gpu(folder, image_size=64, max_samples=500):

    transform = transforms.Compose([
        transforms.Resize((image_size, image_size)),
        transforms.Grayscale(),
        transforms.ToTensor(),
    ])

    all_files = [
        f for f in os.listdir(folder)
        if os.path.isfile(os.path.join(folder, f))
    ]

    if len(all_files) > max_samples:
        random.seed(0)
        all_files = random.sample(all_files, max_samples)

    images = []

    for fname in all_files:
        path = os.path.join(folder, fname)

        try:
            img = Image.open(path).convert("RGB")
            img = transform(img)

            # scale [0,1] -> [0,255]
            img = (img * 255.0).to(torch.uint8)

            images.append(img.squeeze(0))

        except:
            continue

    images = torch.stack(images).to(device)

    return images

def poc_similarity_batch_gpu(images, batch_size=64):

    """
    images: [N,H,W] uint8 GPU tensor
    return: [N,N] similarity matrix
    """

    images = images.float()
    N, H, W = images.shape

    # FFT of all images
    F_imgs = torch.fft.fft2(images)

    K = torch.zeros((N, N), device=device, dtype=torch.float32)

    for i in range(N):

        Fi = F_imgs[i]

        for j_start in range(i, N, batch_size):

            j_end = min(j_start + batch_size, N)
            Fj = F_imgs[j_start:j_end]

            # =================================================
            # Cross-power spectrum (phase only)
            # =================================================
            R = Fi.unsqueeze(0) * torch.conj(Fj)

            R = R / (torch.abs(R) + 1e-8)

            # =================================================
            # Inverse FFT -> correlation map
            # =================================================
            corr = torch.fft.ifft2(R).real

            # =================================================
            # similarity = peak of correlation
            # =================================================
            sim = torch.amax(corr, dim=(-2, -1))

            sim = torch.clamp(sim, min=0.0)

            K[i, j_start:j_end] = sim
            K[j_start:j_end, i] = sim

    # normalize kernel to [0,1]
    K = K / (K.max() + 1e-8)

    return K

def vendi_score_poc_gpu(folder_path,batch_size=64,image_size=64,max_samples=500):

    # -----------------------------------------------------
    # Load images
    # -----------------------------------------------------
    images = load_images_gpu(
        folder_path,
        image_size=image_size,
        max_samples=max_samples
    )

    # -----------------------------------------------------
    # Similarity matrix (POC)
    # -----------------------------------------------------
    K = poc_similarity_batch_gpu(images, batch_size=batch_size)

    # symmetrize
    K = (K + K.T) / 2

    # -----------------------------------------------------
    # Eigen decomposition
    # -----------------------------------------------------
    eigvals = torch.linalg.eigvalsh(K)
    eigvals = torch.clamp(eigvals, min=1e-12)
    eigvals = eigvals / eigvals.sum()

    # -----------------------------------------------------
    # Entropy-based Vendi Score
    # -----------------------------------------------------
    entropy_val = -torch.sum(eigvals * torch.log(eigvals))

    VS = torch.exp(entropy_val)

    EDoF = 1.0 / torch.sum(eigvals ** 2)

    return VS.item(), EDoF.item(), K.detach().cpu().numpy()

# phase domain based on phase correlation 

def load_images_gpu2(folder,image_size=64,max_samples=500):

    transform = transforms.Compose([
        transforms.Resize((image_size, image_size)),
        transforms.Grayscale(),
        transforms.ToTensor(),
    ])

    all_files = [
        f for f in os.listdir(folder)
        if os.path.isfile(os.path.join(folder, f))
    ]

    # Random sampling
    if len(all_files) > max_samples:
        random.seed(0)
        all_files = random.sample(all_files, max_samples)

    images = []

    for fname in all_files:

        path = os.path.join(folder, fname)

        try:
            img = Image.open(path).convert("RGB")

            img = transform(img)

            # [1,H,W] -> [H,W]
            img = img.squeeze(0)

            images.append(img)

        except:
            continue

    images = torch.stack(images).to(device)

    return images

def compute_phase_features(images):
    """
    images: [N,H,W] float tensor on GPU

    return:
        phase_features: [N,P]
    """

    # FFT
    Freq = torch.fft.fft2(images)

    # Phase
    phase = torch.angle(Freq)

    # Flatten
    phase = phase.reshape(phase.shape[0], -1)

    # Normalize each image phase vector
    phase = phase - phase.mean(dim=1, keepdim=True)

    phase = F.normalize(phase, p=2, dim=1)

    return phase

def spectral_phase_similarity_matrix_gpu(images,batch_size=128):
    """
    images : [N,H,W]

    return:
        K : [N,N]
    """

    # -----------------------------------------------------
    # Compute normalized phase vectors
    # -----------------------------------------------------

    phase_features = compute_phase_features(images)

    N = phase_features.shape[0]

    # -----------------------------------------------------
    # Similarity matrix
    # Pearson correlation ~= cosine similarity
    # because vectors are mean-centered + normalized
    # -----------------------------------------------------

    K = torch.zeros(
        (N, N),
        dtype=torch.float32,
        device=device
    )

    for i in (range(0, N, batch_size)):

        i_end = min(i + batch_size, N)

        fi = phase_features[i:i_end]

        for j in range(i, N, batch_size):

            j_end = min(j + batch_size, N)

            fj = phase_features[j:j_end]

            # Matrix multiplication
            sim = torch.matmul(fi, fj.T)

            K[i:i_end, j:j_end] = sim

            # Symmetric fill
            if i != j:
                K[j:j_end, i:i_end] = sim.T

    return K

def vendi_score_spectral_phase_gpu(folder_patch,image_size=64,max_samples=500,batch_size=128):

    # -----------------------------------------------------
    # Load images
    # -----------------------------------------------------

    images = load_images_gpu2(
        folder_patch,
        image_size=image_size,
        max_samples=max_samples
    )

    # -----------------------------------------------------
    # Similarity matrix
    # -----------------------------------------------------

    K = spectral_phase_similarity_matrix_gpu(
        images,
        batch_size=batch_size
    )

    # -----------------------------------------------------
    # Shift similarity to positive range
    # [-1,1] -> [0,1]
    # -----------------------------------------------------

    K = (K + 1.0) / 2.0

    # -----------------------------------------------------
    # Numerical stability
    # -----------------------------------------------------

    K = (K + K.T) / 2

    K = K + 1e-6 * torch.eye(
        K.shape[0],
        device=device
    )

    # -----------------------------------------------------
    # Eigenvalues
    # -----------------------------------------------------

    eigvals = torch.linalg.eigvalsh(K)

    eigvals = torch.clamp(
        eigvals,
        min=1e-12
    )

    eigvals = eigvals / eigvals.sum()

    # -----------------------------------------------------
    # Shannon entropy
    # -----------------------------------------------------

    entropy_val = -torch.sum(
        eigvals * torch.log(eigvals)
    )

    # -----------------------------------------------------
    # Metrics
    # -----------------------------------------------------

    VS = torch.exp(entropy_val)

    EDoF = 1.0 / torch.sum(
        eigvals ** 2
    )

    return (
        VS.item(),
        EDoF.item(),
        K.detach().cpu().numpy()
    )

# spatial domain based on MI

def normalized_mutual_information_batch_gpu(images,bins=64,batch_size=128):
    """
    images : [N,H,W] uint8 GPU tensor

    return:
        K : [N,N] NMI similarity matrix
    """

    device = images.device

    N, H, W = images.shape

    P = H * W

    # -----------------------------------------------------
    # Flatten + quantize once
    # -----------------------------------------------------

    x = images.reshape(N, -1).long()

    x = torch.clamp(
        (x * bins) // 256,
        0,
        bins - 1
    )

    # -----------------------------------------------------
    # Entropy of each image
    # -----------------------------------------------------

    hist_all = torch.zeros(
        (N, bins),
        device=device
    )

    for b in range(bins):

        hist_all[:, b] = (x == b).sum(dim=1)

    px_all = hist_all / P

    H_all = -torch.sum(
        px_all * torch.log(px_all + 1e-12),
        dim=1
    )

    # -----------------------------------------------------
    # Similarity matrix
    # -----------------------------------------------------

    K = torch.zeros(
        (N, N),
        dtype=torch.float32,
        device=device
    )

    # -----------------------------------------------------
    # Pairwise batches
    # -----------------------------------------------------

    for i in range(N):

        xi = x[i]  # [P]

        for j_start in range(i, N, batch_size):

            j_end = min(
                j_start + batch_size,
                N
            )

            xj = x[j_start:j_end]  # [B,P]

            B = xj.shape[0]

            # ---------------------------------------------
            # Joint histogram
            # ---------------------------------------------

            joint_idx = (
                xi.unsqueeze(0) * bins + xj
            )  # [B,P]

            hists = torch.zeros(
                (B, bins * bins),
                device=device
            )

            ones = torch.ones_like(
                joint_idx,
                dtype=torch.float32
            )

            hists.scatter_add_(
                1,
                joint_idx,
                ones
            )

            hists = hists.reshape(
                B,
                bins,
                bins
            )

            # ---------------------------------------------
            # Probabilities
            # ---------------------------------------------

            pxy = hists / P

            px = pxy.sum(dim=2)

            py = pxy.sum(dim=1)

            px_py = (
                px.unsqueeze(2) *
                py.unsqueeze(1)
            )

            # ---------------------------------------------
            # Mutual Information
            # ---------------------------------------------

            nzs = pxy > 0

            mi = torch.sum(
                torch.where(
                    nzs,
                    pxy * torch.log(
                        pxy / (px_py + 1e-12)
                    ),
                    torch.zeros_like(pxy)
                ),
                dim=(1, 2)
            )

            # ---------------------------------------------
            # NMI
            # ---------------------------------------------

            hx = H_all[i]

            hy = H_all[j_start:j_end]

            nmi = (
                2.0 * mi
            ) / (hx + hy + 1e-12)

            # ---------------------------------------------
            # Fill matrix
            # ---------------------------------------------

            K[i, j_start:j_end] = nmi

            K[j_start:j_end, i] = nmi

    return K

def vendi_score_nmi_gpu(folder_patch,bins=64,batch_size=128,image_size=64,max_samples=500):

    # -----------------------------------------------------
    # Load images
    # -----------------------------------------------------

    images = load_images_gpu(
        folder_patch,
        image_size=image_size,
        max_samples=max_samples
    )

    # -----------------------------------------------------
    # Similarity matrix
    # -----------------------------------------------------

    K = normalized_mutual_information_batch_gpu(
        images,
        bins=bins,
        batch_size=batch_size
    )

    # -----------------------------------------------------
    # Symmetry
    # -----------------------------------------------------

    K = (K + K.T) / 2

    # -----------------------------------------------------
    # Eigenvalues
    # -----------------------------------------------------

    eigvals = torch.linalg.eigvalsh(K)

    eigvals = torch.clamp(
        eigvals,
        min=1e-12
    )

    eigvals = eigvals / eigvals.sum()

    # -----------------------------------------------------
    # Shannon entropy
    # -----------------------------------------------------

    entropy_val = -torch.sum(
        eigvals * torch.log(eigvals)
    )

    # -----------------------------------------------------
    # Metrics
    # -----------------------------------------------------

    VS = torch.exp(entropy_val)

    EDoF = 1.0 / torch.sum(
        eigvals ** 2
    )

    return (
        VS.item(),
        EDoF.item(),
        K.detach().cpu().numpy()
    )




