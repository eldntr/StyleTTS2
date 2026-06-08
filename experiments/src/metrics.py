import numpy as np
import librosa

def dtw(x, y, dist):
    """
    Computes Dynamic Time Warping distance and path of two sequences using NumPy.
    """
    len_x, len_y = len(x), len(y)
    cost_matrix = np.zeros((len_x + 1, len_y + 1))
    cost_matrix[0, 1:] = np.inf
    cost_matrix[1:, 0] = np.inf
    
    for i in range(1, len_x + 1):
        for j in range(1, len_y + 1):
            d = dist(x[i-1], y[j-1])
            cost_matrix[i, j] = d + min(cost_matrix[i-1, j], cost_matrix[i, j-1], cost_matrix[i-1, j-1])
            
    # Backtrack
    path = []
    i, j = len_x, len_y
    while i > 0 and j > 0:
        path.append((i-1, j-1))
        option = np.argmin([cost_matrix[i-1, j], cost_matrix[i, j-1], cost_matrix[i-1, j-1]])
        if option == 0:
            i -= 1
        elif option == 1:
            j -= 1
        else:
            i -= 1
            j -= 1
    path.reverse()
    return cost_matrix[len_x, len_y], path

def compute_mcd_and_f0(gt_path, pred_path, sr=24000):
    # 1. Load Audio
    y_gt, _ = librosa.load(gt_path, sr=sr)
    y_pred, _ = librosa.load(pred_path, sr=sr)
    
    # Trim silence
    y_gt, _ = librosa.effects.trim(y_gt, top_db=30)
    y_pred, _ = librosa.effects.trim(y_pred, top_db=30)
    
    # 2. Extract MFCCs
    mfcc_gt = librosa.feature.mfcc(y=y_gt, sr=sr, n_mfcc=13)[1:] # Exclude 0-th coefficient (energy)
    mfcc_pred = librosa.feature.mfcc(y=y_pred, sr=sr, n_mfcc=13)[1:]
    
    # 3. Align MFCCs using DTW
    dist_func = lambda u, v: np.sqrt(np.sum((u - v) ** 2))
    _, path = dtw(mfcc_gt.T, mfcc_pred.T, dist_func)
    
    aligned_gt = np.array([mfcc_gt.T[i] for i, j in path])
    aligned_pred = np.array([mfcc_pred.T[j] for i, j in path])
    
    # Calculate MCD
    # MCD = (10 / ln(10)) * sqrt(2 * sum((c_gt - c_pred)^2))
    diff = aligned_gt - aligned_pred
    mcd = (10.0 / np.log(10.0)) * np.sqrt(2.0 * np.sum(diff ** 2, axis=1))
    avg_mcd = np.mean(mcd)
    
    # 4. F0 Extraction using YIN
    f0_gt = librosa.yin(y_gt, fmin=50, fmax=500, sr=sr)
    f0_pred = librosa.yin(y_pred, fmin=50, fmax=500, sr=sr)
    
    # Align F0 using DTW
    dist_func_f0 = lambda u, v: np.abs(u - v)
    _, path_f0 = dtw(f0_gt, f0_pred, dist_func_f0)
    
    aligned_f0_gt = np.array([f0_gt[i] for i, j in path_f0])
    aligned_f0_pred = np.array([f0_pred[j] for i, j in path_f0])
    
    # Filter unvoiced frames (represented by values close to limits/fmin)
    valid_mask = (aligned_f0_gt > 55) & (aligned_f0_pred > 55)
    if np.sum(valid_mask) > 0:
        f0_rmse = np.sqrt(np.mean((aligned_f0_gt[valid_mask] - aligned_f0_pred[valid_mask]) ** 2))
    else:
        f0_rmse = 0.0
        
    return float(avg_mcd), float(f0_rmse)
