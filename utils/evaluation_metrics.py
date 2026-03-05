import numpy as np
import tensorflow as tf
from scipy.linalg import sqrtm
from scipy.spatial.distance import euclidean as scipy_euclidean
from scipy.spatial.distance import jensenshannon
from sklearn.metrics.pairwise import pairwise_kernels

try:
    from fastdtw import fastdtw
except ImportError:
    print("Warning: fastdtw not installed. DTW metrics will not be available.")
    fastdtw = None


def compute_mmd_tf(x, y, sigma=1.0):
    """
    Compute the Maximum Mean Discrepancy (MMD) between two samples.

    Args:
        x: np.array or tensor, first dataset (n_samples, n_features)
        y: np.array or tensor, second dataset (n_samples, n_features)
        sigma: float, bandwidth parameter for the RBF kernel

    Returns:
        mmd_score: float, the MMD value between x and y
    """
    x = tf.convert_to_tensor(x, dtype=tf.float32)
    y = tf.convert_to_tensor(y, dtype=tf.float32)

    # Compute Gram matrices
    xx = tf.matmul(x, tf.transpose(x))
    yy = tf.matmul(y, tf.transpose(y))
    xy = tf.matmul(x, tf.transpose(y))

    rx = tf.expand_dims(tf.reduce_sum(x * x, axis=1), 0)
    ry = tf.expand_dims(tf.reduce_sum(y * y, axis=1), 0)

    Kxx = tf.exp(- (rx - 2 * xx + tf.transpose(rx)) / (2 * sigma ** 2))
    Kyy = tf.exp(- (ry - 2 * yy + tf.transpose(ry)) / (2 * sigma ** 2))
    Kxy = tf.exp(- (rx - 2 * xy + tf.transpose(ry)) / (2 * sigma ** 2))

    # Compute MMD
    mmd_score = tf.reduce_mean(Kxx) + tf.reduce_mean(Kyy) - 2 * tf.reduce_mean(Kxy)

    return mmd_score.numpy()


def custom_euclidean(u, v):
    """
    Custom Euclidean distance function that handles both scalars and arrays.
    
    Parameters:
    -----------
    u : float or ndarray
        First input value or array.
    v : float or ndarray
        Second input value or array.
    
    Returns:
    --------
    float
        Computed Euclidean distance.
    """
    # Ensure inputs are numpy arrays
    u = np.asarray(u)
    v = np.asarray(v)
    
    # Check if inputs are scalars
    if u.ndim == 0 and v.ndim == 0:
        return abs(u - v)  # Return absolute difference for scalars
    else:
        return scipy_euclidean(u, v)  # Use scipy's Euclidean for arrays

def compute_dtw_distance(real_sensors, fake_sensors):
    """
    Computes the DTW (Dynamic Time Warping) distance between real and fake sensor sequences
    for each channel and averages the distances across all channels.
    
    Parameters:
        real_sensors (numpy.ndarray): Real sensor data of shape (samples, timesteps, channels).
        fake_sensors (numpy.ndarray): Fake sensor data of shape (samples, timesteps, channels).
    
    Returns:
        float: Average DTW distance across all channels.
    """
    # Ensure the input is 3D (samples, timesteps, channels)
    if real_sensors.ndim != 3 or fake_sensors.ndim != 3:
        raise ValueError("Input sensor data must be 3D arrays (samples, timesteps, channels).")
    
    # Check that real and fake data have the same shape
    if real_sensors.shape != fake_sensors.shape:
        raise ValueError("Real and fake sensor data must have the same shape.")
    
    num_samples, num_timesteps, num_channels = real_sensors.shape
    
    dtw_distances = []
    
    # Iterate over each channel
    for channel in range(num_channels):
        channel_distances = []
        
        # Compute DTW distance for each sample in the current channel
        for sample in range(num_samples):
            real_sequence = real_sensors[sample, :, channel]  # 1D time-series for real data
            fake_sequence = fake_sensors[sample, :, channel]  # 1D time-series for fake data
            #print(f"Processing sample {sample}, channel {channel}: real_sequence shape {real_sequence.shape}, fake_sequence shape {fake_sequence.shape}")
            real_sequence = (real_sequence - np.mean(real_sequence)) / (np.std(real_sequence) + 1e-8)
            fake_sequence = (fake_sequence - np.mean(fake_sequence)) / (np.std(fake_sequence) + 1e-8)

            # Compute DTW distance between the two sequences
            distance, _ = fastdtw(real_sequence, fake_sequence, dist=custom_euclidean)#dist=lambda x, y: abs(x - y))
            channel_distances.append(distance)
        
        # Average DTW distance for the current channel
        dtw_distances.append(np.mean(channel_distances))
    
    # Average DTW distance across all channels
    avg_dtw = np.mean(dtw_distances)
    return avg_dtw


def compute_frechet_distance_old(real_samples, fake_samples):
    """
    Computes the Fréchet Distance (FD) between the real and fake sensor data distributions.
    
    Args:
        real_samples (numpy array): Shape (num_samples, time_steps, num_channels)
        fake_samples (numpy array): Shape (num_samples, time_steps, num_channels)
    
    Returns:
        float: Fréchet Distance score (lower is better)
    """
    real_samples = real_samples.reshape(-1, real_samples.shape[-1])  # Flatten time dimension
    fake_samples = fake_samples.reshape(-1, fake_samples.shape[-1])
    
    # Compute mean and covariance of both distributions
    mu_real, sigma_real = np.mean(real_samples, axis=0), np.cov(real_samples, rowvar=False)
    mu_fake, sigma_fake = np.mean(fake_samples, axis=0), np.cov(fake_samples, rowvar=False)
    
    # Compute the squared difference of means
    mean_diff = np.sum((mu_real - mu_fake) ** 2)
    
    # Compute sqrt of product of covariances
    try:
        cov_sqrt = sqrtm(sigma_real.dot(sigma_fake))
        if np.iscomplexobj(cov_sqrt):
            cov_sqrt = cov_sqrt.real
    except Exception as e:
        print("Covariance sqrt error:", e)
        cov_sqrt = np.zeros_like(sigma_real)
    
    # Compute Fréchet Distance
    fd = mean_diff + np.trace(sigma_real + sigma_fake - 2 * cov_sqrt)
    return max(fd, 0)  # Ensure non-negative value


def compute_frechet_distance(real_samples, fake_samples, eps=1e-6, per_channel=False, min_samples=1000):
    """
    Computes the Fréchet Distance (FD) between the real and fake sensor data distributions.
    
    Args:
        real_samples (numpy array): Shape (num_samples, time_steps, num_channels)
        fake_samples (numpy array): Shape (num_samples, time_steps, num_channels)
        eps (float): Small constant for numerical stability
        per_channel (bool): If True, compute FD for each channel separately
        min_samples (int): Minimum recommended samples for stable estimation
    
    Returns:
        float or dict: Overall FD or dict with overall and per-channel FDs
    """
    # Check input shapes
    assert real_samples.shape[2] == fake_samples.shape[2], "Channel dimensions must match"
    
    if per_channel:
        channel_fds = []
        channel_names = [f"Channel_{i}" for i in range(real_samples.shape[2])]
        
        for ch in range(real_samples.shape[2]):
            real_ch = real_samples[:, :, ch].flatten()  # One channel, flattened time
            fake_ch = fake_samples[:, :, ch].flatten()
            
            # Compute mean and variance (1D case)
            mu_real, sigma_real = np.mean(real_ch), np.var(real_ch)
            mu_fake, sigma_fake = np.mean(fake_ch), np.var(fake_ch)
            
            # Add epsilon for stability
            sigma_real += eps
            sigma_fake += eps
            
            # 1D Fréchet distance
            fd_ch = (mu_real - mu_fake)**2 + sigma_real + sigma_fake - 2*np.sqrt(sigma_real * sigma_fake)
            channel_fds.append(max(fd_ch, 0))
        
        # Also compute the joint FD
        joint_fd = compute_joint_fd(real_samples, fake_samples, eps, min_samples)
        
        # Return both overall and per-channel results
        return {
            "overall_fd": joint_fd,
            "channel_fd": dict(zip(channel_names, channel_fds)),
            "average_channel_fd": np.mean(channel_fds)
        }
    else:
        return compute_joint_fd(real_samples, fake_samples, eps, min_samples)

def compute_joint_fd(real_samples, fake_samples, eps=1e-6, min_samples=1000):
    """Helper function to compute joint FD across all channels"""
    real_flat = real_samples.reshape(-1, real_samples.shape[-1])
    fake_flat = fake_samples.reshape(-1, fake_samples.shape[-1])
    
    # Check sample size
    if len(real_flat) < min_samples or len(fake_flat) < min_samples:
        print(f"Warning: Sample size may be too small for stable FD estimation. "
              f"Real: {len(real_flat)}, Fake: {len(fake_flat)}")
    
    # Compute mean and covariance
    mu_real, sigma_real = np.mean(real_flat, axis=0), np.cov(real_flat, rowvar=False)
    mu_fake, sigma_fake = np.mean(fake_flat, axis=0), np.cov(fake_flat, rowvar=False)
    
    # Ensure covariance matrices are 2D (np.cov returns scalar for 1 feature)
    sigma_real = np.atleast_2d(sigma_real)
    sigma_fake = np.atleast_2d(sigma_fake)
    mu_real = np.atleast_1d(mu_real)
    mu_fake = np.atleast_1d(mu_fake)
    
    # Add small epsilon to diagonal for numerical stability
    sigma_real += np.eye(sigma_real.shape[0]) * eps
    sigma_fake += np.eye(sigma_fake.shape[0]) * eps
    
    # Compute squared difference of means
    mean_diff = np.sum((mu_real - mu_fake) ** 2)
    
    # Compute sqrt of product of covariances
    try:
        cov_sqrt = sqrtm(sigma_real.dot(sigma_fake))
        if np.iscomplexobj(cov_sqrt):
            cov_sqrt = cov_sqrt.real
    except Exception as e:
        print(f"Covariance sqrt error: {e}")
        # Return a large value instead of zeros
        return float('inf')
    
    # Compute Fréchet Distance
    fd = mean_diff + np.trace(sigma_real + sigma_fake - 2 * cov_sqrt)
    return max(fd, 0)  # Ensure non-negative value

def compute_mmd(real_samples, fake_samples, kernel='rbf', sigma=1.0):
    """
    Computes Maximum Mean Discrepancy (MMD) between real and fake sensor data distributions.
    
    Args:
        real_samples (numpy array): Shape (num_samples, time_steps, num_channels)
        fake_samples (numpy array): Shape (num_samples, time_steps, num_channels)
        kernel (str): Type of kernel to use ('rbf' or 'linear')
        sigma (float): Kernel bandwidth for RBF kernel
    
    Returns:
        float: MMD score (lower is better)
    """
    real_samples = real_samples.reshape(-1, real_samples.shape[-1])  # Flatten time dimension
    fake_samples = fake_samples.reshape(-1, fake_samples.shape[-1])
    
    if kernel == 'rbf':
        gamma = 1.0 / (2 * sigma ** 2)
        K_real = pairwise_kernels(real_samples, real_samples, metric='rbf', gamma=gamma)
        K_fake = pairwise_kernels(fake_samples, fake_samples, metric='rbf', gamma=gamma)
        K_cross = pairwise_kernels(real_samples, fake_samples, metric='rbf', gamma=gamma)
    else:
        K_real = pairwise_kernels(real_samples, real_samples, metric='linear')
        K_fake = pairwise_kernels(fake_samples, fake_samples, metric='linear')
        K_cross = pairwise_kernels(real_samples, fake_samples, metric='linear')
    
    mmd = np.mean(K_real) + np.mean(K_fake) - 2 * np.mean(K_cross)
    return max(mmd, 0)  # Ensure non-negative value



def spectral_loss_stft(y_true, y_pred, frame_length=64, frame_step=32, fft_length=128):
    """
    Compute spectral loss between real and generated IMU signals using STFT.
    """
    # Get shapes
    num_sensors = tf.shape(y_true)[2]
    
    # Initialize loss accumulators
    total_mse_loss = 0.0
    total_log_mse_loss = 0.0
    
    # Process each channel separately
    for i in range(num_sensors):
        # Extract channel i for all batches
        y_true_channel = y_true[:, :, i]  # Shape: [batch_size, seq_length]
        y_pred_channel = y_pred[:, :, i]  # Shape: [batch_size, seq_length]
        
        # Compute STFT
        stft_true = tf.signal.stft(
            y_true_channel, 
            frame_length=frame_length,
            frame_step=frame_step,
            fft_length=fft_length,
            window_fn=tf.signal.hann_window
        )
        
        stft_pred = tf.signal.stft(
            y_pred_channel,
            frame_length=frame_length,
            frame_step=frame_step,
            fft_length=fft_length,
            window_fn=tf.signal.hann_window
        )
        
        # Process spectrograms and compute loss
        spec_true = tf.abs(stft_true)
        spec_pred = tf.abs(stft_pred)
        
        epsilon = 1e-6
        log_spec_true = tf.math.log(spec_true + epsilon)
        log_spec_pred = tf.math.log(spec_pred + epsilon)
        
        mse_loss = tf.reduce_mean(tf.square(spec_true - spec_pred))
        log_mse_loss = tf.reduce_mean(tf.square(log_spec_true - log_spec_pred))
        
        total_mse_loss += mse_loss
        total_log_mse_loss += log_mse_loss
    
    # Average and combine losses
    avg_mse_loss = total_mse_loss / tf.cast(num_sensors, tf.float32)
    avg_log_mse_loss = total_log_mse_loss / tf.cast(num_sensors, tf.float32)
    combined_loss = avg_mse_loss + 0.5 * avg_log_mse_loss
    
    return combined_loss

def compute_js_divergence_3d_dynamic_bins(real_sensors, fake_sensors):
    """
    Computes the Jensen-Shannon Divergence (JSD) between real and fake sensor data
    for 3D data (samples, timesteps, channels) with dynamic binning.
    
    Parameters:
        real_sensors (numpy.ndarray): Real sensor data of shape (samples, timesteps, channels).
        fake_sensors (numpy.ndarray): Fake sensor data of shape (samples, timesteps, channels).
    
    Returns:
        float: Average Jensen-Shannon Divergence across all sensor channels.
    """
    # Ensure the input is 3D (samples, timesteps, channels)
    if real_sensors.ndim != 3 or fake_sensors.ndim != 3:
        raise ValueError("Input sensor data must be 3D arrays (samples, timesteps, channels).")
    
    # Number of sensor channels
    num_channels = real_sensors.shape[2]
    
    # Initialize list to store JSD for each channel
    js_divergences = []
    
    for channel in range(num_channels):
        # Extract data for the current channel across all samples and timesteps
        real_channel = real_sensors[:, :, channel].flatten()  # Flatten to 1D
        fake_channel = fake_sensors[:, :, channel].flatten()  # Flatten to 1D
        
        # Combine real and fake data to determine dynamic bin edges
        combined_data = np.concatenate([real_channel, fake_channel])
        
        # Compute bin width using Freedman-Diaconis rule
        q75, q25 = np.percentile(combined_data, [75, 25])  # Compute IQR
        iqr = q75 - q25
        bin_width = 2 * iqr / (len(combined_data) ** (1 / 3))  # Freedman-Diaconis rule
        
        # If IQR is zero or bin_width is too small, fall back to a default bin width
        if bin_width <= 0:
            bin_width = (np.max(combined_data) - np.min(combined_data)) / 50  # Default to 50 bins
        
        # Compute the number of bins
        num_bins = int(np.ceil((np.max(combined_data) - np.min(combined_data)) / bin_width))
        
        # Compute histograms (probability distributions) for real and fake data
        real_hist, bin_edges = np.histogram(real_channel, bins=num_bins, density=True)
        fake_hist, _ = np.histogram(fake_channel, bins=bin_edges, density=True)
        
        # Add a small epsilon to avoid division by zero or log(0)
        epsilon = 1e-8
        real_hist = real_hist + epsilon
        fake_hist = fake_hist + epsilon
        
        # Normalize histograms to ensure they sum to 1 (probability distributions)
        real_hist /= np.sum(real_hist)
        fake_hist /= np.sum(fake_hist)
        
        # Compute Jensen-Shannon Divergence using scipy's jensenshannon function
        js_div = jensenshannon(real_hist, fake_hist, base=2)  # Base 2 for information entropy
        js_divergences.append(js_div)
    
    # Average JSD across all channels
    avg_js_divergence = np.mean(js_divergences)
    
    # Convert JSD to a similarity percentage
    similarity = 100 - avg_js_divergence * 100  # Lower JSD means higher similarity
    return max(similarity, 0)  # Ensure it's non-negative


# =====================================================================
# Per-Channel Time-Series-Aware Metrics
# =====================================================================
# These metrics compare real vs synthetic data CHANNEL BY CHANNEL,
# preserving temporal structure.  Each channel is a financial feature
# (e.g. Open, High, Low, Close, Volume, Adj Close).
# =====================================================================

CHANNEL_NAMES_OHLCV = ['Open', 'High', 'Low', 'Close', 'Volume', 'Adj Close']


def _get_channel_name(ch_idx, n_channels):
    """Return a human-readable channel name."""
    if n_channels <= len(CHANNEL_NAMES_OHLCV):
        return CHANNEL_NAMES_OHLCV[ch_idx]
    return f'Channel_{ch_idx}'


# ----- 1.  Per-channel MMD (multi-scale RBF, TF, float64) ------------

def compute_mmd_per_channel(real_data, synthetic_data, num_samples=100,
                            sigmas=None, seed=42):
    """
    Per-channel MMD² using a multi-scale RBF kernel (TF, float64).

    Each channel's seq_len-step sequences are compared as seq_len-dim
    vectors, preserving the temporal ordering within each sequence.

    Args:
        real_data:      (N, seq_len, C) array
        synthetic_data: (N, seq_len, C) array
        num_samples:    how many sequences to subsample (memory safety)
        sigmas:         list of RBF bandwidths; None → [0.1, 0.5, 1, 2, 5]
        seed:           random seed for reproducible subsampling

    Returns:
        dict  {channel_name: mmd_score, ..., 'mean': avg}
    """
    if sigmas is None:
        sigmas = [0.1, 0.5, 1.0, 2.0, 5.0]

    rng = np.random.RandomState(seed)
    n = min(num_samples, len(real_data), len(synthetic_data))
    idx = rng.choice(min(len(real_data), len(synthetic_data)), size=n, replace=False)
    real_sub = np.asarray(real_data)[idx]
    synth_sub = np.asarray(synthetic_data)[idx]
    n_channels = real_sub.shape[2]

    def _mmd_multiscale(x, y):
        x = tf.cast(x, tf.float64)
        y = tf.cast(y, tf.float64)
        xx = tf.reduce_sum(x * x, axis=1, keepdims=True)
        yy = tf.reduce_sum(y * y, axis=1, keepdims=True)
        D_xx = tf.maximum(xx - 2.0 * tf.matmul(x, tf.transpose(x)) + tf.transpose(xx), 0.0)
        D_yy = tf.maximum(yy - 2.0 * tf.matmul(y, tf.transpose(y)) + tf.transpose(yy), 0.0)
        D_xy = tf.maximum(xx - 2.0 * tf.matmul(x, tf.transpose(y)) + tf.transpose(yy), 0.0)
        mmd = tf.constant(0.0, dtype=tf.float64)
        for s in sigmas:
            g = 1.0 / (2.0 * s * s)
            mmd += (tf.reduce_mean(tf.exp(-g * D_xx))
                    + tf.reduce_mean(tf.exp(-g * D_yy))
                    - 2.0 * tf.reduce_mean(tf.exp(-g * D_xy)))
        return (mmd / len(sigmas)).numpy()

    results = {}
    for ch in range(n_channels):
        name = _get_channel_name(ch, n_channels)
        results[name] = float(_mmd_multiscale(real_sub[:, :, ch], synth_sub[:, :, ch]))
    results['mean'] = float(np.mean([v for k, v in results.items() if k != 'mean']))
    return results


# ----- 2.  Per-channel Autocorrelation RMSE ---------------------------

def compute_acf_per_channel(real_data, synthetic_data, max_lag=10):
    """
    Per-channel autocorrelation RMSE on both raw values and squared values
    (proxy for volatility clustering).

    Each channel is treated independently:
      - Flatten all (N × seq_len) values for that channel
      - Compute sample ACF at lags 1..max_lag
      - RMSE between real ACF and synthetic ACF

    Args:
        real_data:      (N, T, C) array
        synthetic_data: (N, T, C) array
        max_lag:        number of lags

    Returns:
        dict with per-channel ACF-RMSE for returns and squared returns
    """
    real_data = np.asarray(real_data)
    synthetic_data = np.asarray(synthetic_data)
    n_channels = real_data.shape[2]

    def _acf(series, max_lag):
        mean = np.mean(series)
        var = np.var(series)
        if var < 1e-12:
            return np.zeros(max_lag)
        return np.array([
            np.mean((series[:-lag] - mean) * (series[lag:] - mean)) / var
            for lag in range(1, max_lag + 1)
        ])

    results = {'returns': {}, 'squared': {}}
    for ch in range(n_channels):
        name = _get_channel_name(ch, n_channels)
        real_ch = real_data[:, :, ch].flatten()
        synth_ch = synthetic_data[:, :, ch].flatten()

        # Raw returns ACF
        r_acf = _acf(real_ch, max_lag)
        s_acf = _acf(synth_ch, max_lag)
        results['returns'][name] = float(np.sqrt(np.mean((r_acf - s_acf) ** 2)))

        # Squared (volatility clustering) ACF
        r_sq_acf = _acf(real_ch ** 2, max_lag)
        s_sq_acf = _acf(synth_ch ** 2, max_lag)
        results['squared'][name] = float(np.sqrt(np.mean((r_sq_acf - s_sq_acf) ** 2)))

    results['mean_returns'] = float(np.mean(list(results['returns'].values())))
    results['mean_squared'] = float(np.mean(list(results['squared'].values())))
    return results


# ----- 3.  Per-channel Distribution Metrics (Wasserstein + KS) --------

def compute_distribution_per_channel(real_data, synthetic_data):
    """
    Per-channel Wasserstein-1 distance and Kolmogorov-Smirnov statistic.

    These measure how well the marginal distribution of each channel is
    reproduced (e.g. does synthetic Close have the same value range,
    skew, and tails as real Close?).

    Args:
        real_data:      (N, T, C)
        synthetic_data: (N, T, C)

    Returns:
        dict with 'wasserstein' and 'ks' sub-dicts per channel + means
    """
    from scipy.stats import ks_2samp, wasserstein_distance

    real_data = np.asarray(real_data)
    synthetic_data = np.asarray(synthetic_data)
    n_channels = real_data.shape[2]

    results = {'wasserstein': {}, 'ks_statistic': {}, 'ks_pvalue': {}}
    for ch in range(n_channels):
        name = _get_channel_name(ch, n_channels)
        real_ch = real_data[:, :, ch].flatten()
        synth_ch = synthetic_data[:, :, ch].flatten()

        results['wasserstein'][name] = float(wasserstein_distance(real_ch, synth_ch))
        ks_stat, ks_p = ks_2samp(real_ch, synth_ch)
        results['ks_statistic'][name] = float(ks_stat)
        results['ks_pvalue'][name] = float(ks_p)

    results['mean_wasserstein'] = float(np.mean(list(results['wasserstein'].values())))
    results['mean_ks'] = float(np.mean(list(results['ks_statistic'].values())))
    return results


# ----- 4.  Cross-Correlation Preservation -----------------------------

def compute_cross_correlation_distance(real_data, synthetic_data):
    """
    Measures whether inter-channel correlations are preserved.

    For financial OHLCV data the correlation structure matters —
    Open/Close should be highly correlated, Volume less so, etc.
    Computes the Frobenius norm between the real and synthetic
    correlation matrices (lower = better).

    Args:
        real_data:      (N, T, C)
        synthetic_data: (N, T, C)

    Returns:
        dict with 'frobenius_distance', 'real_corr', 'synthetic_corr'
    """
    real_data = np.asarray(real_data)
    synthetic_data = np.asarray(synthetic_data)
    n_channels = real_data.shape[2]
    channel_names = [_get_channel_name(i, n_channels) for i in range(n_channels)]

    # Flatten (N, T, C) → (N*T, C) to get per-channel correlation
    real_flat = real_data.reshape(-1, n_channels)
    synth_flat = synthetic_data.reshape(-1, n_channels)

    real_corr = np.corrcoef(real_flat, rowvar=False)
    synth_corr = np.corrcoef(synth_flat, rowvar=False)

    diff = real_corr - synth_corr
    frobenius = float(np.sqrt(np.sum(diff ** 2)))
    # Normalise by matrix size so it's comparable across different C
    frobenius_norm = frobenius / n_channels

    return {
        'frobenius_distance': frobenius,
        'frobenius_normalised': frobenius_norm,
        'channel_names': channel_names,
        'real_corr': real_corr,
        'synthetic_corr': synth_corr,
    }


# ----- 5.  Per-channel Tail / Shape Metrics ---------------------------

def compute_tail_metrics_per_channel(real_data, synthetic_data):
    """
    Per-channel comparison of distribution shape:
      - Mean absolute error of means
      - Std-dev ratio (synthetic / real)
      - Skewness difference
      - Kurtosis difference

    These are especially important for financial data where fat tails
    and skew drive risk metrics.

    Args:
        real_data:      (N, T, C)
        synthetic_data: (N, T, C)

    Returns:
        dict with per-channel shape metrics
    """
    from scipy.stats import skew, kurtosis

    real_data = np.asarray(real_data)
    synthetic_data = np.asarray(synthetic_data)
    n_channels = real_data.shape[2]

    results = {'mean_diff': {}, 'std_ratio': {}, 'skew_diff': {}, 'kurtosis_diff': {}}
    for ch in range(n_channels):
        name = _get_channel_name(ch, n_channels)
        r = real_data[:, :, ch].flatten()
        s = synthetic_data[:, :, ch].flatten()

        results['mean_diff'][name] = float(abs(np.mean(r) - np.mean(s)))
        r_std = np.std(r)
        results['std_ratio'][name] = float(np.std(s) / r_std) if r_std > 1e-12 else float('inf')
        results['skew_diff'][name] = float(abs(skew(r) - skew(s)))
        results['kurtosis_diff'][name] = float(abs(kurtosis(r) - kurtosis(s)))

    for key in list(results.keys()):
        vals = list(results[key].values())
        results[f'mean_{key}'] = float(np.mean(vals))
    return results


# ----- 6.  Convenience: run ALL per-channel metrics at once -----------

def evaluate_per_channel(real_data, synthetic_data, num_samples_mmd=100,
                         n_dtw=50, max_lag=10, verbose=True):
    """
    Run all per-channel metrics and return a single results dict.

    Metrics computed:
      1. MMD (multi-scale RBF, per channel)
      2. Autocorrelation RMSE (returns + squared, per channel)
      3. Distribution (Wasserstein-1 + KS, per channel)
      4. Cross-correlation preservation (Frobenius norm)
      5. Tail / shape metrics (mean, std, skew, kurtosis diffs)

    Args:
        real_data:        (N, T, C) array
        synthetic_data:   (N, T, C) array
        num_samples_mmd:  sequences to subsample for MMD
        n_dtw:            samples for DTW (kept from existing func)
        max_lag:          ACF lags
        verbose:          print summary table

    Returns:
        dict with all results keyed by metric name
    """
    real_data = np.asarray(real_data)
    synthetic_data = np.asarray(synthetic_data)
    n_channels = real_data.shape[2]
    names = [_get_channel_name(i, n_channels) for i in range(n_channels)]

    results = {}

    # 1. MMD
    results['mmd'] = compute_mmd_per_channel(real_data, synthetic_data,
                                             num_samples=num_samples_mmd)

    # 2. ACF RMSE
    results['acf'] = compute_acf_per_channel(real_data, synthetic_data,
                                             max_lag=max_lag)

    # 3. Distribution
    results['distribution'] = compute_distribution_per_channel(real_data,
                                                               synthetic_data)

    # 4. Cross-correlation
    results['cross_corr'] = compute_cross_correlation_distance(real_data,
                                                               synthetic_data)

    # 5. Tail metrics
    results['tail'] = compute_tail_metrics_per_channel(real_data, synthetic_data)

    if verbose:
        sep = '=' * 72
        print(f'\n{sep}')
        print('PER-CHANNEL EVALUATION SUMMARY  (Real vs Synthetic)')
        print(sep)
        header = f'{"Channel":<12}'
        header += f'{"MMD":>9} {"W-Dist":>9} {"KS":>9}'
        header += f'{"ACF-R":>9} {"ACF-S":>9}'
        header += f'{"StdRatio":>9} {"SkewΔ":>9} {"KurtΔ":>9}'
        print(header)
        print('-' * 72)
        for name in names:
            row = f'{name:<12}'
            row += f'{results["mmd"][name]:>9.6f}'
            row += f'{results["distribution"]["wasserstein"][name]:>9.6f}'
            row += f'{results["distribution"]["ks_statistic"][name]:>9.4f}'
            row += f'{results["acf"]["returns"][name]:>9.6f}'
            row += f'{results["acf"]["squared"][name]:>9.6f}'
            row += f'{results["tail"]["std_ratio"][name]:>9.4f}'
            row += f'{results["tail"]["skew_diff"][name]:>9.4f}'
            row += f'{results["tail"]["kurtosis_diff"][name]:>9.4f}'
            print(row)
        print('-' * 72)
        row = f'{"MEAN":<12}'
        row += f'{results["mmd"]["mean"]:>9.6f}'
        row += f'{results["distribution"]["mean_wasserstein"]:>9.6f}'
        row += f'{results["distribution"]["mean_ks"]:>9.4f}'
        row += f'{results["acf"]["mean_returns"]:>9.6f}'
        row += f'{results["acf"]["mean_squared"]:>9.6f}'
        row += f'{results["tail"]["mean_std_ratio"]:>9.4f}'
        row += f'{results["tail"]["mean_skew_diff"]:>9.4f}'
        row += f'{results["tail"]["mean_kurtosis_diff"]:>9.4f}'
        print(row)
        print(f'\nCross-corr Frobenius dist (norm): '
              f'{results["cross_corr"]["frobenius_normalised"]:.6f}')
        print(sep)

    return results


# =====================================================================
# Additional Metrics (added for unified evaluation notebook)
# =====================================================================


# ----- 7.  ACF Vectors (for plotting) ---------------------------------

def compute_acf_vectors(real_data, synthetic_data, max_lag=10):
    """
    Compute per-channel ACF vectors for real and synthetic data.

    Unlike compute_acf_per_channel (which returns only RMSE scalars),
    this returns the actual ACF arrays so they can be plotted.

    Args:
        real_data:      (N, T, C) array
        synthetic_data: (N, T, C) array
        max_lag:        number of lags

    Returns:
        dict with keys:
          'lags':    np.arange(1, max_lag+1)
          'returns': {channel_name: {'real': array, 'synthetic': array, 'rmse': float}}
          'squared': {channel_name: {'real': array, 'synthetic': array, 'rmse': float}}
    """
    real_data = np.asarray(real_data)
    synthetic_data = np.asarray(synthetic_data)
    n_channels = real_data.shape[2]

    def _acf(series, max_lag):
        mean = np.mean(series)
        var = np.var(series)
        if var < 1e-12:
            return np.zeros(max_lag)
        return np.array([
            np.mean((series[:-lag] - mean) * (series[lag:] - mean)) / var
            for lag in range(1, max_lag + 1)
        ])

    results = {'lags': np.arange(1, max_lag + 1), 'returns': {}, 'squared': {}}
    for ch in range(n_channels):
        name = _get_channel_name(ch, n_channels)
        real_ch = real_data[:, :, ch].flatten()
        synth_ch = synthetic_data[:, :, ch].flatten()

        r_acf = _acf(real_ch, max_lag)
        s_acf = _acf(synth_ch, max_lag)
        results['returns'][name] = {
            'real': r_acf, 'synthetic': s_acf,
            'rmse': float(np.sqrt(np.mean((r_acf - s_acf) ** 2)))
        }

        r_sq = _acf(real_ch ** 2, max_lag)
        s_sq = _acf(synth_ch ** 2, max_lag)
        results['squared'][name] = {
            'real': r_sq, 'synthetic': s_sq,
            'rmse': float(np.sqrt(np.mean((r_sq - s_sq) ** 2)))
        }
    return results


# ----- 8.  Discriminative Score (MLP-based) ---------------------------

def compute_discriminative_score(real_data, synthetic_data,
                                 hidden_layers=(64, 32), max_iter=300,
                                 test_size=0.2, seed=42):
    """
    Train-on-synthetic, test-on-real discriminative score
    (from the original TimeGAN paper, Yoon et al. 2019).

    A 2-layer MLP is trained to classify sequences as real (1) or
    synthetic (0).  The discriminative score = |accuracy − 0.5|.
    Lower is better (the classifier cannot tell them apart).

    Args:
        real_data:      (N, T, C) or (N, T) array
        synthetic_data: (N, T, C) or (N, T) array
        hidden_layers:  MLP hidden-layer sizes
        max_iter:       training iterations
        test_size:      fraction held out for testing
        seed:           random state

    Returns:
        dict with 'accuracy', 'score' (= |acc - 0.5|)
    """
    from sklearn.model_selection import train_test_split
    from sklearn.neural_network import MLPClassifier
    from sklearn.metrics import accuracy_score

    real = np.asarray(real_data)
    synth = np.asarray(synthetic_data)

    # Flatten to 2D if needed: (N, T*C)
    if real.ndim == 3:
        real = real.reshape(real.shape[0], -1)
    if synth.ndim == 3:
        synth = synth.reshape(synth.shape[0], -1)

    X = np.vstack([real, synth])
    y = np.concatenate([np.ones(len(real)), np.zeros(len(synth))])

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=test_size, random_state=seed, stratify=y
    )

    clf = MLPClassifier(hidden_layer_sizes=hidden_layers,
                        max_iter=max_iter, random_state=seed)
    clf.fit(X_train, y_train)
    acc = accuracy_score(y_test, clf.predict(X_test))

    return {'accuracy': float(acc), 'score': float(abs(0.5 - acc))}


# ── Entropy-based metrics ────────────────────────────────────────
from math import log2
from collections import Counter


def _discretize(series, n_bins=5, edges=None):
    """Discretize a 1-D array into a string of bin labels ('0','1',...).

    Uses equal-frequency (quantile) binning.  If ``edges`` is provided
    they are reused (so real and synthetic share the same grid).
    Returns (string, edges).
    """
    from numpy import percentile, digitize, linspace, clip
    if edges is None:
        edges = percentile(series, linspace(0, 100, n_bins + 1))
        edges[-1] += 1e-10
        for i in range(1, len(edges)):
            if edges[i] <= edges[i - 1]:
                edges[i] = edges[i - 1] + 1e-10
    bins = digitize(series, edges[1:-1])
    bins = clip(bins, 0, n_bins - 1)
    return ''.join(str(b) for b in bins), edges


def shannon_entropy(message):
    """Shannon entropy of a discrete message string (bits)."""
    if not message:
        return 0.0
    counts = Counter(message)
    n = len(message)
    return -sum((c / n) * log2(c / n) for c in counts.values())


def lempel_ziv_complexity(message):
    """Lempel-Ziv (LZ76) complexity normalised by message length."""
    if not message:
        return 0.0
    library = set()
    n = len(message)
    i = 0
    while i < n:
        j = i
        while j < n and message[i:j + 1] in library:
            j += 1
        library.add(message[i:j + 1])
        i = j + 1
    return len(library) / n


def plug_in_entropy(message, word_length=1):
    """Plug-in (block) entropy estimator for a given word length."""
    if len(message) < word_length or word_length < 1:
        return 0.0
    words = [message[i:i + word_length]
             for i in range(len(message) - word_length + 1)]
    counts = Counter(words)
    n = len(words)
    return -sum((c / n) * log2(c / n) for c in counts.values())


def compute_entropy_metrics(real_data, synthetic_data, n_bins=5):
    """Compare entropy between real and synthetic 3-D arrays.

    Parameters
    ----------
    real_data : ndarray (N, T, C) or (N, T)
        Real sequences.
    synthetic_data : ndarray (N, T, C) or (N, T)
        Synthetic sequences.
    n_bins : int
        Number of quantile bins for discretization.

    Returns
    -------
    dict  with keys per channel:
        'shannon_real', 'shannon_synth', 'shannon_diff',
        'lz_real', 'lz_synth', 'lz_diff',
        'plugin2_real', 'plugin2_synth', 'plugin2_diff',
        'mean_shannon_diff', 'mean_lz_diff', 'mean_plugin2_diff'
    """
    if real_data.ndim == 2:
        real_data = real_data[:, :, np.newaxis]
    if synthetic_data.ndim == 2:
        synthetic_data = synthetic_data[:, :, np.newaxis]

    n_channels = real_data.shape[2]
    results = {
        'shannon_real': {}, 'shannon_synth': {}, 'shannon_diff': {},
        'lz_real': {}, 'lz_synth': {}, 'lz_diff': {},
        'plugin2_real': {}, 'plugin2_synth': {}, 'plugin2_diff': {},
    }

    for c in range(n_channels):
        ch = _get_channel_name(c, n_channels)
        # flatten all samples into one long series per channel
        real_flat = real_data[:, :, c].ravel()
        synth_flat = synthetic_data[:, :, c].ravel()

        # discretize using real distribution's quantile edges
        real_str, real_edges = _discretize(real_flat, n_bins)
        synth_str, _ = _discretize(synth_flat, n_bins, edges=real_edges)

        # Shannon
        sr = shannon_entropy(real_str)
        ss = shannon_entropy(synth_str)
        results['shannon_real'][ch] = sr
        results['shannon_synth'][ch] = ss
        results['shannon_diff'][ch] = abs(sr - ss)

        # Lempel-Ziv
        lr = lempel_ziv_complexity(real_str)
        ls = lempel_ziv_complexity(synth_str)
        results['lz_real'][ch] = lr
        results['lz_synth'][ch] = ls
        results['lz_diff'][ch] = abs(lr - ls)

        # Plug-in (word-length 2) — captures pairwise transitions
        pr = plug_in_entropy(real_str, word_length=2)
        ps = plug_in_entropy(synth_str, word_length=2)
        results['plugin2_real'][ch] = pr
        results['plugin2_synth'][ch] = ps
        results['plugin2_diff'][ch] = abs(pr - ps)

    # mean across channels
    results['mean_shannon_diff'] = float(np.mean(
        list(results['shannon_diff'].values())))
    results['mean_lz_diff'] = float(np.mean(
        list(results['lz_diff'].values())))
    results['mean_plugin2_diff'] = float(np.mean(
        list(results['plugin2_diff'].values())))

    return results


# =====================================================================
# MICROSTRUCTURE METRICS  (require High / Low / Close columns)
# =====================================================================

def _cs_beta(high, low, window=20):
    """Corwin-Schultz β: rolling(2).sum of daily [ln(H/L)]², then
    rolling(window).mean.  Matches RiskLabAI / de Prado snippet 19.1.

    Returns np.ndarray with NaN where insufficient data.
    """
    log_ratio_sq = np.log(high / low) ** 2
    # Sum of 2 consecutive days
    beta_2d = np.full_like(log_ratio_sq, np.nan)
    beta_2d[1:] = log_ratio_sq[1:] + log_ratio_sq[:-1]
    # Rolling mean over window
    beta = np.full_like(beta_2d, np.nan)
    for i in range(window - 1, len(beta_2d)):
        chunk = beta_2d[i - window + 1 : i + 1]
        valid = chunk[~np.isnan(chunk)]
        if len(valid) > 0:
            beta[i] = np.mean(valid)
    return beta


def _cs_gamma(high, low):
    """Corwin-Schultz γ: [ln(H_2day / L_2day)]² where H_2day and L_2day
    are the max-high and min-low over 2 consecutive days.
    Matches RiskLabAI / de Prado snippet 19.1.
    """
    gamma = np.full(len(high), np.nan)
    for i in range(1, len(high)):
        h2 = max(high[i], high[i - 1])
        l2 = min(low[i], low[i - 1])
        gamma[i] = np.log(h2 / l2) ** 2
    return gamma


# Constant d from Corwin-Schultz (2012)
_CS_DENOM = 3.0 - 2.0 * np.sqrt(2.0)


def _cs_alpha(beta, gamma):
    """Corwin-Schultz α, floored at 0.
    α = [(√2 − 1)·√β] / d  −  √(γ / d)
    """
    term1 = (np.sqrt(2.0) - 1.0) * np.sqrt(np.maximum(beta, 0.0)) / _CS_DENOM
    term2 = np.sqrt(np.maximum(gamma / _CS_DENOM, 0.0))
    alpha = np.maximum(term1 - term2, 0.0)
    return alpha


def corwin_schultz_spread(high, low, window=20):
    """Corwin-Schultz (2012) bid-ask spread estimator.

    Matches the canonical RiskLabAI / de Prado (AFML snippet 19.1)
    implementation:
        β = rolling(2).sum of [ln(H/L)]², then rolling(window).mean
        γ = [ln(max(H_t,H_{t-1}) / min(L_t,L_{t-1}))]²
        α = [(√2−1)·√β] / d  − √(γ/d),   floored at 0
        S = 2·(e^α − 1) / (1 + e^α)

    Parameters
    ----------
    high, low : array-like, shape (T,)
        Daily High and Low prices.
    window : int
        Rolling window for averaging β.

    Returns
    -------
    spread : np.ndarray, shape (T,)
        Estimated proportional spread.  NaN for early entries.
    """
    high = np.asarray(high, dtype=np.float64)
    low = np.asarray(low, dtype=np.float64)

    beta = _cs_beta(high, low, window)
    gamma = _cs_gamma(high, low)
    alpha = _cs_alpha(beta, gamma)

    spread = 2.0 * (np.exp(alpha) - 1.0) / (1.0 + np.exp(alpha))
    # Where alpha is NaN, spread is NaN (no explicit floor needed, alpha already ≥ 0)
    return spread


def bekker_parkinson_volatility(high, low, window=20):
    """Bekker-Parkinson volatility estimator (de Prado, AFML snippet 19.2).

    Adjusts Parkinson vol by removing the Corwin-Schultz spread component:
        k₂ = √(8/π)
        σ = [(√2 − 1)·√β] / d  +  √(γ / (k₂²·d))
    where β and γ are from Corwin-Schultz.

    This gives a *spread-adjusted* volatility — more accurate for less
    liquid instruments where H-L range includes bid-ask bounce.

    Parameters
    ----------
    high, low : array-like, shape (T,)
        Daily High and Low prices.
    window : int
        Rolling window for β averaging.

    Returns
    -------
    bp_vol : np.ndarray, shape (T,)
        Bekker-Parkinson volatility.  NaN for early entries.
    """
    high = np.asarray(high, dtype=np.float64)
    low = np.asarray(low, dtype=np.float64)

    beta = _cs_beta(high, low, window)
    gamma = _cs_gamma(high, low)

    k2 = np.sqrt(8.0 / np.pi)
    term1 = (np.sqrt(2.0) - 1.0) * np.sqrt(np.maximum(beta, 0.0)) / _CS_DENOM
    term2 = np.sqrt(np.maximum(gamma / (k2 ** 2 * _CS_DENOM), 0.0))
    bp_vol = np.maximum(term1 + term2, 0.0)
    return bp_vol


def compute_microstructure_metrics(real_high, real_low, real_close,
                                    synth_high, synth_low, synth_close,
                                    window=20):
    """Compare Bekker-Parkinson vol and Corwin-Schultz spread between
    real and synthetic OHLCV data.

    Uses the canonical RiskLabAI / de Prado formulas (AFML Ch. 19).

    Parameters
    ----------
    real_high, real_low, real_close : array-like, shape (T_real,)
    synth_high, synth_low, synth_close : array-like, shape (T_synth,)
    window : int
        Rolling window for both estimators.

    Returns
    -------
    dict with keys:
        bp_vol_real_mean, bp_vol_synth_mean, bp_vol_diff,
        cs_spread_real_mean, cs_spread_synth_mean, cs_spread_diff,
        bp_vol_real, bp_vol_synth, cs_spread_real, cs_spread_synth
    """
    bp_real = bekker_parkinson_volatility(real_high, real_low, window)
    bp_synth = bekker_parkinson_volatility(synth_high, synth_low, window)
    cs_real = corwin_schultz_spread(real_high, real_low, window)
    cs_synth = corwin_schultz_spread(synth_high, synth_low, window)

    bp_r_mean = float(np.nanmean(bp_real))
    bp_s_mean = float(np.nanmean(bp_synth))
    cs_r_mean = float(np.nanmean(cs_real))
    cs_s_mean = float(np.nanmean(cs_synth))

    return {
        'bp_vol_real_mean': bp_r_mean,
        'bp_vol_synth_mean': bp_s_mean,
        'bp_vol_diff': abs(bp_r_mean - bp_s_mean),
        'cs_spread_real_mean': cs_r_mean,
        'cs_spread_synth_mean': cs_s_mean,
        'cs_spread_diff': abs(cs_r_mean - cs_s_mean),
        # Full series for plotting
        'bp_vol_real': bp_real,
        'bp_vol_synth': bp_synth,
        'cs_spread_real': cs_real,
        'cs_spread_synth': cs_synth,
    }
