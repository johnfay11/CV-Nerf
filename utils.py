import torch
import numpy as np

def inv_transform_sampling(pts, weights, n):
    """
    Get of sampled points and corresponding weights, we perform inverse transform sampling:
    https://en.wikipedia.org/wiki/Inverse_transform_sampling. In essence, this technique allows us
    to sample n points from the weight pdf.
    """

    # numerical stability
    EPS = 1e-5
    weights = weights + EPS

    # map weights to a probability distribution
    pdf = weights / torch.sum(weights, -1, keepdim=True)

    # construct cdf (denote F) from pdf
    cdf = torch.cumsum(pdf, -1)
    cdf = torch.cat([torch.zeros_like(cdf[..., :1]), cdf], -1)

    # uniformly sample points
    unif_samp = torch.rand(list(cdf.shape[:-1]) + [n])

    """
    Invert the CDF and compute F^{-1}(U). This reduces to a searching for domain values of F that contain the 
    values of U and then rescaling. See the following source for more information:
    - http://www.columbia.edu/~ks20/4404-Sigman/4404-Notes-ITM.pdf 
    - https://stephens999.github.io/fiveMinuteStats/inverse_transform_sampling.html 
    - http://www.cse.psu.edu/~rtc12/CSE586/lectures/cse586samplingPreMCMC.pdf
    """

    unif_samp = unif_samp.contiguous()
    # luckily, searchsorted implements this searching functionality!
    i = torch.searchsorted(cdf, unif_samp, right=True)

    # upper and lower bounds of U w.r.t. F
    upper = torch.min((cdf.shape[-1] - 1) * torch.ones_like(i), i)
    lower = torch.max(torch.zeros_like(i - 1), i - 1)
    indices = torch.stack([lower, upper], -1)

    # rescale input parameters to match shape of uniformly chosen points
    _new_shape = [indices.shape[0], indices.shape[1], cdf.shape[-1]]
    cdf = cdf.unsqueeze(1).expand(_new_shape)
    cdf = torch.gather(cdf, 2, indices)
    pts = pts.unsqueeze(1).expand(_new_shape)
    pts = torch.gather(pts, 2, indices)

    # rescale into [t_n, t_f] domain
    scale = cdf[..., 1] - cdf[..., 0]
    # need this for numerical stability
    scale = torch.where(scale < EPS, torch.ones_like(scale), scale)
    return (pts[..., 1] - pts[..., 0]) * ((unif_samp - cdf[..., 0]) / scale) + pts[..., 0]


# converts [0, 1] images to 8 byte images
def cont_to_byte8_im(x):
    return (255 * np.clip(x, 0, 1)).astype(np.uint8)
