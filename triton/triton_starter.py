# avoid bottleneck : metric computation crafted on Triton > Parallelization of L2 operation, optimizing Image-space
# 곡률(Curvature)이 낮은 영역에서는 빠르게 L2​ Proxy로 전환하는 'Adaptive Switching'



import torch
import triton
import triton.language as tl



@triton.jit
def riemannian_drift_kernel(
    score_ptr, metric_ptr, out_ptr,
    n_el, BLOCK_SIZE: tl.constexpr
)
    pid = tl.program_id(0)
    offsets = pid * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
    mask = offsets < n_el

    # data loader
    score = tl.load(score_ptr + offsets, mask = mask)
    metric = tl.load(metric_ptr + offsets, mask=mask)


    # Gradient: g^-1 * score
    # local curvarture
    drift = score / (metric + 1e-6)

    tl.store(out_ptr + offsets, drift, mask=mask)

def triton_riemanninan_step(score, metric)
    out = torch.empty_like(score)
    n_el = score.numel()
    grid = lambda meta: (triton.cdiv(n_el, meta['BLCOK_SIZE']))
    riemannian_drift_kernel[grid](score, metric, out, n_el, BLOCK_SIZE=1024)

    return out

# 로컬 곡률(κ) 혹은 메트릭의 변화율(∥∇g∥)을 기준으로 연산 모드
# Mode A (L2​ Proxy): 곡률이 임계값(τ)보다 낮을 때. 선형 근사(Tangent Space Approximation)를 사용하여 속도를 극대화합니다.
# Mode B (Riemannian): 곡률이 높을 때. 실제 Geodesic Drift와 Metric Inverse를 계산하여 정밀도를 확보합니다.


def adaptive_riemannian_kernel(
    x_ptr, g_ptr, out_ptr,
    threshold, n_el,
    BLOCK_SIZE: tl.constexpr
):
    pid = tl.program_id(0)
    offsets = pid * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
    mask = offsets < n_el

    
    # data loader
    x = tl.load(x_ptr + offsets, mask=mask)
    g = tl.load(g_ptr + offsets, mask=mask)

    # ||g||
    cuvarture_estimation = tl.abs(g - 1.0)

    # L2 proxy : x_{t+1} = x_t + space
    l2_step = x + score

    # Riemannian: x_{t+1} = x_t + g^-1 * score
    re_step = x + (1.0 / (g + 1e-6)) * score

    if curvature_estimation > threshold:
        final_step = re_step
    else:
        final_step = l2_step

    tl.store(out_ptr + offsets, final_step, mask=mask)
