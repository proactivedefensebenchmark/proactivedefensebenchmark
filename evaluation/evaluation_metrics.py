import os
import torch
import torch.nn
import torch.nn.functional as F
from torchvision import transforms as T
from skimage.metrics import peak_signal_noise_ratio
from skimage.metrics import structural_similarity
import numpy as np
import lpips
import piq
import warnings
# from deepface import DeepFace # pip install deepface - lazy import to avoid tensorflow issues
# from FaceImageQuality.face_image_quality import SER_FIQ - lazy import
# from facenet_pytorch import InceptionResnetV1 # for facenet - lazy import
# from vgg_loss import VGGLoss 
from utils.utils import *
from evaluation.face_idloss import IDLoss

warnings.filterwarnings("ignore", message="The parameter 'pretrained' is deprecated")
warnings.filterwarnings("ignore", module="torchvision.models._utils")

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# for ArcFace
from configs.paths_config import ARCFACE_CKPT
arcface = IDLoss(ARCFACE_CKPT).to(DEVICE).eval()

# for LPIPS
loss_fn_alex = lpips.LPIPS(net='alex').to(DEVICE).eval()
loss_fn_vgg = lpips.LPIPS(net='vgg').to(DEVICE).eval()

'''
complete list of evaluation metrics:
- MAE
- MSE
- RMSE
- SSIM
- PSNR
- LPIPS(alex, vgg)
- ID LOSS (My face is Mine의 CS_src, CS_att와 동일)
- BRISQUE
- Anti-Forgery ASR (CMUA의 asr, disrupting deepfake, dual defense, DF-RAP, SCOL의 DSR과 동일)
- CMUA : L2_mask
- TAFIM : VGG Loss
- Restricted FSIM
- LEAT CDSR
- Anti-dreambooth : FDFR, ISM (SCOL의 ISM과 동일), SER-FQA
- Dual Defense : SR_mask (CMUA, Scalable의 SR_mask와 동일)
- SCOL : PDS
'''

'''
have to do:
- MagDR : Feature Sim (논문에 detail 내용 없음)
- CMUA : TFHC (hyperFAS 중국 github)
- dual defense : FN_acc (FaceNet 기반)
- FaceSwapGuard : FMR (FR로 평가하는데, arcface, cosface, facenet 3개와 외부 API 3개 호출)
'''

# metric functions
def mae(img1, img2):
    criterion = torch.nn.L1Loss(reduction='mean')
    loss = criterion(img1, img2)
    return loss

def mse(img1, img2):
    criterion = torch.nn.MSELoss(reduction='mean')
    loss = criterion(img1, img2)
    return loss

def rmse(img1, img2):
    criterion = torch.nn.MSELoss(reduction='mean')
    mse = criterion(img1, img2)
    loss = torch.sqrt(mse)
    return loss

def ssim(img1,img2):
    # Convert from [-1, 1] to [0, 1] range
    img1_01 = (img1.clamp(-1, 1) * 0.5 + 0.5).squeeze(0).cpu().numpy()
    img2_01 = (img2.clamp(-1, 1) * 0.5 + 0.5).squeeze(0).cpu().numpy()
    img1_01 = np.transpose(img1_01, (1, 2, 0))
    img2_01 = np.transpose(img2_01, (1, 2, 0))
    
    # Clip to [0, 1] range
    img1_01 = np.clip(img1_01, 0.0, 1.0)
    img2_01 = np.clip(img2_01, 0.0, 1.0)
    
    ssim_val = structural_similarity(img1_01, img2_01, data_range=1.0, channel_axis=-1)
    return ssim_val

def psnr(img1,img2):
    # Convert from [-1, 1] to [0, 1] range
    img1_01 = (img1.clamp(-1, 1) * 0.5 + 0.5).squeeze(0).cpu().numpy()
    img2_01 = (img2.clamp(-1, 1) * 0.5 + 0.5).squeeze(0).cpu().numpy()
    img1_01 = np.transpose(img1_01, (1, 2, 0))
    img2_01 = np.transpose(img2_01, (1, 2, 0))
    
    # Clip to [0, 1] range
    img1_01 = np.clip(img1_01, 0.0, 1.0)
    img2_01 = np.clip(img2_01, 0.0, 1.0)
    
    psnr_val = peak_signal_noise_ratio(img1_01, img2_01, data_range=1.0)
    return psnr_val

def id_loss(img_1, img_2):
    # Ensure tensors are on the same device
    img_1 = img_1.to(DEVICE)
    img_2 = img_2.to(DEVICE)
    img_1 = F.interpolate(denorm(img_1), size=(112,112), mode='bilinear')
    img_2 = F.interpolate(denorm(img_2), size=(112,112), mode='bilinear')
    loss = arcface(img_1, img_2)
    return loss

def vgg_lpips(img1, img2):
    # Ensure tensors are on the same device as the model
    img1 = img1.to(DEVICE)
    img2 = img2.to(DEVICE)
    lpips_vgg = loss_fn_vgg.forward(img1, img2)
    return lpips_vgg

def alex_lpips(img1, img2):
    # Ensure tensors are on the same device as the model
    img1 = img1.to(DEVICE)
    img2 = img2.to(DEVICE)
    lpips_alex = loss_fn_alex.forward(img1, img2)
    return lpips_alex

def brisque_metric(img):
    img = denorm(img.float())
    if img.dim() == 3:
        img = img.unsqueeze(0)
    score = piq.brisque(img, data_range=1.0, reduction='none')
    return float(score.mean().item())

def anti_asr(img1, img2):
    criterion = torch.nn.MSELoss(reduction='mean')
    loss = criterion(img1, img2)
    if loss >= 0.05:
        return 1
    
def fsim(img1, img2):
    img1 = denorm(img1.float())
    img2 = denorm(img2.float())
    if img1.dim() == 3:
        img1 = img1.unsqueeze(0)
    if img2.dim() == 3:
        img2 = img2.unsqueeze(0)
    score = piq.fsim(img1, img2, data_range=1.0, reduction='mean')
    return float(score.item())

def leat_cdsr(img1, img2, l2_threshold: float = 0.05, id_threshold: float = 0.6, lpips_threshold: float = 0.35):
    l2_val = float(rmse(img1, img2).item())
    id_val = float(id_loss(img1, img2))
    lpips_val = float(vgg_lpips(img1, img2).item())

    cond_l2 = l2_val > l2_threshold
    cond_id = id_val > id_threshold
    cond_lpips = lpips_val > lpips_threshold

    success = cond_l2 or cond_id or cond_lpips

    return {"success": int(success)} # 1 if successful disruption, 0 otherwise

# anti_dreambooth fdr, ism
# avg_embedding은 같은 ID의 여러 이미지로부터 미리 계산된 평균 임베딩 벡터
def compute_face_embedding(img_path):
    from deepface import DeepFace  # lazy import
    try:
        resps = DeepFace.represent(img_path = os.path.join(img_path), 
                                   model_name="ArcFace", 
                                   enforce_detection=True, 
                                   detector_backend="retinaface", 
                                   align=True)
        if resps == 1:
            # detect only 1 face
            return np.array(resps[0]["embedding"])
        else:
            # detect more than 1 faces, choose the biggest one
            resps = list(resps)
            resps.sort(key=lambda resp: resp["facial_area"]["h"]*resp["facial_area"]["w"], reverse=True)
            return np.array(resps[0]["embedding"])
    except Exception:
        # no face found
        return None

def matching_score_id(image_path, avg_embedding):
    image_emb = compute_face_embedding(image_path)
    id_emb = avg_embedding
    if image_emb is None:
        return None
    image_emb, id_emb = torch.Tensor(image_emb), torch.Tensor(id_emb)
    ism = F.cosine_similarity(image_emb, id_emb, dim=0)
    return ism

def matching_score_genimage_id(images_path, list_id_path):
    image_list = os.listdir(images_path)
    fail_detection_count = 0
    ave_ism = 0
    avg_embedding = compute_idx_embedding(list_id_path)

    for image_name in image_list:
        image_path = os.path.join(images_path, image_name)
        ism = matching_score_id(image_path, avg_embedding)
        if ism is None:
            fail_detection_count += 1
        else:
            ave_ism += ism
    if fail_detection_count != len(image_list):
        return ave_ism/(len(image_list)-fail_detection_count), fail_detection_count/len(image_list)
    return None, 1

# ism, fdr = matching_score_genimage_id(args.data_dir, args.emb_dirs)

def ser_fiq_single(img_path, gpu_id=0, T=100):
    from FaceImageQuality.face_image_quality import SER_FIQ  # lazy import
    ser_fiq = SER_FIQ(gpu=gpu_id)
    img = cv2.imread(img_path)
    aligned_img = ser_fiq.apply_mtcnn(img)
    if aligned_img is None:
        return None
    return ser_fiq.get_score(aligned_img, T=T)

def l2_mask(img1, img2):
    diff = (img1 - img2).abs()
    mask = (diff.sum(dim=1, keepdim=True) > 0.05).float()  # 1 or 0
    if mask.sum() == 0:
        return 0.0
    masked_l2 = ((img1 * mask - img2 * mask) ** 2).sum() / (mask.sum() * 3.0)
    return float(masked_l2.item())

# Dual Defense
# SR_mask 기준으로 disruption 성공이면 True
def sr_mask_single(x_df: torch.Tensor,
                   x_adv_df: torch.Tensor,
                   thresh_mask: float = 0.5,
                   thresh_l2: float = 0.05) -> bool:
    diff = (x_df - x_adv_df).abs()
    mask = diff[0] + diff[1] + diff[2]
    mask = (mask > thresh_mask).float()   # 1 or 0
    if mask.sum() == 0:
        return False
    masked_l2 = ((x_df * mask - x_adv_df * mask) ** 2).sum() / (mask.sum() * 3.0)
    return float(masked_l2.item()) > thresh_l2

def vgg_loss(img1, img2):
    img1 = denorm(img1.float())
    img2 = denorm(img2.float())
    vgg_loss_fn = VGGLoss().to(DEVICE)
    loss = vgg_loss_fn(img1, img2)
    return loss

def scol_pds(img1, img2, img3, img4):
    # Convert to [0, 1] range
    img1_01 = (img1.clamp(-1, 1) * 0.5 + 0.5).squeeze(0).cpu().numpy()
    img2_01 = (img2.clamp(-1, 1) * 0.5 + 0.5).squeeze(0).cpu().numpy()
    img1_01 = np.transpose(img1_01, (1, 2, 0))
    img2_01 = np.transpose(img2_01, (1, 2, 0))
    img1_01 = np.clip(img1_01, 0.0, 1.0)
    img2_01 = np.clip(img2_01, 0.0, 1.0)
    
    ssim_src = structural_similarity(img1_01, img2_01, data_range=1.0, channel_axis=-1)
    
    img3_01 = (img3.clamp(-1, 1) * 0.5 + 0.5).squeeze(0).cpu().numpy()
    img4_01 = (img4.clamp(-1, 1) * 0.5 + 0.5).squeeze(0).cpu().numpy()
    img3_01 = np.transpose(img3_01, (1, 2, 0))
    img4_01 = np.transpose(img4_01, (1, 2, 0))
    img3_01 = np.clip(img3_01, 0.0, 1.0)
    img4_01 = np.clip(img4_01, 0.0, 1.0)
    
    ssim_adv = structural_similarity(img3_01, img4_01, data_range=1.0, channel_axis=-1)
    return ssim_src - ssim_adv