#!/usr/bin/env python3
"""Transfer filtered differences from a complete relit reference onto a photo.

Everything in this script is local. It neither calls an image model nor scores
aesthetic quality. The complete reference is color-managed, optionally aligned
TO a reduced original, then subtracted in linear sRGB. The native original is
never warped or resized for delivery. Signed changes are retained. Inferred
depth and semantic labels are experimental guides, not measured geometry.
"""
from __future__ import annotations

import argparse
from dataclasses import dataclass
import json
import math
from pathlib import Path
from typing import Any

import cv2
import numpy as np
from PIL import Image
from scipy.fft import dctn, idctn
from scipy.ndimage import gaussian_filter

import relight_layers as common

VERSION = "3.0-delta-lab"
# Keep output declarations next to the writers and guards. Wrappers check all
# destinations before decoding inputs or creating files; auxiliary previews
# and reports must protect originals just as final.png does.
COMPOSE_OUTPUTS = {"final": "final.png", "overlay": "effect_overlay.png", "preview": "preview.jpg"}
METHOD_OUTPUTS = {"signed": "delta-signed.png", "positive": "delta-positive.png",
                  "negative": "delta-negative.png", "field": "field.npz", "report": "report.json"}
WORKING_DIRECTORY = "_working"
WORKING_OUTPUTS = {"source": "source-sRGB.png", "unaligned": "relit-unaligned-sRGB.png",
                   "registered": "relit-registered-sRGB.png", "confidence": "structure-confidence.png",
                   "depth": "depth-working.png", "coverage": "soft-coverage.png", "pair": "pair.json"}
RUN_OUTPUTS = {"source": "original-sRGB.png", "manifest": "manifest.json"}
METHODS = {
    "D00": {"name": "未配准原始差值", "family": "unaligned-linear", "parameters": {}},
    "D01": {"name": "配准后原始差值", "family": "linear", "parameters": {}},
    "D02": {"name": "小尺度高斯", "family": "gaussian", "parameters": {"sigma": 2.0}},
    "D03": {"name": "大尺度高斯", "family": "gaussian", "parameters": {"sigma": 8.0}},
    "D04": {"name": "差值自身双边", "family": "delta-bilateral", "parameters": {"diameter": 9, "sigma_color": .08, "sigma_space": 5.0}},
    "D05": {"name": "原图亮度引导", "family": "source-guided", "parameters": {"radius": 12, "epsilon": .002}},
    "D06": {"name": "深度引导", "family": "depth-guided", "parameters": {"radius": 20, "epsilon": .003}},
    "D07": {"name": "RGB与深度联合双边", "family": "rgb-depth-joint", "parameters": {"radius": 4, "stride": 2, "sigma_rgb": .16, "sigma_depth": .10, "sigma_space": 3.0}},
    "D08": {"name": "材质分区归一卷积", "family": "semantic-normalized", "parameters": {"sigma": 6.0, "palette_classes": 8}},
    "D09": {"name": "低中频与结构置信", "family": "multiscale-confidence", "parameters": {"fine_sigma": 2.0, "coarse_sigma": 12.0, "mid_weight": .65}},
    "D10": {"name": "对数亮度增益加色差", "family": "log-gain-chroma", "parameters": {"epsilon": .005, "gain_sigma": 5.0, "chroma_sigma": 8.0, "max_abs_ev": 6.0}},
    "D11": {"name": "屏蔽泊松平滑", "family": "screened-poisson", "parameters": {"lambda": 20.0, "boundary": "Neumann DCT"}},
    "D12": {"name": "深度加软覆盖融合", "family": "depth-soft-coverage", "parameters": {"depth_radius": 16, "depth_epsilon": .003, "coverage_sigma": 12.0, "depth_floor": .25}},
    "D13": {"name": "受限光流加引导传达", "family": "bounded-flow-guided", "opt_in": True, "parameters": {"flow_max_edge": 720, "max_displacement_working_px": 12.0, "fb_threshold_working_px": 1.5, "guided_radius": 12, "guided_epsilon": .002}},
}
PALETTE = np.array([[128,128,128],[242,170,133],[0,200,220],[208,64,200],
                    [242,207,48],[49,94,234],[245,139,32],[48,48,48]], np.float32)


def reject_input_collisions(inputs, outputs) -> None:
    """Reject an output that resolves to an original, reference, map or cache."""
    protected={Path(path).resolve() for path in inputs if path is not None}
    for destination in outputs:
        if Path(destination).resolve() in protected:
            raise ValueError(f"Output must not overwrite an input file: {destination}")


def run_output_paths(out: Path, methods: list[str]) -> list[Path]:
    """Every static run destination, including optional cache/map outputs."""
    return ([out/name for name in RUN_OUTPUTS.values()]
            +[out/WORKING_DIRECTORY/name for name in WORKING_OUTPUTS.values()]
            +[out/method/name for method in methods for name in (*COMPOSE_OUTPUTS.values(),*METHOD_OUTPUTS.values())])


def resize_rgb_float(value: np.ndarray, size: tuple[int, int]) -> np.ndarray:
    return np.stack([common.resize_float(value[:,:,i], size) for i in range(value.shape[2])], axis=2)


def gaussian(value: np.ndarray, sigma: float) -> np.ndarray:
    return gaussian_filter(value, (sigma,sigma,0) if value.ndim == 3 else sigma, mode="reflect").astype(np.float32)


def guided(value: np.ndarray, guide: np.ndarray, radius: int, epsilon: float) -> np.ndarray:
    a,b = common.guided_coefficients(guide.astype(np.float32), value.astype(np.float32), radius, epsilon)
    return (a * guide[:,:,None] + b).astype(np.float32)


def semantic_labels(rgb: np.ndarray) -> np.ndarray:
    labels=np.empty(rgb.shape[:2],np.uint8)
    for start in range(0,rgb.shape[0],96):
        rows=rgb[start:start+96].astype(np.float32)
        labels[start:start+96]=np.argmin(np.sum((rows[:,:,None,:]-PALETTE[None,None,:,:])**2,axis=3),axis=2)
    return labels


def normalized_semantic(value: np.ndarray, labels: np.ndarray, sigma: float) -> np.ndarray:
    result=np.zeros_like(value)
    for class_id in np.unique(labels):
        mask=(labels==class_id).astype(np.float32)
        denominator=gaussian(mask,sigma)
        numerator=gaussian(value*mask[:,:,None],sigma)
        normalized=numerator/np.maximum(denominator[:,:,None],1e-6)
        result[labels==class_id]=normalized[labels==class_id]
    return result


def joint_bilateral(value: np.ndarray, rgb: np.ndarray, depth: np.ndarray, *, radius: int=4,
                    stride: int=2, sigma_rgb: float=.16, sigma_depth: float=.10,
                    sigma_space: float=3) -> np.ndarray:
    """Sparse 25-tap normalized joint filter, rather than a model depth solver."""
    p=radius
    vv=np.pad(value,((p,p),(p,p),(0,0)),mode="reflect")
    rr=np.pad(rgb,((p,p),(p,p),(0,0)),mode="reflect")
    zz=np.pad(depth,((p,p),(p,p)),mode="reflect")
    h,w=value.shape[:2]; numerator=np.zeros_like(value); denominator=np.zeros((h,w),np.float32)
    for dy in range(-radius,radius+1,stride):
        for dx in range(-radius,radius+1,stride):
            ys=slice(p+dy,p+dy+h);xs=slice(p+dx,p+dx+w)
            color_distance=np.mean((rr[ys,xs]-rgb)**2,axis=2)
            depth_distance=(zz[ys,xs]-depth)**2
            weight=np.exp(-color_distance/(2*sigma_rgb**2)-depth_distance/(2*sigma_depth**2)
                          -(dx*dx+dy*dy)/(2*sigma_space**2)).astype(np.float32)
            numerator += weight[:,:,None]*vv[ys,xs]
            denominator += weight
    return numerator/np.maximum(denominator[:,:,None],1e-6)


def screened_poisson(value: np.ndarray, coefficient: float) -> np.ndarray:
    """Solve (I + lambda*L)u=value with Neumann boundary conditions.

    L is the positive discrete Laplacian. This is a screened smoothing problem,
    not seamless cloning, reconstruction of true lighting, or geometric flow.
    """
    h,w=value.shape[:2]
    eigen=4*np.sin(np.pi*np.arange(h)/(2*h))[:,None]**2+4*np.sin(np.pi*np.arange(w)/(2*w))[None,:]**2
    transformed=dctn(value,axes=(0,1),norm="ortho")
    transformed /= (1+coefficient*eigen[:,:,None])
    return idctn(transformed,axes=(0,1),norm="ortho").astype(np.float32)


def structural_confidence(source: np.ndarray, relit: np.ndarray) -> np.ndarray:
    # Illumination and geometry are inseparable in a single image. This is only
    # a heuristic high-frequency disagreement weight, never a fidelity score.
    weights=np.array([.2126,.7152,.0722],np.float32)
    so=np.einsum('ijk,k->ij',source,weights)
    rr=np.einsum('ijk,k->ij',relit,weights)
    disagreement=np.abs((rr-gaussian(rr,2))-(so-gaussian(so,2)))
    return np.clip(np.exp(-gaussian(disagreement,1)/.045),.03,1).astype(np.float32)


def align_reference(source_codes: np.ndarray, relit_codes: np.ndarray, mode: str,
                    coverage: np.ndarray|None=None) -> tuple[np.ndarray,dict[str,Any]]:
    """Constrained global ECC; ONLY the reference is resampled."""
    h,w=source_codes.shape[:2]
    report={"requested":mode,"applied":"identity","matrix_reference_sampling": [[1.,0.,0.],[0.,1.,0.]],
            "source_warped":False,"flow":"none","boundary":"transparent validity; invalid reference pixels replaced by original"}
    if mode=="identity" or np.array_equal(source_codes,relit_codes):
        report["reason"]="requested identity" if mode=="identity" else "identical decoded working images"
        return relit_codes.copy(),report
    if mode!="ecc":raise ValueError("registration must be identity or ecc")
    scale=min(1.,640/max(h,w)); size=(max(16,round(w*scale)),max(16,round(h*scale)))
    def edges(codes):
        rgb=cv2.resize(codes.astype(np.float32)/255,size,interpolation=cv2.INTER_AREA)
        y=np.einsum('ijk,k->ij',rgb,np.array([.2126,.7152,.0722],np.float32))
        y=cv2.GaussianBlur(y,(0,0),1.2)
        dx=cv2.Sobel(y,cv2.CV_32F,1,0,ksize=3);dy=cv2.Sobel(y,cv2.CV_32F,0,1,ksize=3)
        return np.sqrt(dx*dx+dy*dy).astype(np.float32)
    template=edges(source_codes); target=edges(relit_codes); initial=np.eye(2,3,dtype=np.float32)
    mask=None
    if coverage is not None:
        bg=(cv2.resize(coverage,size,interpolation=cv2.INTER_LINEAR)<.1).astype(np.uint8)*255
        if np.mean(bg>0)>.15:mask=bg;report["ecc_mask"]="background outside optional coverage; alignment only"
    try:
        score,matrix=cv2.findTransformECC(template,target,initial,cv2.MOTION_EUCLIDEAN,
            (cv2.TERM_CRITERIA_EPS|cv2.TERM_CRITERIA_COUNT,100,1e-6),mask,3)
        matrix[:,2]/=scale
        angle=math.degrees(math.atan2(float(matrix[1,0]),float(matrix[0,0])))
        shift=float(np.linalg.norm(matrix[:,2]));limit=.025*max(h,w)
        report.update({"ecc_correlation":float(score),"proposed_shift_pixels":shift,"rotation_degrees":angle,
                       "maximum_allowed_shift_pixels":limit,"maximum_allowed_rotation_degrees":1.5})
        if not np.all(np.isfinite(matrix)) or abs(angle)>1.5 or shift>limit or score<.25:
            report["reason"]="ECC proposal outside constrained bounds; identity retained"
            return relit_codes.copy(),report
        aligned=cv2.warpAffine(relit_codes,matrix,(w,h),flags=cv2.INTER_LINEAR|cv2.WARP_INVERSE_MAP,
            borderMode=cv2.BORDER_CONSTANT,borderValue=(0,0,0))
        validity=cv2.warpAffine(np.ones((h,w),np.float32),matrix,(w,h),flags=cv2.INTER_LINEAR|cv2.WARP_INVERSE_MAP,
            borderMode=cv2.BORDER_CONSTANT,borderValue=0)
        valid=validity>.999
        aligned[~valid]=source_codes[~valid]
        report.update({"applied":"ecc-euclidean","matrix_reference_sampling":matrix.astype(float).tolist(),
                       "valid_fraction":float(valid.mean()),"reason":"bounded global proposal accepted"})
        return aligned,report
    except cv2.error as exc:
        report["reason"]="ECC did not converge; identity retained"
        report["error"]=str(exc).splitlines()[-1]
        return relit_codes.copy(),report


def bounded_flow_reference(source: np.ndarray,reference: np.ndarray,*,max_edge=720,
                           max_displacement=12.,fb_threshold=1.5) -> tuple[np.ndarray,dict]:
    """Experimental local reference warp, constrained by forward/backward flow.

    Illumination can move image features without moving surfaces. Bounds and
    consistency therefore limit displacement; they do not prove correctness.
    Rejected samples retain the globally aligned reference, not new pixels.
    """
    h,w=source.shape[:2];scale=min(1.,max_edge/max(h,w))
    size=(max(16,round(w*scale)),max(16,round(h*scale)))
    report={"source_warped":False,"applied":"identity","max_displacement_working_px":max_displacement,
            "fb_threshold_working_px":fb_threshold,"flow_size":list(size),"invalid_fallback":"global ECC reference"}
    if np.array_equal(source,reference):
        report['reason']='identical images';report['accepted_fraction']=1.;return reference.copy(),report
    def gray(rgb):
        codes=common.linear_to_srgb(rgb)
        small=cv2.resize(codes,size,interpolation=cv2.INTER_AREA)
        y=cv2.cvtColor(small,cv2.COLOR_RGB2GRAY)
        # Contrast normalization helps the registration task but is NEVER
        # applied to the actual photograph or transferred color values.
        return cv2.createCLAHE(clipLimit=1.5,tileGridSize=(8,8)).apply(y)
    try:
        a=gray(source);b=gray(reference)
        options=dict(pyr_scale=.5,levels=4,winsize=21,iterations=4,poly_n=7,poly_sigma=1.5,flags=0)
        forward=cv2.calcOpticalFlowFarneback(a,b,None,**options)
        backward=cv2.calcOpticalFlowFarneback(b,a,None,**options)
        yy,xx=np.mgrid[:size[1],:size[0]].astype(np.float32)
        sx=xx+forward[:,:,0];sy=yy+forward[:,:,1]
        back=cv2.remap(backward,sx,sy,cv2.INTER_LINEAR,borderMode=cv2.BORDER_CONSTANT,borderValue=0)
        error=np.linalg.norm(forward+back,axis=2)/scale
        magnitude=np.linalg.norm(forward,axis=2)/scale
        inside=(sx>=1)&(sy>=1)&(sx<size[0]-2)&(sy<size[1]-2)
        accepted=inside&(error<=fb_threshold)&(magnitude<=max_displacement)&np.isfinite(error)
        confidence=np.clip(1-error/fb_threshold,0,1)*accepted
        flow=cv2.resize(forward,(w,h),interpolation=cv2.INTER_LINEAR)/scale
        confidence=cv2.resize(confidence.astype(np.float32),(w,h),interpolation=cv2.INTER_LINEAR)
        flow*=confidence[:,:,None]
        # Interpolation of accepted bounded vectors cannot exceed the bound.
        yy,xx=np.mgrid[:h,:w].astype(np.float32)
        warped=cv2.remap(reference,xx+flow[:,:,0],yy+flow[:,:,1],cv2.INTER_LINEAR,borderMode=cv2.BORDER_REFLECT_101)
        zero=confidence==0;warped[zero]=reference[zero]
        report.update({"applied":"bounded-forward-backward-flow","accepted_fraction":float(accepted.mean()),
            "rejected_fraction":float(1-accepted.mean()),"actual_max_displacement_working_px":float(np.linalg.norm(flow,axis=2).max()),
            "median_fb_error_working_px":float(np.median(error)),"reason":"experimental confidence-weighted local reference sampling"})
        return warped,report
    except cv2.error as exc:
        report['reason']='optical flow failed; globally aligned reference retained';report['error']=str(exc).splitlines()[-1]
        return reference.copy(),report


@dataclass
class Pair:
    native: np.ndarray
    source_info: dict
    source: np.ndarray
    relit: np.ndarray
    unaligned: np.ndarray
    delta: np.ndarray
    guide: np.ndarray
    confidence: np.ndarray
    depth: np.ndarray|None
    labels: np.ndarray|None
    coverage: np.ndarray|None
    metadata: dict


def prepare_pair(source: str|Path,relit: str|Path,*,depth: str|Path|None=None,
                 materials: str|Path|None=None,coverage: str|Path|None=None,
                 registration: str="ecc",max_edge: int=1600) -> Pair:
    if max_edge<16:raise ValueError("max-edge must be >=16")
    native,source_info=common.load_source(source)
    reference,ref_info=common.load_source(relit)
    native_size=(native.shape[1],native.shape[0]);ref_size=(reference.shape[1],reference.shape[0])
    ratio_error=common.ratio_check(ref_size,native_size,"Complete relit reference")
    factor=min(1.,max_edge/max(ref_size));work_size=(max(1,round(ref_size[0]*factor)),max(1,round(ref_size[1]*factor)))
    low=np.asarray(Image.fromarray(native).resize(work_size,Image.Resampling.LANCZOS),np.uint8).copy()
    ref_codes=np.asarray(Image.fromarray(reference).resize(work_size,Image.Resampling.LANCZOS),np.uint8).copy()
    del reference
    maps={}
    def load_map(path,mode,label):
        codes=common.load_data_image(path,mode)
        size=(codes.shape[1],codes.shape[0]);error=common.ratio_check(size,native_size,label)
        maps[label]={"file_name":Path(path).name,"sha256":common.file_hash(Path(path)),"original_size":list(size),"relative_aspect_error":error}
        return np.asarray(Image.fromarray(codes).resize(work_size,Image.Resampling.BILINEAR),np.uint8).copy()
    z=load_map(depth,"L","depth")/np.float32(255) if depth else None
    labels=semantic_labels(load_map(materials,"RGB","materials")) if materials else None
    cov=load_map(coverage,"L","coverage")/np.float32(255) if coverage else None
    aligned,registration_info=align_reference(low,ref_codes,registration,cov)
    so=common.srgb_to_linear(low);rr=common.srgb_to_linear(aligned);raw=common.srgb_to_linear(ref_codes)
    guide=np.einsum('ijk,k->ij',low.astype(np.float32)/255,np.array([.2126,.7152,.0722],np.float32))
    metadata={"version":VERSION,"source":source_info,"relit_reference":ref_info,"relative_aspect_error":ratio_error,
              "working_size":list(work_size),"native_size":list(native_size),"registration":registration_info,"maps":maps,
              "delta_encoding":"signed linear-sRGB R_aligned - O_low; no negative clamp, no V2 whitening or additive hand-painted light",
              "network_calls_by_engine":0,"original_is_warped":False,"aesthetic_evaluation":"reserved for user"}
    return Pair(native,source_info,so,rr,raw,rr-so,guide,structural_confidence(so,rr),z,labels,cov,metadata)


def filter_method(pair: Pair,method: str) -> dict[str,np.ndarray]:
    if method not in METHODS:raise ValueError(f"Unknown method {method}")
    p=METHODS[method]["parameters"];d=pair.delta
    if method=="D00":return {"delta_linear":pair.unaligned-pair.source}
    if method=="D01":return {"delta_linear":d.copy()}
    if method in ("D02","D03"):return {"delta_linear":gaussian(d,p["sigma"])}
    if method=="D04":return {"delta_linear":cv2.bilateralFilter(d,p["diameter"],p["sigma_color"],p["sigma_space"],borderType=cv2.BORDER_REFLECT_101)}
    if method=="D05":return {"delta_linear":guided(d,pair.guide,p["radius"],p["epsilon"])}
    if method=="D06":
        if pair.depth is None:raise ValueError("D06 requires an inferred depth map")
        return {"delta_linear":guided(d,pair.depth,p["radius"],p["epsilon"])}
    if method=="D07":
        if pair.depth is None:raise ValueError("D07 requires an inferred depth map")
        return {"delta_linear":joint_bilateral(d,pair.source,pair.depth,**p)}
    if method=="D08":
        if pair.labels is None:raise ValueError("D08 requires a semantic material map")
        return {"delta_linear":normalized_semantic(d,pair.labels,p["sigma"])}
    if method=="D09":
        coarse=gaussian(d,p["coarse_sigma"]); fine=gaussian(d,p["fine_sigma"])
        return {"delta_linear":coarse+p["mid_weight"]*pair.confidence[:,:,None]*(fine-coarse)}
    if method=="D10":
        w=np.array([.2126,.7152,.0722],np.float32)
        yo=np.einsum('ijk,k->ij',pair.source,w);yr=np.einsum('ijk,k->ij',pair.relit,w)
        log=np.clip(np.log2((yr+p["epsilon"])/(yo+p["epsilon"])), -p["max_abs_ev"],p["max_abs_ev"])
        chroma=pair.relit-pair.source*np.exp2(log[:,:,None])
        return {"log_gain":gaussian(log,p["gain_sigma"]),"chroma_delta":gaussian(chroma,p["chroma_sigma"])}
    if method=="D11":return {"delta_linear":screened_poisson(d,p["lambda"])}
    if method=="D12":
        if pair.depth is None or pair.coverage is None:raise ValueError("D12 requires depth and coverage maps")
        filtered=guided(d,pair.depth,p["depth_radius"],p["depth_epsilon"])
        soft=gaussian(pair.coverage,p["coverage_sigma"])
        gate=soft*(p["depth_floor"]+(1-p["depth_floor"])*pair.depth)
        return {"delta_linear":filtered*gate[:,:,None]}
    if method=="D13":
        local,flow=bounded_flow_reference(pair.source,pair.relit,max_edge=p['flow_max_edge'],
            max_displacement=p['max_displacement_working_px'],fb_threshold=p['fb_threshold_working_px'])
        pair.metadata.setdefault('experimental_alignment_branches',{})['D13']=flow
        return {"delta_linear":guided(local-pair.source,pair.guide,p['guided_radius'],p['guided_epsilon'])}
    raise AssertionError(method)


def field_size(field: dict) -> tuple[int,int]:
    a=field["delta_linear"] if "delta_linear" in field else field["chroma_delta"]
    return a.shape[1],a.shape[0]


def effective_low_delta(field: dict, source_low: np.ndarray) -> np.ndarray:
    if "delta_linear" in field:return field["delta_linear"]
    return source_low*(np.exp2(field["log_gain"][:,:,None])-1)+field["chroma_delta"]


def compose_field(native: np.ndarray,field: dict,out: str|Path,*,strength: float=1) -> dict:
    if not math.isfinite(strength) or not 0<=strength<=3:raise ValueError("strength must be finite in 0..3")
    out=Path(out);out.mkdir(parents=True,exist_ok=True)
    h,w=native.shape[:2];size=(w,h)
    common.ratio_check(field_size(field),size,"Filtered delta")
    for value in field.values():
        if not np.all(np.isfinite(value)):raise ValueError("Field must contain finite values")
    if "delta_linear" in field:
        large=resize_rgb_float(field["delta_linear"],size);log=None
    else:
        large=resize_rgb_float(field["chroma_delta"],size);log=common.resize_float(field["log_gain"],size)
    final=np.empty_like(native);overlay=np.empty((h,w,4),np.uint8)
    clipped_low=clipped_high=negative_components=0;max_error=0;err_sum=0.;unchanged_count=0
    for start in range(0,h,128):
        stop=min(start+128,h);pixels=native[start:stop];base=common.srgb_to_linear(pixels)
        if log is None:change=strength*large[start:stop]
        else:change=base*np.expm1(strength*log[start:stop,:,None]*np.float32(math.log(2)))+strength*large[start:stop]
        target=base+change
        clipped_low+=int(np.count_nonzero(target<0));clipped_high+=int(np.count_nonzero(target>1))
        negative_components+=int(np.count_nonzero(change<0));unchanged=np.all(change==0,axis=2);unchanged_count+=int(unchanged.sum())
        codes=common.linear_to_srgb(target);codes[unchanged]=pixels[unchanged];final[start:stop]=codes
        block,report=common.make_overlay(pixels,codes);overlay[start:stop]=block
        max_error=max(max_error,report["reconstruction_max_code_error"])
        err_sum+=report["reconstruction_mean_code_error"]*(stop-start)
    Image.fromarray(final,'RGB').save(out/COMPOSE_OUTPUTS['final'],icc_profile=common.SRGB_ICC,compress_level=3)
    Image.fromarray(overlay,'RGBA').save(out/COMPOSE_OUTPUTS['overlay'],icc_profile=common.SRGB_ICC,compress_level=3)
    # Small browsing output only; final.png stays on the original display grid.
    small=Image.fromarray(final);small.thumbnail((900,1200),Image.Resampling.LANCZOS)
    small.save(out/COMPOSE_OUTPUTS['preview'],quality=91,icc_profile=common.SRGB_ICC)
    return {"native_size":[w,h],"strength":strength,"native_source_grid_unchanged":True,
        "clipped_below_zero_component_fraction":clipped_low/(h*w*3),"clipped_above_one_component_fraction":clipped_high/(h*w*3),
        "negative_delta_component_fraction":negative_components/(h*w*3),"zero_delta_pixel_fraction":unchanged_count/(h*w),
        "overlay":{"blend":"normal alpha in encoded sRGB on the exact color-managed base, 100% opacity",
                   "reconstruction_max_code_error":max_error,"reconstruction_mean_code_error":err_sum/h},
        "display_mapping":"clip linear RGB to0..1; sRGB encode; no added V2 shoulder, whitening, or beauty processing"}


def save_field_previews(delta: np.ndarray,out: Path) -> None:
    scale=.25
    signed=np.clip(np.rint(128+delta/scale*127),0,255).astype(np.uint8)
    positive=np.clip(np.rint(np.maximum(delta,0)/scale*255),0,255).astype(np.uint8)
    negative=np.clip(np.rint(np.maximum(-delta,0)/scale*255),0,255).astype(np.uint8)
    for name,codes in [(METHOD_OUTPUTS['signed'],signed),(METHOD_OUTPUTS['positive'],positive),(METHOD_OUTPUTS['negative'],negative)]:
        image=Image.fromarray(codes);image.thumbnail((900,1200),Image.Resampling.BILINEAR);image.save(out/name)


def save_working_field(field: dict,path: Path,source_hash: str,relit_hash: str,method: str) -> None:
    np.savez_compressed(path,**field,source_sha256=np.asarray(source_hash),relit_sha256=np.asarray(relit_hash),
                        method=np.asarray(method),version=np.asarray(VERSION))


def run_lab(source: str|Path,relit: str|Path,out: str|Path,*,depth=None,materials=None,coverage=None,
            registration="ecc",methods=None,strength=1.,max_edge=1600,save_working=False) -> dict:
    selected=list(methods) if methods else [m for m,v in METHODS.items() if not v.get('opt_in')]
    if not selected or len(set(selected))!=len(selected) or any(m not in METHODS for m in selected):
        raise ValueError("Select unique known method IDs D00..D13")
    if not math.isfinite(strength) or not 0<=strength<=3:raise ValueError("strength must be finite in 0..3")
    out=Path(out)
    # Reject output/input collisions before producing any file.
    reject_input_collisions([source,relit,depth,materials,coverage],run_output_paths(out,selected))
    pair=prepare_pair(source,relit,depth=depth,materials=materials,coverage=coverage,registration=registration,max_edge=max_edge)
    if any(m in selected for m in ['D06','D07','D12']) and pair.depth is None:raise ValueError("Selected methods require depth")
    if 'D08' in selected and pair.labels is None:raise ValueError("D08 requires materials")
    if 'D12' in selected and pair.coverage is None:raise ValueError("D12 requires coverage")
    work=out/WORKING_DIRECTORY;work.mkdir(parents=True,exist_ok=True)
    common.save_rgb(pair.native,out/RUN_OUTPUTS['source'])
    common.save_rgb(common.linear_to_srgb(pair.source),work/WORKING_OUTPUTS['source'])
    common.save_rgb(common.linear_to_srgb(pair.unaligned),work/WORKING_OUTPUTS['unaligned'])
    common.save_rgb(common.linear_to_srgb(pair.relit),work/WORKING_OUTPUTS['registered'])
    Image.fromarray(np.rint(pair.confidence*255).astype(np.uint8)).save(work/WORKING_OUTPUTS['confidence'])
    if pair.depth is not None:Image.fromarray(np.rint(pair.depth*255).astype(np.uint8)).save(work/WORKING_OUTPUTS['depth'])
    if pair.coverage is not None:Image.fromarray(np.rint(gaussian(pair.coverage,12)*255).astype(np.uint8)).save(work/WORKING_OUTPUTS['coverage'])
    (work/WORKING_OUTPUTS['pair']).write_text(json.dumps(pair.metadata,ensure_ascii=False,indent=2),encoding='utf-8')
    entries=[]
    for method in selected:
        folder=out/method;folder.mkdir(parents=True,exist_ok=True)
        field=filter_method(pair,method)
        report=compose_field(pair.native,field,folder,strength=strength)
        delta=effective_low_delta(field,pair.source)
        save_field_previews(delta,folder)
        if save_working:save_working_field(field,folder/METHOD_OUTPUTS['field'],pair.source_info['sha256'],pair.metadata['relit_reference']['sha256'],method)
        report.update({"version":VERSION,"method_id":method,**METHODS[method],"pair":pair.metadata,
                       "coverage_applied_to_delta":method=='D12',"field_storage":"working-resolution float32 NPZ" if save_working else "preview only",
                       "source_unchanged":common.file_hash(Path(source))==pair.source_info['sha256'],"aesthetic_evaluation":"user only; no ranking or beauty rejection",
                       "limitations":["Filtered residual may retain reference geometry or texture drift.","Inferred depth/material guides are not calibrated or pixel-exact.","Clipping, sRGB conversion and quantization can change local contrast.","Working field previews are visualization encodings; field.npz is the reusable signed data."]})
        (folder/METHOD_OUTPUTS['report']).write_text(json.dumps(report,ensure_ascii=False,indent=2),encoding='utf-8')
        entries.append({"id":method,"name":METHODS[method]['name'],"family":METHODS[method]['family'],"folder":method,
                        "final":f"{method}/{COMPOSE_OUTPUTS['final']}","overlay":f"{method}/{COMPOSE_OUTPUTS['overlay']}","preview":f"{method}/{COMPOSE_OUTPUTS['preview']}",
                        "field":f"{method}/{METHOD_OUTPUTS['field']}" if save_working else None,"report":f"{method}/{METHOD_OUTPUTS['report']}","parameters":METHODS[method]['parameters']})
        print(f'{method} complete',flush=True)
        del field,delta
    result={"version":VERSION,"pair":pair.metadata,"methods":entries,"count":len(entries),"network_calls_by_engine":0,"aesthetic_selection":"none"}
    (out/RUN_OUTPUTS['manifest']).write_text(json.dumps(result,ensure_ascii=False,indent=2),encoding='utf-8')
    return result


def apply_saved(source: str|Path,field_path: str|Path,out: str|Path,strength: float=1.) -> dict:
    out=Path(out)
    reject_input_collisions([source,field_path],[out/name for name in (*COMPOSE_OUTPUTS.values(),METHOD_OUTPUTS['report'])])
    native,info=common.load_source(source)
    with np.load(field_path,allow_pickle=False) as saved:
        if 'source_sha256' not in saved or str(saved['source_sha256'].item())!=info['sha256']:
            raise ValueError("Working field belongs to a different original source")
        field={k:np.asarray(saved[k],np.float32).copy() for k in ['delta_linear','log_gain','chroma_delta'] if k in saved}
        if not ('delta_linear' in field or ('log_gain' in field and 'chroma_delta' in field)):
            raise ValueError("Saved field lacks signed delta or log-gain/chroma data")
    report=compose_field(native,field,out,strength=strength)
    report.update({"source":info,"field_sha256":common.file_hash(Path(field_path)),"version":VERSION,"aesthetic_evaluation":"user"})
    (out/METHOD_OUTPUTS['report']).write_text(json.dumps(report,ensure_ascii=False,indent=2),encoding='utf-8')
    return report


def main() -> int:
    parser=argparse.ArgumentParser(description=__doc__)
    commands=parser.add_subparsers(dest='command',required=True)
    run=commands.add_parser('run',help='Prepare one reference once and export local filter candidates')
    run.add_argument('--source',type=Path,required=True);run.add_argument('--relit',type=Path,required=True)
    run.add_argument('--out',type=Path,required=True)
    for name in ['depth','materials','coverage']:run.add_argument('--'+name,type=Path)
    run.add_argument('--registration',choices=['ecc','identity'],default='ecc')
    run.add_argument('--methods',nargs='+',choices=list(METHODS));run.add_argument('--strength',type=float,default=1.)
    run.add_argument('--max-edge',type=int,default=1600);run.add_argument('--save-working',action='store_true')
    reuse=commands.add_parser('apply',help='Reapply a cached low-resolution signed field at native size')
    reuse.add_argument('--source',type=Path,required=True);reuse.add_argument('--field',type=Path,required=True)
    reuse.add_argument('--out',type=Path,required=True);reuse.add_argument('--strength',type=float,default=1.)
    commands.add_parser('list-methods')
    args=parser.parse_args()
    try:
        if args.command=='list-methods':print(json.dumps(METHODS,ensure_ascii=False,indent=2));return 0
        if args.command=='run':result=run_lab(args.source,args.relit,args.out,depth=args.depth,materials=args.materials,coverage=args.coverage,
            registration=args.registration,methods=args.methods,strength=args.strength,max_edge=args.max_edge,save_working=args.save_working)
        else:result=apply_saved(args.source,args.field,args.out,args.strength)
    except (ValueError,OSError,cv2.error) as exc:parser.exit(2,f'Error: {exc}\n')
    print(json.dumps({'complete':True,'out':str(args.out.resolve()),'count':result.get('count',1)},ensure_ascii=False))
    return 0


if __name__=='__main__':raise SystemExit(main())
