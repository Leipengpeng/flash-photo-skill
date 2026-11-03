"""Frozen F04/F06 math, exported without paths to private experiments.

Only analytic local fitting. See references/native-fusion.md for provenance.
"""
from pathlib import Path
import hashlib
import io
import json
import math
import sys
import zipfile
import cv2
import numpy as np
from PIL import Image, ImageCms, ImageOps
from scipy.sparse import csr_matrix
from scipy.spatial import cKDTree

EPS=.001
LUMA=np.array([.2126,.7152,.0722],np.float32)
SRGB_PROFILE=ImageCms.ImageCmsProfile(ImageCms.createProfile('sRGB'))
SRGB_ICC=SRGB_PROFILE.tobytes()

def file_hash(path):
    h=hashlib.sha256()
    with Path(path).open('rb') as f:
        for b in iter(lambda:f.read(1024*1024),b''):h.update(b)
    return h.hexdigest()

def write_json(path,value):
    Path(path).parent.mkdir(parents=True,exist_ok=True)
    Path(path).write_text(json.dumps(value,ensure_ascii=False,indent=2)+'\n',encoding='utf-8')

old=sys.modules[__name__]

POLICY={'version': '8.0-detail-separation-six-experiments', 'anchor_stride': 2, 'native_neighbors': 9, 'space_scale_work_px': 4.0, 'color_scale_encoded_RGB': 0.075, 'epsilon_linear': 0.001, 'max_abs_EV': 3.0, 'energy_sigma_work_px': 8.0, 'multi_energy_sigma_work_px': [4.0, 8.0, 16.0], 'guide_sigma_work_px': 3.0, 'recursive_sigma_anchor_px': 4.0, 'recursive_range_EV': 0.25, 'recursive_iterations': 3, 'structure_derivative_sigma_work_px': 1.0, 'structure_statistics_sigma_work_px': 3.0, 'structure_confidence_NCC_low': 0.25, 'structure_confidence_NCC_high': 0.65, 'anti_texture_band_sigma_anchor_px': 1.0, 'anti_texture_statistics_sigma_anchor_px': 4.0, 'negative_projection_bound': -0.8, 'anti_texture_min_correlation': 0.15, 'trusted_fine_gain_bound_EV': 0.25, 'per_image_manual_parameters': False, 'output': 'native linear source times exp2(estimated gain); no RGB blur/warp/resize/sharpen', 'guide_maps_used': [], 'image_generation_upload_download_calls': 0}

def load_source(path):
    with Image.open(path) as im:
        im.load()
        stored = im.size
        orientation = int(im.getexif().get(274, 1))
        profile = im.info.get('icc_profile')
        image = ImageOps.exif_transpose(im)
        if 'A' in image.getbands() and image.getchannel('A').getextrema() != (255, 255):
            raise ValueError('Photographic input must be opaque')
        if profile:
            image = ImageCms.profileToProfile(image, ImageCms.ImageCmsProfile(io.BytesIO(profile)),
                                             SRGB_PROFILE, outputMode='RGB')
        else:
            image = image.convert('RGB')
        codes = np.asarray(image, np.uint8).copy()
    return codes, {'path': str(Path(path).resolve()), 'sha256': file_hash(path),
                   'stored_size': list(stored), 'native_size': [codes.shape[1], codes.shape[0]],
                   'exif_orientation': orientation,
                   'color': 'ICC converted to sRGB' if profile else 'unprofiled RGB assumed sRGB'}

def srgb_to_linear(codes):
    x = np.asarray(codes, np.float32) / np.float32(255)
    return np.where(x <= .04045, x / 12.92, ((x + .055) / 1.055) ** 2.4).astype(np.float32)

def linear_to_float(x):
    x = np.clip(x, 0, 1)
    return np.where(x <= .0031308, x * 12.92, 1.055 * np.power(x, 1 / 2.4) - .055).astype(np.float32)

def linear_to_srgb(x):
    return np.clip(np.rint(linear_to_float(x) * 255), 0, 255).astype(np.uint8)

def area_weights(source_length, target_length):
    """Pixel footprint overlap weights, not interpolated point samples."""
    if not 0 < target_length <= source_length:
        raise ValueError('Exact area integration is only used for downsampling')
    scale = source_length / target_length
    rows, cols, values = [], [], []
    for j in range(target_length):
        lo, hi = j * scale, (j + 1) * scale
        for i in range(int(math.floor(lo)), min(source_length, int(math.ceil(hi)))):
            overlap = max(0., min(hi, i + 1.) - max(lo, i))
            if overlap:
                rows.append(j); cols.append(i); values.append(overlap / scale)
    return csr_matrix((np.asarray(values, np.float32), (rows, cols)),
                      shape=(target_length, source_length), dtype=np.float32)

def median(x,axis):
    mid=(x.shape[axis]-1)//2
    return np.take(np.partition(x,mid,axis=axis),mid,axis=axis)

def make_anchors(low,ref,base):
    rh,rw=low.shape[:2]
    yy,xx=np.mgrid[0:rh:2,0:rw:2];rows=yy.ravel();cols=xx.ravel()
    s=low[rows,cols];t=ref[rows,cols]
    features=np.column_stack((cols/4.,rows/4.,base.linear_to_float(s)/.075)).astype(np.float32)
    q=np.clip(np.log2((t+EPS)/(s+EPS)),-3,3).astype(np.float32)
    y0=s@LUMA;y1=t@LUMA
    return {'features':features,'rows':rows,'cols':cols,'source':s,'target':t,'q':q,
            'luma_q':np.clip(np.log2((y1+EPS)/(y0+EPS)),-3,3).astype(np.float32),
            'source_log_luma':np.log2(y0+EPS).astype(np.float32)}

def model_neighbors(a):
    tree=cKDTree(a['features'],compact_nodes=True,balanced_tree=True)
    n=len(a['source']);k=min(25,n)
    ids=np.empty((n,k),np.uint32);dist=np.empty((n,k),np.float32)
    for start in range(0,n,8192):
        stop=min(n,start+8192);d,i=tree.query(a['features'][start:stop],k=k,workers=2)
        if k==1:d=d[:,None];i=i[:,None]
        ids[start:stop]=i;dist[start:stop]=d
    return ids,dist

def fit_nuisance(a,ids,dist,low):
    log=np.log2(low+EPS).astype(np.float32)
    gx=cv2.Sobel(log,cv2.CV_32F,1,0,ksize=3,scale=.125)
    gy=cv2.Sobel(log,cv2.CV_32F,0,1,ksize=3,scale=.125)
    ax=gx[a['rows'],a['cols']];ay=gy[a['rows'],a['cols']]
    n=len(ids);out=np.empty((n,3),np.float32);motion=np.empty((n,2),np.float32)
    for start in range(0,n,2048):
        stop=min(n,start+2048);ind=ids[start:stop];b,k=ind.shape
        x=np.zeros((b,k,3,5),np.float32)
        for c in range(3):x[:,:,c,c]=1
        x[:,:,:,3]=ax[ind];x[:,:,:,4]=ay[ind]
        x=x.reshape(b,k*3,5);target=a['q'][ind].reshape(b,k*3)
        initial=np.maximum(np.exp(-.5*np.minimum(dist[start:stop]**2,16)),.001).astype(np.float32)
        initial=np.repeat(initial,3,axis=1);weights=initial.copy()
        prior=median(a['q'][ind[:,:min(9,k)]],1)
        for _ in range(3):
            lhs=np.einsum('bki,bk,bkj->bij',x,weights,x)
            lhs[:,np.arange(5),np.arange(5)]+=np.array([.002,.002,.002,.15,.15],np.float32)
            rhs=np.einsum('bki,bk,bk->bi',x,weights,target)
            rhs[:,:3]+=.002*prior
            c=np.linalg.solve(lhs,rhs[...,None])[...,0].astype(np.float32)
            norm=np.maximum(np.linalg.norm(c[:,3:],axis=1)/3.,1.)
            c[:,3:]/=norm[:,None]
            error=np.abs(target-np.einsum('bki,bi->bk',x,c))
            scale=np.maximum(median(error,1)*1.4826,.04)
            weights=initial*np.minimum(1.,1.5*scale[:,None]/np.maximum(error,1e-8))
        corrected=a['q'][ind]-ax[ind]*c[:,None,3,None]-ay[ind]*c[:,None,4,None]
        out[start:stop]=np.clip(median(corrected,1),-3,3)
        motion[start:stop]=c[:,3:]
    return out,motion

def save_npz(path,arrays):
    """Lossless standard NPZ, streamed to avoid a second native-size copy."""
    path=Path(path);building=path.with_name(path.stem+'.building.npz')
    with zipfile.ZipFile(building,'w',compression=zipfile.ZIP_DEFLATED,compresslevel=2,allowZip64=True) as z:
        for key,value in arrays.items():
            value=np.asarray(value)
            with z.open(key+'.npy','w',force_zip64=True) as f:
                np.lib.format.write_array_header_2_0(f,{'descr':np.lib.format.dtype_to_descr(value.dtype),
                    'fortran_order':False,'shape':value.shape})
                if value.ndim==0:f.write(value.tobytes())
                else:
                    for start in range(0,len(value),8):f.write(np.ascontiguousarray(value[start:start+8]).tobytes())
    building.replace(path)

def export_result(folder,final,source,base):
    h,w=source.shape[:2];path=folder/'delta_linear.npz';building=folder/'delta_linear.building.npz'
    limits=[float('inf'),-float('inf')];max_error=0
    with zipfile.ZipFile(building,'w',compression=zipfile.ZIP_DEFLATED,compresslevel=2,allowZip64=True) as z:
        with z.open('delta_linear.npy','w',force_zip64=True) as f:
            np.lib.format.write_array_header_2_0(f,{'descr':'<f4','fortran_order':False,'shape':(h,w,3)})
            for y in range(0,h,8):
                src=base.srgb_to_linear(source[y:y+8]);delta=base.srgb_to_linear(final[y:y+8])-src
                if not np.all(np.isfinite(delta)):raise ValueError('Non-finite Delta')
                limits[0]=min(limits[0],float(delta.min()));limits[1]=max(limits[1],float(delta.max()))
                error=np.max(np.abs(base.linear_to_srgb(src+delta).astype(np.int16)-final[y:y+8].astype(np.int16)))
                max_error=max(max_error,int(error));f.write(delta.astype('<f4',copy=False).tobytes())
    if max_error:raise ValueError('Delta reconstruction failed')
    building.replace(path)
    stride=max(1,math.ceil(max(h,w)/900))
    d=base.srgb_to_linear(final[::stride,::stride])-base.srgb_to_linear(source[::stride,::stride])
    Image.fromarray(np.clip(np.rint(128+127*d),0,255).astype(np.uint8)).save(folder/'delta-preview.png')
    report={'file':path.name,'key':'delta_linear','shape':[h,w,3],'dtype':'float32',
            'space':'linear sRGB','definition':'decode(delivered final)-decode(original sRGB)',
            'range':limits,'max_reconstruction_code_error':max_error,'lossless_compression':True,
            'preview':'128 neutral; round(128+127*delta); integer stride; not an alpha layer'}
    write_json(folder/'delta.json',report)
    return report

def blur(a,sigma):
    return cv2.GaussianBlur(a,(0,0),sigma,borderType=cv2.BORDER_REFLECT_101)

def recursive_regularize(field,guide):
    """Four-direction convex recursions on the fixed anchor grid, three sweeps.

    Custom coarse-guide variant inspired by recursive edge-aware filtering.
    It smooths illumination coefficients, never the delivered photograph.
    """
    value=np.asarray(field,np.float32).copy();guide=np.asarray(guide,np.float32)
    dh=1+POLICY['recursive_sigma_anchor_px']/POLICY['recursive_range_EV']*np.abs(np.diff(guide,axis=1))
    dv=1+POLICY['recursive_sigma_anchor_px']/POLICY['recursive_range_EV']*np.abs(np.diff(guide,axis=0))
    n=POLICY['recursive_iterations']
    for i in range(n):
        sigma=POLICY['recursive_sigma_anchor_px']*math.sqrt(3)*2**(n-i-1)/math.sqrt(4**n-1)
        ah=np.exp(-math.sqrt(2)/sigma*dh).astype(np.float32)
        av=np.exp(-math.sqrt(2)/sigma*dv).astype(np.float32)
        for x in range(1,value.shape[1]):value[:,x]+=ah[:,x-1,None]*(value[:,x-1]-value[:,x])
        for x in range(value.shape[1]-2,-1,-1):value[:,x]+=ah[:,x,None]*(value[:,x+1]-value[:,x])
        for y in range(1,value.shape[0]):value[y]+=av[y-1,:,None]*(value[y-1]-value[y])
        for y in range(value.shape[0]-2,-1,-1):value[y]+=av[y,:,None]*(value[y+1]-value[y])
    return value

def energy_ratio(low,ref,sigma):
    return np.clip(np.log2((blur(ref,sigma)+.001)/(blur(low,sigma)+.001)),-3,3).astype(np.float32)

def structure_confidence(low,ref):
    sy=np.log2(low@old.LUMA+.001).astype(np.float32)
    ry=np.log2(ref@old.LUMA+.001).astype(np.float32)
    s=blur(sy,1.);r=blur(ry,1.)
    sx=cv2.Sobel(s,cv2.CV_32F,1,0,ksize=3,scale=.125);sy=cv2.Sobel(s,cv2.CV_32F,0,1,ksize=3,scale=.125)
    rx=cv2.Sobel(r,cv2.CV_32F,1,0,ksize=3,scale=.125);ry=cv2.Sobel(r,cv2.CV_32F,0,1,ksize=3,scale=.125)
    ss=blur(sx*sx+sy*sy,3.);rr=blur(rx*rx+ry*ry,3.)
    ncc=blur(sx*rx+sy*ry,3.)/np.maximum(np.sqrt(ss*rr),1e-6)
    weight=np.clip((ncc-.25)/.40,0,1)
    # No evidence of matching structure in nearly flat patches.
    weight*=np.clip(np.minimum(ss,rr)/.002,0,1)
    return weight.astype(np.float32),ncc.astype(np.float32)

def make_models(low,ref,light):
    rh,rw=low.shape[:2];shape=(math.ceil(rh/2),math.ceil(rw/2),3)
    original=light.reshape(shape)
    guide=blur(np.log2(low@old.LUMA+.001).astype(np.float32),3.)[::2,::2]
    regular=recursive_regularize(original,guide)
    energy=energy_ratio(low,ref,8.)[::2,::2]
    confidence,ncc=structure_confidence(low,ref);confidence=confidence[::2,::2]
    source_log=np.log2(low[::2,::2]+.001)
    texture=source_log-blur(source_log,1.);gain_hi=original-blur(original,1.)
    var=blur(np.mean(texture*texture,axis=2),4.)
    gain_var=blur(np.mean(gain_hi*gain_hi,axis=2),4.)
    cov=blur(np.mean(texture*gain_hi,axis=2),4.)
    rho=cov/np.maximum(np.sqrt(var*gain_var),1e-6)
    beta=np.clip(cov/np.maximum(var,1e-5),-.8,0.)
    anti_weight=np.clip((-rho-.15)/.50,0,1)
    corrected=np.clip(original-(anti_weight*beta)[:,:,None]*texture,-3,3)
    multi=old.median(np.stack([energy_ratio(low,ref,s)[::2,::2] for s in (4.,8.,16.)]),0)
    multi=recursive_regularize(multi,guide)
    fine=np.clip(original-regular,-.25,.25)
    combined=np.clip(multi+confidence[:,:,None]*fine,-3,3)
    fields={'R08':original.reshape(-1,3),'F01':original.reshape(-1,3),
        'F02':energy.reshape(-1,3),'F03':corrected.reshape(-1,3),'F05':regular.reshape(-1,3),
        'F06':combined.reshape(-1,3),'confidence':confidence.ravel()}
    facts={'mean_structure_confidence':float(confidence.mean()),
           'fraction_confidence_above_half':float(np.mean(confidence>.5)),
           'negative_texture_correction_mean_weight':float(anti_weight.mean()),
           'fraction_negative_texture_correction_active':float(np.mean(anti_weight>0)),
           'source_reference_NCC':ncc,'guide':guide,'anti_correction_weight':anti_weight}
    return fields,facts

def query_pixels(codes,a,rh,rw,y,stop,color_tree,space_tree):
    h,w=codes.shape[:2]
    xx=np.tile((np.arange(w,dtype=np.float32)+.5)*rw/w-.5,stop-y)
    yy=np.repeat((np.arange(y,stop,dtype=np.float32)+.5)*rh/h-.5,w)
    q=np.column_stack((xx/4.,yy/4.,codes[y:stop].reshape(-1,3).astype(np.float32)/(255*.075)))
    _,color=color_tree.query(q,k=9,workers=2)
    _,space=space_tree.query(q[:,:2],k=9,workers=2)
    return color,space

def evaluate_gain(method,fields,color,space):
    if method=='F01':ev=old.median(fields['F01'][space],1)
    elif method=='F02':ev=old.median(fields['F02'][space],1)
    elif method=='F04':
        weight=old.median(fields['confidence'][space],1)[:,None]
        ev=weight*old.median(fields['R08'][color],1)+(1-weight)*old.median(fields['F02'][space],1)
    else:ev=old.median(fields[method][color],1)
    return np.clip(ev,-3,3)

