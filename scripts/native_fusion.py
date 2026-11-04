"""Map lighting from a relit reference back to the original photo."""
from pathlib import Path
import argparse
import json
import math
import time

import numpy as np
from PIL import Image
from scipy.spatial import cKDTree

import _native_fusion_core as core

VERSION='4.0-native-F04-F06'
DEFAULT_METHOD = 'F06'
METHODS = (DEFAULT_METHOD, 'F04')


def area_reduce_codes(codes,size):
    h,w=codes.shape[:2];rw,rh=map(int,size)
    if not (0<rw<=w and 0<rh<=h):raise ValueError('Area integration only reduces images')
    wy=core.area_weights(h,rh);wx=core.area_weights(w,rw)
    low=np.empty((rh,rw,3),np.float32)
    for y in range(0,rh,8):
        stop=min(rh,y+8);y0=math.floor(y*h/rh);y1=min(h,math.ceil(stop*h/rh))
        lin=core.srgb_to_linear(codes[y0:y1])
        v=wy[y:stop,y0:y1].dot(lin.reshape(y1-y0,w*3)).reshape(stop-y,w,3)
        low[y:stop]=wx.dot(v.transpose(1,0,2).reshape(w,(stop-y)*3)).reshape(rw,stop-y,3).transpose(1,0,2)
    return low


def preflight(inputs,out,methods):
    out=Path(out).resolve()
    for path in inputs:
        path=Path(path).resolve()
        if path==out or out in path.parents:raise ValueError('Output contains a protected input')
    for p in [out/'manifest.json',out/'pair.npz',out/'original-sRGB.png',
              *(out/m/'final.png' for m in methods)]:
        if p.exists():raise ValueError('Output already exists; choose a new directory: '+str(p))
    out.mkdir(parents=True,exist_ok=True)
    return out


def query(codes,working_size,y,stop,ct,st):
    h,w=codes.shape[:2];rw,rh=working_size
    xx=np.tile((np.arange(w,dtype=np.float32)+.5)*rw/w-.5,stop-y)
    yy=np.repeat((np.arange(y,stop,dtype=np.float32)+.5)*rh/h-.5,w)
    q=np.column_stack((xx/4.,yy/4.,codes[y:stop].reshape(-1,3).astype(np.float32)/(255*.075)))
    _,color=ct.query(q,k=9,workers=2);_,space=st.query(q[:,:2],k=9,workers=2)
    return color,space


def render(codes,features,working_size,fields,methods,strength=1.):
    ct=cKDTree(features,balanced_tree=True,compact_nodes=True)
    st=cKDTree(features[:,:2],balanced_tree=True,compact_nodes=True)
    h,w=codes.shape[:2];final={m:np.empty_like(codes) for m in methods};clips={m:0 for m in methods}
    for y in range(0,h,8):
        stop=min(h,y+8);color,space=query(codes,working_size,y,stop,ct,st)
        src=core.srgb_to_linear(codes[y:stop]).reshape(-1,3)
        for m in methods:
            ev=np.clip(core.evaluate_gain(m,fields,color,space)*strength,-3,3)
            target=src*np.exp2(ev)
            if not np.isfinite(target).all():raise ValueError('Non-finite composition')
            clips[m]+=int(np.count_nonzero(target>1))
            final[m][y:stop]=core.linear_to_srgb(target).reshape(stop-y,w,3)
        if y%512==0:print('native rows '+str(stop)+'/'+str(h),flush=True)
    return final,clips


def export(out,codes,final,clips,source_info,reference_info,working_size,features,fields,strength):
    results=[];h,w=codes.shape[:2]
    Image.fromarray(codes).save(out/'original-sRGB.png',icc_profile=core.SRGB_ICC,compress_level=3)
    for method,encoded in final.items():
        folder=out/method;folder.mkdir(exist_ok=True)
        Image.fromarray(encoded).save(folder/'final.png',icc_profile=core.SRGB_ICC,compress_level=3)
        delta=core.export_result(folder,encoded,codes,core)
        model={k:fields[k] for k in ('R08','F02','F06','confidence')}
        model.update(features=features,working_size=np.array(working_size,np.int32),method=np.array(method),
            source_sha256=np.array(source_info['sha256']),reference_sha256=np.array(reference_info['sha256']),
            version=np.array(VERSION),policy_json=np.array(json.dumps(core.POLICY,sort_keys=True)))
        core.save_npz(folder/'model.npz',model)
        report={'version':VERSION,'method':method,'native_size':[w,h],'working_size':list(working_size),
            'source':source_info,'reference':reference_info,'strength':strength,'policy':core.POLICY,
            'clip_above_one_component_fraction':clips[method]/(h*w*3),'delta':delta,
            'source_warped':False,'reference_warped':False,'ordinary_spatial_interpolation':False,
            'RGB_blur_or_post_sharpening':False,'actual_numeric_guide_maps_used':[],
            'algorithm_module_sha256':core.file_hash(Path(core.__file__))}
        core.write_json(folder/'report.json',report)
        results.append({'method':method,'final':str(folder/'final.png'),'delta':str(folder/'delta_linear.npz'),
                        'model':str(folder/'model.npz'),'sha256':core.file_hash(folder/'final.png')})
    return results


def run(args):
    start=time.perf_counter();methods=list(dict.fromkeys(args.methods))
    out=preflight((args.source,args.reference),args.out,methods)
    codes,source_info=core.load_source(args.source);rc,reference_info=core.load_source(args.reference)
    h,w=codes.shape[:2];rh,rw=rc.shape[:2]
    error=abs((w/h)/(rw/rh)-1)
    if error>args.max_aspect_error:raise ValueError('Full-FOV aspect mismatch exceeds explicit tolerance: '+str(error))
    if rw>w or rh>h:
        scale=min(w/rw,h/rh,1.);rw=min(w,max(1,round(rw*scale)));rh=min(h,max(1,round(rh*scale)))
    if min(rh,rw)<8:raise ValueError('Working observations are too small for the fixed model')
    ref=area_reduce_codes(rc,(rw,rh));low=area_reduce_codes(codes,(rw,rh))
    core.save_npz(out/'pair.npz',{'source_low_linear':low,'reference_linear':ref,'raw_signed_delta_linear':ref-low,
        'source_sha256':np.array(source_info['sha256']),'reference_sha256':np.array(reference_info['sha256'])})
    Image.fromarray(np.clip(np.rint(128+127*(ref-low)),0,255).astype(np.uint8)).save(out/'raw-delta-preview.png')
    a=core.make_anchors(low,ref,core);print('fit common R08 observations',flush=True)
    ids,dist=core.model_neighbors(a);light,motion=core.fit_nuisance(a,ids,dist,low)
    del ids,dist
    fields,facts=core.make_models(low,ref,light)
    # Only scalar facts are needed for the public report; arrays are in models.
    facts={k:v for k,v in facts.items() if not isinstance(v,np.ndarray)}
    final,clips=render(codes,a['features'],(rw,rh),fields,methods,args.strength)
    results=export(out,codes,final,clips,source_info,reference_info,(rw,rh),a['features'],fields,args.strength)
    if core.file_hash(args.source)!=source_info['sha256'] or core.file_hash(args.reference)!=reference_info['sha256']:
        raise ValueError('An input changed during processing')
    manifest={'version':VERSION,'methods':results,'source':source_info,'reference':reference_info,
        'native_size':[w,h],'working_size':[rw,rh],'relative_aspect_error':error,
        'mapping':'full-FOV pixel-area observations; no pixelwise content registration',
        'raw_delta_definition':'reference_linear-source_low_linear; signed float32',
        'raw_preview_definition':'round(128+127*raw_delta); neutral128; not alpha',
        'case_facts':facts,'inputs_unchanged':True,'policy':core.POLICY,'seconds':time.perf_counter()-start}
    core.write_json(out/'manifest.json',manifest)
    return manifest


def apply(args):
    with np.load(args.model,allow_pickle=False) as z:
        method=str(z['method']);features=z['features'];size=z['working_size'].tolist()
        expected=str(z['source_sha256']);ref_hash=str(z['reference_sha256'])
        if str(z['version'])!=VERSION or json.loads(str(z['policy_json']))!=core.POLICY:
            raise ValueError('Cached model uses a different frozen policy')
        fields={k:z[k] for k in ('R08','F02','F06','confidence')}
    if method not in METHODS:raise ValueError('Unknown approved method')
    codes,source_info=core.load_source(args.source)
    if source_info['sha256']!=expected:raise ValueError('Model belongs to another source file')
    out=preflight((args.source,args.model),args.out,[method])
    reference_info={'path':'cached full-relit reference','sha256':ref_hash}
    final,clips=render(codes,features,size,fields,[method],args.strength)
    results=export(out,codes,final,clips,source_info,reference_info,size,features,fields,args.strength)
    manifest={'version':VERSION,'methods':results,'source':source_info,'from_model':str(Path(args.model).resolve()),
              'strength':args.strength,'inputs_unchanged':core.file_hash(args.source)==expected}
    core.write_json(out/'manifest.json',manifest);return manifest


def cli():
    p=argparse.ArgumentParser(description=__doc__);sub=p.add_subparsers(dest='command',required=True)
    r=sub.add_parser('run');r.add_argument('--source',required=True);r.add_argument('--reference',required=True)
    r.add_argument('--out',required=True);r.add_argument('--methods', nargs='+', choices=METHODS, default=[DEFAULT_METHOD], help=argparse.SUPPRESS)
    r.add_argument('--max-aspect-error',type=float,default=.01);r.add_argument('--strength',type=float,default=1.)
    a=sub.add_parser('apply');a.add_argument('--source',required=True);a.add_argument('--model',required=True)
    a.add_argument('--out',required=True);a.add_argument('--strength',type=float,default=1.)
    args=p.parse_args()
    if not math.isfinite(args.strength) or not 0<=args.strength<=3:raise ValueError('Strength must be finite in [0,3]')
    if args.command=='run' and (not math.isfinite(args.max_aspect_error) or not 0<=args.max_aspect_error<=.01):
        raise ValueError('Aspect tolerance must be in [0,.01]')
    result=run(args) if args.command=='run' else apply(args)
    results = result['methods']
    summary = {
        'out': str(Path(args.out).resolve()),
        'methods': [entry['method'] for entry in results],
    }
    if len(results) == 1:
        summary.update(final=results[0]['final'], model=results[0]['model'])
    print(json.dumps(summary, ensure_ascii=False))


if __name__=='__main__':cli()
