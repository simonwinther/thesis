import argparse, importlib.util, json, os, random, sys, time, traceback
from pathlib import Path
import numpy as np
import torch

p=argparse.ArgumentParser()
p.add_argument('--model',choices=['wfm','wmf'],required=True)
p.add_argument('--steps',type=int,default=10)
p.add_argument('--batch-size',type=int,default=4)
p.add_argument('--tag',default='preflight')
a=p.parse_args()
root=Path('/home/ucloud/dev/thesis/models')/('WaveletMeanFlows' if a.model=='wmf' else 'WaveletFlowMatching')
sys.path.insert(0,str(root))
os.chdir(root)
spec=importlib.util.spec_from_file_location('training',root/'scripts'/('train_'+a.model+'.py'))
training=importlib.util.module_from_spec(spec);spec.loader.exec_module(training)
from guided_diffusion import logger, dist_util
logdir=Path('/work/WMF/b200-20261022')/(a.tag+'_'+a.model)
logdir.mkdir(parents=True,exist_ok=True)
logger.configure(dir=str(logdir),format_strs=['stdout','log'])
torch.set_num_threads(8)
torch.manual_seed(42);np.random.seed(42);random.seed(42)
args=training.create_argparser().parse_args([])
keys=['image_size','num_channels','num_res_blocks','channel_mult','learn_sigma','class_cond','use_checkpoint','attention_resolutions','num_heads','num_head_channels','num_heads_upsample','use_scale_shift_norm','dropout','resblock_updown','use_fp16','use_new_attention_order','dims','num_groups','in_channels','out_channels','bottleneck_attention','resample_2d','additive_skips','use_freq']
dist_util.setup_dist()
model=training.create_model(**{k:getattr(args,k) for k in keys})
model.to(dist_util.dev())
flow=(training.WaveletMeanFlows if a.model=='wmf' else training.WaveletFlowMatching)(sigma_max=.5)
# Preloaded full-resolution synthetic input: measures GPU compute/memory, not data I/O or quality.
batch={m:torch.rand(a.batch_size,1,224,224,160,device='cuda') for m in ['t1n','t1c','t2w','t2f']}
loop=training.BridgeTrainLoop(model,flow,[batch],a.batch_size,1e-5,0.,10,10000,0,a.steps,'all')
result={'model':a.model,'batch_size':a.batch_size,'shape':[224,224,160],'sigma_max':.5,'torch':torch.__version__,'gpu':torch.cuda.get_device_name(0),'synthetic_preloaded':True,'steps':[]}
try:
    torch.cuda.reset_peak_memory_stats()
    for step in range(1,a.steps+1):
        torch.cuda.synchronize();start=time.perf_counter()
        loss,_,_=loop.run_step(batch)
        torch.cuda.synchronize();elapsed=time.perf_counter()-start
        item={'step':step,'seconds':elapsed,'loss':float(loss),'peak_allocated_gib':torch.cuda.max_memory_allocated()/2**30,'peak_reserved_gib':torch.cuda.max_memory_reserved()/2**30}
        result['steps'].append(item)
        print(json.dumps(item),flush=True)
    result['status']='pass'
except Exception as e:
    result['status']='fail';result['error']=str(e);result['traceback']=traceback.format_exc()
    print(result['traceback'],flush=True)
finally:
    (logdir/'result.json').write_text(json.dumps(result,indent=2))
    if torch.distributed.is_initialized():torch.distributed.destroy_process_group()
if result['status']!='pass':sys.exit(1)
