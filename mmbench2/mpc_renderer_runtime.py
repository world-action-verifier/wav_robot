"""Configure a process-local renderer before importing simulator libraries."""
from pathlib import Path
import ctypes.util,os,sys
REVISION='portable-renderer-20261105'

def bootstrap(task):
    if os.environ.get('WM_MPC_GRAPHICS_REVISION')==REVISION:return
    backend=os.environ.get('MMBENCH2_RENDERER','osmesa')
    if backend not in ['osmesa','egl']:raise ValueError('MMBENCH2_RENDERER must be osmesa or egl')
    env=dict(os.environ,MUJOCO_GL=backend,PYOPENGL_PLATFORM=backend,SDL_VIDEODRIVER='dummy',
             LP_NUM_THREADS='2',WM_MPC_GRAPHICS_REVISION=REVISION)
    lib=os.environ.get('MMBENCH2_GRAPHICS_LIB')
    if lib:
        if not Path(lib).is_dir():raise FileNotFoundError(lib)
        env['LD_LIBRARY_PATH']=lib+(':'+env['LD_LIBRARY_PATH']if env.get('LD_LIBRARY_PATH')else'')
    if backend=='osmesa' and not lib and not ctypes.util.find_library('OSMesa'):
        raise RuntimeError('Install libOSMesa or set MMBENCH2_GRAPHICS_LIB; use MMBENCH2_RENDERER=egl for a configured EGL host')
    if task.startswith('ms-'):
        env.update(MUJOCO_GL='egl',PYOPENGL_PLATFORM='egl')
        icd=os.environ.get('MMBENCH2_VULKAN_ICD')
        if icd:env['VK_ICD_FILENAMES']=icd
        vulkan=os.environ.get('MMBENCH2_VULKAN_LIBRARY')
        if vulkan:env['SAPIEN_VULKAN_LIBRARY_PATH']=vulkan
    os.execve(sys.executable,[sys.executable,'-u','-X','faulthandler',*sys.argv],env)

def install_task_overrides(task):
    if not task.startswith('ms-')or os.environ.get('MMBENCH2_MANISKILL_CPU_RENDER')!='1':return
    import envs.maniskill as ms
    from mani_skill.envs.utils.system import backend as bk
    if getattr(bk,'_wav_pci_parser_fixed',False):return
    original_parse=bk.parse_backend_device_id
    bk.parse_backend_device_id=lambda name:(name,None)if isinstance(name,str)and name.startswith('pci:')else original_parse(name)
    original_make=ms.gym.make;env_ids={x['env']for x in ms.MANISKILL_TASKS.values()}
    def make(env_id,*args,**kw):
        if env_id in env_ids:kw['render_backend']='pci:0000:00:00.0'
        return original_make(env_id,*args,**kw)
    ms.gym.make=make;bk._wav_pci_parser_fixed=True

def provenance(task):
    return dict(revision=REVISION,mujoco=os.environ.get('MUJOCO_GL'),
                maniskill_cpu_render=os.environ.get('MMBENCH2_MANISKILL_CPU_RENDER')=='1' if task.startswith('ms-')else None,
                planning_device='cuda',physics_backend_override=False)
