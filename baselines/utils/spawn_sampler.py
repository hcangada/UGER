import multiprocessing as mp
from utils.infer_spawn_worker import run_sampling_worker

def spawn_sampling(*, args, corpus_path: str, idx: int, dyn: str, data_type: str, model_cls, tmpdir: str='/tmp', cublas_ws: str=':4096:8', gpu_for_sampling: str=None):
    ctx = mp.get_context('spawn')
    model_ctor = model_cls.__module__ + '.' + model_cls.__name__
    payload = {'tmpdir': tmpdir, 'cublas_ws': cublas_ws, 'gpu': args.gpu if gpu_for_sampling is None else gpu_for_sampling, 'seed': int(args.random_seed * 10007 + idx), 'corpus_path': corpus_path, 'idx': int(idx), 'dyn': dyn, 'data_type': data_type, 'model_ctor': model_ctor, 'args_dict': vars(args)}
    p = ctx.Process(target=run_sampling_worker, args=(payload,))
    p.start()
    p.join()
    if p.exitcode != 0:
        raise RuntimeError(f'[spawn_sampling] worker failed: exitcode={p.exitcode}')
