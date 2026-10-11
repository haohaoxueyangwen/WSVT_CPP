#!/usr/bin/env python3
"""Native author-compatible DDF fitting; Python handles I/O and derived maps only."""
from __future__ import annotations
import ctypes as ct
import csv
from pathlib import Path
import time
import numpy as np

from wsvt.common.ddf import covariance_from_abc
from wsvt.common.hashing import file_sha256 as sha256
from wsvt.common.json_io import write_json
from wsvt.common.paths import ddf_library, project_root

ROOT = project_root()
DEFAULT_LIBRARY = ddf_library()

class NativeAuthorFit:
    def __init__(self, library=DEFAULT_LIBRARY):
        self.library = Path(library).resolve()
        if not self.library.is_file():
            raise FileNotFoundError(
                f'native DDF library not found: {self.library}. '
                'Build target wsvt_ddf_author or pass --cpp-library.'
            )
        self.lib = ct.CDLL(str(self.library))
        self.fn = self.lib.wsvt_ddf_author_fit
        p = ct.POINTER(ct.c_double)
        common = [p,p,ct.c_size_t,ct.c_size_t,ct.c_size_t,ct.c_size_t,
            ct.c_size_t,ct.c_size_t,ct.c_double,ct.c_double,ct.c_size_t,ct.c_int,p,
            ct.c_double,ct.c_double,ct.c_size_t]
        self.fn.argtypes = common + [p,ct.c_char_p,ct.c_size_t]
        self.fn.restype = ct.c_int
        self.fn_ex = None
        self.fn_tol = None
        try:
            self.fn_ex = self.lib.wsvt_ddf_author_fit_ex
            self.fn_ex.argtypes = common + [ct.c_size_t, p, ct.c_char_p, ct.c_size_t]
            self.fn_ex.restype = ct.c_int
        except AttributeError:
            self.fn_ex = None
        try:
            self.fn_tol = self.lib.wsvt_ddf_author_fit_ex_tol
            self.fn_tol.argtypes = common + [ct.c_size_t, ct.c_double, ct.c_int, p, ct.c_char_p, ct.c_size_t]
            self.fn_tol.restype = ct.c_int
        except AttributeError:
            self.fn_tol = None

    def fit(self, sample, reference, y, x, uy, ux, window_half=8, max_shift=16,
            initial=(0.,0.,0.4), blur_extra=.05, max_sigma=10., maxiter=500,
            kernel_radius=8, objective_atol=None, use_direct_residual=False):
        sample = np.ascontiguousarray(sample, dtype=np.float64)
        reference = np.ascontiguousarray(reference, dtype=np.float64)
        if sample.ndim != 3 or sample.shape != reference.shape:
            raise ValueError('identical frame-major 3D stacks required')
        if y < 0 or x < 0 or window_half < 0 or maxiter <= 0:
            raise ValueError('invalid pixel, window or iteration limit')
        kr = int(kernel_radius)
        if kr < 1:
            raise ValueError('kernel_radius must be >= 1')
        init = np.ascontiguousarray(initial, dtype=np.float64)
        if init.shape != (3,): raise ValueError('initial transform needs three values')
        use_tol = objective_atol is not None
        out = np.empty(18 if use_tol else 13, dtype=np.float64)
        err = ct.create_string_buffer(1024)
        p = ct.POINTER(ct.c_double)
        args = (sample.ctypes.data_as(p), reference.ctypes.data_as(p), sample.size,
            *sample.shape, int(y), int(x), uy, ux, window_half, max_shift,
            init.ctypes.data_as(p), blur_extra, max_sigma, maxiter)
        if use_tol:
            if self.fn_tol is None:
                raise ValueError('native library has no wsvt_ddf_author_fit_ex_tol')
            rc = self.fn_tol(*args, kr, float(objective_atol), 1 if use_direct_residual else 0,
                             out.ctypes.data_as(p), err, len(err))
        elif self.fn_ex is not None:
            rc = self.fn_ex(*args, kr, out.ctypes.data_as(p), err, len(err))
        else:
            if kr != 8:
                raise ValueError(
                    'native library has no wsvt_ddf_author_fit_ex; '
                    'rebuild target wsvt_ddf_author for configurable kernel_radius'
                )
            rc = self.fn(*args, out.ctypes.data_as(p), err, len(err))
        if rc: raise ValueError(err.value.decode())
        names = ['sx','sy','sigma','a','b','c','penalty','transmission','cost',
            'residual_cost','iterations','evaluations','converged']
        result = dict(zip(names, out[:13].tolist()))
        result['kernel_radius'] = kr
        result['window_half'] = int(window_half)
        result['objective'] = float(result['cost']) + float(result['penalty'])
        if use_tol:
            result['final_dx'] = float(out[13])
            result['final_df'] = float(out[14])
            result['author_converged'] = bool(out[15])
            result['data_scale'] = float(out[16])
            result['direct_plus_penalty'] = float(out[17])
            result['objective_atol'] = float(objective_atol)
            result['use_direct_residual'] = bool(use_direct_residual)
        else:
            result['author_converged'] = bool(result['converged'])
            result['objective_atol'] = 1e-12
        return result

def run_cpp_solver(args):
    from wsvt.darkfield.solver import load_stack, load_displacement
    if args.mode != 'gradient_descent' or args.refine_displacement:
        raise ValueError('native backend implements the verified fixed-u joint Nelder-Mead path')
    if not args.transmission_threshold > 1.:
        raise ValueError('fixed-u path initializes T=1; threshold must exceed 1 to fit all pixels')
    output = args.output.resolve()
    if output.exists(): raise FileExistsError(output)
    sam, ref, input_info = load_stack(args.input)
    dx, dy, displacement_info = load_displacement(args.wsvt_result)
    kr = int(getattr(args, 'kernel_radius', 8))
    pad = args.max_shift + args.window_half + kr
    h,w = sam.shape[1:]
    y0,y1,x0,x1 = args.roi_yx or (pad,h-pad-1,pad,w-pad-1)
    if args.step<=0 or not (pad<=y0<=y1<h-pad and pad<=x0<=x1<w-pad):
        raise ValueError('ROI outside official full-halo domain')
    ry,rx = np.arange(y0,y1+1,args.step),np.arange(x0,x1+1,args.step)
    oy,ox = args.displacement_origin_yx
    if oy<0 or ox<0 or ry[0]<oy or rx[0]<ox or ry[-1]-oy>=dx.shape[0] or rx[-1]-ox>=dx.shape[1]:
        raise ValueError('ROI outside WSVT displacement field')
    uy,ux = -dy[np.ix_(ry-oy,rx-ox)],-dx[np.ix_(ry-oy,rx-ox)]
    native = NativeAuthorFit(getattr(args,'cpp_library',DEFAULT_LIBRARY))
    rows=[];start=time.perf_counter()
    for i,y in enumerate(ry):
        for j,x in enumerate(rx):
            r=native.fit(sam,ref,y,x,uy[i,j],ux[i,j],args.window_half,args.max_shift,
                (0.,0.,args.initial_sigma),args.blur_extra,args.max_sig,args.maxiter,
                kernel_radius=kr)
            cov,major,minor,angle=covariance_from_abc(r['a'],r['b'],r['c'])
            r.update(raw_y=int(y),raw_x=int(x),lookup_uy=float(uy[i,j]),lookup_ux=float(ux[i,j]),
                cov_xx_px2=float(cov[0,0]),cov_xy_px2=float(cov[0,1]),cov_yy_px2=float(cov[1,1]),
                sigma_major_px=major,sigma_minor_px=minor,angle_deg=angle,
                rounded_lookup_y=float(np.copysign(np.floor(abs(uy[i,j])+.5),uy[i,j])),
                rounded_lookup_x=float(np.copysign(np.floor(abs(ux[i,j])+.5),ux[i,j])),fit_valid=1.)
            rows.append(r)
    elapsed=time.perf_counter()-start
    output.mkdir(parents=True)
    with (output/'wsvt_umpa_ddf.csv').open('w',newline='') as stream:
        writer=csv.DictWriter(stream,fieldnames=list(rows[0]));writer.writeheader();writer.writerows(rows)
    maps={k:np.array([r[k] for r in rows]).reshape(len(ry),len(rx)) for k in rows[0] if k not in ('raw_y','raw_x')}
    np.savez_compressed(output/'wsvt_umpa_ddf.npz',**maps,raw_y=ry,raw_x=rx)
    params={k:str(v) if isinstance(v,Path) else v for k,v in vars(args).items()}
    params.update(backend='native_cpp_author_nelder_mead',input=input_info,wsvt_result=displacement_info,
        official_python_used=False,library_sha256=sha256(native.library),
        covariance_units='pixel^2',angle_convention='image x-y, modulo 180 degrees',
        rounding='C round: halfway away from zero',xatol=1e-4,fatol=1e-12)
    write_json(output/'params.json',params)
    report=dict(status='completed',sampled_points=len(rows),fit_valid_count=len(rows),
        converged_count=sum(bool(r['converged']) for r in rows),solver_time_s=elapsed,
        formal_timing=False,physical_accuracy_claim=False,backend=params['backend'])
    write_json(output/'bench_report.json',report)
    write_json(output/'artifact_index.json',{p.name:dict(bytes=p.stat().st_size,sha256=sha256(p)) for p in output.iterdir()})
    return report

def main() -> None:
    from wsvt.darkfield.solver import parser
    ap = parser()
    ap.add_argument('--cpp-library', type=Path, default=DEFAULT_LIBRARY)
    ap.add_argument('--kernel-radius', type=int, default=8,
                    help='DF kernel support radius (independent of analysis window)')
    print(run_cpp_solver(ap.parse_args()))


if __name__ == '__main__':
    main()
