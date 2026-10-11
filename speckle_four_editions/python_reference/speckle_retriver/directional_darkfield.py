"""Directional scatter covariance from paired raw intensities and WSVT shifts.

Independent reference implementation of the weighted Fourier-magnitude model
in Lautizi et al., APL 125, 264103 (2024), DOI:10.1063/5.0244120, Eq.3.
No code from the authors' implementation is included.
"""
from dataclasses import dataclass, asdict
import numpy as np
from scipy.ndimage import map_coordinates


@dataclass(frozen=True)
class DirectionalDarkfieldConfig:
    window: int = 16
    stride: int = 8
    apodization: str = "hamming"
    max_condition: float = 1e8
    max_model_residual: float = 0.15
    min_orientation_anisotropy: float = 0.1
    interpolation_order: int = 0

    def validate(self):
        if not isinstance(self.window, int) or self.window < 4:
            raise ValueError("analysis window must be an integer >=4")
        if not isinstance(self.stride, int) or self.stride < 1:
            raise ValueError("stride must be a positive integer")
        if self.apodization not in ("hamming", "rectangular"):
            raise ValueError("apodization must be hamming or rectangular")
        if self.interpolation_order not in (0, 1):
            raise ValueError("reference interpolation is explicit nearest(0) or linear(1)")
        if not np.isfinite(self.max_condition) or self.max_condition < 1:
            raise ValueError("max_condition must be finite and >=1")
        if not np.isfinite(self.max_model_residual) or self.max_model_residual <= 0:
            raise ValueError("max_model_residual must be finite and positive")
        if not 0 <= self.min_orientation_anisotropy <= 1:
            raise ValueError("orientation anisotropy threshold must lie in [0,1]")


def frequency_design(shape):
    """Flattened [mu,Cxx,Cxy,Cyy] design in angular frequency rad/pixel."""
    ky = 2*np.pi*np.fft.fftfreq(shape[0])
    kx = 2*np.pi*np.fft.fftfreq(shape[1])
    y,x = np.meshgrid(ky,kx,indexing="ij")
    return np.column_stack([np.ones(x.size),.5*x.ravel()**2,
                            (x*y).ravel(),.5*y.ravel()**2])


def tensor_observables(covariance, min_anisotropy=0.1):
    c = np.asarray(covariance, dtype=float)
    if c.shape != (2,2) or not np.isfinite(c).all() or not np.allclose(c,c.T):
        raise ValueError("expected a finite symmetric 2x2 covariance")
    eigen,vectors = np.linalg.eigh(c)
    tolerance = 256*np.finfo(float).eps*max(1.,float(np.max(np.abs(eigen))))
    physical = bool(eigen[0] >= -tolerance)
    out = dict(eigenvalue_min_raw_px2=float(eigen[0]),eigenvalue_max_raw_px2=float(eigen[1]),
               physical_valid=physical,roundoff_psd_clipped=bool(physical and eigen[0]<0),
               mean_scatter_variance_px2=np.nan,sigma_major_px=np.nan,sigma_minor_px=np.nan,
               anisotropy=np.nan,fractional_anisotropy_2d=np.nan,
               scattering_angle_rad=np.nan,minor_scattering_angle_rad=np.nan,
               orientation_valid=False)
    if not physical:
        return out
    lo,hi = np.maximum(eigen,0)
    total=lo+hi
    aniso=(hi-lo)/total if total>tolerance else 0.
    fa=(hi-lo)/np.hypot(hi,lo) if total>tolerance else 0.
    orient=bool(total>tolerance and hi-lo>tolerance and aniso>=min_anisotropy)
    angle=float(np.arctan2(vectors[1,1],vectors[0,1])%np.pi) if orient else np.nan
    out.update(mean_scatter_variance_px2=float(total/2),sigma_major_px=float(np.sqrt(hi)),
               sigma_minor_px=float(np.sqrt(lo)),anisotropy=float(aniso),
               fractional_anisotropy_2d=float(fa),orientation_valid=orient,
               scattering_angle_rad=angle,
               minor_scattering_angle_rad=float((angle+np.pi/2)%np.pi) if orient else np.nan)
    return out


def fit_spectral_tensor(sample_amplitude, reference_amplitude, config=None):
    """Fit one window across frames; inputs are full FFT magnitudes, F,H,W."""
    config=config or DirectionalDarkfieldConfig()
    config.validate()
    s=np.asarray(sample_amplitude,dtype=float)
    r=np.asarray(reference_amplitude,dtype=float)
    if s.ndim!=3 or s.shape!=r.shape or min(s.shape)<1 or min(s.shape[1:])<4:
        raise ValueError("paired spectral magnitudes must be F,H,W, H/W>=4")
    if not np.isfinite(s).all() or not np.isfinite(r).all() or np.any(s<0) or np.any(r<0):
        raise ValueError("spectral magnitudes must be finite and nonnegative")
    out=dict(mu=np.nan,transmission=np.nan,cov_xx_px2=np.nan,cov_xy_px2=np.nan,cov_yy_px2=np.nan,
             rank=0,condition=np.inf,observations=0,model_residual=np.nan,log_residual=np.nan,
             numerical_valid=False,physical_valid=False,fit_valid=False,orientation_valid=False,
             status="insufficient_spectrum")
    design=np.tile(frequency_design(s.shape[1:]),(s.shape[0],1))
    sf=s.ravel();rf=r.ravel()
    # A relative numerical floor, not a noise estimate. No synthetic log(0).
    eps=64*np.finfo(float).eps
    valid=(sf>eps*max(float(sf.max()),np.finfo(float).tiny)) & \
          (rf>eps*max(float(rf.max()),np.finfo(float).tiny))
    out["observations"]=int(valid.sum())
    if valid.sum()<4:
        return out
    a=design[valid]
    target=np.log(rf[valid])-np.log(sf[valid])
    weight=sf[valid]/sf[valid].max()
    aw=a*weight[:,None];bw=target*weight
    coef,_,rank,singular=np.linalg.lstsq(aw,bw,rcond=None)
    condition=float(singular[0]/singular[-1]) if singular[-1]>0 else np.inf
    out.update(rank=int(rank),condition=condition)
    if rank!=4 or condition>config.max_condition:
        out["status"]="ill_conditioned"
        return out
    mu,xx,xy,yy=coef
    exponent=-(a@coef)
    if not np.isfinite(coef).all() or np.any(np.abs(exponent)>700) or abs(mu)>700:
        out["status"]="nonfinite_model"
        return out
    # Normalize globally for stable residuals across intensity scales.
    scale=max(float(sf[valid].max()),float(rf[valid].max()))
    observed=sf[valid]/scale;predicted=np.exp(exponent)*(rf[valid]/scale)
    residual=float(np.linalg.norm(observed-predicted)/np.linalg.norm(observed))
    logres=float(np.linalg.norm(aw@coef-bw)/np.linalg.norm(weight))
    obs=tensor_observables([[xx,xy],[xy,yy]],config.min_orientation_anisotropy)
    out.update(obs)
    out.update(mu=float(mu),transmission=float(np.exp(-mu)),cov_xx_px2=float(xx),
               cov_xy_px2=float(xy),cov_yy_px2=float(yy),numerical_valid=True,
               model_residual=residual,log_residual=logres)
    out["fit_valid"]=bool(out["physical_valid"] and residual<=config.max_model_residual)
    out["orientation_valid"]=bool(out["orientation_valid"] and out["fit_valid"])
    out["status"]="ok" if out["fit_valid"] else ("non_psd" if not out["physical_valid"] else "model_mismatch")
    if not out["orientation_valid"]:
        out["scattering_angle_rad"]=out["minor_scattering_angle_rad"]=np.nan
    return out


def fit_raw_patches(sample,reference,config=None):
    config=config or DirectionalDarkfieldConfig()
    config.validate()
    s=np.asarray(sample,dtype=float);r=np.asarray(reference,dtype=float)
    if s.ndim!=3 or s.shape!=r.shape or s.shape[1:]!=(config.window,config.window):
        raise ValueError("patches must have shape F,window,window")
    if not np.isfinite(s).all() or not np.isfinite(r).all():
        raise ValueError("raw intensities must be finite")
    # Apodization of a flat/striped patch can create artificial 2-D frequencies.
    # Require 2-D raw reference texture before applying the window.
    gy,gx=np.gradient(r,axis=(-2,-1))
    gradient_gram=np.array([[np.sum(gx*gx),np.sum(gx*gy)],
                            [np.sum(gx*gy),np.sum(gy*gy)]])
    texture_eigen=np.linalg.eigvalsh(gradient_gram)
    texture_ok=texture_eigen[-1]>0 and texture_eigen[0]>256*np.finfo(float).eps*texture_eigen[-1]
    if not texture_ok:
        out=fit_spectral_tensor(np.zeros_like(s),np.zeros_like(r),config)
        out["status"]="insufficient_2d_texture"
        return out
    w=np.hamming(config.window) if config.apodization=="hamming" else np.ones(config.window)
    kernel=w[:,None]*w[None,:]
    return fit_spectral_tensor(np.abs(np.fft.fft2(s*kernel,axes=(-2,-1))),
                               np.abs(np.fft.fft2(r*kernel,axes=(-2,-1))),config)


def directional_visibility(covariance, angles_rad, frequency_cycles_per_pixel):
    if not np.isfinite(frequency_cycles_per_pixel) or frequency_cycles_per_pixel<0:
        raise ValueError("frequency must be finite and nonnegative")
    c=np.asarray(covariance,dtype=float)
    obs=tensor_observables(c)
    if not obs["physical_valid"]:
        raise ValueError("directional visibility requires a PSD covariance")
    theta=np.asarray(angles_rad,dtype=float)
    if not np.isfinite(theta).all():
        raise ValueError("angles must be finite")
    v=np.stack([np.cos(theta),np.sin(theta)],axis=-1)
    return np.exp(-.5*(2*np.pi*frequency_cycles_per_pixel)**2*np.einsum("...i,ij,...j->...",v,c,v))


def analyze_stack(sample,reference,*,displace_x=None,displace_y=None,
                  displacement_origin_yx=(0,0),centers_y=None,centers_x=None,
                  allow_unregistered=False,config=None,pixel_size_m=None,distance_m=None):
    """Return a sampled grid of DDF fits; no inference/interpolation at missing centers.

    Reference sample at p-public_displacement. Shift is held constant per window.
    Input normalization is the caller's responsibility; raw intensity only.
    """
    config=config or DirectionalDarkfieldConfig()
    config.validate()
    s=np.asarray(sample);r=np.asarray(reference)
    if s.ndim!=3 or s.shape!=r.shape or min(s.shape)<1 or min(s.shape[1:])<config.window:
        raise ValueError("paired raw stacks must be nonempty F,H,W and fit the window")
    if not np.isfinite(s).all() or not np.isfinite(r).all():
        raise ValueError("raw input contains NaN/Inf")
    registered=displace_x is not None and displace_y is not None
    if (displace_x is None)!=(displace_y is None):
        raise ValueError("both displacement components are required")
    if not registered and not allow_unregistered:
        raise ValueError("provide WSVT displacement or explicitly allow unregistered analysis")
    if registered:
        dx=np.asarray(displace_x);dy=np.asarray(displace_y)
        if dx.ndim!=2 or dx.shape!=dy.shape:
            raise ValueError("displacement fields must share a 2D shape")
    origin=np.asarray(displacement_origin_yx)
    if origin.shape!=(2,) or np.any(origin!=origin.astype(int)):
        raise ValueError("displacement origin must be two integers")
    if (pixel_size_m is None)!=(distance_m is None):
        raise ValueError("angular units require both pixel size and distance")
    if pixel_size_m is not None and (not np.isfinite([pixel_size_m,distance_m]).all() or
                                    pixel_size_m<=0 or distance_m<=0):
        raise ValueError("geometry values must be finite and positive")
    _,h,w=s.shape;n=config.window;left=n//2
    default_y=np.arange(left,h-(n-left)+1,config.stride)
    default_x=np.arange(left,w-(n-left)+1,config.stride)
    cy=default_y if centers_y is None else np.asarray(centers_y)
    cx=default_x if centers_x is None else np.asarray(centers_x)
    if cy.ndim!=1 or cx.ndim!=1 or not cy.size or not cx.size or \
       np.any(cy!=cy.astype(int)) or np.any(cx!=cx.astype(int)):
        raise ValueError("centers must be nonempty vectors of integer raw coordinates")
    rows=[];oy,ox=map(int,origin)
    gy,gx=np.indices((n,n),dtype=float)
    for y0 in cy.astype(int):
        for x0 in cx.astype(int):
            y=y0-left;x=x0-left
            row=dict(raw_y=int(y0),raw_x=int(x0),numerical_valid=False,physical_valid=False,
                     fit_valid=False,orientation_valid=False,status="outside_valid_domain")
            if y<0 or x<0 or y+n>h or x+n>w:
                rows.append(row);continue
            ux=uy=0.
            if registered:
                iy,ix=y0-oy,x0-ox
                if iy<0 or ix<0 or iy>=dx.shape[0] or ix>=dx.shape[1]:
                    row["status"]="outside_displacement_map";rows.append(row);continue
                ux=-float(dx[iy,ix]);uy=-float(dy[iy,ix])
                if not np.isfinite([ux,uy]).all():
                    row["status"]="invalid_displacement";rows.append(row);continue
            requested_ux,requested_uy=ux,uy
            if config.interpolation_order==0:
                # Fourier magnitudes absorb residual constant translation.
                # Integer patch shifts avoid injecting interpolation blur.
                ux=float(np.floor(ux+.5));uy=float(np.floor(uy+.5))
            sy=gy+y+uy;sx=gx+x+ux
            if sy.min()<0 or sx.min()<0 or sy.max()>h-1 or sx.max()>w-1:
                rows.append(row);continue
            sample_patch=s[:,y:y+n,x:x+n]
            if ux==int(ux) and uy==int(uy):
                ref_patch=r[:,y+int(uy):y+int(uy)+n,x+int(ux):x+int(ux)+n]
            else:
                ref_patch=np.array([map_coordinates(frame.astype(float),[sy,sx],
                                    order=config.interpolation_order,prefilter=False,
                                    mode="constant",cval=np.nan) for frame in r])
            row.update(fit_raw_patches(sample_patch,ref_patch,config))
            row.update(reference_lookup_x=ux,reference_lookup_y=uy,
                       requested_reference_lookup_x=requested_ux,requested_reference_lookup_y=requested_uy)
            if pixel_size_m is not None:
                factor=(pixel_size_m/distance_m)**2
                for axis in ("xx","xy","yy"):
                    row["cov_"+axis+"_rad2"]=row["cov_"+axis+"_px2"]*factor
            rows.append(row)
    required={"cov_xx_px2","cov_xy_px2","cov_yy_px2","transmission","mu","condition","rank",
              "observations","model_residual","mean_scatter_variance_px2","sigma_major_px",
              "sigma_minor_px","anisotropy","fractional_anisotropy_2d","scattering_angle_rad",
              "minor_scattering_angle_rad","roundoff_psd_clipped"}
    keys=sorted((set().union(*(row.keys() for row in rows))|required)-{"raw_y","raw_x"})
    fields={}
    for k in keys:
        if k=="status":
            value=[row.get(k,"unknown") for row in rows]
        elif k.endswith("_valid") or k=="roundoff_psd_clipped":
            value=[row.get(k,False) for row in rows]
        else:
            value=[row.get(k,np.nan) for row in rows]
        fields[k]=np.array(value).reshape(len(cy),len(cx))
    return dict(fields=fields,centers_y=cy.astype(int),centers_x=cx.astype(int),
                metadata=dict(config=asdict(config),registered=registered,
                    displacement_origin_yx=list(map(int,origin)),frames=int(s.shape[0]),
                    raw_shape=list(s.shape[1:]),covariance_units="pixel^2",
                    angle_convention="major scattering axis modulo pi; +x columns, +y rows",
                    geometry="parallel-beam small-angle" if pixel_size_m is not None else None,
                    pixel_size_m=pixel_size_m,distance_m=distance_m,
                    method="Lautizi2024 Eq3 weighted local Fourier magnitudes; WSVT patch registration",
                    physical_accuracy_claim=False))
