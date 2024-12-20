#include "cones.h"

#define BOX_CONE_MAX_ITERS (25)
#define POW_CONE_TOL (1e-9)
#define POW_CONE_MAX_ITERS (20)

/* Box cone limits (+ or -) taken to be INF */
#define MAX_BOX_VAL (1e15)

/*
 * CUDA Routine to scale the limits of one entry in the box cone by the scaling diagonal mat D > 0
 *
 *  want (t, s) \in K <==> (t', s') \in K'
 *
 *  (t', s') = (d0 * t, D s) (overloading D to mean D[1:])
 *    (up to scalar scaling factor which we can ignore due to conic prooperty)
 *
 *   K = { (t, s) | t * l <= s <= t * u, t >= 0 } =>
 *       { (t, s) | d0 * t * D l / d0 <= D s <= d0 * t D u / d0, t >= 0 } =>
 *       { (t', s') | t' * l' <= s' <= t' u', t >= 0 } = K'
 *  where l' = D l  / d0, u' = D u / d0.
 */
void cuda_normalize_box_cone(scs_int j, ScsCone *k, scs_float *D, scs_int bsize) {
  if (j < bsize - 1){
    if (k->bu[j] >= MAX_BOX_VAL) {
      k->bu[j] = INFINITY;
    } else {
      k->bu[j] = D ? D[j + 1] * k->bu[j] / D[0] : k->bu[j];
    }
    if (k->bl[j] <= -MAX_BOX_VAL) {
      k->bl[j] = -INFINITY;
    } else {
      k->bl[j] = D ? D[j + 1] * k->bl[j] / D[0] : k->bl[j];
    }
  }
}

void cuda_scale_box_cone(ScsCone *k, ScsConeWork *c, ScsScaling *scal) {
  if (k->bsize && k->bu && k->bl) {
    c->box_t_warm_start = 1.;
    if (scal) {
      /* also does some sanitizing */
      scs_int j = blockIdx.x*blockDim.x+threadIdx.x;
      cuda_normalize_box_cone(j,k, &(scal->D[k->z + k->l]), k->bsize);
    }
  }
}

__global__ void cuda_proj_box_cone_grad_hess_kernel(const scs_float *bl,const scs_float* bu,const scs_int bsize,const scs_float t,const scs_float* x, const scs_float* rho, scs_float* gt_ele, scs_float* ht_ele){
  scs_int j = threadIdx.x;
  if (j<bsize-1){
    scs_float r = rho ? 1.0 / rho[j] : 1.;
    if (x[j] > t * bu[j]) {
      gt_ele[j] = r * (t * bu[j] - x[j]) * bu[j]; /* gradient */
      ht_ele[j] = r * bu[j] * bu[j];              /* hessian */
    } else if (x[j] < t * bl[j]) {
      gt_ele[j] = r * (t * bl[j] - x[j]) * bl[j]; /* gradient */
      ht_ele[j] = r * bl[j] * bl[j];              /* hessian */
    }
  }
  //accumulate gradient and hessian into first element of arrays
  for(uint stride=(bsize-1)/2;stride>0;stride>>=1){
    if (j<stride){
      gt_ele[j]+=gt_ele[j+stride];
      ht_ele[j]+=ht_ele[j+stride];
    }
    __syncthreads();
  }
}


/* Project onto { (t, s) | t * l <= s <= t * u, t >= 0 }, Newton's method on t
   tx = [t; s], total length = bsize, under Euclidean metric 1/r_box.
   Using a single CUDA thread
*/
static scs_float cuda_proj_box_cone(scs_float *tx, const scs_float *bl,
                               const scs_float *bu, scs_int bsize,
                               scs_float t_warm_start, scs_float *r_box) {
  scs_float *x, gt, ht, t_prev, t = t_warm_start;
  scs_float rho_t = 1, *rho = SCS_NULL, r;
  scs_int iter, j;

  if (bsize == 1) { /* special case */
    tx[0] = MAX(tx[0], 0.0);
    return tx[0];
  }
  x = &(tx[1]);

  if (r_box) {
    rho_t = 1.0 / r_box[0];
    rho = &(r_box[1]);
  }

  /* should only require about 5 or so iterations, 1 or 2 if warm-started */
  scs_float *gt_dev,*ht_dev;
  cudaMalloc(&gt_dev,(bsize-1)*sizeof(scs_float));
  cudaMalloc(&ht_dev,(bsize-1)*sizeof(scs_float));
  for (iter = 0; iter < BOX_CONE_MAX_ITERS; iter++) {
    t_prev = t;
    gt = rho_t * (t - tx[0]); /* gradient */
    ht = rho_t;               /* hessian */
    cuda_proj_box_cone_grad_hess_kernel<<<1,256>>>(bl,bu,bsize,t,x,rho,gt_dev,ht_dev);//TODO: how should the various cone parameters get to the device?
    gt+=gt_dev[0];
    ht+=ht_dev[0];
    t = MAX(t - gt / MAX(ht, 1e-8), 0.); /* newton step */
#if VERBOSITY > 3
    scs_printf("iter %i, t_new %1.3e, t_prev %1.3e, gt %1.3e, ht %1.3e\n", iter,
               t, t_prev, gt, ht);
    scs_printf("ABS(gt / (ht + 1e-6)) %.4e, ABS(t - t_prev) %.4e\n",
               ABS(gt / (ht + 1e-6)), ABS(t - t_prev));
#endif
    /* TODO: sometimes this check can fail (ie, declare convergence before it
     * should) if ht is very large, which can happen with some pathological
     * problems.
     */
    if (ABS(gt / MAX(ht, 1e-6)) < 1e-12 * MAX(t, 1.) ||
        ABS(t - t_prev) < 1e-11 * MAX(t, 1.)) {
      break;
    }
  }
  cudaFree(gt_dev);
  cudaFree(ht_dev);
  if (iter == BOX_CONE_MAX_ITERS) {
    scs_printf("warning: box cone proj hit maximum %i iters\n", (int)iter);
  }
  for (j = 0; j < bsize - 1; j++) {
    if (x[j] > t * bu[j]) {
      x[j] = t * bu[j];
    } else if (x[j] < t * bl[j]) {
      x[j] = t * bl[j];
    }
    /* x[j] unchanged otherwise */
  }
  tx[0] = t;

#if VERBOSITY > 3
  scs_printf("box cone iters %i\n", (int)iter + 1);
#endif
  return t;
}

/* project onto SOC of size q using a single CUDA thread*/
void cuda_proj_soc(scs_float *x, scs_int q) {
  if (q == 0) {
    return;
  }
  if (q == 1) {
    x[0] = MAX(x[0], 0.);
    return;
  }
  scs_float v1 = x[0];
  scs_float s = 0;
  for (int i=1;i<q;i++){
    s+=x[i]*x[i];
  }
  s=SQRTF(s);
  scs_float alpha = (s + v1) / 2.0;

  if (s <= v1) {
    return;
  } else if (s <= -v1) {
    memset(&(x[0]), 0, q * sizeof(scs_float));
  } else {
    x[0] = alpha;
    for(int i=1;i<q;i++){
      x[i]*=alpha / s;
    }
  }
}

static void proj_power_cone(scs_float *v, scs_float a) {
  scs_float xh = v[0], yh = v[1], rh = ABS(v[2]);
  scs_float x = 0.0, y = 0.0, r;
  scs_int i;
  /* v in K_a */
  if (xh >= 0 && yh >= 0 &&
      POW_CONE_TOL + POWF(xh, a) * POWF(yh, (1 - a)) >= rh) {
    return;
  }

  /* -v in K_a^* */
  if (xh <= 0 && yh <= 0 &&
      POW_CONE_TOL + POWF(-xh, a) * POWF(-yh, 1 - a) >=
          rh * POWF(a, a) * POWF(1 - a, 1 - a)) {
    v[0] = v[1] = v[2] = 0;
    return;
  }

  r = rh / 2;
  for (i = 0; i < POW_CONE_MAX_ITERS; ++i) {
    scs_float f, fp, dxdr, dydr;
    x = pow_calc_x(r, xh, rh, a);
    y = pow_calc_x(r, yh, rh, 1 - a);

    f = pow_calc_f(x, y, r, a);
    if (ABS(f) < POW_CONE_TOL) {
      break;
    }

    dxdr = pow_calcdxdr(x, xh, rh, r, a);
    dydr = pow_calcdxdr(y, yh, rh, r, (1 - a));
    fp = pow_calc_fp(x, y, dxdr, dydr, a);

    r = MAX(r - f / fp, 0);
    r = MIN(r, rh);
  }
  v[0] = x;
  v[1] = y;
  v[2] = (v[2] < 0) ? -(r) : (r);
}

/* project onto the primal K cone in the paper */
/* the r_y vector determines the INVERSE metric, ie, project under the
 * diag(r_y)^-1 norm.
 */
scs_int cuda_proj_cone(scs_int cone_index, scs_float *x, const ScsCone *k, ScsConeWork *c,
                         scs_int normalize, scs_float *r_y) {
  scs_int done=0;
  scs_int vector_index=0;
  scs_int cone_count=0;

  scs_float *r_box = SCS_NULL;

  if (cone_index<k->z) { /* doesn't use r_y */
    /* project onto primal zero / dual free cone */
    x[cone_index]=0;
    done=1;
  }
  else{
    vector_index=k->z;
    cone_count=k->z;
  }
  if (!done && cone_index<cone_count+k->l) { /* doesn't use r_y */
    /* project onto positive orthant */
    x[cone_index] = MAX(x[cone_index], 0.0);
    done=1;
  }
  else{
    vector_index+=k->l;
    cone_index+=k->l;
  }
  if (!done && cone_index<cone_count+k->bsize) { /* DOES use r_y */
    if (r_y) {
      r_box = &(r_y[cone_index]);
    }
    /* project onto box cone */
    c->box_t_warm_start = cuda_proj_box_cone(&(x[cone_index]), k->bl, k->bu, k->bsize,
                                        c->box_t_warm_start, r_box);
    done=1;
  }
  else{
    vector_index += k->bsize; /* since b = (t,s), len(s) = bsize - 1 */
    cone_count+= k->bsize;
  }
  if (!done && k->q && cone_index<cone_count+k->qsize) { /* doesn't use r_y */
    /* project onto second-order cones */

    //figure out what index in the vectors this cone's entries begin at
    scs_int SOC_cone_index=cone_index-cone_count;//current cone is this index into the SOC cones
    for(scs_int j=0;j<SOC_cone_index;k++){
      vector_index+=k->q[j];
    }
    cuda_proj_soc(&(x[vector_index]),k->q[SOC_cone_index]);
    done=1;
  }
  else{
    for(scs_int j=0;j<k->qsize;k++){//accumulate SOC entries if any
      vector_index+=k->q[j];
    }
    cone_count+=k->qsize;
  }
  if (!done && k->s && cone_index<cone_count+k->ssize) { /* doesn't use r_y */
    /* project onto PSD cones */

    //figure out what index in the vectors this cone's entries begin at
    scs_int PSD_cone_index=cone_index-cone_count;//current cone is this index into the PSD cones
    for(scs_int j=0;j<PSD_cone_index;k++){
      vector_index+=get_sd_cone_size(k->s[j]);
    }
    cuda_proj_semi_definite_cone(&(x[vector_index]),k->s[PSD_cone_index],c);
    done=1;
  }
  else{
    for(scs_int j=0;j<k->ssize;k++){
      vector_index+=get_sd_cone_size(k->s[j]);
    }
    cone_count+=k->ssize;
  }
  if (!done && (k->ep || k->ed) && cone_index< cone_count+k->ep+k->ed ) { /* doesn't use r_y */
    scs_int EXP_cone_index=cone_index-cone_count;
      /* provided in exp_cone.c */
    SCS(proj_pd_exp_cone)(&(x[vector_index + 3 * EXP_cone_index]), EXP_cone_index < k->ep);
    done=1;
  }
  else{
    vector_index += 3 * (k->ep + k->ed);
    cone_count+=k->ep+k->ed;
  }
  if (!done && cone_index<cone_count+k->psize && k->p) { /* doesn't use r_y */
    scs_float v[3];
    scs_int idx;
    scs_int PWR_cone_index=cone_index-cone_count;
    idx = vector_index + 3 * PWR_cone_index;
    if (k->p[PWR_cone_index] >= 0) {
      /* primal power cone */
      proj_power_cone(&(x[idx]), k->p[PWR_cone_index]);
    } else {
      /* dual power cone, using Moreau */
      v[0] = -x[idx];
      v[1] = -x[idx + 1];
      v[2] = -x[idx + 2];

      proj_power_cone(v, -k->p[PWR_cone_index]);

      x[idx] += v[0];
      x[idx + 1] += v[1];
      x[idx + 2] += v[2];
    }
    done=1;
  }else{
    vector_index += 3 * k->psize;
    cone_count+=k->psize;
  }
  /* project onto OTHER cones */
  return 0;
}

/* CUDA Kernel for cone projection routine, performs projection in-place.
   If normalize > 0 then will use normalized (equilibrated) cones if applicable.

   Moreau decomposition for R-norm projections:

    `x + R^{-1} \Pi_{C^*}^{R^{-1}} ( - R x ) = \Pi_C^R ( x )`

   where \Pi^R_C is the projection onto C under the R-norm:

    `||x||_R = \sqrt{x ' R x}`.

*/
__global__
void _cuda_proj_dual_cone_kernel(scs_float *x, ScsConeWork *c, ScsScaling *scal,
                            scs_float *r_y) {
  int status, i;
  ScsCone *k = c->k;

  if (!c->scaled_cones) {
    cuda_scale_box_cone(k, c, scal);
    c->scaled_cones = 1;
  }

  /* copy s = x */
  i=blockIdx.x*blockDim.x+threadIdx.x;
  c->s[i]=x[i];

  /* x -> - Rx */
  x[i] *= r_y ? -r_y[i] : -1;

  /* project -x onto cone, x -> \Pi_{C^*}^{R^{-1}}(-x) under r_y metric */
  status = cuda_proj_cone(i,x, k, c, scal ? 1 : 0, r_y);

  /* return x + R^{-1} \Pi_{C^*}^{R^{-1}} ( -x )  */
  if (r_y) {
    x[i] = x[i] / r_y[i] + c->s[i];
  } else {
    x[i] += c->s[i];
  }
}

int _cuda_proj_dual_cone_host(float *x, ScsConeWork *c, ScsScaling *scal,
                            float *r_y) {
    
}