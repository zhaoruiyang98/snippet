## dependencies
- numpy
- matplotlib
- scipy
- [emcee](https://emcee.readthedocs.io/en/stable/)

## description
do sampling **uniformly** on the surface of a ellipsoid

configuration of the ellipsoid is described by 9 parameters
- $x_0,y_0,z_0$: center of the ellipsoid
- $a,b,c$: scale parameters
- $\alpha,\beta,\gamma$: three euler angle

standard ellipsoid equation:
$$
\frac{x^2}{a^2}+\frac{y^2}{b^2}+\frac{z^2}{c^2}=1
$$

then the ellipsoid is rotated by three rotation matrix in the order $z-y-z$(right hand)

$$
R=R_z(\gamma)R_y(\beta)R_z(\gamma)
$$
