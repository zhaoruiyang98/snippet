import numpy as np
from scipy.stats import norm
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import emcee


class emock:
    def __init__(self, parameter, n=1000, sigma=0.1):
        """init a emock instance
        Parameters
        ----------
        parameter: array_like
            shape(parameter) = 9
            [x0,y0,z0,a,b,c,alpha,beta,gamma]
        n: int
            number of mock data
        sigma: float
            std of standard ellipsoid function
        """
        self.parameter = parameter
        self.n = n
        self.sigma = sigma
        self.autocorrelation_time = None
        self.mock = self._get_mock()

    def _get_mock(self):
        """get mock
        Returns
        --------
        result : numpy array
        """
        a = self.parameter[3]
        b = self.parameter[4]
        c = self.parameter[5]
        sqrt = np.sqrt
        cos = np.cos
        sin = np.sin

        def logprob(p):
            """log likelihood
            Parameters
            ----------
            p : array_like
                p = [ratio,theta,phi]
            Returns
            -------
            result : float 
            """
            if p[0] <= 0:
                return -np.inf
            if p[1] < 0 or p[1] >= np.pi:
                return -np.inf
            if p[2] < 0 or p[2] >= 2*np.pi:
                return -np.inf

            # surface element of a ellipsoid
            ds = sqrt(a**2 * b**2 * cos(p[1])**2 * sin(p[1])**2 + c**2 *
                      sin(p[1])**4 * (b**2 * cos(p[2])**2 + a**2 * sin(p[2])**2))
            # do sampling uniformly on ellipsoid's surface
            result = np.log(ds)
            # gaussian fluctuation of ratio
            result += norm.logpdf(p[0], loc=1, scale=self.sigma)
            return result

        nwalkers = 10
        ndim = 3
        p0 = np.random.rand(nwalkers, ndim)
        sampler = emcee.EnsembleSampler(nwalkers, ndim, logprob)
        # burn in
        state = sampler.run_mcmc(p0, 100)
        sampler.reset()

        print("running mcmc")
        sampler.run_mcmc(state, np.ceil(self.n/nwalkers), progress=True)
        result = sampler.get_chain()
        # get autocorrelation_time
        self.autocorrelation_time = emcee.autocorr.integrated_time(
            result, quiet=True)
        # trim
        temp = result.shape
        result = np.reshape(result, (temp[0]*temp[1], temp[2]))[:self.n, :]
        # ratio,theta,phi -> x,y,z
        temp = sqrt(result[:, 0])
        x = a*temp*sin(result[:, 1])*cos(result[:, 2])
        y = b*temp*sin(result[:, 1])*sin(result[:, 2])
        z = c*temp*cos(result[:, 1])
        result = np.array([x, y, z]).T
        result = self._rotate_data(self.parameter, result)
        return result

    def plot(self):
        fig = plt.figure()
        ax = Axes3D(fig)
        ax.scatter3D((self.mock)[:, 0], (self.mock)[
                     :, 1], (self.mock)[:, 2], s=0.5)
        ax.set_xlabel('X')
        ax.set_ylabel('Y')
        ax.set_zlabel('Z')
        plt.show()

    @staticmethod
    def _rotate_data(x, data):
        """rotate data
        Parameters
        ----------
        x : numpy array
            shape(x) = 9, parameters in order [x0,y0,z0,a,b,c,alpha,beta,gamma]
        data : numpy array
            shape(data) = (n,3)

        Returns
        -------
        result : numpy array
            shape(result) = (n,3)

        Notes
        -----
        rotation is performed by z-y-z (right-hand)
        ..math:: R_z(\gamma)R_y(\beta)R_z(\alpha)
        """
        result = data

        alpha = x[-3]
        beta = x[-2]
        gamma = x[-1]

        cos = np.cos
        sin = np.sin

        rotation_matrix = np.array([
            [cos(alpha)*cos(beta)*cos(gamma) - sin(alpha)*sin(gamma), -cos(beta)
             * cos(gamma) * sin(alpha) - cos(alpha)*sin(gamma), cos(gamma)*sin(beta)],
            [cos(gamma)*sin(alpha) + cos(alpha)*cos(beta)*sin(gamma), cos(alpha)
             * cos(gamma)-cos(beta)*sin(alpha)*sin(gamma), sin(beta)*sin(gamma)],
            [-cos(alpha)*sin(beta), sin(alpha)*sin(beta), cos(beta)]
        ])

        result = np.matmul(rotation_matrix, result.T).T
        # translation
        result += x[:3]

        return result
