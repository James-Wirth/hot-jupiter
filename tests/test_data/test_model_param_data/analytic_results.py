from scipy import integrate
from hjmodel.config import *

class Analytic:
    def __init__(self, n_tot, sigma_v, e_init, a_init, m1, m2, total_time):
        self.n_tot = n_tot
        self.sigma_v = sigma_v
        self.m1 = m1
        self.m2 = m2
        self.q = m2/m1
        self.e_init = e_init
        self.a_init = a_init
        self.total_time = total_time
        self.R_td = ETA * R_P * (m1/m2)**(1/3)

        self.m_star_ion = integrate.quad(self.m_star_ion_integrand, M_MIN, M_MAX)[0]
        self.m_hyp = integrate.quad(self.m_hyp_integrand, M_MIN, M_MAX)[0]

        self.Gamma_tide = None
        self.f_td = None
        self.x_max = None
        self.e_td = None
        self.e_age = None
        self.e_min = None
        self.l_age = None
        self.l_max = None
        self.Gamma_ion = None
        self.gamma = None
        self.set_vars()

    def m_star_ion_integrand(self, m_pert):
        return (1 + self.q_pert(m_pert) + self.q) * self.q_pert(m_pert)**(1/3) * self.xi(m_pert)

    def m_hyp_integrand(self, m_pert):
        return self.q_pert(m_pert) * self.xi(m_pert)

    def q_pert(self, m_pert):
        return m_pert/((1+self.q)*self.m1)

    def xi(self, m_pert):
        alpha = 1.8 * (4 * M_BR ** 0.6 - 3 * M_MIN ** 0.6 - M_BR ** 2.4 * M_MAX ** (-1.8)) ** (-1)
        beta = alpha * M_BR ** 2.4
        if M_MIN <= m_pert < M_BR:
            return alpha * m_pert**(-0.4)
        elif M_BR <= m_pert < M_MAX:
            return beta * m_pert**(-2.8)
        else:
            return 0

    def f(self, e):
        num_coeffs = np.array([1, 45/14, 8, 685/224, 255/488, 25/1792])
        denom_coeffs = np.array([1, 3, 3/8])
        vec = np.array([e**(2*i) for i in range(6)])
        return np.dot(num_coeffs, vec)/np.dot(denom_coeffs, vec[:3])

    def set_vars(self):
        self.gamma = 0.046 * np.sqrt(1+self.q) * self.n_tot * self.m_hyp * (self.a_init/5)**(3/2)
        self.Gamma_ion = 0.028 * self.m_star_ion * (self.a_init/5) * self.n_tot * (self.sigma_v/2.108)**(-1)

        # l_max
        tau_delta = 1.04E3
        C = (1 + tau_delta * self.gamma) ** (-1)
        self.l_max = 0.02 * (1.9 * C * (self.a_init / 5) ** (-2) * (self.gamma / 10 ** (-4)) ** (-1)) ** (1 / 6)

        # l_age
        self.l_age = 0.02 * (2400 * self.f(self.e_init) * (self.a_init/5)**(-1/2) * self.e_init * np.sqrt(((1-self.e_init**2)**2)/4 + self.e_init**2))**(2/15)

        # critical e vals
        self.e_min = np.sqrt(1-self.l_max/self.a_init)
        self.e_age = np.sqrt(1-self.l_age/self.a_init)
        self.e_td = np.sqrt(1-self.R_td/self.a_init)

        # helper variables
        self.x_max = (self.l_max/(1+self.e_init))/self.R_td
        self.f_td = 1-np.exp(-1/self.x_max)

        # Gamma_tide
        e_tide = min(self.e_td, max(self.e_min, self.e_age))
        self.Gamma_tide = self.gamma * self.e_init * np.sqrt(1 - self.e_init) / (2 * (e_tide - self.e_init))

    def get_analytic_probabilities(self):
        P_tide = (self.Gamma_tide / (self.Gamma_tide + self.Gamma_ion)) * (1 - np.exp(-(self.Gamma_ion + self.Gamma_tide) * self.total_time))
        ion_frac = (self.Gamma_ion / (self.Gamma_tide + self.Gamma_ion)) * (1 - np.exp(-(self.Gamma_ion + self.Gamma_tide) * self.total_time))

        hj_frac = (1 - self.f_td) * P_tide
        td_frac = self.f_td * P_tide
        nm_frac = 1 - ion_frac - hj_frac - td_frac
        return [nm_frac, ion_frac, td_frac, hj_frac]

if __name__ == '__main__':
    anal = Analytic(n_tot=2.5E-3, sigma_v=1.266, m1=1, m2=1E-3, e_init=0.3, a_init=1, total_time=10000)
    print(anal.get_analytic_probabilities())