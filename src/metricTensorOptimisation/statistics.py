from statsmodels.stats.power import TTestPower
import numpy as np

# For the case of a cohort of 10 participants, we expect an error of 6 (see paper)
# Thus the null hypothesis is that the optimisation won't do anything and we see an error of 10, and the alternative is an error of 6
# That is mu_0 = 10, mu_1 = 6
mu_0 = 10
mu_1 = 6

# From pilot test of 20 cycles we get the errors:
# [6, 6, 6, 6, 6, 6, 6, 6, 6, 6, 6, 6, 6, 7, 6, 6, 6, 6, 6, 6]
# Which has an std of ~0.22
std = 0.22

# We can set the significance level alpha to 0.05 and power to 0.9
alpha = 0.05
power = 0.9

# Using Cohen's f as effect size:
cohens_d = (mu_1 - mu_0) / std
cohens_f = np.sqrt(cohens_d**2 / 2 * 1)

analysis = TTestPower()

required_n = analysis.solve_power(effect_size=cohens_f, alpha=alpha, power=power)

print(f"{required_n} cycles should be ran to show a statistically significant reduction in error.")

