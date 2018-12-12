import sys
import numpy as np
import matplotlib.pyplot as plt
from csv_to_pandas import csv_to_pandas

folder_path = sys.argv[1]
file_name = 'run_log.csv'

data = csv_to_pandas(folder_path, file_name)

grid_comparison = data.loc[data['order'] == 4]
order_comparison = data.loc[data['block0np0'] == 100]

# Plot error v h.
grid_comparison = grid_comparison.sort_values('Delta0Block0')
plt.plot(grid_comparison['Delta0Block0'], grid_comparison['L_1'], 'k-', label='$L_{1}$')
plt.plot(grid_comparison['Delta0Block0'], grid_comparison['L_inf'], 'k--', label='$L_{\infty}$')
plt.xlabel('$\Delta h$')
plt.xlim(0,grid_comparison['Delta0Block0'].max())
plt.ylabel('Error')
plt.legend()
plt.savefig('grid_comparison.png')
plt.gcf().clear()

# Plot log error v log h.
plt.loglog(grid_comparison['Delta0Block0'], grid_comparison['L_1'], 'k-', label='$L_{1}$')
plt.loglog(grid_comparison['Delta0Block0'], grid_comparison['L_inf'], 'k--', label='$L_{\infty}$')
plt.xlabel('$\Delta h$')
plt.ylabel('Error')
plt.legend()
plt.savefig('grid_comparison_log.png')
plt.gcf().clear()

# Plot error v h. (without N=800)
grid_comparison = grid_comparison.loc[grid_comparison['block0np0'] != 800]
grid_comparison = grid_comparison.sort_values('Delta0Block0')
plt.plot(grid_comparison['Delta0Block0'], grid_comparison['L_1'], 'k-', label='$L_{1}$')
plt.plot(grid_comparison['Delta0Block0'], grid_comparison['L_inf'], 'k--', label='$L_{\infty}$')
plt.xlabel('$\Delta h$')
plt.xlim(0,grid_comparison['Delta0Block0'].max())
plt.ylabel('Error')
plt.legend()
plt.savefig('grid_comparison2.png')
plt.gcf().clear()

# Calculate line of best fit.
log_error_1 = np.log(grid_comparison['L_1'])
log_error_inf = np.log(grid_comparison['L_inf'])
log_h = np.log(grid_comparison['Delta0Block0'])
m_1, c_1 = np.polyfit(log_h, log_error_1, 1)
m_inf, c_inf = np.polyfit(log_h, log_error_inf, 1)
text_eqn_1 = 'log($L_{1}$) = %.3flog($\Delta$h) + %.3f' % (np.round(m_1, 3), np.round(c_1, 3))
text_eqn_inf = 'log($L_{\infty}$) = %.3flog($\Delta$h) + %.3f' % (np.round(m_inf, 3), np.round(c_inf, 3))

# Plot log error v log h. (without N=800)
plt.loglog(grid_comparison['Delta0Block0'], grid_comparison['L_1'], 'k-', label='$L_{1}$')
plt.loglog(grid_comparison['Delta0Block0'], grid_comparison['L_inf'], 'k--', label='$L_{\infty}$')
plt.xlabel('$\Delta h$')
plt.ylabel('Error')
plt.legend()
plt.text(0.10, 0.75, text_eqn_1, transform=plt.gca().transAxes)
plt.text(0.10, 0.65, text_eqn_inf, transform=plt.gca().transAxes)
plt.savefig('grid_comparison_log2.png')
plt.gcf().clear()
