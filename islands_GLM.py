import numpy as _np
import pandas as _pd
import matplotlib.pyplot as _plt
import scipy.stats as _stats
from scipy.optimize import minimize as _minimize
from mpl_toolkits import mplot3d as _mplot3d
from mpl_toolkits.mplot3d import axes3d as _axes3d

# just some convenience functions (for plots etc.) for this textbook
# may contain redundant unused functions, as there have been several revisions
# and I may not have removed the old functions!

def r_ify(fig_size = (7,6)):

    "Make plots look like R!"
    _plt.style.use('ggplot')
    _plt.rc('axes', facecolor='white', edgecolor='black',
       axisbelow=True, grid=False)
    _plt.rcParams['figure.figsize'] = fig_size


def normal_pdf(y, y_hat, var):
  
  output = 1/_np.sqrt(var**2 * 2 * _np.pi) * _np.e**(-(y - y_hat)**2/(2*var**2))
  
  return output


def normal_plot():
  
    _np.random.seed(2000)
    n_plots = 4
    mus = _np.random.choice(_np.arange(-20,21), replace = False, size = n_plots)
    sigmas = _np.random.choice(_np.arange(1,11), replace = False, size = n_plots)
    y = _np.arange(-50,50)
    _plt.figure(figsize = (10,10))

    _plt.plot(y, normal_pdf(y, 0, 5)*100,  label = r'$\hat{y_i} $ = '+str(0)+ r' and $\sigma$ = '+str(5)) 

    for i in _np.arange(len(mus)):

        prob = normal_pdf(y, mus[i], sigmas[i])

        _plt.plot(y,prob*100, label = r'$\hat{y_i} $ = '+str(mus[i])+ r' and $\sigma$ = '+str(sigmas[i])) 
        _plt.xlabel('Y')
        _plt.ylabel('Probability %')

    _plt.legend()
    _plt.show()
    

def prestige_wealth_df():

    _np.random.seed(3000)
    
    pop_size = 10000
    
    samp_size = 1000
    
    wealth_pop = _np.round(_np.random.normal(170,10, size = pop_size), 2)
    
    religion_pop = _np.random.choice([0,1], p = [0.3, 0.7], size = pop_size)
    
    prestige_pop = 10 + 4 * wealth_pop + -100 * religion_pop + _np.random.normal(0, 60, size = pop_size)
    
    prestige_pop = prestige_pop.astype('int')
    
    random_ilocs = _np.random.choice(_np.arange(pop_size), samp_size)
    
    wealth = wealth_pop[random_ilocs]
    
    prestige = prestige_pop[random_ilocs]
    
    religion = religion_pop[random_ilocs]
    
    df = _pd.DataFrame({'wealth': wealth, 'religion': religion, 'prestige': prestige})
    
    return wealth_pop, religion_pop, prestige_pop, df


def plot_prestige_wealth(wealth, prestige):

    _plt.figure()
    _plt.scatter(wealth, prestige)
    _plt.xlabel('Wealth')
    _plt.ylabel('Prestige')

    
def plot_prestige_wealth_with_prediction(wealth, prestige, predictions):

    _plt.figure()
    _plt.scatter(wealth, prestige)
    _plt.xlabel('Wealth')
    _plt.ylabel('Prestige')
    _plt.scatter(wealth, predictions, label = 'predicted prestige scores (linear regression)')
    _plt.legend(bbox_to_anchor = (1,1))
    _plt.show()  


def three_D_lin_reg_plot():

	x = _np.outer(_np.linspace(-3, 3, 32), _np.ones(32))
	y = x.copy().T # transpose
	z = 2*x + 3*y

	data_x = _np.random.choice(_np.linspace(-3, 3, 32), size = 100)
	data_y = _np.random.choice(_np.linspace(-3, 3, 32), size = 100)
	data_z = 2*data_x + 3*data_y + _np.random.normal(0, 3, size = 100)

	fig = _plt.figure(figsize = (12,12))
	ax1 = fig.add_subplot(111, projection='3d')
	ax1.plot_wireframe(x,y,z, color = 'blue', label = 'linear regression model')
	ax1.scatter(data_x, data_y, data_z, color = 'red', label = 'data'  )
	_plt.xlabel('Predictor 1')
	_plt.ylabel('Predictor 2')
	ax1.set_zlabel('Outcome Variable')
	ax1.set_xticks([])
	ax1.set_yticks([])
	ax1.set_zticks([])
	_plt.legend(bbox_to_anchor = (1.1,0.9))


def plot_prestige_wealth_with_religion(df):

	_plt.scatter(df[df['religion'] == 1]['wealth'], df[df['religion'] == 1]['prestige'], label = 'Religion 1', color = 'darkred')
	_plt.scatter(df[df['religion'] == 0]['wealth'], df[df['religion'] == 0]['prestige'], label = 'Religion 2', color = 'darkgreen')
	_plt.xlabel('Wealth')
	_plt.ylabel('Prestige')
	_plt.legend()
	_plt.show()


def three_D_prestige_wealth_religion_plot(df):

	religion_1 = df[df['religion'] == 1]
	religion_2 = df[df['religion'] == 0]

	fig = _plt.figure(figsize = (12, 12))
	ax1 = fig.add_subplot(111, projection='3d')
	ax1.scatter(religion_1['wealth'], religion_1['religion'] , religion_1['prestige'], color = 'darkred', label = 'religion 1'  )
	ax1.scatter(religion_2['wealth'], religion_2['religion'] , religion_2['prestige'], color = 'darkgreen', label = 'religion 2'  )
	ax1.set_yticks([0,1])
	_plt.xlabel('Wealth')
	_plt.ylabel('Religion')
	ax1.set_zlabel('Prestige')
	ax1.view_init(azim = 185)
	_plt.legend(bbox_to_anchor = (1.2,1))

def three_D_prestige_wealth_religion_plot_with_surface(df, lin_reg_model):

	religion_1 = df[df['religion'] == 1]
	religion_2 = df[df['religion'] == 0]

	intercept, religion_slope, height_slope = lin_reg_model.params[:3]

	height_x = _np.linspace(_np.min(df['wealth']), _np.max(df['wealth']), 8)
	religion_y = _np.linspace(_np.min(df['religion']), _np.max(df['religion']), 8)
	height_x, religion_y = _np.meshgrid(height_x, religion_y)
	prestige_z = intercept + height_slope * height_x.ravel() + religion_slope * religion_y.ravel()

	fig = _plt.figure(figsize = (12, 12))
	ax1 = fig.add_subplot(111, projection='3d')
	ax1.plot_wireframe(height_x, religion_y,
                prestige_z.reshape(height_x.shape), label = 'linear regression model', color = 'blue')
	ax1.scatter(religion_1['wealth'], religion_1['religion'] , religion_1['prestige'], color = 'darkred', label = 'religion 1'  )
	ax1.scatter(religion_2['wealth'], religion_2['religion'] , religion_2['prestige'], color = 'darkgreen', label = 'religion 2'  )
	ax1.view_init(azim = 185)
	ax1.set_yticks([0,1])
	_plt.xlabel('Wealth')
	_plt.ylabel('Religion')
	ax1.set_zlabel('Prestige')
	ax1.legend(bbox_to_anchor = (1.1,0.85))

   