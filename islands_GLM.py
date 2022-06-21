import numpy as _np
import pandas as _pd
import matplotlib.pyplot as _plt
import scipy.stats as _stats
from scipy.optimize import minimize as _minimize
from mpl_toolkits import mplot3d as _mplot3d
from mpl_toolkits.mplot3d import axes3d as _axes3d
from scipy.special import factorial as _factorial

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
  
  
def normal_labels():
    _plt.xlabel('Y')
    _plt.ylabel('Probability %')
    _plt.legend()


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

    _np.random.seed(300)
    
    pop_size = 10000
    
    samp_size = 1000
    
    wealth_pop = _np.round(_np.random.normal(170,10, size = pop_size), 2)
    
    religion_pop = _np.random.choice([0,1], p = [0.3, 0.7], size = pop_size)
    
    prestige_pop = 10 + 4 * wealth_pop + -100 * religion_pop + _np.random.normal(0, 40, size = pop_size)
    
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


def three_D_lin_reg_plot(interaction = False):

	x = _np.outer(_np.linspace(-3, 3, 32), _np.ones(32))
	y = x.copy().T # transpose

	if interaction == False:
		z = 2*x + 3*y
	if interaction == True:
		z = 2*x + 3*y + x*y

	data_x = _np.random.choice(_np.linspace(-3, 3, 32), size = 100)
	data_y = _np.random.choice(_np.linspace(-3, 3, 32), size = 100)
		
	if interaction == False:
		data_z = 2*data_x + 3*data_y + _np.random.normal(0, 3, size = 100)

	if interaction == True:
		data_z = 2*data_x + 3*data_y + data_x*data_y + _np.random.normal(0, 3, size = 100)

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

	_plt.scatter(df[df['religion'] == 1]['wealth'], df[df['religion'] == 1]['prestige'], label = 'religion A', color = 'darkred')
	_plt.scatter(df[df['religion'] == 0]['wealth'], df[df['religion'] == 0]['prestige'], label = 'religion B', color = 'darkgreen')
	_plt.xlabel('Wealth')
	_plt.ylabel('Prestige')
	_plt.legend()


def three_D_prestige_wealth_religion_plot(df, azim = 185):

	religion_1 = df[df['religion'] == 1]
	religion_2 = df[df['religion'] == 0]

	fig = _plt.figure(figsize = (12, 12))
	ax1 = fig.add_subplot(111, projection='3d')
	ax1.scatter(religion_1['wealth'], religion_1['religion'] , religion_1['prestige'], color = 'darkred', label = 'religion A'  )
	ax1.scatter(religion_2['wealth'], religion_2['religion'] , religion_2['prestige'], color = 'darkgreen', label = 'religion B'  )
	ax1.set_yticks([0,1])
	_plt.xlabel('Wealth')
	_plt.ylabel('Religion')
	ax1.set_zlabel('Prestige')
	ax1.view_init(azim = azim)
	_plt.legend(bbox_to_anchor = (1.2,1))

def three_D_prestige_wealth_religion_plot_with_surface(df, lin_reg_model, azim = 185):

	religion_1 = df[df['religion'] == 1]
	religion_2 = df[df['religion'] == 0]

	intercept = lin_reg_model.params['Intercept']
	religion_slope = lin_reg_model.params['religion']
	wealth_slope = lin_reg_model.params['wealth']


	wealth_x = _np.linspace(_np.min(df['wealth']), _np.max(df['wealth']), 8)
	religion_y = _np.linspace(_np.min(df['religion']), _np.max(df['religion']), 8)
	wealth_x, religion_y = _np.meshgrid(wealth_x, religion_y)
	prestige_z = intercept + wealth_slope * wealth_x.ravel() + religion_slope * religion_y.ravel()

	fig = _plt.figure(figsize = (12, 12))
	ax1 = fig.add_subplot(111, projection='3d')
	ax1.plot_wireframe(wealth_x, religion_y,
                prestige_z.reshape(wealth_x.shape), label = 'linear regression model', color = 'blue')
	ax1.scatter(religion_1['wealth'], religion_1['religion'] , religion_1['prestige'], color = 'darkred', label = 'religion A'  )
	ax1.scatter(religion_2['wealth'], religion_2['religion'] , religion_2['prestige'], color = 'darkgreen', label = 'religion B'  )
	ax1.view_init(azim = azim)
	ax1.set_yticks([0,1])
	_plt.xlabel('Wealth')
	_plt.ylabel('Religion')
	ax1.set_zlabel('Prestige')
	ax1.legend(bbox_to_anchor = (1.1,0.85))

def three_D_prestige_wealth_religion_plot_with_surface_with_int(df, lin_reg_model, azim = 185):

	religion_1 = df[df['religion'] == 1]
	religion_2 = df[df['religion'] == 0]

	intercept = lin_reg_model.params['Intercept']
	religion_slope = lin_reg_model.params['religion']
	wealth_slope = lin_reg_model.params['wealth']	
	interaction_slope = lin_reg_model.params['religion:wealth']	

	wealth_x = _np.linspace(_np.min(df['wealth']), _np.max(df['wealth']), 8)
	religion_y = _np.linspace(_np.min(df['religion']), _np.max(df['religion']), 8)
	wealth_x, religion_y = _np.meshgrid(wealth_x, religion_y)
	prestige_z = intercept + wealth_slope * wealth_x.ravel() + religion_slope * religion_y.ravel() + interaction_slope * wealth_x.ravel() * religion_y.ravel()

	fig = _plt.figure(figsize = (12, 12))
	ax1 = fig.add_subplot(111, projection='3d')
	ax1.plot_wireframe(wealth_x, religion_y,
                prestige_z.reshape(wealth_x.shape), label = 'linear regression model', color = 'blue')
	ax1.scatter(religion_1['wealth'], religion_1['religion'] , religion_1['prestige'], color = 'darkred', label = 'religion A'  )
	ax1.scatter(religion_2['wealth'], religion_2['religion'] , religion_2['prestige'], color = 'darkgreen', label = 'religion B'  )
	ax1.view_init(azim = azim)
	ax1.set_yticks([0,1])
	_plt.xlabel('Wealth')
	_plt.ylabel('Religion')
	ax1.set_zlabel('Prestige')
	ax1.legend(bbox_to_anchor = (1.1,0.85))



def poisson_func(lambd, x):
  
    return (lambd**x * _np.exp(-lambd))/_factorial(x)
    
def poisson_plot(lambdas, max_x):

    x_values = _np.linspace(0, max_x)
    
    for lambd in lambdas:
        _plt.plot(x_values, poisson_func(lambd, x_values), label = 'mean = %.d' % lambd)
        _plt.xlabel('Count')
        _plt.ylabel('Probability')
        _plt.legend();
   
def generate_poisson_data():

    pop_size = 100
    
    _np.random.seed(100)
    
    hormone_level = _np.random.normal(0, 80, pop_size)

    bio_sex = _np.random.choice([0,1], size = pop_size)

    number_of_predation_events =  _np.random.uniform(low =0.03, high = 0.05) * hormone_level + 5 * bio_sex

    error = _np.random.poisson(3, pop_size)


    number_of_predation_events = number_of_predation_events  + error

    number_of_predation_events[number_of_predation_events < 0] = _np.random.poisson(3, 
                                                                                   len(number_of_predation_events[number_of_predation_events < 0]))


    df = _pd.DataFrame({'hormone_level_change': _np.round(hormone_level,2),
                       'biological_sex': bio_sex,
                       'number_of_predation_events': number_of_predation_events.astype('int') })
                       
    return df
    
def count_hist(df):
    _plt.hist(df['number_of_predation_events'])
    _plt.xlabel('Number of Predation Events')
    _plt.ylabel('Frequency');
    
def hormone_predation_plot(df, with_predictions = False, predictions = None, model_name = 'linear regression', log_scale = False):

    if log_scale == False:

        _plt.scatter(df['hormone_level_change'], df['number_of_predation_events'])
        _plt.xlabel('Hormone Level Change')
        _plt.ylabel('Number of Predation Events')
    
        if with_predictions == True:
            _plt.scatter(df['hormone_level_change'], predictions, color = 'gold', label = 'predicted number of predation events\n('+model_name+')')
            _plt.legend(bbox_to_anchor = (1,1))
            
    if log_scale == True:
    
        _plt.scatter(df['hormone_level_change'], _np.log(df['number_of_predation_events']))
        _plt.xlabel('Hormone Level Change')
        _plt.ylabel('Number of Predation Events \n(log scale)')
    
        if with_predictions == True:
            _plt.scatter(df['hormone_level_change'], predictions, color = 'gold', label = 'predicted number of predation events\n('+model_name+')')
            _plt.legend(bbox_to_anchor = (1,1))
    
def addiction_data_gen():

	_np.random.seed(200)

	slopes =  [-2, 1.2] # <-- this works!

	pop_size = 100

	number_of_social_contacts = _np.random.gamma(5, size = pop_size) 

	number_of_social_contacts[number_of_social_contacts > _np.quantile(number_of_social_contacts, .25)] = number_of_social_contacts[number_of_social_contacts > 	_np.quantile(number_of_social_contacts, .25)] + _np.abs(_np.random.normal(0, 5, size = len(number_of_social_contacts[number_of_social_contacts > _np.quantile			(number_of_social_contacts, .25)])))
        
	drug_alone = _np.random.choice([0,1], p = [0.5, 0.5], size = pop_size) 

	linear_predictor =  slopes[0] * number_of_social_contacts + slopes[1] * drug_alone + _np.random.normal(0, 20, size = pop_size)

	addiction_p = (_np.exp(linear_predictor))/(1 + _np.exp(linear_predictor))

	addiction_status = _np.repeat('', len(addiction_p))
	addiction_status = _np.where(addiction_p >= 0.5, 'addict', addiction_status)
	addiction_status = _np.where(addiction_p < 0.5, 'not_addict', addiction_status)

        
	df = _pd.DataFrame({'number_of_social_contacts': number_of_social_contacts.astype('int'), 'drug_alone': drug_alone,
                   'addiction_status': addiction_status})

	df['addiction_status'] =  df['addiction_status'].replace(['addict', 'not_addict'], [1,0])

	return df

def addiction_plot(df, predictions = [], plot_predictions = False, log_scale = False, plot_other = False):

	addiction_color = {0: 'blue',
              1: 'red'}

	if log_scale == False:
		fig, ax = _plt.subplots()
		ax.scatter(df['number_of_social_contacts'], df['addiction_status'] , c = df['addiction_status'].map(addiction_color))
		ax.set_yticks([0,1])
		ax.set_ylabel('Addiction Status\n (1 == Addict)')
		ax.set_xlabel('Number of Social Contacts')
		ax.scatter([], [], color = 'blue', label = 'NOT addict' )
		ax.scatter([], [],  color = 'red', label = 'Addict')
		if plot_predictions == True:
			ax.scatter(df['number_of_social_contacts'], predictions, color = 'darkred', label = 'Predicted probability of being an addict')

		if plot_other == True:
			ax.scatter(df['number_of_social_contacts'], 1 - predictions, color = 'black', label = 'Predicted probability of NOT being an addict')

	if log_scale == True:
		fig, ax = _plt.subplots()
		ax.scatter(df['number_of_social_contacts'], predictions, color = 'darkred', label = 'Predicted log odds of being an addict')
		ax.set_ylabel('Log Odds (Addiction Status == 1)')
		ax.set_xlabel('Number of Social Contacts')


	_plt.legend(bbox_to_anchor = (1,1))
	