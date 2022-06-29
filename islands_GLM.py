import numpy as _np
import pandas as _pd
import matplotlib.pyplot as _plt
import scipy.stats as _stats
from scipy.optimize import minimize as _minimize
from mpl_toolkits import mplot3d as _mplot3d
from mpl_toolkits.mplot3d import axes3d as _axes3d
from scipy.special import factorial as _factorial
import statsmodels.api as _sm

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

def three_D_model_plot(x_name, y_name, z_name, intercept, x_slope, y_slope, df, model_name, y1_name, y2_name, legend_loc = (1.1,0.85), azim = 185):
    group_1 = df[df[y_name] == 1]
    group_2 = df[df[y_name] == 0]
    x = _np.linspace(_np.min(df[x_name]), _np.max(df[x_name]), 8)
    y = _np.linspace(_np.min(df[y_name]), _np.max(df[y_name]), 8)
    x, y = _np.meshgrid(x, y)
    
    if model_name == 'linear_regression_model':
        z = intercept + x_slope * x.ravel() + y_slope * y.ravel()
    
    if model_name == 'poisson_regression_model':
        z = _np.exp(intercept + x_slope * x.ravel() + y_slope * y.ravel())
    
    if model_name == 'logistic_regression_model':
        z = (_np.exp(intercept + x_slope * x.ravel() + y_slope * y.ravel()))/(1 + _np.exp(intercept + x_slope * x.ravel() + y_slope * y.ravel()))

    fig = _plt.figure(figsize = (12, 12))
    ax1 = fig.add_subplot(111, projection='3d')
    ax1.plot_wireframe(x, y,
                    z.reshape(x.shape), label = model_name, color = 'blue')
    ax1.scatter( group_1[x_name],  group_1[y_name] ,  group_1[z_name], color = 'darkred', label = y1_name  )
    ax1.scatter(group_2[x_name],  group_2[y_name] ,  group_2[z_name], color = 'darkgreen', label =  y2_name  )
    ax1.view_init(azim = azim)
    ax1.set_yticks([0,1])
    _plt.xlabel(x_name)
    _plt.ylabel(y_name)
    ax1.set_zlabel(z_name)
    ax1.legend(bbox_to_anchor = legend_loc)
    _plt.show()


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

def odds_original_intercept_plot(b0, b1, df):

	_plt.figure(figsize = (10, 5))
	_plt.subplot(1,2,1)
	_plt.title('Log scale')
	_plt.xlabel('x1')
	_plt.ylabel('log(y)')
	for intercept in [-6, -7, -8]:
    		_plt.plot(df['hormone_level_change'],
            intercept + b1 * df['hormone_level_change'],
            linewidth=1,
            linestyle=':')
    
	_plt.subplot(1,2,2)
	_plt.title('Original scale')
	_plt.xlabel('x1')
	_plt.ylabel('y')
	for intercept in [-6, -7, -8]:
    		_plt.scatter(df['hormone_level_change'],
           _np.exp(intercept + b1 * df['hormone_level_change']),
            linewidth=1,
            linestyle=':',
            label='intercept=%d' % intercept)

	_plt.legend(bbox_to_anchor = (1,1))

def odds_original_slope_plot(b0, b1, df):

	_plt.figure(figsize = (10, 5))
	_plt.subplot(1,2,1)
	_plt.title('Log scale')
	_plt.xlabel('x1')
	_plt.ylabel('log(y)')
	for slope in [0.01, 0.02, 0.025]:
    		_plt.plot(df['hormone_level_change'],
            	b0 + slope * df['hormone_level_change'],
            	linewidth=1,
            	linestyle=':')
    
	_plt.subplot(1,2,2)
	_plt.title('Original scale')
	_plt.xlabel('x1')
	_plt.ylabel('y')
	for slope in [0.01, 0.02, 0.025]:
    		_plt.scatter(df['hormone_level_change'],
            	_np.exp(b0 + slope * df['hormone_level_change']),
            	linewidth=1,
            	linestyle=':',
            	label='slope =%.3f' % slope)

	_plt.legend(bbox_to_anchor = (1,1))


def three_D_pois_reg_plot(interaction = False):

	x = _np.outer(_np.linspace(-3, 3, 32), _np.ones(32))
	y = x.copy().T # transpose
	if interaction == False:
		z = _np.exp(0.2*x + 0.3*y)
	if interaction == True:
		z = _np.exp(0.2*x + 0.3*y + -0.2*x*y)

	data_x = _np.random.choice(_np.linspace(-3, 3, 32), size = 100)
	data_y = _np.random.choice(_np.linspace(-3, 3, 32), size = 100)

	if interaction == False:
		data_z = _np.exp(0.2*data_x + 0.3*data_y + _np.random.normal(0, 0.3, size = 100))

	if interaction == True:
		data_z = _np.exp(0.2*data_x + 0.3*data_y + -0.2 *data_x*data_y + _np.random.normal(0, 0.3, size = 100))



	fig = _plt.figure(figsize = (12,12))
	ax1 = fig.add_subplot(111, projection='3d')
	ax1.plot_wireframe(x,y,z, color = 'blue', label = 'poisson regression model')
	ax1.scatter(data_x, data_y, data_z, color = 'red', label = 'data'  )
	_plt.xlabel('Predictor 1')
	_plt.ylabel('Predictor 2')
	ax1.set_zlabel('Outcome Variable')
	ax1.set_xticks([])
	ax1.set_yticks([])
	ax1.set_zticks([])
	_plt.legend(bbox_to_anchor = (1.1,0.9))
	_plt.show()

def pois_group_plot(df):

	_plt.scatter(df[df['biological_sex'] == 1]['hormone_level_change'], df[df['biological_sex'] == 1]['number_of_predation_events'], label = 'Males', color = 'darkred')
	_plt.scatter(df[df['biological_sex'] == 0]['hormone_level_change'], df[df['biological_sex'] == 0]['number_of_predation_events'], label = 'Females', color = 	'darkgreen')
	_plt.xlabel('average_hormone_change')
	_plt.ylabel('number_of_predation_events')
	_plt.legend()
	_plt.show()
    
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
	

def odds_prob_intercept_plot(b0, b1, df):

	_plt.figure(figsize = (10, 5))
	_plt.subplot(1,2,1)
	_plt.title('Log odds scale')
	_plt.xlabel('x1')
	_plt.ylabel('log odds of being in category 1')
	for intercept in [-6, -7, -8]:
   		_plt.plot(df['number_of_social_contacts'],
            	intercept + b1 * df['number_of_social_contacts'],
            	linewidth=1,
            	linestyle=':')
    
	_plt.subplot(1,2,2)
	_plt.title('Probability scale')
	_plt.xlabel('x1')
	_plt.ylabel('y')
	for intercept in [-6, -7, -8]:
    		_plt.scatter(df['number_of_social_contacts'],
            	_np.exp(intercept + b1 * df['number_of_social_contacts'])/(1 + _np.exp(intercept + b1 * df['number_of_social_contacts'])),
            	linewidth=1,
            	linestyle=':',
            	label='intercept=%d' % intercept)

	_plt.legend(bbox_to_anchor = (1,1))

def odds_prob_slope_plot(b0, b1, df):

	_plt.figure(figsize = (10, 5))
	_plt.subplot(1,2,1)
	_plt.title('Log scale')
	_plt.xlabel('x1')
	_plt.ylabel('log(y)')
	for slope in [0.3, 0.5, 0.7]:
    		_plt.plot(df['number_of_social_contacts'],
            	b0 + slope * df['number_of_social_contacts'],
            	linewidth=1,
            	linestyle=':')
    
	_plt.subplot(1,2,2)
	_plt.title('Probability scale')
	_plt.xlabel('x1')
	_plt.ylabel('y')
	for slope in [0.3, 0.5, 0.7]:
    		_plt.scatter(df['number_of_social_contacts'],
            	_np.exp(b0 + slope * df['number_of_social_contacts'])/(1 + _np.exp(b0 + slope * df['number_of_social_contacts'])),
            	linewidth=1,
            	linestyle=':',
            	label='slope =%.3f' % slope)

	_plt.legend(bbox_to_anchor = (1,1))

def bin_log_reg_plot(interaction = False):

	x_slope = 0.2
	y_slope = 3
 
	x = _np.outer(_np.linspace(-3, 3, 32), _np.ones(32))
	y = x.copy().T # transpose
	
	if interaction == False:
		lin_pop_z = x_slope*x + y_slope*y

	if interaction == True:
		lin_pop_z = x_slope*x + y_slope*y + 1*x*y
	z = _np.exp(lin_pop_z)/(1 + _np.exp(lin_pop_z))

	data_x = _np.random.choice(_np.linspace(-3, 3, 32), size = 100)
	data_y = _np.random.choice(_np.linspace(-3, 3, 32), size = 100)
	if interaction == False:
		lin_pred = x_slope*data_x + y_slope*data_y + _np.random.normal(0, 0.3, size = 100)
	if interaction == True:
		lin_pred = x_slope*data_x + y_slope*data_y + 1*data_x*data_y + _np.random.normal(0, 0.3, size = 100)
	data_z = (_np.exp(lin_pred))/(1 + _np.exp(lin_pred))
	data_z = _np.where(data_z >= 0.5, 1, 0)

	fig = _plt.figure(figsize = (12,12))
	ax1 = fig.add_subplot(111, projection='3d')
	ax1.plot_wireframe(x,y,z, color = 'blue', label = 'binary logistic regression model')
	ax1.scatter(data_x[data_z >= 0.5], data_y[data_z >= 0.5], data_z[data_z >= 0.5], color = 'red', label = 'outcome = 1')
	ax1.scatter(data_x[data_z < 0.5], data_y[data_z < 0.5], data_z[data_z < 0.5], color = 'green', label = 'outcome = 0'  )
	ax1.set_zticks([0,1])
	_plt.xlabel('Predictor 1')
	_plt.ylabel('Predictor 2')
	ax1.set_zlabel('Outcome Variable')
	ax1.set_xticks([])
	ax1.set_yticks([])
	ax1.set_zticks([])
	_plt.legend(bbox_to_anchor = (1.1,0.9))
	_plt.show()

def data_gen_multinomial(seed = 1000):
	_np.random.seed(seed)

	pop_size = 100

	religions = _np.array(['Symmetrianity', 'Communionism', 'Lamothianism'])

	income = _np.random.gamma(1, size = pop_size) * 100

	religion = _np.array([])
	bio_sex = _np.array([])

	for inc in income:
    
    		if inc < _np.median(income):
        
        		religion = _np.append(religion, _np.random.choice(['Symmetrianity', 'Communionism', 'Lamothianism'], 
                                                       p = [3.5/6, 2.4/6, 0.1/6]))
        		bio_sex = _np.append(bio_sex, _np.random.choice(['male', 'female'], p = [0.7, 0.3]))
        
    		elif inc >= _np.median(income):
        
        		religion = _np.append(religion, _np.random.choice(['Symmetrianity', 'Communionism', 'Lamothianism'], 
                                                       p = [0.5/6, 2/6, 3.5/6]))
        		bio_sex = _np.append(bio_sex, _np.random.choice(['male', 'female'], p = [0.3, 0.7]))
        
        
	df = _pd.DataFrame({'income': income.astype('int'), 'religion': religion, 'biological_sex': bio_sex})

	return df

def relig_scatter(df, legend_loc = (1.4,1)):

	relig_color = {0: 'darkblue',
              1: 'darkred',
              2: 'darkgreen'}

	fig, ax = _plt.subplots()
	ax.scatter(df['income'], df['religion_dummy'] , c = df['religion_dummy'].map(relig_color))
	ax.set_yticks([0,1,2])
	ax.set_ylabel('Religion Dummy')
	ax.set_xlabel('Income')
	ax.scatter([], [], color = 'darkblue', label =  'Communionism' )
	ax.scatter([], [],  color = 'darkred', label = 'Symmetrianity')
	ax.scatter([], [],  color = 'darkgreen', label =  'Lamothianism')
	_plt.legend(bbox_to_anchor = legend_loc)

def scatter_prob_subplots(mod, df):
	
	log_odds_predictions_1 = mod.params.loc['Intercept', 0] +  mod.params.loc['income', 0] * df['income']
	log_odds_predictions_2 = mod.params.loc['Intercept', 1] +  mod.params.loc['income', 1] * df['income']

	probability_predictions_1 = _np.exp(log_odds_predictions_1)/(1 + _np.exp(log_odds_predictions_1) + _np.exp(log_odds_predictions_2))

	probability_predictions_2 = _np.exp(log_odds_predictions_2)/(1 + _np.exp(log_odds_predictions_1) + _np.exp(log_odds_predictions_2))

	probability_predictions_0 = 1 - probability_predictions_1 - probability_predictions_2

	relig_color = {0: 'darkblue',
              	1: 'darkred',
              	2: 'darkgreen'}

	fig, ax = _plt.subplots(nrows =1, ncols=2, figsize = (16, 8))
	ax[0].scatter(df['income'], df['religion_dummy'] , c = df['religion_dummy'].map(relig_color))
	ax[0].set_yticks([0,1,2])
	ax[0].set_ylabel('Religion Dummy')
	ax[0].set_xlabel('Income')
	ax[1].scatter(df['income'], probability_predictions_0, color = 'darkblue', label =  'Communionism')
	ax[1].scatter(df['income'], probability_predictions_1, color = 'darkred', label =  'Symmetrianity')
	ax[1].scatter(df['income'], probability_predictions_2, color = 'darkgreen', label ='Lamothianism')
	ax[1].set_ylabel('Probability')
	ax[1].set_xlabel('Income')
	_plt.legend(bbox_to_anchor = (0.1,-0.1))

def three_D_model_plot_multinomial(x_name, y_name, z_name, intercept, x_slope, y_slope,  intercept_2, x_slope_2, y_slope_2, 
                                   df, legend_loc = (1.2,0)):
  
    relig_color = {0: 'darkblue',
              1: 'darkred',
              2: 'darkgreen'}
    
    x = _np.linspace(_np.min(df[x_name]), _np.max(df[x_name]), 8)
    y = _np.linspace(_np.min(df[y_name]), _np.max(df[y_name]), 8)
    x, y = _np.meshgrid(x, y)
  
        
    
    z = (_np.exp(intercept + x_slope * x.ravel() + y_slope * y.ravel()))/(1 + _np.exp(intercept + x_slope * x.ravel() + y_slope * y.ravel()) + _np.exp(intercept_2 + x_slope_2 	* x.ravel() + y_slope_2 * y.ravel()))
    z2 = (_np.exp(intercept_2 + x_slope_2 * x.ravel() + y_slope_2 * y.ravel()))/(1 + _np.exp(intercept + x_slope * x.ravel() + y_slope * y.ravel()) + _np.exp(intercept_2 + 		x_slope_2 * x.ravel() + y_slope_2 * y.ravel()))
    z3 = 1 - z - z2

    
    fig = _plt.figure(figsize = (16, 8))
    ax1 = fig.add_subplot(121, projection='3d')
    ax1.plot_wireframe(x, y,
                    z.reshape(x.shape), label =  'Symmetrianity', color = 'red', alpha = 0.5)
    ax1.plot_wireframe(x, y,
                    z2.reshape(x.shape), label = 'Lamothianism', color = 'green', alpha = 0.5)
    ax1.plot_wireframe(x, y,
                    z3.reshape(x.shape), label = 'Communionism', color = 'blue')
    ax1.view_init(azim = 30)
    ax1.set_yticks([0, 1])
    ax1.set_yticklabels(['Female', 'Male'])
    ax1.set_zticks([0,1])
    _plt.xlabel(x_name)
    _plt.ylabel('biological_sex')
    ax1.set_zlabel('Probability')
    _plt.legend(bbox_to_anchor = legend_loc)
    
    ax2 = fig.add_subplot(122, projection='3d')
    ax2.scatter(df[x_name], df[y_name], df[z_name], c = df['religion_dummy'].map(relig_color))
    ax2.view_init(azim = 30)
    ax2.set_yticks([0, 1])
    ax2.set_yticklabels(['Female', 'Male'])
    ax2.set_zticks([0,1, 2])
    _plt.xlabel(x_name)
    _plt.ylabel('biological_sex')
    ax2.set_zlabel(z_name)
 
   
def multinomial_wireframe():

	x_slope = 0.2
	y_slope = 3
 
	x = _np.outer(_np.linspace(-3, 3, 32), _np.ones(32))
	y = x.copy().T # transpose
	lin_pop_z = x_slope*x + y_slope*y
	z = _np.exp(lin_pop_z)/(1 + _np.exp(lin_pop_z))

	x_slope2 = -1
	y_slope2 = -6
 
	lin_pop_z2 = x_slope2*x + y_slope2*y
	z2 = _np.exp(lin_pop_z2)/(1 + _np.exp(lin_pop_z2))

	x_slope3 = -0.1
	y_slope3 = -1.5
 
	lin_pop_z3 = x_slope3*x + y_slope3*y
	z3 = _np.exp(lin_pop_z3)/(1 + _np.exp(lin_pop_z3))

	fig = _plt.figure(figsize = (12,12))
	ax1 = fig.add_subplot(111, projection='3d')
	ax1.plot_wireframe(x,y,z, color = 'blue', label = 'multinomial logistic regression model (category 1)')
	ax1.plot_wireframe(x,y,z2, color = 'red', label = 'multinomial logistic regression model (category 2)')
	ax1.plot_wireframe(x,y,z3, color = 'green', label = 'multinomial logistic regression model (category 3)')
	_plt.xlabel('Predictor 1')
	_plt.ylabel('Predictor 2')
	ax1.set_zlabel('Probability')
	_plt.legend(bbox_to_anchor = (1.4,0.9))


def multinomial_illustration(legend_loc = (1.3,1)):

	x_axis = _np.linspace(-500, 500)
	lin_pred_1 = -0.2535 + 0.0058 * x_axis 	
	lin_pred_2 = -0.3686 + 0.0038 * x_axis 

	pp_1 = _np.e**(lin_pred_1)/(1 + _np.e**(lin_pred_1) + _np.e**(lin_pred_2))
	pp_2 = _np.e**(lin_pred_2)/(1 + _np.e**(lin_pred_1) + _np.e**(lin_pred_2))
	pp_ref = 1 - pp_1 - pp_2


	_plt.plot(x_axis, pp_1, label = 'Category A', color = 'purple')
	_plt.plot(x_axis, pp_2, label = 'Category B', color = 'red')
	_plt.plot(x_axis, pp_ref, label = 'Category C', color = 'blue')
	_plt.xticks([])
	_plt.yticks([])
	_plt.xlabel('Predictor')
	_plt.ylabel('Predicted Probability')
	_plt.legend(bbox_to_anchor = legend_loc)
	
