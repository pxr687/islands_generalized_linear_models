import numpy as _np
import pandas as _pd
import matplotlib.pyplot as _plt
import scipy.stats as _stats
from scipy.optimize import minimize as _minimize

def r_ify(fig_size = (7,6)):
    "Make plots look like R!"
    _plt.style.use('ggplot')
    _plt.rc('axes', facecolor='white', edgecolor='black',
       axisbelow=True, grid=False)
    _plt.rcParams['figure.figsize'] = fig_size
    
def good_plot_bad_plot():
    _np.random.seed(3000)

    x = _np.random.normal(100, 10, 100)

    y = 2*x + _np.random.normal(0,20, 100)

    lin_reg = _stats.linregress(x,y)
    preds = lin_reg[1] + lin_reg[0] * x 
    other =  160 + -0.2 * x

    _plt.figure(figsize = (16,6))
    _plt.subplot(1,2,1)
    _plt.scatter(x,y, color = 'blue')
    _plt.plot(x, preds, color = 'red')
    _plt.ylabel('Y variable')
    _plt.xlabel('X variable')
    _plt.xticks([], [])
    _plt.yticks([], [])

    _plt.subplot(1,2,2)
    _plt.scatter(x,y, color = 'blue')
    _plt.plot(x, other, color = 'darkred')
    _plt.ylabel('Y variable')
    _plt.xlabel('X variable')
    _plt.xticks([], [])
    _plt.yticks([], [])
    _plt.show()

def prestige_height_df():
    _np.random.seed(3000)
    
    pop_size = 10000
    
    samp_size = 1000
    
    height_pop = _np.round(_np.random.normal(170,10, size = pop_size), 2)
    
    religion_pop = _np.random.choice([0,1], p = [0.3, 0.7], size = pop_size)
    
    prestige_pop = 10 + 4 * height_pop + 30 * religion_pop + _np.random.normal(0, 30, size = pop_size)
    
    prestige_pop = prestige_pop.astype('int')
    
    random_ilocs = _np.random.choice(_np.arange(pop_size), samp_size)
    
    height = height_pop[random_ilocs]
    
    prestige = prestige_pop[random_ilocs]
    
    religion = religion_pop[random_ilocs]
    
    df = _pd.DataFrame({'height': height, 'religion': religion, 'prestige': prestige})
    
    return height_pop, religion_pop, prestige_pop, df

def plot_prestige_height(height, prestige):
    _plt.figure()
    _plt.scatter(height, prestige)
    _plt.xlabel('Height')
    _plt.ylabel('Prestige')
    _plt.show()
    
    
def plot_prestige_height_with_prediction(height, prestige, predictive_value):
    _plt.figure()
    _plt.scatter(height, prestige)
    _plt.xlabel('Height')
    _plt.ylabel('Prestige')
    _plt.scatter(height, predictive_value*height, label = 'predicted prestige scores')
    _plt.legend(bbox_to_anchor = (1,1))
    _plt.show()  
    
    
def minimize_plot(df):

    intercepts = []
    slopes = []
    SSEs = []

    def sos_error_for_minimize(intercept_and_slope):
        intercept = intercept_and_slope[0]
        slope = intercept_and_slope[1]
        predicted = intercept + df['height'] * slope
        error = df['prestige'] - predicted
        intercepts.append(intercept)
        slopes.append(slope)
        SSEs.append(_np.sum(error ** 2))
        return _np.sum(error ** 2)

    min_res = _minimize(sos_error_for_minimize, [1, 1])
    best_predicted = min_res.x[0] + min_res.x[1] * df['height']

    tracker = _pd.DataFrame({ 'intercept': _np.round(intercepts, 3), 'slope': _np.round(slopes,3), 'SSE': _np.round(SSEs, 3)})
    tracker_filt = tracker.drop_duplicates(subset = 'slope')
    tracker_filt = tracker_filt.sort_values(by = 'SSE', ascending = False)
                                                     
    fig = _plt.figure(figsize =(14,6))
    ax1 = fig.add_subplot(121)
    ax2 = fig.add_subplot(122)
    fig2 = _plt.figure(figsize = (14,6))
    ax3 = fig2.add_subplot(121)
    ax_best = fig2.add_subplot(122)
    sub_plots = [ax1, ax2, ax3, ax_best]


    for i in _np.arange(len(sub_plots)):
        if i != len(sub_plots)-1:
            sub_plots[i].scatter(df['height'], df['prestige'], color = 'red', label = 'actual data')
            predicted = tracker_filt['intercept'].iloc[i] + tracker_filt['slope'].iloc[i] * df['height']
            sub_plots[i].scatter(df['height'], predicted, color = 'blue', label = 'predicted data')
            for ind in _np.arange(len(df['height'])):
                x = df['height'].iloc[ind]
                y_0 = predicted.iloc[ind]
                y_1 = df['prestige'].iloc[ind]
                sub_plots[i].plot([x, x], [y_0, y_1], '--', color='orange', linewidth=1, alpha = 0.6)
            sub_plots[i].plot([x, x], [y_0, y_1], '--', color='orange', linewidth=2, alpha = 0.6, label = 'prediction error')
            sub_plots[i].legend()
            sub_plots[i].set_xlabel('Height')
            sub_plots[i].set_ylabel('Prestige')
            sub_plots[i].set_title('Intercept = '+str(_np.round(tracker_filt['intercept'].iloc[i],2))+' \nSlope = '+str(_np.round(tracker_filt['slope'].iloc[i],2))+' \nSSE = '+str(_np.round(tracker_filt['SSE'].iloc[i],2)))
            sub_plots[i].set_ylim([100, 1100])
        else:
            ax_best.scatter(df['height'], df['prestige'], color = 'red', label = 'actual data')
            ax_best.scatter(df['height'], best_predicted, color = 'blue', label = 'predicted data')
            for ind in _np.arange(len(df['height'])):
                x = df['height'].iloc[ind]
                y_0 = best_predicted.iloc[ind]
                y_1 = df['prestige'].iloc[ind]
                ax_best.plot([x, x], [y_0, y_1], '--', color='orange', linewidth=1, alpha = 0.6)
            ax_best.plot([x, x], [y_0, y_1], '--', color='orange', linewidth=2, alpha = 0.6, label = 'prediction error')
            ax_best.legend()
            ax_best.set_xlabel('Height')
            ax_best.set_ylabel('Prestige')
            ax_best.set_title('Intercept = '+str(_np.round(min_res.x[0],2))+' \nSlope = '+str(_np.round(min_res.x[1],2))+' \nSSE = '+str(_np.round(min_res.fun,2)))
            ax_best.set_ylim([100, 1100])

        
    _plt.show()
    print('The intercept and slope of the best fitting line =', min_res.x)
    
def prestige_hist(prestige):
    _plt.figure()
    _plt.hist(prestige, color = 'blue')
    _plt.xlabel('Prestige')
    _plt.ylabel('Number of Scores')
    _plt.show()

    
def split_plot_prestige(height, prestige, hist_reverse = False, scatter = False, no_unconditional = False):
        
    height = _np.array(height)
    prestige = _np.array(prestige)
        
    n = len(height)

    group_1 = _np.where(height < _np.quantile(height, q = 1/3))
    group_2 = _np.where((height > _np.quantile(height, q = 1/3)) & (height < _np.quantile(height, q = 2/3)) )
    group_3 = _np.where((height > _np.quantile(height, q = 2/3)) & (height < _np.quantile(height, q = 3/3)) )
        
    if no_unconditional == False:
        
        if hist_reverse == False:
            _plt.figure()
            _plt.hist(prestige, color = 'blue',  label = 'prestige scores, ignoring height')
            _plt.hist(prestige[group_1], color = 'orange', label = 'prestige scores associated with LOW values of height')
            _plt.hist(prestige[group_2], color = 'green', label = 'prestige scores associated with MEDIUM values of height')
            _plt.hist(prestige[group_3], color = 'crimson', label = 'prestige scores associated with HIGH values of height')
            _plt.legend(bbox_to_anchor = (1, 1 ))
            _plt.xlabel('Prestige')
            _plt.ylabel('Number of Scores')
            _plt.show()
                
        if hist_reverse == True:
            _plt.figure()
            _plt.hist(prestige, color = 'blue',  label = 'prestige scores, ignoring height')
            _plt.hist(prestige[group_3], color = 'crimson', label = 'prestige scores associated with HIGH values of height')
            _plt.hist(prestige[group_2], color = 'green', label = 'prestige scores associated with MEDIUM values of height')
            _plt.hist(prestige[group_1], color = 'orange', label = 'prestige scores associated with LOW values of height')
            _plt.legend(bbox_to_anchor = (1, 1 ))
            _plt.xlim((_np.max(prestige)+5, _np.min(prestige)-5))
            _plt.xlabel('Prestige')
            _plt.ylabel('Number of Scores')
            _plt.show()
                
    if no_unconditional == True:
    
        if hist_reverse == False:
            _plt.figure()
            _plt.hist(prestige[group_1], color = 'orange', label = 'prestige scores associated with LOW values of height')
            _plt.hist(prestige[group_2], color = 'green', label = 'prestige scores associated with MEDIUM values of height')
            _plt.hist(prestige[group_3], color = 'crimson', label = 'prestige scores associated with HIGH values of height')
            _plt.legend(bbox_to_anchor = (1, 1 ))
            _plt.xlabel('Prestige')
            _plt.ylabel('Number of Scores')
            _plt.show()
                
        if hist_reverse == True:
            _plt.figure()
            _plt.hist(prestige[group_3], color = 'crimson', label = 'prestige scores associated with HIGH values of height')
            _plt.hist(prestige[group_2], color = 'green', label = 'prestige scores associated with MEDIUM values of height')
            _plt.hist(prestige[group_1], color = 'orange', label = 'prestige scores associated with LOW values of height')
            _plt.legend(bbox_to_anchor = (1, 1 ))
            _plt.xlim((_np.max(prestige)+5, _np.min(prestige)-5))
            _plt.xlabel('Prestige')
            _plt.ylabel('Number of Scores')
            _plt.show()
        
    if scatter == True:
        _plt.figure()
        _plt.scatter(height[group_1], prestige[group_1], color = 'orange', label = 'prestige scores associated with LOW values of height')
        _plt.scatter(height[group_2], prestige[group_2], color = 'green', label = 'prestige scores associated with MEDIUM values of height')
        _plt.scatter(height[group_3], prestige[group_3], color = 'crimson', label = 'prestige scores associated with HIGH values of height')
        _plt.legend(bbox_to_anchor = (1, 1))
        _plt.xlabel('Height')
        _plt.ylabel('Prestige')
        _plt.show()
        
        
def null_hist(prestige, pop_size, no_unconditional = False):
    _np.random.seed(1000)
    height_none = _np.random.normal(100,10, size = pop_size)

    prestige_none = _np.mean(prestige) + 0 * height_none + _np.random.normal(0, 30, size = pop_size)

    prestige_none = prestige_none.astype('int')

    n = len(height_none)

    group_1_none = _np.where(height_none < _np.quantile(height_none, q = 1/3))
    group_2_none = _np.where((height_none > _np.quantile(height_none, q = 1/3)) & (height_none < _np.quantile(height_none, q = 2/3)) )
    group_3_none = _np.where((height_none > _np.quantile(height_none, q = 2/3)) & (height_none < _np.quantile(height_none, q = 3/3)) )


    if no_unconditional == False:
        _plt.figure()
        _plt.hist(prestige_none, color = 'blue', label = 'prestige scores, ignoring height')
        _plt.hist(prestige_none[group_3_none], color = 'crimson', label = 'prestige scores associated with HIGH values of height')
        _plt.hist(prestige_none[group_2_none], color = 'green', label = 'prestige scores associated with MEDIUM values of height')
        _plt.hist(prestige_none[group_1_none], color = 'orange', label = 'prestige scores associated with LOW values of height')
        _plt.xlim((_np.max(prestige_none), _np.min(prestige_none)))
        _plt.legend(bbox_to_anchor = (1, 1 ))
        _plt.xlabel('Prestige')
        _plt.ylabel('Number of Scores')
        _plt.show()
        
    if no_unconditional == True:
        _plt.figure()
        _plt.hist(prestige_none[group_3_none], color = 'crimson', label = 'prestige scores associated with HIGH values of height')
        _plt.hist(prestige_none[group_2_none], color = 'green', label = 'prestige scores associated with MEDIUM values of height')
        _plt.hist(prestige_none[group_1_none], color = 'orange', label = 'prestige scores associated with LOW values of height')
        _plt.xlim((_np.max(prestige_none), _np.min(prestige_none)))
        _plt.legend(bbox_to_anchor = (1, 1 ))
        _plt.xlabel('Prestige')
        _plt.ylabel('Number of Scores')
        _plt.show()

def null_scatter(prestige, pop_size):
    _np.random.seed(1000)
    height_none = _np.random.normal(100,10, size = pop_size)

    prestige_none = _np.mean(prestige) + 0 * height_none + _np.random.normal(0, 30, size = pop_size)

    prestige_none = prestige_none.astype('int')

    n = len(height_none)

    group_1_none = _np.where(height_none < _np.quantile(height_none, q = 1/3))
    group_2_none = _np.where((height_none > _np.quantile(height_none, q = 1/3)) & (height_none < _np.quantile(height_none, q = 2/3)) )
    group_3_none = _np.where((height_none > _np.quantile(height_none, q = 2/3)) & (height_none < _np.quantile(height_none, q = 3/3)) )
    
    _plt.figure()
    _plt.scatter(height_none[group_1_none], prestige_none[group_1_none], color = 'orange', label = 'prestige scores associated with LOW values of height')
    _plt.scatter(height_none[group_2_none], prestige_none[group_2_none], color = 'green', label = 'prestige scores associated with MEDIUM values of height')
    _plt.scatter(height_none[group_3_none], prestige_none[group_3_none], color = 'crimson', label = 'prestige scores associated with HIGH values of height')
    _plt.legend(bbox_to_anchor = (1, 1 ))
    _plt.xlabel('Height')
    _plt.ylabel('Prestige')
    _plt.show()