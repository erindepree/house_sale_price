"""
Module housing

Contains all functions needed to clean the dataframe, df, extracts features, basic column EDA
"""

import pandas as pd
import numpy as np
from datetime import date

import matplotlib.pyplot as plt
import seaborn as sns


def garage_age(row):
    """
        calculates the age of the garage.  If the garage doesn't exist, returns a value of 0

        Parameters
        ----------
        row : row in a dataframe

        Returns
        -------
        age of garage | int
            zero if garage doesn't exist
    """
    if row['garage_type'] not in [0, '0']:
        return int(row['yr_sold'] - row['garage_yr_blt'])
    else:
        return 0

def sale_date(row):
    """
        takes a row in a data frame and returns the sale date constructed from the year and month of the sale

        Parameters
        ----------
        row : row in a dataframe

        Returns
        date of sale : date object
    """
    return date(row['yr_sold'], row['mo_sold'], 1)


def railroad_near(row):
    if row['condition_1'] in ['RRNn', 'RRNe'] or row['condition_2'] in ['RRNn', 'RRNe']:
        return 1
    else:
        return 0

def railroad_adj(row):
    if row['condition_1'] in ['RRAn', 'RRAe'] or row['condition_2'] in ['RRAn', 'RRAe']:
        return 1
    else:
        return 0

def feeder(row):
    if row['condition_1'] == 'feedr' or row['condition_2'] == 'feedr':
        return 1
    else:
        return 0

def artery(row):
    if row['condition_1'] == 'artery' or row['condition_2'] == 'artery':
        return 1
    else:
        return 0

def normal(row):
    if row['condition_1'] == 'norm' or row['condition_2'] == 'norm':
        return 1
    else:
        return 0

def positive_near(row):
    if row['condition_1'] == 'PosN' or row['condition_2'] == 'PosN':
        return 1
    else:
        return 0

def positive_adj(row):
    if row['condition_1'] == 'PosA' or row['condition_2'] == 'PosA':
        return 1
    else:
        return 0

def ext_vinyl(row):
    if row['exterior_1st'] == 'VinylSd' or row['exterior_2nd'] == 'VinylSd':
        return 1
    else: 
        return 0

def ext_metal(row):
    if row['exterior_1st'] == 'MetalSd' or row['exterior_2nd'] == 'MetalSd':
        return 1
    else: 
        return 0

def ext_hardboard(row):
    if row['exterior_1st'] == 'HdBoard' or row['exterior_2nd'] == 'HdBoard':
        return 1
    else: 
        return 0

def ext_wood_side(row):
    if row['exterior_1st'] == 'Wd Sdng' or row['exterior_2nd'] == 'Wd Sdng':
        return 1
    else: 
        return 0

def ext_plywood(row):
    if row['exterior_1st'] == 'Plywood' or row['exterior_2nd'] == 'Plywood':
        return 1
    else: 
        return 0

def ext_cement(row):
    if row['exterior_1st'] == 'CemntBd' or row['exterior_2nd'] == 'CemntBd':
        return 1
    else: 
        return 0

def ext_brick_face(row):
    if row['exterior_1st'] == 'BrkFace' or row['exterior_2nd'] == 'BrkFace':
        return 1
    else: 
        return 0

def ext_wood_shingle(row):
    if row['exterior_1st'] == 'WdShing' or row['exterior_2nd'] == 'WdShing':
        return 1
    else: 
        return 0

def ext_asbestos_shingle(row):
    if row['exterior_1st'] == 'AsbShng' or row['exterior_2nd'] == 'AsbShng':
        return 1
    else: 
        return 0

def ext_stucco(row):
    if row['exterior_1st'] == 'Stucco' or row['exterior_2nd'] == 'Stucco':
        return 1
    else: 
        return 0



def data_cleaning(df):
    """
        takes a dataframe and returns a cleaned dataframe, 
        reports number of null values in returned dataframe

        Parameters
        ----------
        df : pandas dataframe
            the data to be cleaned

        Returns
        -------
        data : pandas dataframe
            the cleaned data in a dataframe
    """
    # null replacements to use while cleaning data.  When a null was used to indicate a missing feature, 
    # replacing that null with a zero
    null_replacements = {
        'Alley': 'no',
        'Mas Vnr Type': 'no',
        'Mas Vnr Area': 0,
        'Bsmt Qual': 0, 
        'Bsmt Cond': 0, 
        'Bsmt Exposure': 0, 
        'BsmtFin Type 1': 'no', 
        'BsmtFin SF 1': 0, 
        'BsmtFin Type 2': 'no', 
        'BsmtFin SF 2': 0, 
        'Bsmt Unf SF': 0,
        'Total Bsmt SF': 0,
        'Bsmt Full Bath': 0, 
        'Bsmt Half Bath': 0,
        'Fireplace Qu': 'no',
        'Garage Type': 'no',
        'Garage Yr Blt': 0,
        'Garage Finish': 'no',
        'Garage Cars': 0,
        'Garage Area': 0,
        'Garage Qual': 0,
        'Garage Cond': 0,
        'Pool QC': 'no',
        'Fence': 'no',
        'Misc Feature': 'no'
    }

    data = df.fillna(value=null_replacements)

    # changing a few data types to be integers where it makes sense to do so
    data = data.astype({
        'Bsmt Full Bath': 'int', 
        'Bsmt Half Bath': 'int', 
        'Garage Area': 'int', 
        'Garage Cars': 'int'
    })

    for row in data[data['Garage Area'] == 0]:
        data['Garage Type'] = 0

    # dropping the lot frontage feature, many missing entries from both training and testing data, 
    # so it won't be a helpful feature to model with
    data = data.drop(columns='Lot Frontage')

    # report total number null entries in returned dataframe
    print(f"Number of nulls present in data after cleaning: {data.isnull().sum().sum()}")


    # convert columns to snake case
    data.columns= [col.replace(' ', '_').lower() for col in data.columns]
    
    # calculate ages
    data['garage_age'] = data.apply(lambda row: garage_age(row), axis=1)        
    data['house_age'] = data.apply(lambda row: row['yr_sold'] - row['year_built'], axis=1)
    data['remod_add_age'] = data.apply(lambda row: row['yr_sold'] - row['year_remod/add'], axis=1)
    data['time_between'] = data.apply(lambda row: row['house_age']-row['remod_add_age'], axis=1)
    data['is_remod'] = data.apply(lambda row: int(bool(row['house_age']-row['remod_add_age'])), axis=1)
    
    # drop entries with negative ages (built in the future!)
    index_neg_age = data[(data['remod_add_age']<0) | (data['house_age']< 0) | (data['garage_age']<0)].index
    data = data.drop(index=index_neg_age)

    # determine date sold
    data['sale_date']= data.apply(lambda row: date(row['yr_sold'], row['mo_sold'],1), axis=1)
    
    # drop the unneeded year columns no longer needed
    data = data.drop(columns=['garage_yr_blt', 'year_built', 'year_remod/add', 'mo_sold', 'yr_sold'])

    # I'm combining 'condition_1' and 'condition_2' into new columns: 'is_rr_near', 'is_rr_adj', 
    # 'is_feeder_st', 'is_artery_st', 'is_pos_near', and 'is_pos_adj'.  'Normal' will be treated 
    # as the 0 state in all categories. Similar to OneHotEncoding (bascially ignoring the differences 
    # between the railroads, since there are few houses with those conditions)
    data['is_rr_near']= data.apply(lambda row: railroad_near(row), axis=1)
    data['is_rr_adj'] = data.apply(lambda row: railroad_adj(row), axis=1)
    data['is_feeder_st'] = data.apply(lambda row: feeder(row), axis= 1)
    data['is_artery_st'] = data.apply(lambda row: artery(row), axis= 1)
    data['is_pos_near'] = data.apply(lambda row: positive_near(row), axis= 1)
    data['is_pos_adj'] = data.apply(lambda row: positive_adj(row), axis= 1)

    # drop the 'condition_1' and 'condition_2' columns
    data = data.drop(columns=['condition_1', 'condition_2'])

    # all sidings that appear in at least 1% of housing.  Default state is other siding.
    data['ext_vinyl'] = data.apply(lambda row: ext_vinyl(row), axis= 1)
    data['ext_metal'] = data.apply(lambda row: ext_metal(row), axis= 1)
    data['ext_hardboard'] = data.apply(lambda row: ext_hardboard(row), axis= 1)
    data['ext_wood_side'] = data.apply(lambda row: ext_wood_side(row), axis= 1)
    data['ext_plywood'] = data.apply(lambda row: ext_plywood(row), axis= 1)
    data['ext_cement_bd'] = data.apply(lambda row: ext_cement(row), axis= 1)
    data['ext_brick_face'] = data.apply(lambda row: ext_brick_face(row), axis= 1)
    data['ext_wood_shingle'] = data.apply(lambda row: ext_wood_shingle(row), axis= 1)
    data['ext_asbestos_shingle'] = data.apply(lambda row: ext_asbestos_shingle(row), axis= 1)
    data['ext_stucco'] = data.apply(lambda row: ext_stucco(row), axis= 1)

    # drop exterior_1st and exterior_2nd columns
    data = data.drop(columns=['exterior_1st', 'exterior_2nd'])


    # return our cleaned dataframe!  :-)
    return data


def feat_eda(df, feat, feat_type, save_fig= False):
    """
        takes a dataframe, feature, type, correlation matrix, and an option to save any visuals 
        and returns EDA visualization and statistics on the feature

        Parameters
        ----------
        df : DataFrame
            dataframe holding the features of the data
        feat : str
            feature interested in, must be a column in df
        feat_type : str
            type of feature, dictates the EDA done
        corr_target : Series
            correlation of features with target
        save_fig : bool
            whether to save any figures produced (True), or not (False)

        Returns
        -------
        None
        prints information and visualizations
    """
    try:
        if feat_type in ['continuous', 'numeric', 'discrete', 'ordinal']:
            mean= df[feat].mean()
            std= df[feat].std()
            corr= df.corr(numeric_only=True)['saleprice'][feat]
            print(f"Correlation with sale price:  {np.round(corr,4)}")
            print(f"\n{feat} mean:  {np.round(mean,2)}")
            print(f"\n{feat} standard deviation:  {np.round(std,2)}")
            print(f"\nNumber of outliers in {feat}:   {np.round(((df[feat] < mean - 3*std) | (df[feat] > mean + 3*std)).sum(),1)}")
            if std < 0.01:
                print(f"\n{feat} has low variance, std = {std}")
            print('\n\n\n ')

            if feat_type in ['discrete', 'ordinal']:
                fig, axes = plt.subplots(1, 2, figsize=(12,6))

                ax= axes[0]
                sns.histplot(df[feat], ax=ax)
                ax.set_title(f"Histogram: {feat}")

                ax= axes[1]
                sns.boxplot(y= df[feat], ax=ax)
                ax.set_title(f"Box plot: {feat}")

                plt.suptitle(f"Feature: {feat}")
                plt.tight_layout()
                plt.show()
                if save_fig == True:
                    plt.savefig(f"../images/{feat}_eda.jpg")
                plt.close(fig)
            
                return

            else:
                # visualization of feature
                fig, axes = plt.subplots(1, 2, figsize=(12, 6))
            
                ax= axes[0]
                sns.histplot(df[feat], kde=True, ax=ax)
                ax.set_title(f'Histogram with KDE: {feat}')
            
                ax = axes[1]
                sns.boxplot(y=df[feat], ax=ax)
                ax.set_title(f'Box plot: {feat}')
            
                plt.suptitle(f'Feature: {feat}');
                plt.tight_layout()
                plt.show()
                if save_fig == True:
                    plt.savefig(f"../images/{feat}_eda.jpg")
                plt.close(fig)

                return 
        
        else:
            # nominal
            print(f"{feat}\n\n")
            print(df[feat].value_counts(normalize=True))
        
            return 
    except:
        print(f"{feat} experienced an error.")


def corr_heatmap(df, target='saleprice', save_fig=False):
    """
        returns a heatmap of the dataframe correlations with target

        parameters
        ----------
        df : DataFrame
        target : str, optional
        save_fig : bool, default = False
        
    """
    fig, ax= plt.subplots(figsize=(4,16))

    sns.heatmap(
        df.corr(numeric_only=True)[[target]].sort_values(target, ascending=False),
        vmin=-1,
        vmax=1,
        annot= True
    )
    
    ax.set_title('Numeric Feature Correlations with Sale Price')
    ax.set_ylabel('Features (numeric only)');

    plt.show()
    if save_fig==True:
        plt.savefig(f"../images/corr_feature_target.jpg")
    plt.close(fig)

    return


def OrdEncode(df, feat):
    """
        Takes a dataframe and feat and uses the ord_maps below to add a new column to df 
        and drops the previous column.
        
        Parameters
        ----------
        df : pandas DataFrame
        feat : str
            one of df's columns
        
        Returns
        -------
        data : pandas DataFrame
            encoded pandas series of integers determined by the ord_maps below
    """
    # housing.apply(lambda row: hous.feat_class['bsmt_qual']['map'][row['bsmt_qual']], axis=1)
    #    for feat in data.columns:
    #    if feat_class[feat]['type'] == 'ordinal':
    #        new_col = f"{feat}_ord"
    #        data[new_col] = OrdEncode(data, feat)
    #        data = data.drop(feat)

    new_col = feat + '_ord'
    df[new_col] = df.apply(lambda row: feat_class[feat]['map'][row[feat]], axis=1)
    data = df.drop(columns= [feat])
    
    return data






# ord_maps used to take the feature entries as keys and return an integer to rank them
qual_cond = { # features: 'exter_qual', 'exter_cond', 'bsmt_qual', 'bsmt_cond', 'heatingqc', 
    # 'kitchenqual', 'fireplacequ', 'garage_qual', 'garage_cond', 'pool_qc'
    'Ex': 5, # excellent
    'Gd': 4, # good
    'TA': 3, # typical/average
    'Fa': 2, # fair
    'Po': 1, # poor
    0: 0
}

street = { # feature: 'street'
    'Pave': 2,    # paved street
    'Grvl': 1,     # gravel street
    0: 0
}

shape = { # feature: 'lot_shape'
    'Reg': 4, 	# Regular	
    'IR1': 3, 	# Slightly irregular
    'IR2': 2, 	# Moderately Irregular
    'IR3':1, 	# Irregular
    0: 0
}

utilities = { # feature: 'utilities'
    'AllPub': 4,	# All public Utilities (E,G,W,& S)	
    'NoSewr': 3,	# Electricity, Gas, and Water (Septic Tank)
    'NoSeWa': 2,	# Electricity and Gas Only
    'ELO': 1,    	# Electricity only
    0: 0
}

slope = { # feature: 'lot_slope'
    'Gtl': 3,	# Gentle slope
    'Mod': 2,	# Moderate Slope	
    'Sev': 1,	# Severe Slope
    0: 0
}

bsmt_exposure = { # feature: 'bsmt_exposure'
    'Gd': 4,	# Good Exposure
    'Av': 3,	# Average Exposure (split levels or foyers typically score average or above)	
    'Mn': 2,	# Mimimum Exposure
    'No': 1,	    # No Exposure
    0: 0
}

bsmt_finish = { # features: 'bsmtfin_type_1', 'bsmtfin_type_2'
    'GLQ': 6,	# Good Living Quarters
    'ALQ': 5,	# Average Living Quarters
    'BLQ': 4,	# Below Average Living Quarters	
    'Rec': 3,	# Average Rec Room
    'LwQ': 2,	# Low Quality
    'Unf': 1,	# Unfinished
    0: 0
}

electrical = { # feature: 'electrical'
    'SBrkr': 5,	  # Standard Circuit Breakers & Romex
    'FuseA': 4,   # Fuse Box over 60 AMP and all Romex wiring (Average)	
    'FuseF': 3,   # 60 AMP Fuse Box and mostly Romex wiring (Fair)
    'FuseP': 2,	  # 60 AMP Fuse Box and mostly knob & tube wiring (poor)
    'Mix': 1,	  # Mixed
    0: 0
}

functional= { # feature: 'functional'
    'Typ': 8,	# Typical Functionality
    'Min1': 7,	# Minor Deductions 1
    'Min2': 6,	# Minor Deductions 2
    'Mod': 5,	# Moderate Deductions
    'Maj1': 4,	# Major Deductions 1
    'Maj2': 3,	# Major Deductions 2
    'Sev': 2,	# Severely Damaged
    'Sal': 1,	# Salvage only
    0: 0
}

garage_fin = { # feature: 'garage_finish'
    'Fin': 3,	# Finished
    'RFn': 2, 	# Rough Finished	
    'Unf': 1,	# Unfinished
    0: 0
}

paved = {
    'Y': 3, # paved
    'P': 2, # partial pavement
    'N': 1,  # dirt or gravel
    0: 0
}

fence = { # feature: 'fence'
    'GdPrv': 4, # Good Privacy
    'MnPrv': 3, # Minimum Privacy
    'GdWo': 2,  # Good Wood
    'MnWw': 1,   # Minimum Wood/Wire
    0: 0
}


feat_class = {'ms_subclass': {'type': 'nominal', 'encoder': 'OneHotEncoder'}, 
              'ms_zoning': {'type': 'nominal', 'encoder': 'OneHotEncoder'}, 
              'lot_area': {'type': 'continuous', 'encoder': 'StandardScaler'}, 
              'street': {'type': 'ordinal', 'encoder': 'OrdEncode', 'map': street}, 
              'alley': {'type': 'nominal', 'encoder': 'OneHotEncoder'},
              'lot_shape': {'type': 'ordinal', 'encoder': 'OrdEncode', 'map': shape}, 
              'land_contour': {'type': 'nominal', 'encoder': 'OneHotEncoder'}, 
              'utilities': {'type': 'ordinal', 'encoder': 'OrdEncode', 'map': utilities}, 
              'lot_config': {'type': 'nominal', 'encoder': 'OneHotEncoder'}, 
              'land_slope': {'type': 'ordinal', 'encoder': 'OrdEncode', 'map': slope},
              'neighborhood': {'type': 'nominal', 'encoder': 'OneHotEncoder'}, 
              'condition_1': {'type': 'nominal', 'encoder': 'OneHotEncoder'}, 
              'condition_2': {'type': 'nominal', 'encoder': 'OneHotEncoder'}, 
              'bldg_type': {'type': 'nominal', 'encoder': 'OneHotEncoder'},
              'house_style': {'type': 'nominal', 'encoder': 'OneHotEncoder'}, 
              'overall_qual': {'type': 'ordinal', 'encoder': 'None'}, 
              'overall_cond': {'type': 'ordinal', 'encoder': 'None'}, 
              'roof_style': {'type': 'nominal', 'encoder': 'OneHotEncoder'},
              'roof_matl': {'type': 'nominal', 'encoder': 'OneHotEncoder'}, 
              'exterior_1st': {'type': 'nominal', 'encoder': 'OneHotEncoder'}, 
              'exterior_2nd': {'type': 'nominal', 'encoder': 'OneHotEncoder'}, 
              'mas_vnr_type': {'type': 'nominal', 'encoder': 'OneHotEncoder'},
              'mas_vnr_area': {'type': 'continuous', 'encoder': 'StandardScaler'}, 
              'exter_qual': {'type': 'ordinal', 'encoder': 'OrdEncode', 'map': qual_cond}, 
              'exter_cond': {'type': 'ordinal', 'encoder': 'OrdEncode', 'map': qual_cond}, 
              'foundation': {'type': 'nominal', 'encoder': 'OneHotEncoder'}, 
              'bsmt_qual': {'type': 'ordinal', 'encoder': 'OrdEncode', 'map': qual_cond},
              'bsmt_cond': {'type': 'ordinal', 'encoder': 'OrdEncode', 'map': qual_cond}, 
              'bsmt_exposure': {'type': 'ordinal', 'encoder': 'OrdEncode', 'map': bsmt_exposure}, 
              'bsmtfin_type_1': {'type': 'ordinal', 'encoder': 'OrdEncode', 'map': bsmt_finish}, 
              'bsmtfin_sf_1': {'type': 'continuous', 'encoder': 'StandardScaler'},
              'bsmtfin_type_2': {'type': 'ordinal', 'encoder': 'OrdEncode', 'map': bsmt_finish}, 
              'bsmtfin_sf_2': {'type': 'continuous', 'encoder': 'StandardScaler'}, 
              'bsmt_unf_sf': {'type': 'continuous', 'encoder': 'StandardScaler'}, 
              'total_bsmt_sf': {'type': 'continuous', 'encoder': 'StandardScaler'},
              'heating': {'type': 'nominal', 'encoder': 'OneHotEncoder'}, 
              'heating_qc': {'type': 'ordinal', 'encoder': 'OrdEncode', 'map': qual_cond}, 
              'central_air': {'type': 'nominal', 'encoder': 'OneHotEncoder'}, 
              'electrical': {'type': 'ordinal', 'encoder': 'OrdEncode', 'map': electrical}, 
              '1st_flr_sf': {'type': 'continuous', 'encoder': 'StandardScaler'},
              '2nd_flr_sf': {'type': 'continuous', 'encoder': 'StandardScaler'}, 
              'low_qual_fin_sf': {'type': 'continuous', 'encoder': 'StandardScaler'}, 
              'gr_liv_area': {'type': 'continuous', 'encoder': 'StandardScaler'}, 
              'bsmt_full_bath': {'type': 'discrete'},
              'bsmt_half_bath': {'type': 'discrete'}, 
              'full_bath': {'type': 'discrete'}, 
              'half_bath': {'type': 'discrete'}, 
              'bedroom_abvgr': {'type': 'discrete'},
              'kitchen_abvgr': {'type': 'discrete'}, 
              'kitchen_qual': {'type': 'ordinal', 'encoder': 'OrdEncode', 'map': qual_cond}, 
              'totrms_abvgrd': {'type': 'discrete'}, 
              'functional': {'type': 'ordinal', 'encoder': 'OrdEncode', 'map': functional},
              'fireplaces': {'type': 'discrete'}, 
              'fireplace_qu': {'type': 'ordinal', 'encoder': 'OrdEncode', 'map': qual_cond}, 
              'garage_type': {'type': 'nominal', 'encoder': 'OneHotEncoder'}, 
              'garage_finish': {'type': 'ordinal', 'encoder': 'OrdEncode', 'map': garage_fin},
              'garage_cars': {'type': 'discrete'}, 
              'garage_area': {'type': 'continuous', 'encoder': 'StandardScaler'}, 
              'garage_qual': {'type': 'ordinal', 'encoder': 'OrdEncode', 'map': qual_cond}, 
              'garage_cond': {'type': 'ordinal', 'encoder': 'OrdEncode', 'map': qual_cond},
              'paved_drive': {'type': 'ordinal', 'encoder': 'OrdEncode', 'map': paved}, 
              'wood_deck_sf': {'type': 'continuous', 'encoder': 'StandardScaler'}, 
              'open_porch_sf': {'type': 'continuous', 'encoder': 'StandardScaler'}, 
              'enclosed_porch': {'type': 'continuous', 'encoder': 'StandardScaler'},
              '3ssn_porch': {'type': 'continuous', 'encoder': 'StandardScaler'}, 
              'screen_porch': {'type': 'continuous', 'encoder': 'StandardScaler'}, 
              'pool_area': {'type': 'continuous', 'encoder': 'StandardScaler'}, 
              'pool_qc': {'type': 'ordinal', 'encoder': 'OrdEncode', 'map': qual_cond}, 
              'fence': {'type': 'ordinal', 'encoder': 'OrdEncode', 'map': fence},
              'misc_feature': {'type': 'nominal', 'encoder': 'OneHotEncoder'},
              'misc_val': {'type': 'continuous', 'encoder': 'StandardScaler'}, 
              'sale_type': {'type': 'nominal', 'encoder': 'OneHotEncoder'}, 
              'saleprice': {'type': 'target', 'encoder': 'None'}, 
              'garage_age': {'type': 'continuous', 'encoder': 'StandardScaler'},
              'house_age': {'type': 'continuous', 'encoder': 'StandardScaler'}, 
              'remod_add_age': {'type': 'continuous', 'encoder': 'StandardScaler'}, 
              'time_between': {'type': 'continuous', 'encoder': 'StandardScaler'}, 
              'is_remod': {'type': 'bool', 'encoder': 'None'}, 
              'sale_date': {'type': 'date'},
              'is_rr_near': {'type': 'bool_int'}, 
              'is_rr_adj': {'type': 'bool_int'}, 
              'is_feeder_st': {'type': 'bool_int'}, 
              'is_artery_st': {'type': 'bool_int'}, 
              'is_pos_near': {'type': 'bool_int'}, 
              'is_pos_adj': {'type': 'bool_int'}, 
              'ext_vinyl': {'type': 'bool_int'},
              'ext_metal': {'type': 'bool_int'},
              'ext_hardboard': {'type': 'bool_int'},
              'ext_wood_side': {'type': 'bool_int'},
              'ext_plywood': {'type': 'bool_int'},
              'ext_cement_bd': {'type': 'bool_int'},
              'ext_brick_face': {'type': 'bool_int'},
              'ext_wood_shingle': {'type': 'bool_int'},
              'ext_asbestos_shingle': {'type': 'bool_int'},
              'ext_stucco': {'type': 'bool_int'},
              'bsmt_qual_ord': {'type': 'discrete', 'encoder': 'StandardScaler'},
              'street_ord': {'type': 'discrete', 'encoder': 'StandardScaler'},
              'lot_shape_ord': {'type': 'discrete', 'encoder': 'StandardScaler'},
              'utilities_ord': {'type': 'discrete', 'encoder': 'StandardScaler'},
              'land_slope_ord': {'type': 'discrete', 'encoder': 'StandardScaler'},
              'exter_qual_ord': {'type': 'discrete', 'encoder': 'StandardScaler'},
              'exter_cond_ord': {'type': 'discrete', 'encoder': 'StandardScaler'},
              'bsmt_cond_ord': {'type': 'discrete', 'encoder': 'StandardScaler'},
              'bsmt_exposure_ord': {'type': 'discrete', 'encoder': 'StandardScaler'},
              'heating_qc_ord': {'type': 'discrete', 'encoder': 'StandardScaler'},
              'electrical_ord': {'type': 'discrete', 'encoder': 'StandardScaler'},
              'kitchen_qual_ord': {'type': 'discrete', 'encoder': 'StandardScaler'},
              'functional_ord': {'type': 'discrete', 'encoder': 'StandardScaler'},
              'garage_qual_ord': {'type': 'discrete', 'encoder': 'StandardScaler'},
              'garage_cond_ord': {'type': 'discrete', 'encoder': 'StandardScaler'},
              'paved_drive_ord': {'type': 'discrete', 'encoder': 'StandardScaler'}
             }



