import numpy as np # for mathematical operations
import pandas as pd # to manipulate dataset

def predict(row):
    '''
    Function that calculates P(Detect | a,d) and P(No Detect | a, d) and returns a prediction 
    based on which probability is higher

    P(Detect | a,d) = P(a | Detect) * P(d | Detect) * P(Detect)\n
                      ------------------------------------------\n
                                        P(a, d)

    P(No Detect | a,d) = P(a | No Detect) * P(d | No Detect) * P(No Detect)\n
                        ----------------------------------------------------\n
                                            P(a, d)        

    P(a, d) = P(a, d | Detect) * P(Detect) + P(a, d | No Detect) * P(No Detect)
            = P(a | Detect) * P(d | Detect) * P(Detect) + P(a | No Detect) * P(d | No Detect) * P(No Detect)

    Parameters
    ----------
    row : Pandas series objects (e.g. a row of pandas dataframe)
    '''
    p_a_D = gaussian(row['Amplitude'], detect_means['Amplitude'], detect_vars['Amplitude'])
    p_a_ND = gaussian(row['Amplitude'], no_detect_means['Amplitude'], no_detect_vars['Amplitude'])

    p_d_D = gaussian(row['Distance'], detect_means['Distance'], detect_vars['Distance'])
    p_d_ND = gaussian(row['Distance'], no_detect_means['Distance'], no_detect_vars['Distance'])

    p_ad = p_a_D * p_d_D * p_D + p_a_ND * p_d_ND * (1 - p_D)

    p_D_ad = p_a_D * p_d_D * p_D / p_ad             # P(Detect | a,d)
    p_ND_ad = p_a_ND * p_d_ND * (1 - p_D) / p_ad    # P(No Detect | a,d)

    return 'Detect' if p_D_ad >= p_ND_ad else 'No Detect'

def gaussian(x, mean, var):
        '''
        Probability density function for normal distribution
        '''
        return (np.pow(np.e, -np.pow(x-mean,2)/(2*var)))/np.sqrt(2*np.pi*var)

if __name__ == '__main__':
    data = pd.read_csv('detection_data.csv')

    # Split detection dataset into two
    data_detect = data[data['Detection'] == 'Detect'].drop('Detection', axis=1)
    data_no_detect = data[data['Detection'] == 'No Detect'].drop('Detection', axis=1)

    # Obtain means and variances for the features of the 2 dataset
    detect_means = dict()
    detect_vars = dict()

    for feature in data_detect.columns:
        detect_means[feature] = data_detect[feature].mean()
        detect_vars[feature] = data_detect[feature].var()

    no_detect_means = dict()
    no_detect_vars = dict()

    for feature in data_no_detect.columns:
        no_detect_means[feature] = data_no_detect[feature].mean()
        no_detect_vars[feature] = data_no_detect[feature].var()

    p_D = len(data_detect) / len(data)  # Probability of detection
    p_ND = len(data_no_detect) / len(data)  # Probability of no detection

    # Make predictions for each data point in the dataset
    data['Prediction'] = data.apply(predict, axis=1)

    print(f'Number of correct predictions: {len(data[data['Detection'] == data['Prediction']])}')
    print(f'Number of incorrect predictions: {len(data[data['Detection'] != data['Prediction']])}\n')

    # Make predictions for each data point in the extra dataset
    new_data = pd.read_csv('detection_data_extra.csv')
    new_data['Prediction'] = new_data.apply(predict, axis=1)

    print(f'Number of correct predictions for the new dataset: {len(new_data[new_data['Detection'] == new_data['Prediction']])}')
    print(f'Number of incorrect predictions for the new dataset: {len(new_data[new_data['Detection'] != new_data['Prediction']])}')
