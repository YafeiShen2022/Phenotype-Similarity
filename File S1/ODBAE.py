from tensorflow.keras.models import Model
from tensorflow.keras.layers import Input, Dense, Dropout
from tensorflow.keras import regularizers
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau
import shap
import warnings
import logging
import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler     
from sklearn.preprocessing import MinMaxScaler
import matplotlib.pyplot as plt
from sklearn.utils import check_array
from tensorflow.keras.losses import mean_squared_error
import seaborn as sns
from sklearn.utils.validation import check_is_fitted
from tensorflow.keras.models import Sequential
from tensorflow.keras.regularizers import l2
from mpl_toolkits.mplot3d import Axes3D
import tensorflow as tf
from scipy.stats import normaltest, gaussian_kde
import tensorflow.keras.backend as K

class ODBAE:
    '''
    This class implements method to detect and explain anomalies.
    '''

    autoencoder = None
    hidden_neurons = None
    reconstruction_error_percent = None
    category_filter_ratio = None
    batch_parameter = None
    category_parameter = None

    def __init__(self, reconstruction_error_percent = 0.5, category_filter_ratio = 0.5):
        """
        Args:
            reconstruction_error_percent (float): Number between 0 to 1.
            category_filter_ratio (float): Threshold for filtering category-related anomalies.
        """
        self.reconstruction_error_percent = reconstruction_error_percent
        self.category_filter_ratio = category_filter_ratio
        
    def remove_batch(self, x_train, x_test):
        """
        Remove batch effects from the dataset.

        Args:
            x_train (data frame): Set of records we can use to train the model (training dataset).
            x_test (data frame): Set of records we want to detect anomalies (test dataset).

        Returns:
            data frame: Batch effect removed training dataset and test dataset.
        """
        x_test_batch=pd.DataFrame()
        x_train_batch=pd.DataFrame()
        x_train_normalize1_final=pd.DataFrame()
        x_test_normalize1_final=pd.DataFrame()
        if self.category_parameter is None:
            for i in range(x_train.shape[1]):
                x_train_batch[x_train.columns[i]]=x_train[x_train.columns[i]]
            for i in range(x_test.shape[1]):
                x_test_batch[x_test.columns[i]]=x_test[x_test.columns[i]]
        else:
            for i in range(x_train.shape[1]-1):
                x_train_batch[x_train.columns[i]]=x_train[x_train.columns[i]]
            for i in range(x_test.shape[1]-1):
                x_test_batch[x_test.columns[i]]=x_test[x_test.columns[i]]
        x_train_batch.rename(columns={x_test.columns[x_train_batch.shape[1]-1]: 'batch'}, inplace=True)
        x_test_batch.rename(columns={x_train.columns[x_test_batch.shape[1]-1]: 'batch'}, inplace=True)
        x_train_batch['original_index']=list(x_train_batch.index)
        x_test_batch['original_index']=list(x_test_batch.index)
        train_batch_list=x_train_batch.batch.unique().tolist()
        test_batch_list=x_test_batch.batch.unique().tolist()
        batch_list=list(set(train_batch_list).intersection(set(test_batch_list)))
        self.normalize1=StandardScaler()
        x_train_batch=x_train_batch.set_index('batch')
        x_test_batch=x_test_batch.set_index('batch')
        
        for i in range(len(batch_list)):
            batch=batch_list[i]
            x_train_batch_tmp1=x_train_batch.loc[batch]
            x_test_batch_tmp1=x_test_batch.loc[batch]
            x_train_batch_tmp1=x_train_batch_tmp1.reset_index(drop=True)
            x_test_batch_tmp1=x_test_batch_tmp1.reset_index(drop=True)
            x_train_batch_tmp2=pd.DataFrame()
            x_train_batch_tmp2=pd.concat([x_train_batch_tmp2, x_train_batch_tmp1.iloc[:,0:x_train_batch_tmp1.shape[1]-1]], axis=0)
            x_test_batch_tmp2=pd.DataFrame()
            x_test_batch_tmp2=pd.concat([x_test_batch_tmp2, x_test_batch_tmp1.iloc[:,0:x_test_batch_tmp1.shape[1]-1]], axis = 0)
            x_train_normalize1=pd.DataFrame(self.normalize1.fit_transform(x_train_batch_tmp2),columns=x_train_batch_tmp2.columns)
            x_test_normalize1=pd.DataFrame(self.normalize1.transform(x_test_batch_tmp2),columns=x_test_batch_tmp2.columns)
            x_train_normalize1[x_train_batch_tmp1.columns[x_train_batch_tmp1.shape[1]-1]]=x_train_batch_tmp1[x_train_batch_tmp1.columns[x_train_batch_tmp1.shape[1]-1]]
            x_test_normalize1[x_test_batch_tmp1.columns[x_test_batch_tmp1.shape[1]-1]]=x_test_batch_tmp1[x_test_batch_tmp1.columns[x_test_batch_tmp1.shape[1]-1]]
            x_train_normalize1_final=pd.concat([x_train_normalize1_final, x_train_normalize1], axis = 0)
            x_test_normalize1_final=pd.concat([x_test_normalize1_final, x_test_normalize1])
        x_train_normalize1_final=x_train_normalize1_final.sort_values(by=[x_train_normalize1_final.columns[x_train_normalize1_final.shape[1]-1]])
        x_test_normalize1_final=x_test_normalize1_final.sort_values(by=[x_test_normalize1_final.columns[x_test_normalize1_final.shape[1]-1]])
        x_train_normalize1_final=x_train_normalize1_final.reset_index(drop=True).iloc[:,0:x_train_normalize1_final.shape[1]-1]
        x_test_normalize1_final=x_test_normalize1_final.reset_index(drop=True).iloc[:,0:x_test_normalize1_final.shape[1]-1]          
        
        return x_train_normalize1_final,x_test_normalize1_final
    
    def min_max_normalization(self, x_train_normalize1_final, x_test_normalize1_final):
        """
        Standardize the dataset.

        Args:
            x_train_normalize1_final (data frame): Training dataset without batch effects.
            x_test_normalize1_final (data frame): Test dataset without batch effects.

        Returns:
            data frame: Standardized training and test datasets.
        """

        self.normalize2=MinMaxScaler()
        x_train_normalize2_final=pd.DataFrame(self.normalize2.fit_transform(x_train_normalize1_final),columns=x_train_normalize1_final.columns)
        x_test_normalize2_final=pd.DataFrame(self.normalize2.transform(x_test_normalize1_final),columns=x_test_normalize1_final.columns)
        
        return x_train_normalize2_final, x_test_normalize2_final
    
    def estimate_intrinsic_dim(self,x_train):
        """
        Estimate the intrinsic dimensionality of the dataset.

        Args:
            x_train (data frame): Standardized training datasets.

        Returns:
            int: The intrinsic dimensionality of training dataset.
        """
        cov_result_h=np.cov(x_train,rowvar=False)  
        eigenvalue, featurevector = np.linalg.eig(cov_result_h)
        eigenvalue_sort=sorted(eigenvalue,reverse=True)
        eigenvalue_sum=0
        for i in range(x_train.shape[1]):
            eigenvalue_sum=eigenvalue_sum+eigenvalue_sort[i]
        eigenvalue_count=0
        for i in range(x_train.shape[1]):
            eigenvalue_count=eigenvalue_count+eigenvalue_sort[i]
            if eigenvalue_count>0.98*eigenvalue_sum:
                data_real_dim=i+1
                break
                
        return data_real_dim

    def train_until_val_loss_threshold(self, X_train, batch_size = 64, loss_fn = 'mse', 
                                       val_loss_threshold = 0.001, max_restarts = 10, epochs = 500):
        """
        Train Autoencoder model on the given training dataset.

        Args:
            x_train (data frame): The training dataset to train the Autoencoder model on.
            batch_size (int): Size of each batch of data enter to the model.
            loss_fn: Training loss function.
            val_loss_threshold (float): A threshold on the validation loss used to determine whether the model training is successful.
            max_restarts (int): The maximum number of retraining attempts if the model training is unsuccessful.

        Returns:
            model: Trained autoencoder. 
        """
        restart_count = 0
        final_history = None
        def reset_weights_set_weights(model):
            for layer in model.layers:
                if hasattr(layer, 'kernel') and hasattr(layer, 'bias'):
                    kernel_shape = layer.kernel.shape
                    bias_shape = layer.bias.shape
                    kernel_initializer = layer.kernel_initializer
                    bias_initializer = layer.bias_initializer
                    new_kernel = kernel_initializer(kernel_shape)
                    new_bias = bias_initializer(bias_shape)
                    new_kernel_np = K.get_value(new_kernel)
                    new_bias_np = K.get_value(new_bias)
                    layer.set_weights([new_kernel_np, new_bias_np])

        while restart_count < max_restarts:

            reset_weights_set_weights(self.encoder)
            reset_weights_set_weights(self.decoder)
            self.autoencoder = Sequential([self.encoder, self.decoder])
            self.autoencoder.compile(loss=loss_fn, optimizer='adam')

            lr_reducer = ReduceLROnPlateau(monitor='val_loss', factor=0.5, patience=5, min_lr=1e-6, verbose=1)

            history = self.autoencoder.fit(X_train, X_train, epochs = epochs, batch_size = batch_size, shuffle = True,
                                           validation_split = 0.1, callbacks = [lr_reducer], verbose = 1)

            current_val_loss = min(history.history['val_loss'])
            print(f"{current_val_loss:.6f}")

            if current_val_loss < val_loss_threshold:
                final_history = history
                return self.autoencoder

            restart_count += 1
        return self.autoencoder

    def build_model(self, x_train, data_real_dim):
        """
        Build Autoencoder model.

        Args:
            x_train (data frame): The training dataset to train the Autoencoder model on.
            data_real_dim (int): The intrinsic dimensionality of training dataset.

        Returns:
            model: The untrained encoder and decoder. 
        """
        if self.hidden_neurons is None:
            tf.keras.backend.clear_session()   
            self.encoder = Sequential(name='encoder')
            # Input layer
            self.encoder.add(Dense(
               data_real_dim, activation='tanh',  
                input_shape=(x_train.shape[1],),activity_regularizer=l2(10e-7)))   

            self.decoder = Sequential(name='decoder')

            self.decoder.add(Dense(  
                x_train.shape[1],activation='sigmoid',activity_regularizer=l2(10e-7)
                ))

            self.autoencoder = Sequential([self.encoder, self.decoder])
            self.encoder.summary()
            self.decoder.summary()

            return self.encoder, self.decoder
        
        else:
            if len(self.hidden_neurons)>=3:
                tf.keras.backend.clear_session()
                self.encoder = Sequential(name='encoder')
                self.encoder.add(Dense(
                    self.hidden_neurons[0], activation='tanh',  
                    input_shape=(x_train.shape[1],),activity_regularizer=l2(10e-7)))
    
                for i in range(1, int((len(self.hidden_neurons)-1)/2+1)):
                    self.encoder.add(Dense(
                            self.hidden_neurons[i],activation='tanh',activity_regularizer=l2(10e-7)))
                
                self.decoder = Sequential(name='decoder')
            
                for i in range(int((len(self.hidden_neurons)-1)/2+1),len(self.hidden_neurons)):
                    self.decoder.add(Dense(
                            self.hidden_neurons[i],activation='tanh',activity_regularizer=l2(10e-7)))
                
                self.decoder.add(Dense(  
                        x_train.shape[1],activation='sigmoid',activity_regularizer=l2(10e-7)
                        ))
            
            if len(self.hidden_neurons)<3:
                tf.keras.backend.clear_session() 
                self.encoder = Sequential(name='encoder')
            
                self.encoder.add(Dense(
                     self.hidden_neurons[0], activation='tanh',  
                     input_shape=(x_train.shape[1],),activity_regularizer=l2(10e-7)))
            
                self.decoder = Sequential(name='decoder')
            
                self.decoder.add(Dense(  
                       x_train.shape[1],activation='sigmoid',activity_regularizer=l2(10e-7)
                        ))
            self.autoencoder=Sequential([self.encoder, self.decoder])
            self.encoder.summary()
            self.decoder.summary()

            return self.encoder, self.decoder

    def get_outliers(self, x_test):
        """
        Identify anomalies in the test dataset.

        Args:
            x_test (data frame): Test dataset.

        Returns:
            x_test_ob_final (data frame): Test dataset, the last column represents the reconstruction error.
            x_test_abnormal_final (data frame): The set of identified anomalies.
            rec_error_ratio_df_final (data frame): The proportion of reconstruction error for each attribute of each record in the test dataset.
            rec_error_ratio_df_abnormal_final (data frame): The proportion of reconstruction error for each attribute of identified anomalies.
        """
        predictions_test = self.autoencoder.predict(x_test)
        square_errors_test = np.power(x_test - predictions_test, 2)
        mse_series_test = pd.Series(np.mean(square_errors_test, axis=1))
        x_test_predict = pd.DataFrame(predictions_test,columns=x_test.columns)
        mse_test_df = pd.DataFrame(square_errors_test,columns=x_test.columns)

        if self.abnormal_rate is None:
            errors = np.asarray(mse_series_test)
            stat, p_value = normaltest(errors)
            is_normal = p_value > 0.05
            if is_normal:
                mean = np.mean(errors)
                std = np.std(errors)
                threshold = mean + 3 * std
                anomaly_mask = errors > threshold
            else:
                kde = gaussian_kde(errors)
                error_range = np.linspace(min(errors), max(errors), 1000)
                cdf_vals = np.array([kde.integrate_box_1d(-np.inf, x) for x in error_range])
                threshold_index = np.searchsorted(cdf_vals, 0.95)
                threshold = error_range[threshold_index]
                anomaly_mask = errors > threshold
            abnormal_rate = np.sum(anomaly_mask) / len(errors)
            abnormal_number = int(abnormal_rate*x_test.shape[0])

        if self.abnormal_rate is not None:
            abnormal_number = int(self.abnormal_rate*x_test.shape[0])
        
        x_test_ob=pd.DataFrame()
        for i in range(x_test.shape[1]):
            x_test_ob[x_test.columns[i]]=x_test[x_test.columns[i]]
        x_test_ob['loss']=mse_series_test
        x_test_ob_sort=x_test_ob.sort_values(by=['loss'],ascending=False)
        x_test_abnormal=x_test_ob_sort.head(abnormal_number)
        x_test_normal=x_test_ob_sort.tail(x_test.shape[0]-abnormal_number)
        
        rec_error_ratio_tmp=pd.DataFrame()
        for i in range(mse_test_df.shape[1]):
            rec_error_ratio_tmp[mse_test_df.columns[i]]=mse_test_df[mse_test_df.columns[i]]
        rec_error_ratio_tmp['loss']=mse_series_test
        rec_error_ratio=pd.DataFrame()
        for i in range(rec_error_ratio_tmp.shape[1]-1):
            tmp1=rec_error_ratio_tmp[rec_error_ratio_tmp.columns[i]]/((rec_error_ratio_tmp.shape[1]-1)*rec_error_ratio_tmp[rec_error_ratio_tmp.columns[rec_error_ratio_tmp.shape[1]-1]])
            rec_error_ratio[rec_error_ratio_tmp.columns[i]]=tmp1
        rec_error_ratio_df=pd.DataFrame()
        for i in range(rec_error_ratio.shape[1]):
            rec_error_ratio_df[rec_error_ratio.columns[i]]=rec_error_ratio[rec_error_ratio.columns[i]]
        rec_error_ratio_df['loss']=mse_series_test
        rec_error_ratio_df_sort=rec_error_ratio_df.sort_values(by=['loss'],ascending=False)
        rec_error_ratio_df_abnormal=rec_error_ratio_df_sort.head(abnormal_number)
#         x_test_ob_final=x_test_ob.iloc[:,0:x_test_ob.shape[1]-1]
        x_test_ob_final=x_test_ob.iloc[:,0:x_test_ob.shape[1]]
        x_test_abnormal_final=x_test_abnormal.iloc[:,0:x_test_abnormal.shape[1]-1]
        rec_error_ratio_df_final=rec_error_ratio_df.iloc[:,0:rec_error_ratio_df.shape[1]-1]
        rec_error_ratio_df_abnormal_final=rec_error_ratio_df_abnormal.iloc[:,0:rec_error_ratio_df_abnormal.shape[1]-1]
            
        return x_test_ob_final, x_test_abnormal_final, rec_error_ratio_df_final, rec_error_ratio_df_abnormal_final

    def outlier_to_category(self, x_test_ob_final, x_test_abnormal_final, rec_error_ratio_df_final, 
                            rec_error_ratio_df_abnormal_final, x_test, category_filter_ratio):
        """
        Classify anomalies into category-related and category-unrelated.

        Args:
            x_test_ob_final (data frame): Test dataset, the last column represents the reconstruction error.
            x_test_abnormal_final (data frame): The set of identified anomalies.
            rec_error_ratio_df_final (data frame): The proportion of reconstruction error for each attribute of each record in the test dataset.
            rec_error_ratio_df_abnormal_final (data frame): The proportion of reconstruction error for each attribute of identified anomalies.
            x_test (data frame): Test dataset.
            category_filter_ratio (float): Number between 0 to 1. If the proportion of anomalies within a category exceeds this threshold, 
            then these anomalies are considered category-related.

        Returns:
            category_related_abnormal_data (data frame): The set of category-related anomalies.
            category_unrelated_abnormal_data (data frame): The set of category-unrelated anomalies.
            category_related_error_ratio (data frame): The proportion of reconstruction error for each attribute of category-related anomalies.
            category_unrelated_error_ratio (data frame): The proportion of reconstruction error for each attribute of category-unrelated anomalies
        """
        x_test_abnormal_index=list(x_test_abnormal_final.index)
        rec_error_ratio_df_abnormal_index=list(rec_error_ratio_df_abnormal_final.index)
        x_test.rename(columns={x_test.columns[x_test.shape[1]-1]: 'category'}, inplace=True)
        abnormal_cor_category=x_test['category'][x_test_abnormal_index]
        abnormal_cor_category_rec=x_test['category'][rec_error_ratio_df_abnormal_index]
        x_test_abnormal_final['category']=abnormal_cor_category
        x_test_abnormal_final['category_col']=abnormal_cor_category
        x_test_abnormal_final['original_index']=list(x_test_abnormal_final.index)
        rec_error_ratio_df_abnormal_final['category']=abnormal_cor_category_rec
        rec_error_ratio_df_abnormal_final['category_col']=abnormal_cor_category_rec
        rec_error_ratio_df_abnormal_final['original_index']=list(rec_error_ratio_df_abnormal_final.index)
        abnormal_category_counts=x_test_abnormal_final.category.value_counts()
        abnormal_category=list(abnormal_category_counts.index)
        all_category_counts=x_test.category.value_counts()
        abnormal_category_ratio=[]
        category_ratio_count=pd.DataFrame()
        category_record=[]
        abnormal_category_number=[]
        original_category_number=[]
        for i in range(len(abnormal_category)):
            category_record.append(abnormal_category[i])
            abnormal_category_ratio.append(abnormal_category_counts[abnormal_category[i]]/all_category_counts[abnormal_category[i]])
            abnormal_category_number.append(abnormal_category_counts[abnormal_category[i]])
            original_category_number.append(all_category_counts[abnormal_category[i]])
        category_ratio_count['category']=category_record
        category_ratio_count['ratio']=abnormal_category_ratio
        category_ratio_count['abnormal_number']=abnormal_category_number
        category_ratio_count['original_number']=original_category_number
        category_ratio_count_sort=category_ratio_count.sort_values(by=['ratio'],ascending=False)
        category_ratio_count_sort=category_ratio_count_sort.reset_index(drop=True)
        x_test_abnormal_category_index=x_test_abnormal_final.set_index('category')
        rec_error_ratio_df_abnormal_category_index=rec_error_ratio_df_abnormal_final.set_index('category')
        abnormal_category_filter=[]
        category_related_abnormal_data=pd.DataFrame()
        category_unrelated_abnormal_data=pd.DataFrame()
        category_related_error_ratio=pd.DataFrame()
        category_unrelated_error_ratio=pd.DataFrame()
        if max(category_ratio_count_sort['ratio']) < category_filter_ratio:
            judge = 0
            for i in range(category_ratio_count_sort.shape[0]):
                category_unrelated_abnormal_data=pd.concat([category_unrelated_abnormal_data, x_test_abnormal_category_index.loc[category_ratio_count_sort['category'][i]]],axis=0)
                category_unrelated_error_ratio=pd.concat([category_unrelated_error_ratio, rec_error_ratio_df_abnormal_category_index.loc[category_ratio_count_sort['category'][i]]],axis=0)
            category_unrelated_abnormal_data=category_unrelated_abnormal_data.set_index('original_index')
            category_unrelated_error_ratio=category_unrelated_error_ratio.set_index('original_index')

            return category_related_abnormal_data, category_unrelated_abnormal_data, category_related_error_ratio, category_unrelated_error_ratio, judge
        else:
            judge = 1 
            for i in range(category_ratio_count_sort.shape[0]):
                if category_ratio_count_sort['ratio'][i]>=category_filter_ratio:
                    abnormal_category_filter.append(category_ratio_count_sort['category'][i])
                    category_related_abnormal_data=pd.concat([category_related_abnormal_data,x_test_abnormal_category_index.loc[category_ratio_count_sort['category'][i]]],axis = 0)
                    category_related_error_ratio=pd.concat([category_related_error_ratio, rec_error_ratio_df_abnormal_category_index.loc[category_ratio_count_sort['category'][i]]], axis = 0)
                else:
                    category_unrelated_abnormal_data=pd.concat([category_unrelated_abnormal_data, x_test_abnormal_category_index.loc[category_ratio_count_sort['category'][i]]],aixs=0)
                    category_unrelated_error_ratio=pd.concat([category_unrelated_error_ratio, rec_error_ratio_df_abnormal_category_index.loc[category_ratio_count_sort['category'][i]]], axis= 0)
            category_related_abnormal_data=category_related_abnormal_data.set_index('original_index')
            category_unrelated_abnormal_data=category_unrelated_abnormal_data.set_index('original_index')
            category_related_error_ratio=category_related_error_ratio.set_index('original_index')
            category_unrelated_error_ratio=category_unrelated_error_ratio.set_index('original_index')
        
            return category_related_abnormal_data, category_unrelated_abnormal_data, category_related_error_ratio, category_unrelated_error_ratio, judge
    
    def get_mean_value_of_each_category(self, category_related_abnormal_data, category_related_error_ratio):
        """
        Average the values and proportions of reconstruction error for each attribute of category-related anomalies by category.

        Args:
            category_related_abnormal_data (data frame): The set of category-related anomalies.
            category_related_error_ratio (data frame): The proportion of reconstruction error for each attribute of category-related anomalies.

        Returns:
            category_related_abnormal_data_mean_df (data frame): The results of averaging the values of each attribute of category-related anomalies by category.
            category_related_rer_ob_mean_df (data frame): The results of averaging the proportion of reconstruction error for each attribute 
            of category-related anomalies by category.
            category_related_abnormal_category_list (list): List of categories involved in category-related anomalies.
        """
        category_related_abnormal_category_list=category_related_abnormal_data.category_col.unique().tolist()
        category_related_abnormal_category_count=category_related_abnormal_data.category_col.value_counts()
        category_related_abnormal_data_ob=category_related_abnormal_data.set_index('category_col').iloc[:,0:category_related_abnormal_data.shape[1]]
        category_related_abnormal_data_mean_df=pd.DataFrame()
        for i in range(len(category_related_abnormal_category_list)):
            if category_related_abnormal_category_count.loc[category_related_abnormal_category_list[i]]>1:
                category_related_abnormal_data_ob_filter=category_related_abnormal_data_ob.loc[category_related_abnormal_category_list[i]]
                category_related_abnormal_data_ob_filter_mean=np.mean(category_related_abnormal_data_ob_filter,axis=0)
                ob_value=list(category_related_abnormal_data_ob_filter_mean)
                category_related_abnormal_data_ob_filter_mean_df_tmp=pd.DataFrame()
                for j in range(len(ob_value)):
                    tmp=[]
                    tmp.append(category_related_abnormal_data_ob_filter_mean[j])
                    category_related_abnormal_data_ob_filter_mean_df_tmp[category_related_abnormal_data_ob.columns[j]]=tmp
                category_related_abnormal_data_mean_df=category_related_abnormal_data_mean_df.append(category_related_abnormal_data_ob_filter_mean_df_tmp)
            else:
                category_related_abnormal_data_ob_filter_mean=category_related_abnormal_data_ob.loc[category_related_abnormal_category_list[i]]
                ob_value=list(category_related_abnormal_data_ob_filter_mean)
                category_related_abnormal_data_ob_filter_mean_df_tmp=pd.DataFrame()
                for j in range(len(ob_value)):
                    tmp=[]
                    tmp.append(category_related_abnormal_data_ob_filter_mean[j])
                    category_related_abnormal_data_ob_filter_mean_df_tmp[category_related_abnormal_data_ob.columns[j]]=tmp
                category_related_abnormal_data_mean_df=category_related_abnormal_data_mean_df.append(category_related_abnormal_data_ob_filter_mean_df_tmp)
        category_related_abnormal_data_mean_df=category_related_abnormal_data_mean_df.reset_index(drop=True)
        
        category_related_rer_category_list=category_related_error_ratio.category_col.unique().tolist()
        category_related_rer_category_count=category_related_error_ratio.category_col.value_counts()
        category_related_rer_ob=category_related_error_ratio.set_index('category_col').iloc[:,0:category_related_error_ratio.shape[1]]
        category_related_rer_ob_mean_df=pd.DataFrame()
        for i in range(len(category_related_rer_category_list)):
            if category_related_rer_category_count.loc[category_related_rer_category_list[i]]>1:
                category_related_rer_ob_filter=category_related_rer_ob.loc[category_related_rer_category_list[i]]
                category_related_rer_ob_filter_mean=np.mean(category_related_rer_ob_filter,axis=0)
                ob_value=list(category_related_rer_ob_filter_mean)
                category_related_rer_ob_filter_mean_df_tmp=pd.DataFrame()
                for j in range(len(ob_value)):
                    tmp=[]
                    tmp.append(category_related_rer_ob_filter_mean[j])
                    category_related_rer_ob_filter_mean_df_tmp[category_related_rer_ob.columns[j]]=tmp
                category_related_rer_ob_mean_df=pd.concat([category_related_rer_ob_mean_df, category_related_rer_ob_filter_mean_df_tmp],axis=0)
            else:
                category_related_rer_ob_filter_mean=category_related_rer_ob.loc[category_related_rer_category_list[i]]
                ob_value=list(category_related_rer_ob_filter_mean)
                category_related_rer_ob_filter_mean_df_tmp=pd.DataFrame()
                for j in range(len(ob_value)):
                    tmp=[]
                    tmp.append(category_related_rer_ob_filter_mean[j])
                    category_related_rer_ob_filter_mean_df_tmp[category_related_rer_ob.columns[j]]=tmp
                category_related_rer_ob_mean_df=pd.concat([category_related_rer_ob_mean_df, category_related_rer_ob_filter_mean_df_tmp], axis=0)
        category_related_rer_ob_mean_df=category_related_rer_ob_mean_df.reset_index(drop=True)
        
        return category_related_abnormal_data_mean_df, category_related_rer_ob_mean_df, category_related_abnormal_category_list
    
    def outlier_degree_of_each_category(self, category_related_abnormal_data_mean_df):
        """
        Calculate the anomaly degree of each attribute for anomalies in each category of category-related anomalies.

        Args:
            category_related_abnormal_data_mean_df (data frame): The results of averaging the values of each attribute of category-related anomalies by category.

        Returns:
            category_related_error_ratio (data frame): Proportion of reconstruction error for each attribute of anomalies in each category of category-related anomalies.
        """
        category_related_abnormal_data_pre = self.autoencoder.predict(category_related_abnormal_data_mean_df)
        category_related_square_errors = np.power(category_related_abnormal_data_pre- category_related_abnormal_data_mean_df, 2)
        category_related_mse_series = pd.Series(np.mean(category_related_square_errors, axis=1))
        category_related_mse_df=pd.DataFrame(category_related_square_errors,columns=category_related_abnormal_data_mean_df.columns)
        category_related_error_ratio_tmp=pd.DataFrame()
        for i in range(category_related_mse_df.shape[1]):
            category_related_error_ratio_tmp[category_related_mse_df.columns[i]]=category_related_mse_df[category_related_mse_df.columns[i]]
        category_related_error_ratio_tmp['loss']=category_related_mse_series
        category_related_error_ratio=pd.DataFrame()
        for i in range(category_related_error_ratio_tmp.shape[1]-1):
            tmp1=category_related_error_ratio_tmp[category_related_error_ratio_tmp.columns[i]]/((category_related_error_ratio_tmp.shape[1]-1)*category_related_error_ratio_tmp[category_related_error_ratio_tmp.columns[category_related_error_ratio_tmp.shape[1]-1]])
            category_related_error_ratio[category_related_error_ratio_tmp.columns[i]]=tmp1
        
        return category_related_error_ratio
    
    def record_feature_predict(self, record):
        """
        Reconstruct specific attributes for each record.

        Args:
            record (data frame): Records that need to be reconstructed.

        Returns:
            record_predict (array): Reconstruction results for specific attributes.
        """
        record_predict=self.autoencoder.predict(record)[:,self.HLP_index]
        
        return record_predict
    
    def get_abnormal_features(self, error_ratio, abnormal_data_df, rer_ob_df, x_train):
        """
        Obtain the abnormal attributes for anomalies.

        Args:
            error_ratio (data frame): For category-related anomalies, proportion of reconstruction error for each attribute of anomalies in each category of 
            category-related anomalies. For category-unrelated anomalies, Proportion of reconstruction error for each attribute of anomalies.
            abnormal_data_df (data frame): For category-related anomalies, the results of averaging the values of each attribute of category-related anomalies 
            by category. For category-unrelated anomalies, set of anomalies.
            rer_ob_df (data frame): For category-related anomalies, average proportion of reconstruction error for each attribute of category-related anomalies.
            For category-unrelated anomalies, Proportion of reconstruction error for each attribute of anomalies.
            x_train (data frame): Training dataset.

        Returns:
            anomaly_expalnation_df (data frame): Anomalies and their abnormal attributes.
        """
        anomaly_expalnation_df=pd.DataFrame()
        shap_explain_all=[]
        shap_explain_value_all=[]
        HLP_value_all=[]
        HLP_parameter_all=[]
        HLP_explain_all=[]
        original_index=[]
        error_ratio_index=list(error_ratio.index)
        for i in range(error_ratio.shape[0]):
            print(i+1,error_ratio.shape[0])
            original_index.append(error_ratio_index[i])
            record_err_ratio_df=pd.DataFrame()
            col_index=[]
            col_value=[]
            shap_explain=[]
            shap_explain_value=[]
            HLP_value=[]
            HLP_index=[]
            HLP_parameter=[]
            HLP_explain=[]
            col_HLP_value=[]
            for j in range(error_ratio.shape[1]):
                col_index.append(j)
                col_value.append(error_ratio[error_ratio.columns[j]][error_ratio_index[i]])  
                col_HLP_value.append(rer_ob_df[rer_ob_df.columns[j]][error_ratio_index[i]])  
            record_err_ratio_df['col_index']=col_index
            record_err_ratio_df['col_value']=col_value
            record_err_ratio_df['col_HLP_value']=col_HLP_value
            record_err_ratio_df_sort=record_err_ratio_df.sort_values(by=['col_value'],ascending=False)
            record_err_ratio_df_sort=record_err_ratio_df_sort.reset_index(drop=True)
            count=0
            for k in range(record_err_ratio_df_sort.shape[0]):
                count=count+record_err_ratio_df_sort['col_value'][k]
                HLP_value.append(record_err_ratio_df_sort['col_HLP_value'][k])
                HLP_index.append(record_err_ratio_df_sort['col_index'][k])
                HLP_parameter.append(rer_ob_df.columns[record_err_ratio_df_sort['col_index'][k]])
                if count>self.reconstruction_error_percent:
                    break
            if len(HLP_parameter)>1:
                HLP_explain.append(HLP_parameter[0])
                HLP_explain.append(HLP_parameter[1])
                HLP_explain_all.append(sorted(HLP_explain))
            if len(HLP_parameter)==1:
                HLP_explain_all.append(HLP_parameter)
            x_train_filter=x_train.head(200)
            self.HLP_index = HLP_index[0]
            explainer = shap.KernelExplainer(self.record_feature_predict, x_train_filter)
            shap_values = explainer.shap_values(abnormal_data_df.loc[error_ratio_index[i]], nsamples='auto') 
            shap_values_fabs=np.fabs(shap_values)
            shap_values_df=pd.DataFrame()
            col_shap_value=[]
            col_shap_index=[]
            col_shap_name=[]
            for q in range(len(shap_values_fabs)):
                col_shap_value.append(shap_values_fabs[q])
                col_shap_index.append(q)
                col_shap_name.append(abnormal_data_df.columns[q])
            shap_values_df['col_shap_index']=col_shap_index
            shap_values_df['col_shap_name']=col_shap_name
            shap_values_df['col_shap_value']=col_shap_value
            shap_values_df_sort=shap_values_df.sort_values(by=['col_shap_value'],ascending=False)
            shap_values_df_sort=shap_values_df_sort.reset_index(drop=True)
            for s in range(shap_values_df_sort.shape[0]):
                if shap_values_df_sort['col_shap_index'][s]!=HLP_index[0] and shap_values_df_sort['col_shap_index'][s]>np.mean(col_shap_value):
                    shap_explain.append(HLP_parameter[0])
                    shap_explain.append(shap_values_df_sort['col_shap_name'][s])
                    shap_explain_value.append(HLP_value[0]*shap_values_df_sort['col_shap_value'][s])
                    break
                if shap_values_df_sort['col_shap_index'][s]!=HLP_index[0] and shap_values_df_sort['col_shap_index'][s]<=np.mean(col_shap_value):
                    shap_explain.append('None')
                    shap_explain_value.append(0)
                    break
            shap_explain_all.append(sorted(shap_explain))
            shap_explain_value_all.append(shap_explain_value[0])
            HLP_parameter_all.append(HLP_parameter)
        anomaly_expalnation_df['original_index']=original_index
        anomaly_expalnation_df['HLP_parameter']=HLP_parameter_all
        anomaly_expalnation_df['HLP_expalin']=HLP_explain_all
        anomaly_expalnation_df['shap_explain']=shap_explain_all
        anomaly_expalnation_df['shap_explain_value']=shap_explain_value_all
                    
        return anomaly_expalnation_df
    
    def category_based_anomaly_explanation(self, anomaly_expalnation_df, category_related_abnormal_data, category_related_abnormal_category_list):
        """
        Obtain explanation results for category-related anomalies.

        Args:
            anomaly_expalnation_df (data frame): Anomalies and their abnormal attributes.
            category_related_abnormal_data (data frame): Category-related anomalies.
            category_related_abnormal_category_list (list): List of categories involved in category-related anomalies.

        Returns:
            category_related_abnormal_data (data frame): Explanation results for category-related anomalies.
        """
        anomaly_expalnation_df['category']=category_related_abnormal_category_list
        anomaly_expalnation_df_category_index=anomaly_expalnation_df.set_index('category')
        HLP_parameter_final=[]
        HLP_expalin_final=[]
        shap_explain_final=[]
        shap_explain_value_final=[]
        original_index=[]
        category_related_abnormal_data_index=list(category_related_abnormal_data.index)
        for i in range(category_related_abnormal_data.shape[0]):
            original_index.append(category_related_abnormal_data_index[i])
            HLP_parameter_final.append(anomaly_expalnation_df_category_index['HLP_parameter'][category_related_abnormal_data['category_col'][category_related_abnormal_data_index[i]]])
            HLP_expalin_final.append(anomaly_expalnation_df_category_index['HLP_expalin'][category_related_abnormal_data['category_col'][category_related_abnormal_data_index[i]]])
            shap_explain_final.append(anomaly_expalnation_df_category_index['shap_explain'][category_related_abnormal_data['category_col'][category_related_abnormal_data_index[i]]])
            shap_explain_value_final.append(anomaly_expalnation_df_category_index['shap_explain_value'][category_related_abnormal_data['category_col'][category_related_abnormal_data_index[i]]])
        category_related_abnormal_data['original_index']=original_index
        category_related_abnormal_data['HLP_parameter']=HLP_parameter_final
        category_related_abnormal_data['HLP_expalin']=HLP_expalin_final
        category_related_abnormal_data['shap_explain']=shap_explain_final
        category_related_abnormal_data['shap_explain_value']=shap_explain_value_final
        
        shap_value_threshold=np.mean(shap_explain_value_final)
        explain_final=[]
        for i in range(category_related_abnormal_data.shape[0]):
            if category_related_abnormal_data['shap_explain_value'][category_related_abnormal_data_index[i]]>=shap_value_threshold:
                explain_final.append(category_related_abnormal_data['shap_explain'][category_related_abnormal_data_index[i]])
            else:
                explain_final.append(category_related_abnormal_data['HLP_expalin'][category_related_abnormal_data_index[i]])
        category_related_abnormal_data['explain_result']=explain_final
        
        return category_related_abnormal_data
    
    def instance_based_anomaly_explanation(self, anomaly_expalnation_df, category_unrelated_abnormal_data):
        """
            Obtain explanation results for category-unrelated anomalies.

        Args:
            anomaly_expalnation_df (data frame): Anomalies and their abnormal attributes.
            category_unrelated_abnormal_data (data frame): Category-unrelated anomalies.

        Returns:
            category_unrelated_abnormal_data (data frame): Explanation results for category-unrelated anomalies.
        """
        anomaly_expalnation_df_original_index=anomaly_expalnation_df.set_index('original_index')
        category_unrelated_abnormal_data_index=list(category_unrelated_abnormal_data.index)
        HLP_parameter_final=[]
        HLP_expalin_final=[]
        shap_explain_final=[]
        shap_explain_value_final=[]
        original_index=[]
        for i in range(len(category_unrelated_abnormal_data_index)):
            original_index.append(category_unrelated_abnormal_data_index[i])
            HLP_parameter_final.append(anomaly_expalnation_df_original_index['HLP_parameter'][category_unrelated_abnormal_data_index[i]])
            HLP_expalin_final.append(anomaly_expalnation_df_original_index['HLP_expalin'][category_unrelated_abnormal_data_index[i]])
            shap_explain_final.append(anomaly_expalnation_df_original_index['shap_explain'][category_unrelated_abnormal_data_index[i]])
            shap_explain_value_final.append(anomaly_expalnation_df_original_index['shap_explain_value'][category_unrelated_abnormal_data_index[i]])
        category_unrelated_abnormal_data['original_index']=original_index
        category_unrelated_abnormal_data['HLP_parameter']=HLP_parameter_final
        category_unrelated_abnormal_data['HLP_expalin']=HLP_expalin_final
        category_unrelated_abnormal_data['shap_explain']=shap_explain_final
        category_unrelated_abnormal_data['shap_explain_value']=shap_explain_value_final
        
        shap_value_threshold=np.mean(shap_explain_value_final)
        explain_final=[]
        for i in range(len(category_unrelated_abnormal_data_index)):
            if category_unrelated_abnormal_data['shap_explain_value'][category_unrelated_abnormal_data_index[i]]>=shap_value_threshold: 
                explain_final.append(category_unrelated_abnormal_data['shap_explain'][category_unrelated_abnormal_data_index[i]]) 
            else:
                explain_final.append(category_unrelated_abnormal_data['HLP_expalin'][category_unrelated_abnormal_data_index[i]])  
        category_unrelated_abnormal_data['explain_result']=explain_final
        
        return category_unrelated_abnormal_data
    
    def outlier_analysis(self, x_train_raw, x_test_raw, autoencoder=None, abnormal_rate=None, hidden_neurons=None, def_fun = True,
                         batch_parameter=None, category_parameter=None, normalize=True, batch_size = 64, epochs = 100):
        """
            First, if the dataset contains batch effects ('batch_parameter' is not None) , perform batch correction. 
            Then, if Autoencoder model not provided ('autoencoder' is None), train Autoencoder model on given training dataset.
            'hidden_neurons' can be used to set the structure of the autoencoder, for example, [12,6,3,1,3,6,12], where 12 represents
            the number of features in the dataset. Furthermore, obtain anomalies based on the output of 'get_outliers'. Eventually,
            If records belong to different categories, classify anomalies into category-related and category-unrelated anomalies, 
            and obtain the specific anomalous attributes based on the output of 'category_based_anomaly_explanation' or 'instance_based_anomaly_explanation'.

        Args:
            x_train_raw (data frame): The training dataset used for training the autoencoder model.
            x_test_raw (data frame): The test dataset used for identifying and explaining anomalies.
            autoencoder (model): Trained Autoencoder model, if None, we will train an autoencoder model.
            abnormal_rate (float): The preset proportion of anomalies in the dataset.
            hidden_neurons (list): If 'autoencoder=None', 'hidden_neurons' can be used to set the structure of the autoencoder we are training.
            batch_parameter (bool): If None, the dataset does not have batch effects. Otherwise, the dataset needs to be batch-corrected.
            category_parameter (bool): If None, explain each anomaly individually. Otherwise, anomalies need to be classified into category-related 
            and category-unrelated anomalies and explained accordingly. 
            normalize (bool): If True, normalize the training and test datasets to remove the dimensions.
            batch_size (int): Size of each batch of data enter to the model.

        Returns:
            instance_based_anomaly_explanation (data frame): Explanation results for category-unrelated anomalies.
            category_based_anomaly_explanation (data frame): Explanation results for category-related anomalies, 
            this result is only output when 'category_parameter' is not None.
        """
        self.batch_parameter = batch_parameter
        self.category_parameter = category_parameter
        self.abnormal_rate = abnormal_rate
        self.autoencoder = autoencoder
        self.hidden_neurons = hidden_neurons
        self.epochs = epochs
        self.def_fun = def_fun

        if self.batch_parameter is not None:
            x_train_tmp, x_test_tmp = self.remove_batch(x_train_raw, x_test_raw)
            if normalize:
                x_train, x_test = self.min_max_normalization(x_train_tmp, x_test_tmp)
            else:
                x_train = x_train_tmp
                x_test = x_test_tmp

        if self.batch_parameter is None:
            if self.category_parameter is None:
                if normalize:
                    x_train, x_test = self.min_max_normalization(x_train_raw, x_test_raw)
                else:
                    x_train = x_train_raw
                    x_test = x_test_raw
            else:
                x_train_raw_value=x_train_raw.iloc[:,0:x_train_raw.shape[1]-1]
                x_test_raw_value=x_test_raw.iloc[:,0:x_test_raw.shape[1]-1]
                if normalize:
                    x_train, x_test = self.min_max_normalization(x_train_raw_value, x_test_raw_value)
                else:
                    x_train = x_train_raw_value
                    x_test = x_test_raw_value
        
        if self.autoencoder is None:

            def def_loss(y_actual,y_pred):
                y_actual_tensor=tf.subtract(y_actual,0,name=None)
                y_actual_tensor=tf.cast(y_actual_tensor, tf.float64)
                y_pred_tensor=tf.subtract(y_pred,0,name=None)
                y_pred_tensor=tf.cast(y_pred_tensor, tf.float64)
                mean_actual_tensor = tf.reduce_mean(y_actual_tensor, axis=0, keepdims=True)
                mean_pred_tensor = tf.reduce_mean(y_pred_tensor, axis=0, keepdims=True)

                cov_actual= tf.matmul(tf.transpose(y_actual_tensor-mean_actual_tensor), 
                                  y_actual_tensor-mean_actual_tensor)/tf.cast(tf.shape(y_actual_tensor)[0]-1, tf.float64)
                cov_pred= tf.matmul(tf.transpose(y_pred_tensor-mean_pred_tensor), 
                                 y_pred_tensor-mean_pred_tensor)/tf.cast(tf.shape(y_pred_tensor)[0]-1, tf.float64)

                cov_actual = (cov_actual + tf.transpose(cov_actual)) / 2.0
                cov_pred = (cov_pred + tf.transpose(cov_pred)) / 2.0
                epsilon = 1e-6
                cov_actual += epsilon * tf.eye(tf.shape(cov_actual)[0], dtype=tf.float64)
                cov_pred += epsilon * tf.eye(tf.shape(cov_pred)[0], dtype=tf.float64)

                eigenvalues_actual, eigenvectors_actual=tf.linalg.eigh(cov_actual)
                eigenvalues_pred, eigenvectors_pred=tf.linalg.eigh(cov_pred)
                eigenvalues_actual_sort_to_control=tf.sort(eigenvalues_actual,direction='DESCENDING')[0:data_real_dim]
                eigenvalues_pred_sort_to_control=tf.sort(eigenvalues_pred,direction='DESCENDING')[0:data_real_dim]
                sqrt_eigenvalues_actual_to_control=tf.sqrt(eigenvalues_actual_sort_to_control, name=None)
                sqrt_eigenvalues_pred_to_control=tf.sqrt(eigenvalues_pred_sort_to_control, name=None)
                eigenvalues_actual_sort=tf.sort(eigenvalues_actual,direction='DESCENDING')
                sqrt_eigenvalues_actual=tf.sqrt(eigenvalues_actual_sort, name=None)
                beta_control_list=0.3*sqrt_eigenvalues_actual
                beta_list=sqrt_eigenvalues_actual
                max_beta_control_list=tf.reduce_max(beta_control_list)
                max_beta_control_list_reshape=tf.reshape(max_beta_control_list,[1])
                min_beta_list=tf.reduce_min(beta_list)
                min_beta_list_reshape=tf.reshape(min_beta_list,[1])
                combine_beta_list=tf.concat([max_beta_control_list_reshape,min_beta_list_reshape], axis=0)
                beta_control_value=tf.reduce_min(combine_beta_list)
                eigenvalues_diff=tf.subtract(sqrt_eigenvalues_actual_to_control,sqrt_eigenvalues_pred_to_control,name=None)
                eigenvalues_diff_cast=tf.cast(eigenvalues_diff,tf.float64)
                eigenvalues_diff_control=tf.subtract(eigenvalues_diff_cast,beta_control_value,name=None)
                eigenvalues_diff_control_square=tf.reduce_mean(tf.square(eigenvalues_diff_control))
                mean_loss=tf.reduce_mean(tf.square(y_actual_tensor-y_pred_tensor))
                final_loss=0.008*mean_loss+eigenvalues_diff_control_square
                return final_loss
        
            data_real_dim=self.estimate_intrinsic_dim(x_train)
            self.build_model(x_train, data_real_dim)

            if self.def_fun:
                self.autoencoder = self.train_until_val_loss_threshold(x_train, batch_size, loss_fn = def_loss, 
                                                                  val_loss_threshold = 0.001, max_restarts = 10, epochs= self.epochs)
            else:
                self.autoencoder = self.train_until_val_loss_threshold(x_train, batch_size, loss_fn = 'mse', 
                                                                  val_loss_threshold = 0.001, max_restarts = 10, epochs= self.epochs)

            
        x_test_loss, x_test_abnormal, rec_error_ratio_loss, rec_error_ratio_abnormal=self.get_outliers(x_test)
        
        if self.category_parameter is None:
            anomaly_expalnation_df = self.get_abnormal_features(rec_error_ratio_abnormal, x_test_abnormal, 
                                                                rec_error_ratio_abnormal, x_train)
            instance_based_anomaly_explanation = self.instance_based_anomaly_explanation(anomaly_expalnation_df, x_test_abnormal)

            return instance_based_anomaly_explanation
        
        if self.category_parameter is not None:
            x_test[x_test_raw.columns[x_test_raw.shape[1]-1]]=x_test_raw[x_test_raw.columns[x_test_raw.shape[1]-1]]
            category_related_abnormal_data, category_unrelated_abnormal_data, category_related_error_ratio, category_unrelated_error_ratio, judge = self.outlier_to_category(x_test_loss, 
                                                                                                                                                                      x_test_abnormal, rec_error_ratio_loss, rec_error_ratio_abnormal, x_test, self.category_filter_ratio)
            if judge:
                category_related_abnormal_data_mean_df, category_related_rer_ob_mean_df, category_related_abnormal_category_list = self.get_mean_value_of_each_category(category_related_abnormal_data, category_related_error_ratio)
                category_related_error_ratio = self.outlier_degree_of_each_category(category_related_abnormal_data_mean_df)
                category_anomaly_expalnation_df = self.get_abnormal_features(category_related_error_ratio, category_related_abnormal_data_mean_df, category_related_rer_ob_mean_df, x_train)
                category_based_anomaly_explanation = self.category_based_anomaly_explanation(category_anomaly_expalnation_df, 
                                                                                         category_related_abnormal_data, category_related_abnormal_category_list)
                tmp_df=pd.DataFrame()
                tmp_df['category_col'] = category_unrelated_error_ratio['category_col']
                category_unrelated_error_ratio=category_unrelated_error_ratio.drop('category_col', axis=1)
                category_unrelated_abnormal_data=category_unrelated_abnormal_data.drop('category_col', axis=1)
                instance_anomaly_expalnation_df = self.get_abnormal_features(category_unrelated_error_ratio, category_unrelated_abnormal_data, category_unrelated_error_ratio, x_train)
                instance_based_anomaly_explanation = self.instance_based_anomaly_explanation(instance_anomaly_expalnation_df, category_unrelated_abnormal_data)
                instance_based_anomaly_explanation['category_col'] = tmp_df['category_col'] 
            
                return category_based_anomaly_explanation, instance_based_anomaly_explanation
            else:
                category_based_anomaly_explanation=pd.DataFrame()
                tmp_df=pd.DataFrame()
                tmp_df['category_col'] = category_unrelated_error_ratio['category_col']
                category_unrelated_error_ratio=category_unrelated_error_ratio.drop('category_col', axis=1)
                category_unrelated_abnormal_data=category_unrelated_abnormal_data.drop('category_col', axis=1)
                instance_anomaly_expalnation_df = self.get_abnormal_features(category_unrelated_error_ratio, category_unrelated_abnormal_data, category_unrelated_error_ratio, x_train)
                instance_based_anomaly_explanation = self.instance_based_anomaly_explanation(instance_anomaly_expalnation_df, category_unrelated_abnormal_data)
                instance_based_anomaly_explanation['category_col'] = tmp_df['category_col']
                print('No category\'s anomaly proportion reaches the set threshold')

                return category_based_anomaly_explanation, instance_based_anomaly_explanation
