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

class ODBAE_detection:
    '''
    This class implements method to detect anomalies.
    '''

    autoencoder = None
    hidden_neurons = None
    category_filter_ratio = None
    batch_parameter = None
    category_parameter = None

    def __init__(self, task_repetitions = 1):
        """
        Args:
            task_repetitions (int): The number of repetitions of the anomaly detection task, 
            where the final model identifies anomalies based on the median reconstruction error 
            across multiple detection runs, thereby reducing the impact of uncertainty in the machine learning model.
        """
        self.task_repetitions = task_repetitions

        
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
            x_train_batch_tmp2=x_train_batch_tmp2.append(x_train_batch_tmp1.iloc[:,0:x_train_batch_tmp1.shape[1]-1])
            x_test_batch_tmp2=pd.DataFrame()
            x_test_batch_tmp2=x_test_batch_tmp2.append(x_test_batch_tmp1.iloc[:,0:x_test_batch_tmp1.shape[1]-1])
            x_train_normalize1=pd.DataFrame(self.normalize1.fit_transform(x_train_batch_tmp2),columns=x_train_batch_tmp2.columns)
            x_test_normalize1=pd.DataFrame(self.normalize1.transform(x_test_batch_tmp2),columns=x_test_batch_tmp2.columns)
            x_train_normalize1[x_train_batch_tmp1.columns[x_train_batch_tmp1.shape[1]-1]]=x_train_batch_tmp1[x_train_batch_tmp1.columns[x_train_batch_tmp1.shape[1]-1]]
            x_test_normalize1[x_test_batch_tmp1.columns[x_test_batch_tmp1.shape[1]-1]]=x_test_batch_tmp1[x_test_batch_tmp1.columns[x_test_batch_tmp1.shape[1]-1]]
            x_train_normalize1_final=x_train_normalize1_final.append(x_train_normalize1)
            x_test_normalize1_final=x_test_normalize1_final.append(x_test_normalize1)
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

    def reconstruction(self, x_test):
        """
        Identify anomalies in the test dataset.

        Args:
            x_test (data frame): Test dataset.

        Returns:
            x_test_ob (data frame): Test dataset, the last column represents the reconstruction error.
            abnormal_number (int): The number of anomalies.
        """
        predictions_test = self.autoencoder.predict(x_test)
        square_errors_test = np.power(x_test - predictions_test, 2)
        mse_series_test = pd.Series(np.mean(square_errors_test, axis=1))
        x_test_predict = pd.DataFrame(predictions_test,columns=x_test.columns)

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

        return x_test_ob, abnormal_number, x_test_predict
    
    def outlier_detection(self, x_train_raw, x_test_raw, abnormal_rate=None, hidden_neurons=None, def_fun = True,
                         batch_parameter=None, normalize=True, batch_size = 64, epochs = 100):
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
        self.abnormal_rate = abnormal_rate
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
        
        df_columns = [f'col{i+1}' for i in range(self.task_repetitions)]
        rec_error_df = pd.DataFrame(columns=df_columns)

        for i in range(self.task_repetitions):

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
            
            tf.keras.backend.clear_session()
            data_real_dim=self.estimate_intrinsic_dim(x_train)
            self.build_model(x_train, data_real_dim)

            if self.def_fun:
                self.autoencoder = self.train_until_val_loss_threshold(x_train, batch_size, loss_fn = def_loss, 
                                                                  val_loss_threshold = 0.001, max_restarts = 10, epochs= self.epochs)
            else:
                self.autoencoder = self.train_until_val_loss_threshold(x_train, batch_size, loss_fn = 'mse', 
                                                                  val_loss_threshold = 0.001, max_restarts = 10, epochs= self.epochs)
            
            x_test_loss, abnormal_number, x_test_predict = self.reconstruction(x_test)
            rec_error_df[rec_error_df.columns[i]] = x_test_loss['loss']

        mean_rec_error = pd.Series(np.median(rec_error_df, axis=1))
        rec_error_df['median_loss'] = mean_rec_error
        x_test['median_loss'] = mean_rec_error

        x_test_sort = x_test.sort_values(by=['median_loss'],ascending=False)
        x_test_abnormal = x_test_sort.head(abnormal_number)
        x_test_normal = x_test_sort.tail(x_test.shape[0]-abnormal_number)
        
        return x_test_abnormal, x_test_normal, x_test_predict, rec_error_df, self.autoencoder