import numpy as np
import matplotlib.pyplot as plt
from scipy import fftpack
import scipy.stats
import pandas as pd
from sklearn import preprocessing
import statistics
from sklearn.model_selection import train_test_split
import seaborn as sns
from sklearn import svm
from sklearn.metrics import classification_report, accuracy_score, confusion_matrix

# defining some paths and feature size
IRF_filename = "dataset used for training/IRF NL 955 rpm.csv"
BF_filename = "dataset used for training/BF NL 945 rpm.csv"
array_feature_size = 800
threshold = 4000  # frequency threshold above which is considered noise


# These are some utility functions that I used
def find_time_step(time_vector: np.ndarray) -> float:
    """Finds the discrete step size of the time vector.

    Args:
        time_vector(np.ndarray): 1 dimensional array of time that has a const time_step.

    Returns:
        float: average discrete time step of the array.
    """
    # timef = time_vector[0]
    # timel = time_vector[1]
    avg_time_step = statistics.mean(np.diff(time_vector))
    return avg_time_step


def decompose_from_filename(filename: str):
    df = pd.read_csv(filename)
    time_vec = np.array(df.iloc[:, 0])
    signal = np.array(df.iloc[:, 1])
    return time_vec, signal


def fft_filter(signal_fft, sample_freq, threshold):
    signal_fft_copy = signal_fft.copy()
    indices_to_zero = np.abs(sample_freq) > threshold
    signal_fft_copy[indices_to_zero] = 0
    filtered_signal = np.fft.ifft(signal_fft_copy).real
    return signal_fft_copy, filtered_signal


def feature_extraction(filtered_signal_reshaped, IRF):
    rms_array = np.sqrt(np.mean(filtered_signal_reshaped**2, axis=1))
    skew_array = scipy.stats.skew(filtered_signal_reshaped, axis=1)
    mean_array = np.mean(filtered_signal_reshaped, axis=1)
    std_array = np.std(filtered_signal_reshaped, axis=1, ddof=1)  # highly corr
    amplitude_square_array = np.sum(
        filtered_signal_reshaped**2, axis=1
    )  # highly corr
    root_amplitude_array = (
        np.mean(np.sqrt(np.abs(filtered_signal_reshaped)), axis=1) ** 2
    )  # highly corr
    n = filtered_signal_reshaped.size
    array_feature_size = filtered_signal_reshaped.shape[1]
    label_size = n // array_feature_size
    if IRF == True:
        label_array = np.ones(label_size)  # one means IRF
    else:
        label_array = np.zeros(label_size)  # zero means BF
    feature_array = np.stack(
        (
            rms_array,
            skew_array,
            mean_array,
            std_array,
            amplitude_square_array,
            root_amplitude_array,
            label_array,
        ),
        axis=1,
    )
    return feature_array


# with the following funciton we can select highly correlated features
def correlation_thresh(
    dataset,
    cor_threshold,
):
    col_corr = set()  # set of all names of correlated columns
    corr_matrix = dataset.corr()
    for i in range(len(corr_matrix.columns)):
        for j in range(i):
            if (corr_matrix.iloc[i, j]) > cor_threshold:
                colname = corr_matrix.columns[i]
                col_corr.add(colname)
    return col_corr


def supp_vec_fault_diag(filename, classifier):
    time_vec, signal = decompose_from_filename(filename)
    time_step = find_time_step(time_vec)
    n = time_vec.size
    signal_fft = fftpack.fft(signal)
    sample_freq = fftpack.fftfreq(signal.size, d=time_step)

    _, filtered_signal = fft_filter(signal_fft, sample_freq, threshold)
    filtered_signal_reshaped = np.reshape(filtered_signal, (-1, array_feature_size))

    full_features_arr = feature_extraction(filtered_signal_reshaped, IRF=True)[
        :, :-1
    ]  # last col is label
    feature_array = full_features_arr[
        :, :3
    ]  # first three are the uncorrelated features
    dataset_as_df = pd.DataFrame(feature_array, columns=["rms", "skew", "mean"])
    dataset_as_df_normalized = preprocessing.normalize(dataset_as_df, axis=0)
    y_predict = classifier.predict(dataset_as_df_normalized)
    a = np.ones(y_predict.shape[0])
    confidence = accuracy_score(a, y_predict)
    answer = statistics.mode(y_predict)
    if answer == 1:
        answer = "Inner Race Fault"
    else:
        answer = "Ball Fault"
    return answer, max(confidence, 1 - confidence)


def main():
    ### IRF data
    time_vec, signal = decompose_from_filename(IRF_filename)
    time_step = find_time_step(time_vec)
    signal_fft = fftpack.fft(signal)
    sample_freq = fftpack.fftfreq(signal.size, d=time_step)
    _, filtered_signal = fft_filter(signal_fft, sample_freq, threshold)
    filtered_signal_reshaped = np.reshape(filtered_signal, (-1, array_feature_size))
    feature_array = feature_extraction(filtered_signal_reshaped, IRF=True)
    #### BF data
    _, signal_BF = decompose_from_filename(BF_filename)
    signal_BF_fft = fftpack.fft(signal_BF)
    sample_freq_BF = fftpack.fftfreq(signal_BF.size, d=time_step)
    _, filtered_signal_BF = fft_filter(signal_BF_fft, sample_freq_BF, threshold)
    filtered_signal_BF_reshaped = np.reshape(
        filtered_signal_BF, (-1, array_feature_size)
    )
    feature_array_BF = feature_extraction(filtered_signal_BF_reshaped, IRF=False)
    dataset_array = np.concatenate((feature_array_BF, feature_array))
    dataset_as_df = pd.DataFrame(
        dataset_array,
        columns=[
            "rms",
            "skew",
            "mean",
            "std",
            "amp_squared",
            "root_amp",
            "class",
        ],
    )
    feature_df = dataset_as_df.loc[:, dataset_as_df.columns != "class"]
    # independent variable x
    x = feature_df

    # dependent variable y
    label_df = dataset_as_df["class"]
    y = label_df
    # dividing the data as train/test data
    """
    dataset(256rows)-->train(200rows)/test(56rows)
    train (x,y) #note that x is a two dimensional array while y is a 1 dimensional array
    test (x,y)
    """
    x_train, x_test, y_train, y_test = train_test_split(
        x, y, test_size=0.21875, random_state=4
    )
    corr_features = correlation_thresh(x_train, 0.8)
    x_train_new = x_train.drop(corr_features, axis=1)
    x_test_new = x_test.drop(corr_features, axis=1)

    x_train_new_normalized = preprocessing.normalize(x_train_new, axis=0)
    x_test_new_normalized = preprocessing.normalize(x_test_new, axis=0)
    classifier = svm.SVC(kernel="linear", gamma="auto", C=2)
    classifier.fit(x_train_new_normalized, y_train)
    y_predict = classifier.predict(x_test_new_normalized)
    accuracy = accuracy_score(y_test, y_predict)
    print(accuracy)
    # testing on new data
    filename_test = "dataset used for testing\BF HL 830 rpm.csv"
    supp_vec_fault_diag(filename_test)


if __name__ == "__main__":
    main()
