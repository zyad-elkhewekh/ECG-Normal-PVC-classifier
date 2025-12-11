#keep for imports
import numpy as np
#------------------------------------------------------------------------------
#read data into lists
# norm_train = []
# norm_test = []
# pvc_train = []
# pvc_test = []
# with open("data/Normal_Train.txt") as file:
#     for line in file:
#         norm_train = line.strip().split('|')
# with open("data/Normal_Test.txt") as file:
#     for line in file:
#         norm_test = line.strip().split('|')
# with open("data/PVC_Train.txt") as file:
#     for line in file:
#         pvc_train = line.strip().split('|')
# with open("data/PVC_Test.txt") as file:
#     for line in file:
#         pvc_test = line.strip().split('|')
def read_sig_file(file_path):
    with open(file_path, "r") as f:
        data = f.read().strip().split("|")

    # cast to float
    return [float(x) for x in data if x.strip() != ""]

normal_train=r"data/Normal_Train.txt"
pvc_train=r"data/PVC_Train.txt"
pvc_test=r"data/PVC_Test.txt"
normal_test=r"data/Normal_Test.txt"

normal_train_sig=read_sig_file(normal_train)
pvc_train_sig=read_sig_file(pvc_train)
pvc_test_sig=read_sig_file(pvc_test)
normal_test_sig=read_sig_file(normal_test)
print(len(pvc_train_sig))
#-----------------------------------------------------------------------------

#design bandpass filter
#get attenuation and transition band from where??????
def choose_window(delta_s):
    if delta_s <= 21:
        return "rect", lambda n, N: np.ones(N)
    elif delta_s <= 44:
        return "hann", lambda n, N: 0.5 + 0.5 * np.cos(2*np.pi*n/(N-1))
    elif delta_s <= 53:
        return "hamming", lambda n, N: 0.54 + 0.46 * np.cos(2*np.pi*n/(N-1))
    else:
        return "blackman", lambda n, N: (
            0.42 - 0.5*np.cos(2*np.pi*n/(N-1)) + 0.08*np.cos(4*np.pi*n/(N-1))
        )

def estimate_N(delta_f, window_name):
    if window_name == "rect":
        N = int(np.ceil(0.9 / delta_f))
    elif window_name == "hann":
        N = int(np.ceil(3.1 / delta_f))
    elif window_name == "hamming":
        N = int(np.ceil(3.3 / delta_f))
    else:
        N = int(np.ceil(5.5 / delta_f))

    if N % 2 == 0:
        N += 1  

    return N
#constant???
#fs = 360
#f1 = 0.5
#f2 = 40
def bandpass_filter(f1, f2, fs, N):
    mid = (N - 1) / 2
    omega1 = 2*f1*np.pi
    omega2 = 2*f2*np.pi
    f1 = f1 / (fs/2)
    f2 = f2 / (fs/2)
    hd = np.zeros(N)
    for i in range(N):
        n = i - mid
        if n == 0:
            hd[i] = 2*(f2 - f1)
        else:
            hd[i] = 2 * f2 * np.sin(omega2*n)/(omega2*n) - np.sin(omega1*n)/np.pi*n
    return hd

#apply the filter
def convolve_signals(indices1: list[int], samples1: list[float], 
                    indices2: list[int], samples2: list[float]) -> tuple[list[int], list[float]]:

    N1 = len(samples1)
    N2 = len(samples2)
    
    if N1 == 0 or N2 == 0:
        return [], []
    
    output_length = N1 + N2 - 1
    

    start_idx = int(indices1[0] + indices2[0])
    end_idx = int(indices1[-1] + indices2[-1])
    
    result_samples = [0.0] * output_length
    result_indices = list(range(start_idx, end_idx + 1))
    
    for n in range(output_length):
        for k in range(N1):
            if 0 <= n - k < N2:
                result_samples[n] += samples1[k] * samples2[n - k]
    
    return result_indices, result_samples

def apply_filter_no_specs():
    #drop box to choose from the 4 windows
    #text box to manually add N
    if N % 2 == 0:
        N += 1 
    bandpass_filter('''use constant frequencies''')
    convolve_signals('''indices1 = indices norm_train''', norm_train, '''indices2 dont know''', )




def apply_filter():
    delta_f, delta_s = input("enter transition band and stop band attenution: ")
    if(delta_f == '' and delta_s == ''):
        pass