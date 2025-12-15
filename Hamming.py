import numpy as np
import random as rand
from matplotlib import pyplot as plt
import math
from typing import Literal


def invert_bit(bit: int) -> int:
    bit = bit ^ 1
    return bit


def decode(Received_Message: np.ndarray) -> np.ndarray:
    H = np.array([[1,0,1,0,1,0,1],[0,1,1,0,0,1,1],[0,0,0,1,1,1,1]])
    z = np.dot(Received_Message,np.transpose(H)) % 2
    if np.any(z):
        corrupted_position = (z[0] + 2*z[1] + 4*z[2]) - 1 #z1+2*z2+4*z3
        Received_Message[corrupted_position] = invert_bit(Received_Message[corrupted_position])
    Decoded_Message = np.array([Received_Message[2], Received_Message[4], Received_Message[5], Received_Message[6]])
    return Decoded_Message


def code(Message: np.ndarray) -> np.ndarray:
    G = np.array([[1,1,1,0,0,0,0],[1,0,0,1,1,0,0],[0,1,0,1,0,1,0],[1,1,0,1,0,0,1]])
    Coded_Message = np.dot(Message, G) % 2
    return Coded_Message


def transfer(Coded_Message: np.ndarray, P: float) -> np.ndarray:
    for i, bit in enumerate(Coded_Message):
        if (rand.random() < P):
            Coded_Message[i] = invert_bit(bit) #corrupt 1 bit with probability P
    return Coded_Message


def Count_Mistakes (Start_Message: np.ndarray, Decoded_Message: np.ndarray) -> int:
    k = 0
    for i in range(len(Start_Message)):
        if (Start_Message[i] != Decoded_Message[i]):
            k += 1
    return k


def Code_Transfer_Decode(Message: np.ndarray, P: float) -> np.ndarray: #Code Transfer decode 1 time
    Coded_Message = code(Message)
    Received_Message = transfer(Coded_Message, P)
    Decoded_Message = decode(Received_Message)
    return Decoded_Message


def Code_Transfer_Decode_N(Message: np.ndarray, P: float, N: int) -> tuple[float, np.ndarray]:
    All_mistakes = np.zeros((N,))
    correctly_decoded = 0
    Q = 1-P
    for i in range(0, N): #Simulate Code-Transfer-Decode N times
        Coded_Message = code(Message)
        Received_Message = transfer(Coded_Message, P)
        Uncorrected_Message = np.array([Received_Message[2], Received_Message[4], Received_Message[5], Received_Message[6]])
        Decoded_Message = decode(Received_Message)
        if(np.array_equal(Message, Decoded_Message)):
            correctly_decoded += 1
        All_mistakes[i] = Count_Mistakes(Message, Uncorrected_Message)
    Prob_estimation = correctly_decoded/N
    #Formula for real probability is Q^6*(7*P+Q)
    print(f"Real Probability: {(Q**6) * (7*P + Q)}")
    return Prob_estimation, All_mistakes

    
def Calculate_Shenon(Message: np.ndarray, P: int, mode: Literal["eq","!eq"]) -> float: #Calculates Shenon Entropy
    Q = 1-P
    P_m = 1/2**4
    Decoded_Message = Code_Transfer_Decode(Message, P)
    Entropy_Coded =  -1*P_m*math.log2(P_m)
    if (mode == "eq"):
        if (np.array_equal(Message, Decoded_Message)):
            Entropy_Received = -1*((Q**6 * (7*P + Q)))*math.log2((Q**6 * (7*P + Q)))
        else:
            if 1-(Q**6 * (7*P + Q)) <= 0:
                raise Exception("log|x|, x = 0 or smaller")
            Entropy_Received = -1*(1-(Q**6 * (7*P + Q)))*math.log2(1-(Q**6 * (7*P + Q)))
    else:
        Entropy_Received = -1*(1-(Q**6 * (7*P + Q)))*math.log2(1-(Q**6 * (7*P + Q)))
    return Entropy_Received/Entropy_Coded


def Calculate_Information(Message: np.ndarray, P: int, mode: Literal["eq", "!eq"]) -> float: #Calculates Information for 1 bit of message
    Decoded_Message = Code_Transfer_Decode(Message, P)
    Q = 1-P
    P_m = 0.5
    if mode == "eq":
        Information_Received = 0
        for i in range(1, len(Message)):
            if Message[i] == Message[i-1]:
                P_m *= (1/3)
            else:
                P_m *= (2/3)
            if Message[i] == Decoded_Message[i]:
                Information_Received += -1*math.log2(Q + 7*P*Q**6)
            else:
                Information_Received += -1*math.log2(P*(1-(Q**6 * (7*P + Q))))
        Information_Coded =  (-1*math.log2(P_m))
    else: 
        for i in range(1, len(Message)):
            if Message[i] == Message[i-1]:
                P_m *= (1/3)
            else:
                P_m *= (2/3)
        Information_Coded =  (-1*math.log2(P_m))/4
        Information_Received = -1*math.log2(P*(1-(Q**6 * (7*P + Q))))
    return Information_Received/Information_Coded


def Calculate_Transmission_Efficiency(Message: np.ndarray, m: int, P: int, mode:Literal["diff","intersection"] , n = 7) -> float: #Calculates valuable information for n code length
    while 2**m > (2**n)/(n+1):
        n += 1
    Q = 1-P
    if mode == "diff":
        P_m = (1/2)**m
        P_r = 1-((Q**(n-1))*(Q+n*P))
        I_m = -1*math.log2(P_m)
        I_r = -1*math.log2(P_r)
        I_val = I_r - I_m
    else: 
        Decoded_Message = Code_Transfer_Decode(Message, P)
        I_val = 0
        while 2**m > (2**I)/(I+1):
            I += 1
        for i in range(len(Message)):
            if Decoded_Message[i] == Message[i]:
                I_val += -1*math.log2(Q + n*P*Q**(n-1))
            else: 
                I_val += 0
    return I_val/n


def Find_Zero_P(Efficiency_values: np.ndarray, P_values: np.ndarray) -> float:
    zero_P = None
    for k in range(len(Efficiency_values)-1):
        if Efficiency_values[k] >= 0 and Efficiency_values[k+1] < 0:
            y1 = Efficiency_values[k]
            y2 = Efficiency_values[k+1]
            x1 = P_values[k]
            x2 = P_values[k+1]
            zero_P = x1 - y1 * (x2 - x1) / (y2 - y1)
    if zero_P is None:
        return print("No zero crossing found in the provided P range.")
    else: 
        return zero_P


def Entropy_Information_Efficiency_Dependencies(Message: np.ndarray, P_values: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]: #Gives arrays for Graphs of entropy, information/bit, valuable information/n
    Efficiency_values = np.zeros((np.size(P_values),))
    Shenon_Entropy_Values = np.zeros((np.size(P_values),))
    Information_Values = np.zeros((np.size(P_values),))
    I = 0
    for i,P in enumerate(P_values, 1):
        while 2**i > (2**I)/(I+1):
                I += 1
        Shenon_Entropy_Values[i-1] = Calculate_Shenon(Message, P, '!eq')
        Information_Values[i-1] = Calculate_Information(Message, P, '!eq')
        Efficiency_values[i-1] = Calculate_Transmission_Efficiency(Message, 4, P, 'diff')
    
    return Shenon_Entropy_Values, Information_Values, Efficiency_values


if __name__ == '__main__':
    N = 100000
    P_values = np.arange(.01 , 1 , .01)
    Message = np.array([1,0,1,1])
    fig, ax = plt.subplots(2, 2)
    
    P_estimation, M_array = Code_Transfer_Decode_N(Message, P_values[rand.randint(0 , np.size(P_values)-1)], N)
    Shenon_Entropy_Values, Information_Values, Efficiency_values = Entropy_Information_Efficiency_Dependencies(Message, P_values)
    
    print(f"Estimated Probability: {P_estimation}")
    #Mistakes Graph
    unique_k, counts = np.unique(M_array, return_counts=True)
    bars = ax[0][0].bar(unique_k, counts, align='center', width=0.6)
    ax[0][0].set_yscale('log')
    ax[0][0].set_xticks(unique_k)
    ax[0][0].set_xlabel('(k)')
    ax[0][0].set_ylabel('Quantity(Log Scale)')
    ax[0][0].bar_label(bars)
    #Entropy Graph
    ax[0][1].set_xlabel('P')
    ax[0][1].set_ylabel('H(m*)/H(m)')
    ax[0][1].plot(P_values, Shenon_Entropy_Values, linewidth = 2)
    #Information Graph
    ax[1][0].set_xlabel('P')
    ax[1][0].set_ylabel('I(m*)/I(m)')
    ax[1][0].plot(P_values, Information_Values, linewidth = 2)
    #Efficiency Graph
    ax[1][1].set_xlabel('P')
    ax[1][1].set_ylabel('I_val/I')
    ax[1][1].plot(P_values, Efficiency_values, linewidth = 2)
    zero_P = Find_Zero_P(Efficiency_values, P_values)
    if zero_P:
        ax[1][1].plot(zero_P, 0, 'ro', markersize=8, label='Zero Efficiency')
        ax[1][1].annotate(f'P ≈ {zero_P:.4f}', xy=(zero_P, 0), xytext=(zero_P + 0.1, 0.2),arrowprops=dict(facecolor='red', shrink=0.05))
    plt.show()
